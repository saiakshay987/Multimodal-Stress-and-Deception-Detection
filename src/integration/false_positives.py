#!/usr/bin/env python3
"""Do deception false alarms concentrate among truthful clips with high predicted stress?

Example (run from the repository root):
    python src/integration/false_positives.py

Inputs (all read-only)
    results/integration/multitask_predictions.csv      (a) plain two-head model, test-fold predictions
    results/integration/stress_aware_predictions.csv   (b) stress-conditioned model, test-fold predictions
    results/stress/stress_scores_on_deception.csv      P(stress) per Bag-of-Lies / Real-Life Trial clip
    features/<dataset>/metadata.csv                    group_id and fold of every clip

Predicted stress: --stress-scores if that file exists (one fixed score per clip, from the stress model trained on
MAS only, so it never saw a deception label). Otherwise the stress head's probability for deception clips is taken
from the two prediction files (rows with task == stress on bol / rlt). The source used is printed and written to
the output. Each deception clip is a test clip in every seed (3 seeds x 5 folds = 15 runs per model).

Method
    * Truthful clips (label 0) are split per dataset at the median predicted stress of that dataset's truthful
      clips: high = stress > median, low = stress <= median. The split is by clip, so it is identical for both
      models and for every run. "bol+rlt" pools the two per-dataset splits.
    * False-positive rate = share of truthful test predictions with P(deceptive) >= 0.5 (same threshold as
      src/common/train.compute_metrics). Counts are over clip x run predictions (each clip is tested once per
      seed), so they are not independent samples. Run-level mean +- std are also reported.
    * High minus low FPR gets a 95 % percentile CI from a bootstrap over people / source videos (group_id), which
      respects the dependence between clips of one person and between seeds.
    * Correlation: point-biserial r and Mann-Whitney U between predicted stress and the deception label, one
      value per clip (the stress score does not change between runs).
    * Plain vs stress-conditioned on the high-stress truthful band: paired by (fold, seed), diff = conditioned -
      plain FPR; runs whose fold has no high-band clip are dropped. Wilcoxon signed-rank p is indicative only
      (the 15 runs share clips, and the 3 seeds share folds).

Outputs
    results/integration/false_positives_by_stress.csv          tidy table, one row per result (see `section`)
    results/integration/false_positives_by_stress_runs.csv     FPR of every (dataset, fold, seed, model, band)
    results/integration/false_positives_by_stress_summary.txt  one-paragraph summary (numbers filled in by this script)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = ("bol", "rlt")
POOLED = "bol+rlt"
MODELS = ("plain", "stress_conditioned")
THRESHOLD = 0.5


# ----------------------------------------------------------------------------------------------- loading
def load_stress(args, pred_tables):
    """(clip_id, dataset, stress_prob) for bol / rlt and the name of the source."""
    path = Path(args.stress_scores)
    if path.exists():
        s = pd.read_csv(path)[["clip_id", "dataset", "stress_prob"]]
        return s[s["dataset"].isin(DATASETS)], str(path)
    parts = []
    for name, df in pred_tables.items():
        d = df[(df["task"] == "stress") & df["dataset"].isin(DATASETS)]
        parts.append(d.groupby(["clip_id", "dataset"], as_index=False)["prob"].mean().rename(columns={"prob": name}))
    if not parts or any(p.empty for p in parts):
        raise SystemExit(f"{path} does not exist and the prediction files hold no stress-head rows for bol / rlt.")
    s = parts[0].merge(parts[1], on=["clip_id", "dataset"])
    s["stress_prob"] = s[list(pred_tables)].mean(axis=1)
    return s[["clip_id", "dataset", "stress_prob"]], "stress head of " + " and ".join(pred_tables)


def load_predictions(path):
    df = pd.read_csv(path)
    return df


def build_frame(args):
    raw = {"plain": load_predictions(args.plain), "stress_conditioned": load_predictions(args.conditioned)}
    stress, source = load_stress(args, raw)
    meta = pd.concat([pd.read_csv(Path(args.features_root) / d / "metadata.csv") for d in DATASETS])
    meta = meta[["clip_id", "dataset", "group_id", "fold"]].rename(columns={"fold": "meta_fold"})
    meta["group"] = meta["dataset"] + ":" + meta["group_id"].astype(str)

    frames = []
    for model, df in raw.items():
        d = df[(df["task"] == "deception") & df["dataset"].isin(DATASETS)].copy()
        d["model"] = model
        frames.append(d[["clip_id", "dataset", "label", "prob", "fold", "seed", "model"]])
    pred = pd.concat(frames, ignore_index=True)

    pred = pred.merge(meta, on=["clip_id", "dataset"], how="left", validate="m:1")
    assert pred["group"].notna().all(), "prediction clips missing from metadata.csv"
    assert (pred["fold"] == pred["meta_fold"]).all(), "fold in predictions differs from metadata.csv"
    pred = pred.merge(stress, on=["clip_id", "dataset"], how="left", validate="m:1")
    missing = pred.loc[pred["stress_prob"].isna(), "clip_id"].nunique()
    assert missing == 0, f"{missing} prediction clips have no stress score"

    # both models must have been tested on the same clips in every (dataset, fold, seed)
    sets = {m: {k: frozenset(g["clip_id"]) for k, g in pred[pred["model"] == m].groupby(["dataset", "fold", "seed"])}
            for m in MODELS}
    assert sets["plain"] == sets["stress_conditioned"], "plain and stress-conditioned test clips differ"
    for m in MODELS:  # every clip once per seed
        assert (pred[pred["model"] == m].groupby(["dataset", "clip_id", "seed"]).size() == 1).all()
    return pred, source


def assign_bands(pred):
    """Per-dataset median split of the truthful clips; deceptive clips get band 'deceptive'."""
    clips = pred.drop_duplicates(["dataset", "clip_id"])
    truthful = clips[clips["label"] == 0]
    median = truthful.groupby("dataset")["stress_prob"].median()
    pred = pred.copy()
    pred["median"] = pred["dataset"].map(median)
    pred["band"] = np.where(pred["label"] == 1, "deceptive",
                            np.where(pred["stress_prob"] > pred["median"], "high", "low"))
    return pred, median


# ----------------------------------------------------------------------------------------------- results
def with_pooled(df):
    return pd.concat([df, df.assign(dataset=POOLED)], ignore_index=True)


def run_table(truthful):
    """FPR per (dataset, fold, seed, model, band); band 'all' = every truthful clip."""
    t = with_pooled(truthful)
    t = pd.concat([t, t.assign(band="all")], ignore_index=True)
    r = (t.groupby(["dataset", "fold", "seed", "model", "band"])
         .agg(n_pred=("fp", "size"), n_fp=("fp", "sum")).reset_index())
    r["fpr"] = r["n_fp"] / r["n_pred"]
    return r


def band_rows(truthful, runs):
    t = with_pooled(truthful)
    t = pd.concat([t, t.assign(band="all")], ignore_index=True)
    rows = []
    for (ds, model, band), g in t.groupby(["dataset", "model", "band"], sort=False):
        rr = runs[(runs["dataset"] == ds) & (runs["model"] == model) & (runs["band"] == band)]
        rows.append({"section": "fpr_by_band", "dataset": ds, "model": model, "band": band,
                     "n_clips": g["clip_id"].nunique(), "n_groups": g["group"].nunique(),
                     "n_predictions": len(g), "n_fp": int(g["fp"].sum()), "fpr": g["fp"].mean(),
                     "n_runs": len(rr), "fpr_run_mean": rr["fpr"].mean(), "fpr_run_std": rr["fpr"].std(ddof=1)})
    return rows


def bootstrap_high_minus_low(truthful, n_boot, seed):
    """High-minus-low FPR per dataset and model with a 95 % cluster bootstrap CI over group_id."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in (*DATASETS, POOLED):
        t = truthful if ds == POOLED else truthful[truthful["dataset"] == ds]
        groups = np.sort(t["group"].unique())
        idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))  # same resamples for both models
        for model in MODELS:
            m = t[t["model"] == model]
            cnt = {b: m[m["band"] == b].groupby("group")["fp"].agg(["sum", "size"]).reindex(groups, fill_value=0)
                   for b in ("high", "low")}
            rate = {}
            for b in cnt:
                fp, n = cnt[b]["sum"].to_numpy(float), cnt[b]["size"].to_numpy(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rate[b] = (fp[idx].sum(1) / n[idx].sum(1), fp.sum() / n.sum())
            diff = rate["high"][0] - rate["low"][0]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            rows.append({"section": "high_minus_low_fpr", "dataset": ds, "model": model,
                         "n_groups": len(groups), "diff": rate["high"][1] - rate["low"][1],
                         "ci_low": lo, "ci_high": hi, "n_boot": n_boot})
    return rows


def correlation_rows(pred):
    clips = pred.drop_duplicates(["dataset", "clip_id"])
    rows = []
    for ds in (*DATASETS, POOLED):
        c = clips if ds == POOLED else clips[clips["dataset"] == ds]
        y, s = c["label"].to_numpy(), c["stress_prob"].to_numpy()
        r, p_r = stats.pointbiserialr(y, s)
        u, p_u = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
        rows.append({"section": "stress_vs_label", "dataset": ds, "n_clips": len(c), "n_deceptive": int(y.sum()),
                     "n_truthful": int((y == 0).sum()), "stress_mean_deceptive": s[y == 1].mean(),
                     "stress_mean_truthful": s[y == 0].mean(), "pointbiserial_r": r, "pointbiserial_p": p_r,
                     "mannwhitney_U": u, "mannwhitney_p": p_u,
                     "auc_stress_predicts_deceptive": u / ((y == 1).sum() * (y == 0).sum())})
    return rows


def paired_rows(runs):
    rows = []
    for ds in (*DATASETS, POOLED):
        for band in ("high", "low"):
            r = runs[(runs["dataset"] == ds) & (runs["band"] == band)]
            w = r.pivot(index=["fold", "seed"], columns="model", values="fpr").dropna()
            n = r.pivot(index=["fold", "seed"], columns="model", values="n_pred").loc[w.index]
            assert (n["plain"] == n["stress_conditioned"]).all()
            d = (w["stress_conditioned"] - w["plain"]).to_numpy()
            try:
                p = stats.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else np.nan
            except ValueError:
                p = np.nan
            all_runs = r["fold"].nunique() * r["seed"].nunique()
            rows.append({"section": f"paired_{band}_band", "dataset": ds, "band": band, "n_pairs": len(d),
                         "n_runs_dropped_empty_band": all_runs - len(d),
                         "plain_fpr_mean": w["plain"].mean(), "stress_conditioned_fpr_mean": w["stress_conditioned"].mean(),
                         "diff_mean": d.mean(), "diff_std": d.std(ddof=1) if len(d) > 1 else np.nan,
                         "n_conditioned_lower": int((d < 0).sum()), "n_conditioned_higher": int((d > 0).sum()),
                         "n_equal": int((d == 0).sum()), "wilcoxon_p": p})
    return rows


# ----------------------------------------------------------------------------------------------- summary
def pct(x):
    return f"{100 * x:.1f}%"


def write_summary(res, source, median, path):
    def one(section, ds, **kw):
        q = res[(res["section"] == section) & (res["dataset"] == ds)]
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0]

    def p_txt(p):
        return "p = n/a" if pd.isna(p) else (f"p = {p:.3f}" if p >= 0.001 else "p < 0.001")

    cor = {ds: one("stress_vs_label", ds) for ds in (*DATASETS, POOLED)}
    cor_txt = "; ".join(f"{ds} r = {cor[ds].pointbiserial_r:+.2f} ({p_txt(cor[ds].pointbiserial_p)}), Mann-Whitney "
                        f"{p_txt(cor[ds].mannwhitney_p)}, AUC {cor[ds].auc_stress_predicts_deceptive:.2f}" for ds in DATASETS)

    def fpr_txt(model, ds):
        lo, hi = one("fpr_by_band", ds, model=model, band="low"), one("fpr_by_band", ds, model=model, band="high")
        return (f"low {int(lo.n_fp)}/{int(lo.n_predictions)} ({pct(lo.fpr)}) vs high {int(hi.n_fp)}/"
                f"{int(hi.n_predictions)} ({pct(hi.fpr)})")

    gap = {m: one("high_minus_low_fpr", POOLED, model=m) for m in MODELS}

    def verdict(m):
        g = gap[m]
        if g.ci_low > 0:
            return f"higher in the high-stress band (difference {100 * g['diff']:+.1f} points, 95% CI {100 * g.ci_low:+.1f} to {100 * g.ci_high:+.1f})"
        if g.ci_high < 0:
            return f"LOWER in the high-stress band (difference {100 * g['diff']:+.1f} points, 95% CI {100 * g.ci_low:+.1f} to {100 * g.ci_high:+.1f})"
        return (f"not reliably different between bands (difference {100 * g['diff']:+.1f} points, "
                f"95% CI {100 * g.ci_low:+.1f} to {100 * g.ci_high:+.1f}, which includes 0)")

    pr = {ds: one("paired_high_band", ds) for ds in (*DATASETS, POOLED)}
    pooled = pr[POOLED]
    conc = ("do not show a reliable concentration of false alarms among stressed truthful speakers"
            if not (gap["plain"].ci_low > 0 or gap["plain"].ci_high < 0)
            else ("do concentrate among stressed truthful speakers" if gap["plain"].ci_low > 0
                  else "are, if anything, LESS frequent among stressed truthful speakers"))
    text = (
        f"Predicted stress came from {source} (one fixed score per clip; high/low split at each dataset's truthful-clip "
        f"median: bol {median['bol']:.3f}, rlt {median['rlt']:.3f}, so the bands are relative). Predicted stress vs the "
        f"deception label (per clip; point-biserial r with p, Mann-Whitney U p, and the AUC of stress as a deception "
        f"score): {cor_txt}. "
        f"For the plain two-head model, false positives among truthful test predictions were, per dataset, "
        f"bol {fpr_txt('plain', 'bol')} and rlt {fpr_txt('plain', 'rlt')}; pooled over both, the FPR is {verdict('plain')}. "
        f"For the stress-conditioned model the counts were bol {fpr_txt('stress_conditioned', 'bol')} and rlt "
        f"{fpr_txt('stress_conditioned', 'rlt')}; pooled, the FPR is {verdict('stress_conditioned')}. "
        f"So the data {conc}. Run by run on the high-stress truthful band (same folds and seeds), the conditioned model's "
        f"FPR minus the plain model's was {100 * pooled.diff_mean:+.1f} +/- {100 * pooled.diff_std:.1f} points over "
        f"{int(pooled.n_pairs)} pooled runs (lower in {int(pooled.n_conditioned_lower)}, higher in "
        f"{int(pooled.n_conditioned_higher)}, equal in {int(pooled.n_equal)}; Wilcoxon {p_txt(pooled.wilcoxon_p)}); "
        f"per dataset: bol {100 * pr['bol'].diff_mean:+.1f} points ({p_txt(pr['bol'].wilcoxon_p)}), "
        f"rlt {100 * pr['rlt'].diff_mean:+.1f} points ({p_txt(pr['rlt'].wilcoxon_p)}). "
        f"Caveats: clips of the same person and the three seeds are not independent (hence the group bootstrap), the "
        f"stress score is itself a noisy MAS-trained model applied to other datasets, and the bands are small "
        f"(bol {int(one('fpr_by_band', 'bol', model='plain', band='high').n_clips)} and rlt "
        f"{int(one('fpr_by_band', 'rlt', model='plain', band='high').n_clips)} high-stress truthful clips).")
    Path(path).write_text(text + "\n", encoding="utf-8")
    return text


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plain", default="results/integration/multitask_predictions.csv")
    ap.add_argument("--conditioned", default="results/integration/stress_aware_predictions.csv")
    ap.add_argument("--stress-scores", default="results/stress/stress_scores_on_deception.csv")
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out", default="results/integration/false_positives_by_stress.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    np.random.seed(a.seed)

    pred, source = build_frame(a)
    print(f"stress source: {source}")
    pred, median = assign_bands(pred)
    pred["fp"] = (pred["prob"] >= THRESHOLD).astype(int)
    truthful = pred[pred["label"] == 0]
    print("truthful-clip median stress:", {k: round(float(v), 4) for k, v in median.items()})

    runs = run_table(truthful)
    rows = (band_rows(truthful, runs) + bootstrap_high_minus_low(truthful, a.n_boot, a.seed)
            + correlation_rows(pred) + paired_rows(runs))
    res = pd.DataFrame(rows)
    res.insert(1, "stress_source", source)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    runs.to_csv(out.with_name(out.stem + "_runs.csv"), index=False)
    text = write_summary(res, source, median, out.with_name(out.stem + "_summary.txt"))

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        for section in res["section"].unique():
            t = res[res["section"] == section].dropna(axis=1, how="all").drop(columns=["section", "stress_source"])
            print(f"\n[{section}]\n{t.to_string(index=False)}")
    print("\nSUMMARY\n" + text)
    print(f"\nwrote {out}, {out.with_name(out.stem + '_runs.csv')}, {out.with_name(out.stem + '_summary.txt')}")


if __name__ == "__main__":
    main()
