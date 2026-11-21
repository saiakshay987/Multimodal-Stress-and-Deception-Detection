#!/usr/bin/env python3
"""Error analysis and a "no sentiment-only clips" ablation for stress on MAS.

Three subcommands (all paths are arguments; nothing under data/ or src/common/ is touched):

    python src/stress/analysis.py errors
    python src/stress/analysis.py variant
    python src/stress/run_all.py --features-root features/_variants/nosent --suffix _nosent --modes audio
    python src/stress/analysis.py compare

errors   Pick the mode with the highest macro_f1_mean in <results-dir>/deep.csv, average its test
         probabilities over seeds PER CLIP, threshold at 0.5, and join with features/mas/metadata.csv.
         Every stressed clip (label 1) is tagged ``sentiment_only`` when mas_labels.strip() is exactly
         "audio stress", else ``nonverbal``. Prints the confusion matrix, the stress recall (overall /
         sentiment_only / nonverbal) and the false-positive rate. Writes
             <results-dir>/errors.csv          the 5 most confident false positives and 5 false negatives
             <results-dir>/errors_summary.csv  the recall / FPR numbers
variant  Builds <variant-root>/mas/: metadata.csv without the clips whose mas_labels.strip() is exactly
         "audio stress" (all other columns, in particular ``fold``, copied verbatim, no re-split) and one
         absolute symlink <clip_id>.npz per kept clip pointing into <features-root>/mas. run_all.py then
         works unchanged with ``--features-root <variant-root> --suffix _nosent --modes audio``.
compare  Audio row of deep.csv vs deep_nosent.csv, plus the ORIGINAL audio model scored only on the clips
         kept in the variant (per run = fold x seed, then mean ± std). Writes
             <results-dir>/nosent_comparison.csv
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

SEED = 42
SENTIMENT_ONLY_LABEL = "audio stress"
THRESHOLD = 0.5
METRICS = ("accuracy", "macro_f1", "roc_auc")
DATASET = "mas"
META_COLS = ["clip_id", "label", "fold", "group_id", "video_path", "mas_labels", "duration_s"]


# ------------------------------------------------------------------------------------ helpers
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def read_csv_checked(path, **kw):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"error: {path} not found")
    return pd.read_csv(path, **kw)


def is_sentiment_only(mas_labels):
    """Exact match on the whole (stripped) label string; never a substring test."""
    return mas_labels.fillna("").astype(str).str.strip() == SENTIMENT_ONLY_LABEL


def load_mas_metadata(features_root):
    """features/mas/metadata.csv with typed columns (clip_id / group_id as str)."""
    df = read_csv_checked(Path(features_root) / DATASET / "metadata.csv",
                          dtype={"clip_id": str, "group_id": str})
    if df["clip_id"].duplicated().any():
        raise SystemExit("error: clip_id is not unique in metadata.csv")
    df["mas_labels"] = df["mas_labels"].fillna("").astype(str)
    return df


def ratio(num, den):
    return float(num) / den if den else float("nan")


# ------------------------------------------------------------------------------------- errors
def best_mode(deep_csv):
    deep = read_csv_checked(deep_csv)
    row = deep.loc[deep["macro_f1_mean"].idxmax()]
    print(f"Mode with the highest macro_f1_mean in {deep_csv}: {row['mode']} "
          f"(macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f})")
    return str(row["mode"])


def cmd_errors(args):
    res = Path(args.results_dir)
    mode = best_mode(res / "deep.csv")

    pred = read_csv_checked(res / "deep_predictions.csv", dtype={"clip_id": str})
    pred = pred[pred["mode"] == mode]
    if pred.empty:
        raise SystemExit(f"error: no rows for mode {mode!r} in deep_predictions.csv")
    n_seeds = pred["seed"].nunique()
    # one row per clip: average prob over seeds FIRST, threshold afterwards
    per_clip = (pred.groupby("clip_id", sort=False)
                .agg(prob=("prob", "mean"), y_true=("y_true", "first"),
                     n_pred=("prob", "size"), n_y=("y_true", "nunique"), n_fold=("fold", "nunique"))
                .reset_index())
    if (per_clip["n_y"] != 1).any() or (per_clip["n_fold"] != 1).any():
        raise SystemExit("error: a clip has inconsistent y_true or appears in several folds")
    print(f"{len(pred)} prediction rows -> {len(per_clip)} clips, "
          f"{per_clip['n_pred'].min()}-{per_clip['n_pred'].max()} predictions per clip (seeds: {n_seeds})")

    meta = load_mas_metadata(args.features_root)
    df = per_clip.drop(columns=["n_pred", "n_y", "n_fold"]).merge(meta[META_COLS], on="clip_id", how="left",
                                                                  validate="one_to_one")
    if df["label"].isna().any():
        raise SystemExit(f"error: {int(df['label'].isna().sum())} predicted clips are not in metadata.csv")
    if (df["label"].astype(int) != df["y_true"].astype(int)).any():
        raise SystemExit("error: y_true in the predictions disagrees with label in metadata.csv")
    if len(df) != len(meta):
        print(f"warning: {len(meta) - len(df)} metadata clips have no prediction")
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["pred"] = (df["prob"] >= THRESHOLD).astype(int)

    # tag stressed clips only; unstressed clips keep an empty tag
    df["tag"] = ""
    stressed = df["label"] == 1
    df.loc[stressed, "tag"] = np.where(is_sentiment_only(df.loc[stressed, "mas_labels"]),
                                       "sentiment_only", "nonverbal")

    tp = int(((df["label"] == 1) & (df["pred"] == 1)).sum())
    fn = int(((df["label"] == 1) & (df["pred"] == 0)).sum())
    fp = int(((df["label"] == 0) & (df["pred"] == 1)).sum())
    tn = int(((df["label"] == 0) & (df["pred"] == 0)).sum())
    print(f"\nConfusion matrix, mode {mode}, seed-averaged prob >= {THRESHOLD} (rows = true, columns = predicted)")
    print(pd.DataFrame([[tn, fp], [fn, tp]], index=["true no stress (0)", "true stress (1)"],
                       columns=["pred 0", "pred 1"]).to_string())

    summary = [("stress_recall_overall", tp, tp + fn)]
    for tag in ("sentiment_only", "nonverbal"):
        sub = df[df["tag"] == tag]
        summary.append((f"stress_recall_{tag}", int((sub["pred"] == 1).sum()), len(sub)))
    summary.append(("false_positive_rate", fp, fp + tn))
    summary = pd.DataFrame([{"mode": mode, "metric": m, "numerator": n, "denominator": d,
                             "value": ratio(n, d)} for m, n, d in summary])
    print("\nRecall / false-positive rate")
    for r in summary.itertuples():
        print(f"  {r.metric:<30s} {r.numerator:>4d} / {r.denominator:<4d} = {r.value:.3f}")

    fps = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("prob", ascending=False).head(5)
    fns = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("prob", ascending=True).head(5)
    if len(fps) < 5 or len(fns) < 5:
        print(f"warning: only {len(fps)} false positives and {len(fns)} false negatives exist")
    errors = pd.concat([fps.assign(error_type="false_positive"), fns.assign(error_type="false_negative")])
    cols = ["clip_id", "group_id", "fold", "label", "prob", "mas_labels", "tag", "duration_s",
            "video_path", "error_type"]
    errors = errors[cols]
    print("\nMost confident mistakes (video_path is relative to data/: open data/<video_path>)")
    print(errors.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    res.mkdir(parents=True, exist_ok=True)
    errors.to_csv(res / "errors.csv", index=False)
    summary.to_csv(res / "errors_summary.csv", index=False)
    print(f"\nWrote {res / 'errors.csv'}\nWrote {res / 'errors_summary.csv'}")


# ------------------------------------------------------------------------------------ variant
def stress_table(df):
    t = df.groupby("fold")["label"].agg(clips="size", stressed="sum")
    t.loc["all"] = [len(df), int(df["label"].sum())]
    t["stress_share"] = t["stressed"] / t["clips"]
    return t.astype({"clips": int, "stressed": int})


def cmd_variant(args):
    src_dir = Path(args.features_root) / DATASET
    out_dir = Path(args.variant_root) / DATASET
    # all cells as text so every kept row is written back exactly as it was read
    raw = read_csv_checked(src_dir / "metadata.csv", dtype=str, keep_default_na=False)
    for c in ("clip_id", "label", "fold", "mas_labels"):
        if c not in raw.columns:
            raise SystemExit(f"error: metadata.csv has no column {c!r}")
    drop = is_sentiment_only(raw["mas_labels"])
    kept = raw[~drop]
    typed = lambda d: d.assign(label=d["label"].astype(int), fold=d["fold"].astype(int))  # noqa: E731

    print(f"Removing clips whose mas_labels.strip() == {SENTIMENT_ONLY_LABEL!r}: {int(drop.sum())} of {len(raw)}")
    removed_labels = raw.loc[drop, "label"].value_counts().to_dict()
    print(f"  labels of the removed clips: {removed_labels}")
    if removed_labels.get("0", 0):
        print("  warning: some removed clips have label 0")
    before, after = stress_table(typed(raw)), stress_table(typed(kept))
    print("\nBefore (stress share per fold)")
    print(before.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nAfter")
    print(after.to_string(float_format=lambda v: f"{v:.3f}"))
    for fold in before.index:
        if fold != "all" and (after.loc[fold, "stressed"] == 0 or after.loc[fold, "stressed"] == after.loc[fold, "clips"]):
            print(f"warning: fold {fold} has a single class after the removal")

    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_dir / "metadata.csv", index=False)
    keep_names = set()
    for clip_id in kept["clip_id"]:
        name = f"{clip_id}.npz"  # clip ids contain dots: never Path.with_suffix
        target = Path(os.path.abspath(src_dir / name))
        if not target.is_file():
            raise SystemExit(f"error: feature file {target} does not exist")
        link = out_dir / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise SystemExit(f"error: {link} exists and is not a symlink; refusing to overwrite it")
        link.symlink_to(target)
        keep_names.add(name)
    stale = [p for p in out_dir.glob("*.npz") if p.is_symlink() and p.name not in keep_names]
    for p in stale:  # links left by an earlier run with different removal rules
        p.unlink()
    print(f"\nWrote {out_dir / 'metadata.csv'} ({len(kept)} rows) and {len(keep_names)} symlinks"
          + (f"; removed {len(stale)} stale symlinks" if stale else ""))

    # the same call run_all.py makes, so a failure shows up here and not hours into training
    from data import load_metadata
    check = load_metadata(args.variant_root, [DATASET], task="stress")
    assert len(check) == len(kept)
    print(f"load_metadata({args.variant_root!r}, ['mas'], task='stress') -> {len(check)} clips, "
          f"{check['group_key'].nunique()} groups: OK")
    print("\nNext: python src/stress/run_all.py --features-root " f"{args.variant_root} --suffix _nosent --modes audio")


# ------------------------------------------------------------------------------------ compare
def per_run_metrics(pred, clip_ids=None):
    """accuracy / macro-F1 / ROC-AUC of every (fold, seed) run, optionally on a subset of clips."""
    from train import compute_metrics
    if clip_ids is not None:
        pred = pred[pred["clip_id"].isin(clip_ids)]
    rows = []
    for (fold, seed), g in pred.groupby(["fold", "seed"], sort=True):
        rows.append({"fold": fold, "seed": seed, "n_test": len(g),
                     **compute_metrics(g["y_true"].to_numpy(), g["prob"].to_numpy(), THRESHOLD)})
    return pd.DataFrame(rows)


def summary_row(name, n_runs, n_test, stats):
    row = {"name": name, "n_runs": n_runs, "n_test_clips_mean": n_test}
    for m in METRICS:
        mean, std = stats[m]
        row[f"{m}_mean"], row[f"{m}_std"] = mean, std
        row[m] = fmt(mean, std)
    return row


def cmd_compare(args):
    res = Path(args.results_dir)

    def audio_row(path):
        deep = read_csv_checked(path)
        sel = deep[deep["mode"] == "audio"]
        if sel.empty:
            raise SystemExit(f"error: no audio row in {path}")
        return sel.iloc[0]

    orig, nosent = audio_row(res / "deep.csv"), audio_row(res / f"deep{args.nosent_suffix}.csv")
    from_csv = lambda r: {m: (r[f"{m}_mean"], r[f"{m}_std"]) for m in METRICS}  # noqa: E731

    pred = read_csv_checked(res / "deep_predictions.csv", dtype={"clip_id": str})
    pred = pred[pred["mode"] == "audio"]
    kept_meta = read_csv_checked(Path(args.variant_root) / DATASET / "metadata.csv", dtype={"clip_id": str})
    kept_ids = set(kept_meta["clip_id"])
    missing = kept_ids - set(pred["clip_id"])
    if missing:
        print(f"warning: {len(missing)} kept clips have no prediction of the original audio model")

    all_runs = per_run_metrics(pred)
    kept_runs = per_run_metrics(pred, kept_ids)
    # sanity check: per-run metrics recomputed from the predictions must reproduce deep.csv
    for m in METRICS:
        diff = abs(all_runs[m].mean() - orig[f"{m}_mean"])
        if diff > 1e-6:
            print(f"warning: recomputed {m} over all clips differs from deep.csv by {diff:.2e}")
    stats = lambda runs: {m: (runs[m].mean(), runs[m].std()) for m in METRICS}  # noqa: E731

    nosent_runs_csv = res / f"deep_runs{args.nosent_suffix}.csv"
    n_nosent = float("nan")
    if nosent_runs_csv.is_file():
        r = pd.read_csv(nosent_runs_csv)
        n_nosent = float(r.loc[r["mode"] == "audio", "n_test"].mean())

    rows = [
        summary_row("audio_original_all", int(orig["n_runs"]), all_runs["n_test"].mean(), from_csv(orig)),
        summary_row("audio_original_on_kept_clips", len(kept_runs), kept_runs["n_test"].mean(), stats(kept_runs)),
        summary_row("audio_nosent_retrained", int(nosent["n_runs"]), n_nosent, from_csv(nosent)),
    ]
    out = pd.DataFrame(rows)
    print("\nAudio model on MAS stress, mean ± std over runs (fold x seed)")
    print(out[["name", "n_runs", "n_test_clips_mean", "accuracy", "macro_f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nFair comparison: audio_original_on_kept_clips vs audio_nosent_retrained "
          "(same test clips per fold; only the training set differs).")
    for m in METRICS:
        d = out.loc[2, f"{m}_mean"] - out.loc[1, f"{m}_mean"]
        print(f"  {m:<9s} retrained - original-on-kept = {d:+.3f}")
    out.to_csv(res / "nosent_comparison.csv", index=False)
    print(f"\nWrote {res / 'nosent_comparison.csv'}")


# --------------------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features", help="folder that contains mas/")
    ap.add_argument("--results-dir", default="results/stress")
    ap.add_argument("--variant-root", default="features/_variants/nosent",
                    help="features root of the no-sentiment variant (contains mas/)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("errors", help="confusion matrix, recall by tag and the most confident mistakes")
    sub.add_parser("variant", help="build the variant without sentiment-only clips")
    p = sub.add_parser("compare", help="original vs no-sentiment audio model")
    p.add_argument("--nosent-suffix", default="_nosent", help="--suffix used for run_all.py on the variant")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(SEED)
    {"errors": cmd_errors, "variant": cmd_variant, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
