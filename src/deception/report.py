#!/usr/bin/env python3
"""Collect the runs written by run_all.py into results/deception/deep.csv and fusion_comparison.png.

One row per (setting, test dataset, mode). For the pooled setting the predictions of the test
folds are split by dataset, so Bag-of-Lies and Real-Life Trial are reported separately.

Columns of deep.csv
    n_runs                 runs (folds x seeds); cross-dataset: seeds
    n_clips                distinct test clips (one prediction per clip and seed)
    accuracy / macro_f1 / roc_auc  _mean and _std over the runs (metrics recomputed per run on
                           the clips of that dataset only, threshold 0.5)
    p_binom                two-sided binomial test of H0: P(correct) = 0.5, with k = correct
                           predictions averaged over seeds (rounded), n = n_clips
    p_binom_worst_seed     the largest of the per-seed p-values (conservative)
Caveat: clips of one person are not independent, so these p-values are optimistic.

    python src/deception/report.py [--results-dir results/deception]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from train import MODES, compute_metrics  # noqa: E402

# (setting, test dataset, panel title) in plotting order
PANELS = [("within_bol", "bol", "Within Bag-of-Lies"), ("within_rlt", "rlt", "Within Real-Life Trial"),
          ("pooled", "bol", "Pooled -> Bag-of-Lies"), ("pooled", "rlt", "Pooled -> Real-Life Trial"),
          ("rlt_to_bol", "bol", "Train RLT, test Bag-of-Lies"), ("bol_to_rlt", "rlt", "Train BoL, test Real-Life Trial")]
# categorical slots 1-5 of the reference palette (fixed order, one per mode); light / dark
COLORS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
          "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]}


def per_run_metrics(results_dir):
    """DataFrame: one row per (setting, test_dataset, mode, fold, seed) with metrics and k/n."""
    rows = []
    for setting in dict.fromkeys(p[0] for p in PANELS):
        path = results_dir / "runs" / f"{setting}_predictions.csv"
        if not path.is_file():
            print(f"note: {path} not found, skipping {setting}")
            continue
        pred = pd.read_csv(path)
        for ds in [p[1] for p in PANELS if p[0] == setting]:
            sub = pred[pred["dataset"] == ds]
            for (mode, fold, seed), g in sub.groupby(["mode", "fold", "seed"]):
                m = compute_metrics(g["label"], g["prob"])
                correct = int(((g["prob"] >= 0.5).astype(int) == g["label"]).sum())
                rows.append({"setting": setting, "test_dataset": ds, "mode": mode, "fold": fold, "seed": seed,
                             "n": len(g), "correct": correct, **m})
    return pd.DataFrame(rows)


def aggregate(runs):
    out = []
    for (setting, ds, mode), g in runs.groupby(["setting", "test_dataset", "mode"], sort=False):
        by_seed = g.groupby("seed")[["correct", "n"]].sum()  # all folds of a seed = every test clip once
        n_clips = int(by_seed["n"].iloc[0])
        k_mean = int(round(by_seed["correct"].mean()))
        p_worst = max(binomtest(int(k), int(n), 0.5).pvalue for k, n in by_seed.itertuples(index=False))
        row = {"setting": setting, "test_dataset": ds, "mode": mode, "n_runs": len(g), "n_seeds": len(by_seed),
               "n_clips": n_clips}
        for m in ("accuracy", "macro_f1", "roc_auc"):
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        row["p_binom"] = binomtest(k_mean, n_clips, 0.5).pvalue
        row["p_binom_worst_seed"] = p_worst
        out.append(row)
    df = pd.DataFrame(out)
    order = {p[:2]: i for i, p in enumerate(PANELS)}
    df["_o"] = [order[(s, d)] * 10 + MODES.index(m) for s, d, m in zip(df.setting, df.test_dataset, df["mode"])]
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def plot(deep, path, dark=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = ("#1a1a19", "#f0efec", "#c3c2b7", "#3a3a38") if dark else \
        ("#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1")
    cols = COLORS["dark" if dark else "light"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey=True, facecolor=surface)
    for ax, (setting, ds, title) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(surface)
        g = deep[(deep.setting == setting) & (deep.test_dataset == ds)].set_index("mode")
        for i, mode in enumerate(MODES):
            if mode not in g.index:
                continue
            r = g.loc[mode]
            ax.bar(i, r.accuracy_mean, width=0.62, color=cols[i], zorder=2)
            ax.errorbar(i, r.accuracy_mean, yerr=r.accuracy_std, color=ink2, capsize=3, lw=1.2, zorder=3)
            star = "*" if r.p_binom < 0.05 else ""
            ax.text(i, r.accuracy_mean + r.accuracy_std + 0.012, f"{r.accuracy_mean:.2f}{star}", ha="center",
                    va="bottom", fontsize=8.5, color=ink)
        ax.axhline(0.5, color=ink2, lw=1, ls="--", zorder=1)
        n = int(g["n_clips"].iloc[0]) if len(g) else 0
        ax.set_title(f"{title}  (n={n} clips)", fontsize=10.5, color=ink, loc="left")
        ax.set_xticks(range(len(MODES)), MODES, color=ink2, fontsize=9)
        ax.set_ylim(0.3, 0.9)
        ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
        ax.tick_params(colors=ink2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy", color=ink2)
    fig.suptitle("Deception detection: fusion modes (mean accuracy, error bar = std over folds x seeds; "
                 "dashed = chance; * = binomial p < 0.05)", fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/deception")
    ap.add_argument("--baseline", default=None, help="default: <results-dir>/baseline.csv")
    ap.add_argument("--dark", action="store_true", help="dark-surface version of the plot")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rdir = Path(args.results_dir)

    runs = per_run_metrics(rdir)
    if runs.empty:
        raise SystemExit("no predictions found; run src/deception/run_all.py first")
    deep = aggregate(runs)
    deep.to_csv(rdir / "deep.csv", index=False)
    plot(deep, rdir / "fusion_comparison.png", dark=args.dark)

    show = deep.copy()
    for m in ("accuracy", "macro_f1", "roc_auc"):
        show[m] = [f"{a:.3f}±{s:.3f}" for a, s in zip(show[f"{m}_mean"], show[f"{m}_std"])]
    show["p_binom"] = show["p_binom"].map(lambda p: f"{p:.4f}")
    show["p_worst_seed"] = show["p_binom_worst_seed"].map(lambda p: f"{p:.4f}")
    cols = ["setting", "test_dataset", "mode", "n_runs", "n_clips", "accuracy", "macro_f1", "roc_auc",
            "p_binom", "p_worst_seed"]
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 500)
    print("\n=== deep.csv (deep models, this work) ===")
    print(show[cols].to_string(index=False))

    base_path = Path(args.baseline) if args.baseline else rdir / "baseline.csv"
    print(f"\n=== {base_path} (baseline) ===")
    if base_path.is_file():
        print(pd.read_csv(base_path).to_string(index=False))
    else:
        print("not found: no baseline table to print next to the deep results")
    print(f"\nWrote {rdir / 'deep.csv'} and {rdir / 'fusion_comparison.png'}")


if __name__ == "__main__":
    main()
