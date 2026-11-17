#!/usr/bin/env python3
"""Print the LoRA fine-tuning results next to the frozen-feature results for the same modes and settings.

Frozen-feature numbers come from results/deception/deep.csv (mean / std over its runs: 5 folds x 3 seeds).
Fine-tuning numbers are recomputed from the predictions file of finetune.py / finetune_pooled.py, exactly
as report.py does for the frozen runs: metrics per (mode, fold, seed) on the clips of one test dataset,
then mean / std over the runs (here 5 folds x 1 seed, so the std only reflects the folds).

    python src/deception/compare_finetune.py                    # from the repository root
    python src/deception/compare_finetune.py --seed 0 --modes xattn audio video
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from train import compute_metrics  # noqa: E402  (src/common/train.py, unchanged)

SETTINGS = {"within_bol": ("train=bol;", ["bol"]), "pooled": ("train=bol+rlt;", ["bol", "rlt"])}
METRICS = ("accuracy", "macro_f1", "roc_auc")


def finetune_runs(pred_csv, seed):
    pred = pd.read_csv(pred_csv)
    if seed is not None:
        pred = pred[pred["seed"] == seed]
    rows = []
    for setting, (key, datasets) in SETTINGS.items():
        sub_setting = pred[pred["setup"].str.contains(key, regex=False)]
        for ds in datasets:
            for (mode, fold, sd), g in sub_setting[sub_setting["dataset"] == ds].groupby(["mode", "fold", "seed"]):
                rows.append({"setting": setting, "test_dataset": ds, "mode": mode, "fold": fold, "seed": sd,
                             "n": len(g), **compute_metrics(g["label"], g["prob"])})
    return pd.DataFrame(rows)


def fmt(mean, std):
    return f"{mean:.3f}±{0.0 if pd.isna(std) else std:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", default="results/deception/deep.csv")
    ap.add_argument("--finetune", default="results/deception/finetune_predictions.csv",
                    help="predictions CSV written by finetune.py (beside finetune.csv)")
    ap.add_argument("--seed", type=int, default=0, help="fine-tuning seed to use (default 0)")
    ap.add_argument("--modes", nargs="+", default=["xattn", "audio", "video"])
    args = ap.parse_args()

    deep = pd.read_csv(args.deep)
    runs = finetune_runs(args.finetune, args.seed)
    rows = []
    for setting, (_, datasets) in SETTINGS.items():
        for ds in datasets:
            for mode in args.modes:
                fz = deep[(deep["setting"] == setting) & (deep["test_dataset"] == ds) & (deep["mode"] == mode)]
                ft = runs[(runs["setting"] == setting) & (runs["test_dataset"] == ds) & (runs["mode"] == mode)]
                row = {"setting": setting, "test": ds, "mode": mode}
                for m in METRICS:
                    f = fz.iloc[0] if len(fz) else None
                    row[f"frozen {m}"] = fmt(f[f"{m}_mean"], f[f"{m}_std"]) if f is not None else "n/a"
                    row[f"finetune {m}"] = fmt(ft[m].mean(), ft[m].std()) if len(ft) else "n/a"
                row["n_ft_runs"] = len(ft)
                row["d_macro_f1"] = (f"{ft['macro_f1'].mean() - fz.iloc[0]['macro_f1_mean']:+.3f}"
                                     if len(ft) and len(fz) else "n/a")
                rows.append(row)
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", None):
        print(out.to_string(index=False))
    print("\nfrozen: mean±std over 15 runs (5 folds x 3 seeds) from deep.csv; "
          f"finetune: mean±std over n_ft_runs folds, seed {args.seed}. d_macro_f1 = finetune - frozen mean.")


if __name__ == "__main__":
    main()
