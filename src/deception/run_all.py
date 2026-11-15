#!/usr/bin/env python3
"""Run every deception experiment (5 modes x seeds 0 1 2 x 5 settings) with src/common/train.py.

Settings (one results CSV each, under --results-dir/runs/):
    within_bol    5-fold CV inside Bag-of-Lies
    within_rlt    5-fold CV inside Real-Life Trial
    pooled        5-fold CV on Bag-of-Lies + Real-Life Trial pooled, dataset-balanced batches
                  (the per-dataset test numbers are computed from the predictions by report.py)
    bol_to_rlt    train on Bag-of-Lies, test on all of Real-Life Trial
    rlt_to_bol    train on Real-Life Trial, test on all of Bag-of-Lies

train.py is used as it is (it is imported and its ``main`` is called). It already skips runs that
are saved in the results CSV (key = setup, mode, fold, seed) and removes the predictions of
half-finished runs, so this script can be killed at any time and started again with the same
command. Order: Bag-of-Lies fold 0 (all modes and seeds) first, so those checkpoints exist early,
then the other Bag-of-Lies folds, then the remaining settings.

    python src/deception/run_all.py                       # from the repository root
    python src/deception/run_all.py --dry-run             # only print the plan
    python src/deception/run_all.py --only pooled bol_to_rlt
Checkpoints: <ckpt-dir>/<setting>/<mode>_fold<k>_seed<s>.pt  (fold "all" for cross-dataset runs).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import train  # noqa: E402  (src/common/train.py, unchanged)

SETTINGS = {
    "within_bol": ["--datasets", "bol"],
    "within_rlt": ["--datasets", "rlt"],
    "pooled": ["--datasets", "bol", "rlt", "--balance", "dataset"],
    "bol_to_rlt": ["--train-datasets", "bol", "--test-datasets", "rlt"],
    "rlt_to_bol": ["--train-datasets", "rlt", "--test-datasets", "bol"],
}
CROSS = {"bol_to_rlt", "rlt_to_bol"}


def plan(only):
    """Ordered list of (setting, folds); folds is None for cross-dataset runs."""
    jobs = [("within_bol", [0]), ("within_bol", [1, 2, 3, 4])]
    jobs += [(s, [0, 1, 2, 3, 4]) for s in ("within_rlt", "pooled")]
    jobs += [(s, None) for s in ("bol_to_rlt", "rlt_to_bol")]
    return [(s, f) for s, f in jobs if s in only]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--results-dir", default="results/deception")
    ap.add_argument("--ckpt-dir", default="checkpoints/deception")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", choices=train.MODES, default=list(train.MODES))
    ap.add_argument("--only", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = ap.parse_args()

    jobs = plan(set(args.only))
    runs_dir = Path(args.results_dir) / "runs"
    for setting, folds in jobs:
        print(f"plan: {setting:11s} folds {folds if folds is not None else 'cross-dataset'} "
              f"modes {args.modes} seeds {args.seeds} -> {runs_dir / (setting + '.csv')}")
    if args.dry_run:
        return 0

    failed, t0 = [], time.perf_counter()
    for setting, folds in jobs:
        argv = ["--task", "deception", *SETTINGS[setting],
                "--modes", *args.modes, "--seeds", *map(str, args.seeds),
                "--out", str(runs_dir / f"{setting}.csv"),
                "--features-root", args.features_root, "--ckpt-dir", args.ckpt_dir,
                "--workers", str(args.workers), "--preload"]
        if folds is not None:
            argv += ["--folds", *map(str, folds)]
        print(f"\n######## {setting} folds {folds if folds is not None else 'cross'} "
              f"({(time.perf_counter() - t0) / 60:.1f} min elapsed) ########", flush=True)
        try:
            train.main(argv)
        except Exception:  # keep going with the other settings; the failed job resumes on restart
            traceback.print_exc()
            failed.append(setting)
    print(f"\nFinished in {(time.perf_counter() - t0) / 60:.1f} min; failed jobs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
