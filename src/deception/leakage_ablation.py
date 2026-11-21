#!/usr/bin/env python3
"""Leakage ablation for deception: person-grouped folds versus clip-level folds.

For Real-Life Trial (rlt) and Bag-of-Lies (bol), mode ``late``, seeds 0 1 2, two schemes:

    person  the folds of features/<dataset>/metadata.csv (StratifiedGroupKFold on group_id).
            NOT retrained: the 15 runs (5 folds x 3 seeds) are read from
            <results-dir>/runs/within_<dataset>.csv, which src/deception/run_all.py wrote.
    clip    StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the clips, ignoring
            group_id, so a speaker can be in train and test. The validation split (15 % of the
            non-test rows, drawn with the run seed) is clip-level too.

Everything else is exactly what src/common/train.py does for a within-dataset run (its helpers
are imported and used unchanged): dropout 0.3, weight decay 1e-2, lr 1e-3, batch 16,
pos_weight = n_neg / n_pos, fp16 autocast on CUDA, early stopping on validation macro-F1
(patience 10, 100 epochs), train-only z-scoring. Only the way the rows are assigned to
train / val / test differs.

Clip-level runs are saved to <results-dir>/runs/leakage_clip.csv (+ _predictions.csv) with the
key (setup, mode, fold, seed); runs already there are skipped, so the script can be stopped and
restarted. Outputs:

    <results-dir>/leakage_ablation.csv          mean and std per dataset and scheme, and the
                                                difference (clip minus person)
    <results-dir>/leakage_ablation_overlap.csv  per dataset / scheme / fold / seed: how many test
                                                clips have their speaker among the training clips

    python src/deception/leakage_ablation.py                 # from the repository root
    python src/deception/leakage_ablation.py --dry-run       # splits, overlap and plan only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import train  # noqa: E402  (src/common/train.py, unchanged)
from data import compute_stats, load_metadata  # noqa: E402

DATASETS = ("rlt", "bol")
DATASET_NAMES = {"rlt": "Real-Life Trial", "bol": "Bag-of-Lies"}
N_FOLDS = 5
FOLD_SEED = 42           # random_state of the clip-level StratifiedKFold
DROPOUT, WEIGHT_DECAY = 0.3, 1e-2  # train.py defaults, as used by run_all.py
VAL_FRAC = 0.15
SCHEMES = ("person", "clip")


# ------------------------------------------------------------------------------------ splits
def add_clip_folds(df):
    """Add ``fold_clip`` (clip-level stratified 5-fold, ignoring group_id) and ``clip_key``."""
    df = df.copy()
    df["clip_key"] = df["dataset"] + ":" + df["clip_id"]
    df["fold_clip"] = -1
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(df)), df["label"])):
        df.iloc[test_idx, df.columns.get_loc("fold_clip")] = k
    assert (df["fold_clip"] >= 0).all()
    return df


def clip_splits(df, fold, seed, val_frac=VAL_FRAC):
    """(train, val, test) for a clip-level fold: test = fold_clip == fold, val = ``val_frac`` of the
    remaining CLIPS (train.split_groups with every clip as its own group)."""
    test = df[df["fold_clip"] == fold].reset_index(drop=True)
    rest = df[df["fold_clip"] != fold]
    train_df, val = train.split_groups(rest, val_frac, seed, group_col="clip_key")
    assert not (set(train_df["clip_key"]) & set(test["clip_key"])), "train/test share a clip"
    assert not (set(val["clip_key"]) & set(test["clip_key"])), "val/test share a clip"
    assert not (set(train_df["clip_key"]) & set(val["clip_key"])), "train/val share a clip"
    return train_df, val, test


def speaker_overlap(train_df, test_df):
    """Number of test clips whose speaker (group_key) also has clips in the training set."""
    return int(test_df["group_key"].isin(set(train_df["group_key"])).sum())


def overlap_table(df, dataset, seeds):
    """One row per (scheme, fold, seed); no training needed, the splits are deterministic."""
    rows = []
    for fold in range(N_FOLDS):
        for seed in seeds:
            for scheme in SCHEMES:
                if scheme == "person":
                    tr, va, te = train.make_fold_splits(df, fold, VAL_FRAC, seed)
                else:
                    tr, va, te = clip_splits(df, fold, seed)
                rows.append({"dataset": dataset, "scheme": scheme, "fold": fold, "seed": seed,
                             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                             "n_test_speakers": te["group_key"].nunique(),
                             "n_test_speaker_in_train": speaker_overlap(tr, te),
                             "n_test_speaker_in_train_or_val": int(te["group_key"].isin(
                                 set(tr["group_key"]) | set(va["group_key"])).sum())})
    out = pd.DataFrame(rows)
    out["frac_test_speaker_in_train"] = out["n_test_speaker_in_train"] / out["n_test"]
    return out


# ---------------------------------------------------------------------------------- training
def setup_for(dataset, scheme):
    base = train.setup_id("deception", [dataset], [dataset], DROPOUT, WEIGHT_DECAY)
    return base if scheme == "person" else base + ";split=clip_stratified_kfold5_rs42"


def run_clip_experiments(df, dataset, args, device, finished, out_csv, pred_csv, ckpt_dir, cache):
    """Train every missing (fold, seed) of the clip-level scheme for one dataset."""
    setup = setup_for(dataset, "clip")
    for fold in range(N_FOLDS):
        for seed in args.seeds:
            if train.run_key(setup, args.mode, fold, seed) in finished:
                print(f"[{dataset} {args.mode} clip fold {fold} seed {seed}] already finished, skipping")
                continue
            train_df, val_df, test_df = clip_splits(df, fold, seed)
            n_overlap = speaker_overlap(train_df, test_df)
            print(f"\n== {dataset} clip fold {fold}, seed {seed}: train {len(train_df)} / val {len(val_df)} / "
                  f"test {len(test_df)} clips; {n_overlap} of {len(test_df)} test clips have their speaker "
                  f"in train")
            stats = compute_stats(train_df, args.features_root, cache)  # training rows only
            n_pos = int(train_df["label"].sum())
            pos_weight = (len(train_df) - n_pos) / n_pos if n_pos else 1.0
            val_loader = train.build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache,
                                            workers=args.workers, device=device)
            test_loader = train.build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache,
                                             workers=args.workers, device=device)
            t0 = time.perf_counter()
            train.seed_everything(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_loader = train.build_loader(train_df, args.features_root, stats, args.batch_size, train=True,
                                              seed=seed, workers=args.workers, cache=cache, device=device)
            model = train.build_model(args.mode, DROPOUT)
            info = train.train_model(model, train_loader, val_loader, device, pos_weight=pos_weight,
                                     weight_decay=WEIGHT_DECAY)
            preds = train.predict(model, test_loader, device)  # best epoch's weights
            metrics = train.compute_metrics(preds["label"], preds["prob"])

            ckpt_path = ckpt_dir / f"{dataset}_{args.mode}_fold{fold}_seed{seed}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "stats": stats.to_dict(), "setup": setup, "mode": args.mode, "fold": fold,
                        "seed": seed, "best_epoch": info["best_epoch"],
                        "best_val_macro_f1": info["best_val_macro_f1"]}, ckpt_path)

            elapsed = time.perf_counter() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
            # Predictions first, the results row last: the row marks the run as finished.
            train.append_rows(pred_csv, preds.assign(mode=args.mode, fold=fold, seed=seed, setup=setup))
            train.append_rows(out_csv, [{
                "setup": setup, "task": "deception", "train_datasets": dataset, "test_datasets": dataset,
                "mode": args.mode, "fold": fold, "seed": seed, "dropout": DROPOUT,
                "weight_decay": WEIGHT_DECAY, "n_train": len(train_df), "n_val": len(val_df),
                "n_test": len(test_df), "n_test_speaker_in_train": n_overlap,
                "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                "val_macro_f1": info["best_val_macro_f1"], **metrics,
                "time_s": round(elapsed, 1), "peak_gpu_mb": round(peak_mb, 1), "device": device.type}])
            finished.add(train.run_key(setup, args.mode, fold, seed))
            print(f"[{dataset} {args.mode} clip fold {fold} seed {seed}] test acc {metrics['accuracy']:.3f}  "
                  f"macro-F1 {metrics['macro_f1']:.3f}  AUC {metrics['roc_auc']:.3f}  | best epoch "
                  f"{info['best_epoch']}/{info['epochs_run']}  | {elapsed:.1f} s  | "
                  + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda"
                     else "peak GPU memory n/a (cpu)"))
            del model, train_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------- report
def load_runs(dataset, scheme, args, clip_csv):
    """Result rows of one (dataset, scheme): mode and seeds as requested, one row per fold/seed."""
    path = clip_csv if scheme == "clip" else Path(args.results_dir) / "runs" / f"within_{dataset}.csv"
    if not path.is_file():
        raise SystemExit(f"error: {path} not found")
    runs = pd.read_csv(path)
    runs = runs[(runs["setup"] == setup_for(dataset, scheme)) & (runs["mode"] == args.mode)
                & runs["seed"].isin(args.seeds) & runs["fold"].isin(range(N_FOLDS))]
    runs = runs.drop_duplicates(["fold", "seed"], keep="last")
    expected = N_FOLDS * len(args.seeds)
    if len(runs) != expected:
        print(f"warning: {dataset} {scheme}: {len(runs)} runs found in {path}, expected {expected}")
    return runs


def build_ablation(args, clip_csv):
    """DataFrame with a row per (dataset, scheme) and a 'difference' row per dataset."""
    rows = []
    for dataset in args.datasets:
        by_scheme = {}
        for scheme in SCHEMES:
            runs = load_runs(dataset, scheme, args, clip_csv)
            summ = train.summarize(runs).iloc[0]  # mean and std over folds and seeds, as train.py does
            by_scheme[scheme] = summ
            row = {"dataset": dataset, "scheme": scheme, "mode": args.mode, "n_runs": int(summ["n_runs"])}
            for m in train.METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = summ[f"{m}_mean"], summ[f"{m}_std"]
                row[m] = f"{summ[f'{m}_mean']:.3f} ± {summ[f'{m}_std']:.3f}"
            rows.append(row)
        diff = {"dataset": dataset, "scheme": "difference (clip - person)", "mode": args.mode,
                "n_runs": ""}
        for m in train.METRICS:
            d = by_scheme["clip"][f"{m}_mean"] - by_scheme["person"][f"{m}_mean"]
            diff[f"{m}_mean"], diff[f"{m}_std"], diff[m] = d, float("nan"), f"{d:+.3f}"
        rows.append(diff)
    cols = ["dataset", "scheme", "mode", "n_runs", *train.METRICS,
            *[f"{m}_{s}" for m in train.METRICS for s in ("mean", "std")]]
    return pd.DataFrame(rows)[cols]


def print_report(ablation, overlap):
    print("\n=== Leakage ablation: late fusion, mean ± std over 5 folds x seeds ===")
    for dataset, g in ablation.groupby("dataset", sort=False):
        print(f"\n{DATASET_NAMES[dataset]}")
        print(f"  {'scheme':28s} {'n':>3s}  {'accuracy':>13s}  {'macro-F1':>13s}  {'AUC':>13s}")
        for r in g.itertuples():
            print(f"  {r.scheme:28s} {str(r.n_runs):>3s}  {r.accuracy:>13s}  {r.macro_f1:>13s}  {r.roc_auc:>13s}")
    print("\n=== Test clips whose speaker is in the training set (clip-level folds) ===")
    for dataset, g in overlap[overlap["scheme"] == "clip"].groupby("dataset", sort=False):
        print(f"\n{DATASET_NAMES[dataset]} (clips = {g['n_test'].sum() // g['seed'].nunique()}), "
              f"mean over seeds {[int(s) for s in sorted(g['seed'].unique())]}")
        per = g.groupby("fold").agg(n_test=("n_test", "mean"), speakers=("n_test_speakers", "mean"),
                                    overlap=("n_test_speaker_in_train", "mean"),
                                    overlap_tv=("n_test_speaker_in_train_or_val", "mean"))
        for fold, r in per.iterrows():
            print(f"  fold {fold}: {r.overlap:5.1f} of {r.n_test:4.1f} test clips ({r.overlap / r.n_test:5.1%}) "
                  f"have their speaker in train  [train+val: {r.overlap_tv:.1f}]")
        tot = per["overlap"].sum() / per["n_test"].sum()
        print(f"  all folds: {per['overlap'].sum():.1f} of {per['n_test'].sum():.0f} ({tot:.1%})")
    person = overlap[overlap["scheme"] == "person"]
    print(f"\nPerson-grouped folds: {int(person['n_test_speaker_in_train'].sum())} test clips share a speaker "
          f"with the training set over all {len(person)} (dataset, fold, seed) splits (must be 0).")


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--results-dir", default="results/deception")
    ap.add_argument("--ckpt-dir", default="checkpoints/deception/leakage_clip")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--mode", choices=train.MODES, default="late")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print splits and overlap, train nothing")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    clip_csv = results_dir / "runs" / "leakage_clip.csv"
    pred_csv = results_dir / "runs" / "leakage_clip_predictions.csv"
    ckpt_dir = Path(args.ckpt_dir)

    # Sanity: the saved person-grouped runs must come from the splits that metadata.csv gives now.
    overlaps, frames = [], {}
    for dataset in args.datasets:
        df = load_metadata(args.features_root, [dataset], task="deception")
        df = add_clip_folds(df)
        frames[dataset] = df
        print(f"{dataset}: {len(df)} clips, {df['group_key'].nunique()} speakers, "
              f"{df['group_key'].value_counts().eq(1).sum()} with a single clip")
        overlaps.append(overlap_table(df, dataset, args.seeds))
        person_csv = results_dir / "runs" / f"within_{dataset}.csv"
        if person_csv.is_file():
            old = load_runs(dataset, "person", args, clip_csv)
            new = overlaps[-1].query("scheme == 'person'")[["fold", "seed", "n_train", "n_val", "n_test"]]
            merged = old.merge(new, on=["fold", "seed"], suffixes=("_saved", "_now"))
            for c in ("n_train", "n_val", "n_test"):
                bad = int((merged[f"{c}_saved"] != merged[f"{c}_now"]).sum())
                if bad:
                    print(f"warning: {dataset}: {bad} saved person-grouped runs differ in {c} from the "
                          f"splits of the current metadata.csv")
    overlap = pd.concat(overlaps, ignore_index=True)

    if args.dry_run:
        for dataset in args.datasets:
            todo = train.load_finished_runs(clip_csv)
            n = sum(train.run_key(setup_for(dataset, "clip"), args.mode, f, s) not in todo
                    for f in range(N_FOLDS) for s in args.seeds)
            print(f"plan: {dataset} clip-level {args.mode}: {n} of {N_FOLDS * len(args.seeds)} runs to train")
        print(overlap[overlap["scheme"] == "clip"].groupby(["dataset", "fold"])[
            ["n_test", "n_test_speaker_in_train"]].mean().round(1).to_string())
        return 0

    device = train.choose_device()
    finished = train.load_finished_runs(clip_csv)
    train.drop_unfinished_predictions(pred_csv, finished)
    print(f"{len(finished)} finished clip-level runs found in {clip_csv}")
    for dataset in args.datasets:
        run_clip_experiments(frames[dataset], dataset, args, device, finished, clip_csv, pred_csv, ckpt_dir,
                             cache={})

    ablation = build_ablation(args, clip_csv)
    results_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(results_dir / "leakage_ablation.csv", index=False)
    overlap.to_csv(results_dir / "leakage_ablation_overlap.csv", index=False)
    print_report(ablation, overlap)
    print(f"\nWrote {results_dir / 'leakage_ablation.csv'}, {results_dir / 'leakage_ablation_overlap.csv'}; "
          f"runs in {clip_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
