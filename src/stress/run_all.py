#!/usr/bin/env python3
"""Stress on MAS: audio / video / concat / late / xattn, 5 folds x seeds 0 1 2.

Uses src/common/train.py and src/common/models.py unchanged. Runs already saved are skipped, so a
stopped job resumes where it stopped.

    python src/stress/run_all.py                       # main experiment (dropout 0.3, wd 1e-2)
    python src/stress/run_all.py --dropout 0.5 --weight-decay 1e-2 --suffix _reg --modes concat
                                                       # regularised rerun of chosen modes

Files (OUT = --out-dir, default results/stress; SUF = --suffix, "" for the main experiment)
    OUT/deep_runs{SUF}.csv         one row per run (metrics, train/val accuracy, time) - the resume key
    OUT/deep_predictions{SUF}.csv  clip_id, mode, fold, seed, y_true, prob  (test clips of every run)
    OUT/deep{SUF}.csv              mean and std over the runs of each mode
    OUT/fusion_comparison{SUF}.png bar chart (mean +- std over folds and seeds)
After the main experiment the script flags modes that overfit (see --overfit-train-acc /
--overfit-val-margin) and, with --auto-rerun, reruns them with dropout 0.5 / weight decay 1e-2.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import torch  # noqa: E402
from data import compute_stats, load_metadata  # noqa: E402
from train import (MODES, append_rows, build_loader, build_model, choose_device,  # noqa: E402
                   compute_metrics, make_fold_splits, predict, seed_everything, train_model)

METRICS = ("accuracy", "macro_f1", "roc_auc")


def key(mode, fold, seed):
    return (str(mode), int(fold), int(seed))


def finished_runs(runs_csv):
    if not runs_csv.is_file() or runs_csv.stat().st_size == 0:
        return set()
    return {key(r.mode, r.fold, r.seed) for r in pd.read_csv(runs_csv).itertuples()}


def drop_unfinished_predictions(pred_csv, finished):
    """A run killed after writing predictions but before its results row would be duplicated."""
    if not pred_csv.is_file() or pred_csv.stat().st_size == 0:
        return
    df = pd.read_csv(pred_csv)
    keep = np.array([key(r.mode, r.fold, r.seed) in finished for r in df.itertuples()])
    if not keep.all():
        print(f"Removing {int((~keep).sum())} prediction rows of unfinished runs")
        df[keep].to_csv(pred_csv, index=False)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def summarize(runs):
    rows = []
    for mode, g in runs.groupby("mode", sort=False):
        row = {"mode": mode, "n_runs": len(g)}
        for m in METRICS + ("train_acc", "val_acc"):
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        for m in METRICS:
            row[m] = fmt(row[f"{m}_mean"], row[f"{m}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(args, suffix, modes, dropout, weight_decay, device, df):
    out = Path(args.out_dir)
    runs_csv, pred_csv = out / f"deep_runs{suffix}.csv", out / f"deep_predictions{suffix}.csv"
    ckpt_dir = Path(args.ckpt_dir) / f"stress_deep{suffix}"
    finished = finished_runs(runs_csv)
    drop_unfinished_predictions(pred_csv, finished)
    cache = {}
    n_total = len(modes) * len(args.folds) * len(args.seeds)
    print(f"\n##### dropout {dropout:g}, weight decay {weight_decay:g}, modes {modes}: "
          f"{n_total} runs, {len(finished)} already saved in {runs_csv}")
    done = 0
    for fold in args.folds:
        for seed in args.seeds:
            todo = [m for m in modes if key(m, fold, seed) not in finished]
            if not todo:
                continue
            train_df, val_df, test_df = make_fold_splits(df, fold, args.val_frac, seed)
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                assert not (set({"train": train_df, "val": val_df, "test": test_df}[a]["group_key"])
                            & set({"train": train_df, "val": val_df, "test": test_df}[b]["group_key"])), \
                    f"{a}/{b} share a group"
            stats = compute_stats(train_df, args.features_root, cache)  # training rows only
            n_pos = int(train_df["label"].sum())
            pos_weight = (len(train_df) - n_pos) / n_pos if n_pos else 1.0
            mk = lambda rows, train=False: build_loader(  # noqa: E731
                rows, args.features_root, stats, args.batch_size, train=train, seed=seed, cache=cache, device=device)
            val_loader, test_loader, train_eval_loader = mk(val_df), mk(test_df), mk(train_df)
            for mode in todo:
                t0 = time.perf_counter()
                seed_everything(seed)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model = build_model(mode, dropout)
                info = train_model(model, mk(train_df, train=True), val_loader, device, pos_weight=pos_weight,
                                   lr=args.lr, weight_decay=weight_decay, max_epochs=args.max_epochs,
                                   patience=args.patience, accum_steps=args.accum_steps, log_every=10)
                preds = predict(model, test_loader, device)  # best-epoch weights
                m = compute_metrics(preds["label"], preds["prob"])
                tr = predict(model, train_eval_loader, device)
                va = predict(model, val_loader, device)
                train_acc = compute_metrics(tr["label"], tr["prob"])["accuracy"]
                val_acc = compute_metrics(va["label"], va["prob"])["accuracy"]
                ckpt = ckpt_dir / f"{mode}_fold{fold}_seed{seed}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "stats": stats.to_dict(),
                            "mode": mode, "fold": fold, "seed": seed, "dropout": dropout,
                            "weight_decay": weight_decay, "best_epoch": info["best_epoch"]}, ckpt)
                elapsed = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                # predictions first, the runs row last: the row marks the run as finished
                append_rows(pred_csv, pd.DataFrame({
                    "clip_id": preds["clip_id"], "mode": mode, "fold": fold, "seed": seed,
                    "y_true": preds["label"], "prob": preds["prob"]}))
                append_rows(runs_csv, [{
                    "mode": mode, "fold": fold, "seed": seed, "dropout": dropout, "weight_decay": weight_decay,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                    "train_acc": train_acc, "val_acc": val_acc, "val_macro_f1": info["best_val_macro_f1"],
                    **m, "time_s": round(elapsed, 1), "peak_gpu_mb": round(peak_mb, 1), "device": device.type}])
                finished.add(key(mode, fold, seed))
                done += 1
                print(f"[{mode} fold {fold} seed {seed}] ({done} new, {len(finished)}/{n_total}) "
                      f"train acc {train_acc:.3f}  val acc {val_acc:.3f}  test acc {m['accuracy']:.3f}  "
                      f"F1 {m['macro_f1']:.3f}  AUC {m['roc_auc']:.3f}  best epoch {info['best_epoch']}/"
                      f"{info['epochs_run']}  TIME {elapsed:.1f} s  "
                      + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    runs = pd.read_csv(runs_csv)
    runs = runs[runs["mode"].isin(modes)]
    summary = summarize(runs)
    summary.to_csv(out / f"deep{suffix}.csv", index=False)
    return runs, summary


def show_table(title, summary):
    print(f"\n{title}")
    cols = ["mode", "n_runs", "accuracy", "macro_f1", "roc_auc", "train_acc_mean", "val_acc_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def compare_with_baseline(summary, baseline_csv):
    baseline_csv = Path(baseline_csv)
    if not baseline_csv.is_file():
        print(f"\n{baseline_csv} does not exist yet: baseline comparison skipped.")
        return
    base = pd.read_csv(baseline_csv)
    print(f"\nBaseline ({baseline_csv}):")
    print(base.to_string(index=False))
    print("\nDeep models (mean ± std):")
    print(summary[["mode", "accuracy", "macro_f1", "roc_auc"]].to_string(index=False))


def plot(summaries, path):
    """Grouped bars (accuracy, macro-F1, ROC-AUC) per mode with std error bars; reruns drawn as hatched outlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summaries[0][1]["mode"])
    width = 0.26
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for gi, (_, sm) in enumerate(summaries):
        sm = sm.set_index("mode")
        for mi, metric in enumerate(METRICS):
            xs = [i + (mi - 1) * width for i, n in enumerate(names) if n in sm.index]
            sub = [n for n in names if n in sm.index]
            kw = dict(color=colors[mi], label=metric) if gi == 0 else \
                dict(facecolor="none", edgecolor="black", hatch="//", label="dropout 0.5 rerun" if mi == 0 else None)
            ax.bar(xs, sm.loc[sub, f"{metric}_mean"], width, yerr=sm.loc[sub, f"{metric}_std"], capsize=3, **kw)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test score, mean ± std over 5 folds x 3 seeds")
    ax.set_title("Stress on MAS: fusion comparison")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out-dir", default="results/stress")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--baseline", default=None, help="baseline CSV (default <out-dir>/baseline.csv)")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--suffix", default="", help="file-name suffix for this configuration, e.g. _reg")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--overfit-train-acc", type=float, default=0.95,
                    help="overfitting if mean train accuracy is at least this ...")
    ap.add_argument("--overfit-val-margin", type=float, default=0.05,
                    help="... and mean val accuracy is within this margin of chance (majority-class rate)")
    ap.add_argument("--auto-rerun", action="store_true",
                    help="after the main run, rerun overfitting modes with dropout 0.5 / weight decay 1e-2")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    device = choose_device()
    df = load_metadata(args.features_root, ["mas"], task="stress")
    print(f"Loaded {len(df)} stress clips from mas, {df['group_key'].nunique()} groups")
    t0 = time.perf_counter()
    runs, summary = run_experiment(args, args.suffix, args.modes, args.dropout, args.weight_decay, device, df)
    print(f"\nMean time per run: {runs['time_s'].mean():.1f} s (all {len(runs)} runs listed in deep_runs{args.suffix}.csv); "
          f"this invocation took {time.perf_counter() - t0:.0f} s")
    show_table(f"Deep models, dropout {args.dropout:g}, wd {args.weight_decay:g} (mean ± std over folds x seeds)", summary)
    print(runs.groupby("mode", sort=False)["time_s"].agg(["mean", "sum"]).round(1).rename(
        columns={"mean": "s/run", "sum": "total s"}).to_string())
    compare_with_baseline(summary, args.baseline or out / "baseline.csv")
    summaries = [("", summary)]
    if args.suffix == "":
        # chance = rate of the majority class among the validation clips of all runs (approximated by the data)
        chance = max(df["label"].mean(), 1 - df["label"].mean())
        flagged = [r.mode for r in summary.itertuples()
                   if r.train_acc_mean >= args.overfit_train_acc and r.val_acc_mean <= chance + args.overfit_val_margin]
        print(f"\nOverfitting rule: mean train acc >= {args.overfit_train_acc:.2f} and mean val acc <= "
              f"{chance + args.overfit_val_margin:.3f} (majority rate {chance:.3f} + {args.overfit_val_margin:.2f}).")
        print(f"Flagged as overfitting: {flagged if flagged else 'none'}")
        if flagged and args.auto_rerun:
            _, reg = run_experiment(args, "_reg", flagged, 0.5, 1e-2, device, df)
            show_table("Regularised rerun (dropout 0.5, wd 1e-2)", reg)
            both = pd.concat([summary[summary["mode"].isin(flagged)].assign(config="dropout 0.3, wd 1e-2"),
                              reg.assign(config="dropout 0.5, wd 1e-2")])
            print("\nBefore vs after:")
            print(both[["mode", "config", "accuracy", "macro_f1", "roc_auc", "train_acc_mean", "val_acc_mean"]]
                  .sort_values(["mode", "config"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            summaries.append((" (reg)", reg))
        elif flagged:
            print("Rerun with: python src/stress/run_all.py --dropout 0.5 --weight-decay 1e-2 --suffix _reg --modes "
                  + " ".join(flagged))
    plot(summaries, out / f"fusion_comparison{args.suffix}.png")


if __name__ == "__main__":
    main()
