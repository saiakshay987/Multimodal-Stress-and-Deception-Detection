#!/usr/bin/env python3
"""Transfer: stress pre-training on MAS, then fine-tuning on deception (bol + rlt) with a fresh head.

Examples (run from the repository root):
    python src/integration/transfer.py --smoke      # 2 epochs on a few clips, writes *_smoke files
    python src/integration/transfer.py --preload    # 5 test folds x seeds 0 1 2, concat fusion

Per test fold k and seed s
    Stage 1  MultimodalModel(heads=("stress",)) trained on the MAS training folds (the single-task stress recipe of
             train.py: val = 15 % of the remaining MAS groups, pos_weight, AdamW lr 1e-3, early stopping on val macro-F1).
             The stage-1 model is also scored on MAS test fold k (columns stage1_*), as a sanity check.
    Stage 2  the stage-1 weights WITHOUT the stress head go into MultimodalModel(heads=("deception",)) (fresh, randomly
             initialised deception head); every parameter is fine-tuned on the bol + rlt training folds with exactly
             the recipe of the from-scratch deception runs (run_all.py "pooled": train.py --datasets bol rlt
             --balance dataset, same splits, same stats from the deception training rows, same hyper-parameters).
             The only difference from the from-scratch run is the initialisation of the backbone.
    Test     fold k of bol + rlt (the same clips as the from-scratch runs), scored on bol, rlt and bol+rlt.

Comparison (paired, same mode, fold, seed): from-scratch results of the same mode in
    results/deception/runs/pooled.csv + pooled_predictions.csv  (per-run numbers behind deep.csv; deep.csv itself
    only holds mean / std over the 15 runs, so it is used as a cross-check of the recomputed means).

Outputs (OUT = --out, default results/integration/transfer.csv)
    OUT                          one row per (fold, seed, test_dataset): bol, rlt, bol+rlt (+ stage-1 MAS metrics)
    OUT stem + _summary.csv      mean / std over runs per test_dataset
    OUT stem + _predictions.csv  every test clip
    OUT stem + _vs_scratch.csv   paired difference (transfer - scratch)
    <ckpt-dir>/<OUT stem>/stage1_<mode>_fold<k>_seed<s>.pt (kept, a restarted run reuses it) and
    <ckpt-dir>/<OUT stem>/<mode>_fold<k>_seed<s>.pt (the fine-tuned model)
Runs already present in OUT are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import compute_stats, load_metadata  # noqa: E402
from models import MultimodalModel  # noqa: E402
from train import (METRICS, append_rows, build_loader, choose_device, compute_metrics,  # noqa: E402
                   drop_unfinished_predictions, forward_multimodal, load_finished_runs, make_fold_splits,
                   output_paths, predict, run_key, seed_everything, train_model)
import train_multitask as tm  # noqa: E402
from exp_utils import assert_disjoint, eval_rows, paired_table, summarize  # noqa: E402

DECEPTION = ["bol", "rlt"]
EVAL_SETS = [("bol", ["bol"]), ("rlt", ["rlt"]), ("bol+rlt", ["bol", "rlt"])]
HEAD_PREFIX = "heads."


def setup_id(args):
    return f"transfer;stage1=mas(stress);stage2=bol+rlt(deception,balance=dataset);dropout={args.dropout:g};wd={args.weight_decay:g}"


def pos_weight(df):
    n_pos = int(df["label"].sum())
    return (len(df) - n_pos) / n_pos if n_pos else 1.0


def stage1(args, mas_df, fold, seed, device, cache, ckpt_path, smoke):
    """Stress model trained on the MAS training folds -> (state_dict without heads, info dict)."""
    if ckpt_path.is_file() and not smoke:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (ck["mode"], ck["dropout"], ck["weight_decay"]) == (args.mode, args.dropout, args.weight_decay):
            print(f"  stage 1: reusing {ckpt_path}")
            return ck["backbone"], ck["info"]
    train_df, val_df, test_df = make_fold_splits(mas_df, fold, args.val_frac, seed)
    if smoke:
        train_df, val_df, test_df = tm.smoke_subsample(train_df, val_df, test_df, seed)
    assert_disjoint(train_df, val_df, test_df)
    print(f"  stage 1 (MAS stress): train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
          f"val {len(val_df)}, test {len(test_df)}")
    stats = compute_stats(train_df, args.features_root, cache)
    pw = pos_weight(train_df)
    val_loader = build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache, workers=args.workers, device=device)
    test_loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache, workers=args.workers, device=device)
    seed_everything(seed)
    train_loader = build_loader(train_df, args.features_root, stats, args.batch_size, train=True, seed=seed,
                                workers=args.workers, cache=cache, device=device)
    model = MultimodalModel(fusion=args.mode, heads=("stress",), dropout=args.dropout)
    fit = train_model(model, train_loader, val_loader, device, pos_weight=pw, lr=args.lr, weight_decay=args.weight_decay,
                      max_epochs=2 if smoke else args.max_epochs, patience=args.patience, accum_steps=args.accum_steps,
                      forward=forward_multimodal, log_every=1 if smoke else 10)
    preds = predict(model, test_loader, device, forward_multimodal)
    m = compute_metrics(preds["label"], preds["prob"])
    info = {"stage1_best_epoch": fit["best_epoch"], "stage1_epochs_run": fit["epochs_run"],
            "stage1_val_macro_f1": fit["best_val_macro_f1"], "stage1_n_train": len(train_df),
            **{f"stage1_mas_{k}": v for k, v in m.items()}}
    print(f"  stage 1 done: best epoch {fit['best_epoch']}/{fit['epochs_run']}, MAS test fold {fold}: "
          + " ".join(f"{k} {v:.3f}" for k, v in m.items()))
    backbone = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith(HEAD_PREFIX)}
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "info": info, "mode": args.mode, "dropout": args.dropout,
                "weight_decay": args.weight_decay, "fold": fold, "seed": seed}, ckpt_path)
    del model, train_loader
    return backbone, info


def build_transfer_model(backbone, mode, dropout):
    """Deception model: stage-1 backbone, fresh deception head (random init from the current RNG state)."""
    model = MultimodalModel(fusion=mode, heads=("deception",), dropout=dropout)
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    assert set(missing) == {"heads.deception.weight", "heads.deception.bias"} and not unexpected, (missing, unexpected)
    return model


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MAS stress pre-training, then deception fine-tuning.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="concat", choices=["audio", "video", "concat", "late", "xattn"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--preload", action="store_true", help="keep loaded clips in RAM (about 1 MB per clip)")
    ap.add_argument("--drop-missing", action="store_true")
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out", default="results/integration/transfer.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--deception-runs", default="results/deception/runs/pooled.csv")
    ap.add_argument("--deception-preds", default="results/deception/runs/pooled_predictions.csv")
    ap.add_argument("--deception-deep", default="results/deception/deep.csv")
    ap.add_argument("--compare-only", action="store_true", help="skip training, rebuild the comparison")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a few clips, 2 folds x 2 seeds; *_smoke files")
    return ap.parse_args(argv)


def compare(args, out_csv, pred_csv, vs_csv, smoke):
    new = pd.read_csv(out_csv)
    new = new[(new["setup"] == setup_id(args)) & (new["mode"] == args.mode)]
    print(f"\n== Paired comparison with deception training from scratch (mode {args.mode}, "
          f"dropout {args.dropout:g}, weight decay {args.weight_decay:g})")
    scratch = tm.single_task_deception(args.deception_preds, args.deception_runs, args.mode, args.dropout,
                                       args.weight_decay)
    # the per-dataset means over all runs must reproduce deep.csv (pooled setting)
    deep_path = Path(args.deception_deep)
    if deep_path.is_file():
        deep = pd.read_csv(deep_path)
        deep = deep[(deep["setting"] == "pooled") & (deep["mode"] == args.mode)].set_index("test_dataset")
        for name in ("bol", "rlt", "bol+rlt"):
            if name in deep.index:
                g = scratch[scratch["test_dataset"] == name]
                d = {m: abs(g[m].mean() - deep.loc[name, m + "_mean"]) for m in METRICS}
                print(f"  scratch vs deep.csv [{name}] n_runs {len(g)} (deep.csv {int(deep.loc[name, 'n_runs'])}): "
                      "max |diff of means| " + ", ".join(f"{m} {v:.1e}" for m, v in d.items()))
    pred_new = pd.read_csv(pred_csv).query("mode == @args.mode")
    pred_old = pd.read_csv(args.deception_preds).query("mode == @args.mode")[["clip_id", "fold", "seed", "dataset"]]
    table = paired_table(new, scratch, [n for n, _ in EVAL_SETS], "transfer", "scratch", pred_new, pred_old, smoke)
    table.to_csv(vs_csv, index=False)
    print(f"Wrote {vs_csv}")


def main(argv=None):
    args = parse_args(argv)
    out_csv, summary_csv, pred_csv, stem = output_paths(args.out, args.smoke)
    vs_csv = out_csv.with_name(stem + "_vs_scratch.csv")
    setup = setup_id(args)
    if args.compare_only:
        compare(args, out_csv, pred_csv, vs_csv, args.smoke)
        return

    folds, seeds = (args.folds[:2], args.seeds[:2]) if args.smoke else (args.folds, args.seeds)
    max_epochs = 2 if args.smoke else args.max_epochs
    log_every = 1 if args.smoke else 10
    ckpt_dir = Path(args.ckpt_dir) / stem
    if args.smoke:
        for p in (out_csv, summary_csv, pred_csv, vs_csv):
            p.unlink(missing_ok=True)

    device = choose_device()
    missing = "drop" if args.drop_missing else "error"
    mas_df = load_metadata(args.features_root, ["mas"], task="stress", missing=missing)
    dec_df = load_metadata(args.features_root, DECEPTION, task="deception", missing=missing)
    task_of_dataset = {"mas": "stress", "bol": "deception", "rlt": "deception"}
    print(f"Loaded MAS {len(mas_df)} stress clips, bol+rlt {len(dec_df)} deception clips")
    cache = {} if args.preload else None
    finished = set() if args.smoke else load_finished_runs(out_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Setup: {setup}\nResults: {out_csv}  ({len(finished)} finished runs found)")

    for fold in folds:
        for seed in seeds:
            if run_key(setup, args.mode, fold, seed) in finished:
                print(f"[{args.mode} fold {fold} seed {seed}] already finished, skipping")
                continue
            print(f"\n== fold {fold}, seed {seed}")
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            backbone, s1 = stage1(args, mas_df, fold, seed, device, cache,
                                  ckpt_dir / f"stage1_{args.mode}_fold{fold}_seed{seed}.pt", args.smoke)

            # stage 2: the same splits and recipe as the from-scratch pooled deception runs
            train_df, val_df, test_df = make_fold_splits(dec_df, fold, args.val_frac, seed)
            if args.smoke:
                train_df, val_df, test_df = tm.smoke_subsample(train_df, val_df, test_df, seed)
            assert_disjoint(train_df, val_df, test_df)
            print(f"  stage 2 (deception): train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
                  f"val {len(val_df)}, test {len(test_df)}")
            stats = compute_stats(train_df, args.features_root, cache)
            val_loader = build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache,
                                      workers=args.workers, device=device)
            test_loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache,
                                       workers=args.workers, device=device)
            seed_everything(seed)
            train_loader = build_loader(train_df, args.features_root, stats, args.batch_size, train=True,
                                        balance_by=("dataset",), seed=seed, workers=args.workers, cache=cache,
                                        device=device)
            model = build_transfer_model(backbone, args.mode, args.dropout)
            info = train_model(model, train_loader, val_loader, device, pos_weight=pos_weight(train_df), lr=args.lr,
                               weight_decay=args.weight_decay, max_epochs=max_epochs, patience=args.patience,
                               accum_steps=args.accum_steps, forward=forward_multimodal, log_every=log_every)
            preds = predict(model, test_loader, device, forward_multimodal)
            run_rows = eval_rows(preds, EVAL_SETS, task_of_dataset, setup, args.mode, fold, seed)

            ckpt_path = ckpt_dir / f"{args.mode}_fold{fold}_seed{seed}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "stats": stats.to_dict(),
                        "setup": setup, "mode": args.mode, "fold": fold, "seed": seed,
                        "best_epoch": info["best_epoch"], "args": vars(args)}, ckpt_path)

            elapsed = time.perf_counter() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
            run_info = {"dropout": args.dropout, "weight_decay": args.weight_decay, "n_train": len(train_df),
                        "n_val": len(val_df), "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                        "val_score": info["best_val_macro_f1"], **s1, "time_s": round(elapsed, 1),
                        "peak_gpu_mb": round(peak_mb, 1), "device": device.type}
            # Predictions first, the results rows last: they mark the run as finished.
            append_rows(pred_csv, preds.assign(mode=args.mode, fold=fold, seed=seed, setup=setup))
            append_rows(out_csv, [{**r, **run_info} for r in run_rows])
            finished.add(run_key(setup, args.mode, fold, seed))
            print(f"[{args.mode} fold {fold} seed {seed}] "
                  + "  ".join(f"{r['test_dataset']}: acc {r['accuracy']:.3f} F1 {r['macro_f1']:.3f} AUC {r['roc_auc']:.3f}"
                              for r in run_rows)
                  + f"\n    stage 2 best epoch {info['best_epoch']}/{info['epochs_run']}  | time per run (both stages) "
                  f"{elapsed:.1f} s  | "
                  + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
            del model, train_loader, backbone
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not out_csv.is_file():
        print("No results were produced.")
        return
    results = pd.read_csv(out_csv)
    results = results[results["setup"] == setup]
    print("\nTransfer summary (mean ± std over runs)")
    summarize(results, summary_csv)
    s1 = results.drop_duplicates(["fold", "seed"])
    print(f"  stage-1 MAS test: macro_f1 {s1['stage1_mas_macro_f1'].mean():.3f} ± {s1['stage1_mas_macro_f1'].std():.3f} "
          f"over {len(s1)} runs")
    compare(args, out_csv, pred_csv, vs_csv, args.smoke)
    print(f"\nWrote {out_csv}, {summary_csv}, {pred_csv}; checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
