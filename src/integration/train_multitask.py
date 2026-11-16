#!/usr/bin/env python3
"""Multi-task (stress + deception) training of MultimodalModel and paired comparison with single-task runs.

Examples (run from the repository root):
    # quick check on CPU: 2 epochs on a few clips, writes *_smoke files, never resumes
    python src/integration/train_multitask.py --smoke
    # full run: 5 test folds x seeds 0 1 2, concat fusion
    python src/integration/train_multitask.py
    # only rebuild the comparison table from an existing results file
    python src/integration/train_multitask.py --compare-only

Protocol (per test fold k and seed)
    * One model with two linear heads ("stress", "deception") on a shared concat-fusion encoder.
    * test  = clips with fold == k in each of mas, bol, rlt (the same clips as the single-task runs);
      each dataset is evaluated on its own fold k, and bol+rlt is also reported pooled.
    * val   = 15 % of the remaining GROUPS (dataset:group_id) drawn over all three datasets with the run
      seed (train.make_fold_splits); train = the rest. Feature statistics come from train rows only.
    * Batches come from BalancedBatchSampler(balance_by=("task", "dataset")): half stress (mas), half
      deception, the deception half split between bol and rlt.
    * Masked loss: total = lambda_s * BCE(stress clips, pos_weight_s) + lambda_d * BCE(deception clips,
      pos_weight_d); each clip's logit comes from the head of its own task (batch["task"]), and each
      pos_weight is n_neg / n_pos of that task's training clips (as in the single-task runs).
    * Early stopping on the mean over the two tasks of validation macro-F1 (patience 10, <= 100 epochs).
    * AdamW lr 1e-3, weight decay 1e-2, dropout 0.3, fp16 autocast on CUDA.

Comparison with the single-task results (same mode, fold and seed; paired difference multitask - single)
    stress:    results/stress/deep_runs.csv  (mode, fold, seed -> accuracy, macro_f1, roc_auc)
    deception: results/deception/runs/pooled_predictions.csv (mode, fold, seed, dataset). The per-run
               file pooled.csv has no per-dataset columns (one row per pooled bol+rlt test fold), so
               per-dataset metrics are recomputed from the per-clip predictions and cross-checked
               against pooled.csv.
    Folds, seeds, test-set sizes and test clip ids are checked between the files; mismatches are printed.

Outputs (OUT = --out, default results/integration/multitask.csv)
    OUT                          one row per (fold, seed, test_dataset): mas, bol, rlt and bol+rlt
    OUT stem + _summary.csv      mean / std over runs per test_dataset
    OUT stem + _predictions.csv  every test clip: clip_id, dataset, task, label, prob, fold, seed, ...
    OUT stem + _vs_single.csv    the paired comparison
    <ckpt-dir>/<OUT stem>/<mode>_fold<k>_seed<s>.pt
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import TASK_TO_ID, compute_stats, load_metadata  # noqa: E402
from models import MultimodalModel  # noqa: E402
from train import (METRICS, append_rows, build_loader, choose_device, compute_metrics,  # noqa: E402
                   drop_unfinished_predictions, load_finished_runs, make_fold_splits, output_paths,
                   predict, run_key, seed_everything, train_model)

DATASETS = ["mas", "bol", "rlt"]
HEADS = ("stress", "deception")
BALANCE_BY = ("task", "dataset")
SMOKE_PER_DATASET = {"train": 6, "val": 2, "test": 3}
EVAL_SETS = [("mas", ["mas"]), ("bol", ["bol"]), ("rlt", ["rlt"]), ("bol+rlt", ["bol", "rlt"])]
DECEPTION_TEST_SETS = ("bol", "rlt", "bol+rlt")


# ------------------------------------------------------------------ model, loss, score
def build_multitask_model(mode, dropout):
    return MultimodalModel(fusion=mode, heads=HEADS, dropout=dropout, return_dict=True)


def forward_multitask(model, batch, device):
    """(B,) logits: each clip's logit comes from the head of its own task (batch["task"])."""
    out = model(batch["audio"].to(device, non_blocking=True), batch["video"].to(device, non_blocking=True),
                batch["audio_mask"].to(device, non_blocking=True),
                batch["video_mask"].to(device, non_blocking=True), return_dict=True)
    is_stress = (batch["task"].to(device) == TASK_TO_ID["stress"])
    return torch.where(is_stress, out["stress"], out["deception"])


def make_masked_loss(lambda_s, lambda_d, pos_weight_s, pos_weight_d):
    """loss_fn(logits, batch, device) = lambda_s * BCE(stress clips) + lambda_d * BCE(deception clips)."""
    terms = ((TASK_TO_ID["stress"], lambda_s, pos_weight_s), (TASK_TO_ID["deception"], lambda_d, pos_weight_d))

    def loss_fn(logits, batch, device):
        y, task, logits = batch["label"].to(device), batch["task"].to(device), logits.float()
        total = logits.new_zeros(())
        for task_id, lam, pw in terms:
            m = task == task_id
            if m.any():
                total = total + lam * F.binary_cross_entropy_with_logits(
                    logits[m], y[m], pos_weight=torch.tensor(float(pw), device=device))
        return total
    return loss_fn


def make_score_fn(task_of_dataset):
    """Early-stopping score: mean over the tasks present of validation macro-F1."""
    def score_fn(val):
        task = val["dataset"].map(task_of_dataset)
        scores = [compute_metrics(g["label"], g["prob"])["macro_f1"] for _, g in val.groupby(task)]
        return float(np.mean(scores))
    return score_fn


def task_pos_weight(train_df, task):
    d = train_df[train_df["task"] == task]
    n_pos = int(d["label"].sum())
    return (len(d) - n_pos) / n_pos if n_pos else 1.0


def smoke_subsample(train, val, test, seed):
    """A few clips of every dataset, so that both tasks are in every split."""
    out = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        parts = [g.sample(n=min(len(g), SMOKE_PER_DATASET[name]), random_state=seed) for _, g in df.groupby("dataset")]
        out.append(pd.concat(parts).reset_index(drop=True))
    return tuple(out)


def setup_id(args):
    return (f"task=stress+deception;train=mas+bol+rlt;test=mas+bol+rlt;dropout={args.dropout:g};"
            f"wd={args.weight_decay:g};lambda_s={args.lambda_s:g};lambda_d={args.lambda_d:g};balance=task+dataset")


# ----------------------------------------------------------------------- comparison
def _paired_check(name, mt_keys, st_keys, smoke=False):
    only_mt, only_st = sorted(mt_keys - st_keys), sorted(st_keys - mt_keys)
    if only_mt or (only_st and not smoke):
        print(f"  !! {name}: (fold, seed) only in multitask: {only_mt[:8]}; only in single-task: {only_st[:8]}")
        return False
    return True


def single_task_stress(path, mode, dropout, wd):
    s = pd.read_csv(path)
    s = s[s["mode"] == mode]
    for col, val in (("dropout", dropout), ("weight_decay", wd)):
        if col in s:
            s = s[np.isclose(s[col], val)]
    return s[["fold", "seed", "n_test", *METRICS]].assign(test_dataset="mas").reset_index(drop=True)


def single_task_deception(pred_path, runs_path, mode, dropout, wd):
    """Per-run, per-dataset metrics (bol, rlt, pooled bol+rlt) from the pooled-setting predictions."""
    p = pd.read_csv(pred_path)
    p = p[p["mode"] == mode]
    if "setup" in p:
        p = p[p["setup"].str.contains(f"dropout={dropout:g};wd={wd:g}", regex=False)]
    rows = []
    for (fold, seed), g in p.groupby(["fold", "seed"]):
        for name, members in EVAL_SETS[1:]:
            d = g[g["dataset"].isin(members)]
            if len(d):
                rows.append({"fold": fold, "seed": seed, "test_dataset": name, "n_test": len(d),
                             **compute_metrics(d["label"], d["prob"])})
    single = pd.DataFrame(rows)
    # cross-check the recomputed pooled rows against the per-run file pooled.csv
    r = pd.read_csv(runs_path)
    r = r[r["mode"] == mode]
    pooled = single[single["test_dataset"] == "bol+rlt"].merge(r, on=["fold", "seed"], suffixes=("", "_file"))
    diffs = {m: float(np.nanmax(np.abs(pooled[m] - pooled[m + "_file"]))) for m in METRICS}
    n_bad = int((pooled["n_test"] != pooled["n_test_file"]).sum())
    print(f"  pooled.csv vs recomputed from pooled_predictions.csv ({len(pooled)} runs): max |diff| "
          + ", ".join(f"{m} {v:.1e}" for m, v in diffs.items()) + f"; n_test mismatches {n_bad}")
    if n_bad or max(diffs.values()) > 1e-6:
        print("  !! pooled_predictions.csv does not reproduce pooled.csv")
    return single


def compare(mt_csv, mt_pred_csv, setup, args, smoke=False):
    """Paired difference (multitask - single-task) per test dataset and metric. Returns a DataFrame."""
    print("\n== Paired comparison with the single-task results (mode %s)%s" %
          (args.mode, "  [SMOKE: subsampled test clips, numbers are meaningless]" if smoke else ""))
    mt = pd.read_csv(mt_csv)
    mt = mt[mt["setup"] == setup]
    single = pd.concat([single_task_stress(args.stress_runs, args.mode, args.dropout, args.weight_decay),
                        single_task_deception(args.deception_preds, args.deception_runs, args.mode,
                                              args.dropout, args.weight_decay)], ignore_index=True)

    # alignment of folds, seeds, test sizes and test clips
    ok = True
    for ds in ("mas", *DECEPTION_TEST_SETS):
        a, b = mt[mt["test_dataset"] == ds], single[single["test_dataset"] == ds]
        ok &= _paired_check(ds, set(zip(a["fold"], a["seed"])), set(zip(b["fold"], b["seed"])), smoke)
    m = mt.merge(single, on=["fold", "seed", "test_dataset"], suffixes=("_mt", "_st"))
    bad = m[m["n_test_mt"] != m["n_test_st"]]
    if len(bad) and not smoke:
        ok = False
        print(f"  !! {len(bad)} paired runs have different n_test, e.g. "
              f"{bad[['test_dataset', 'fold', 'seed', 'n_test_mt', 'n_test_st']].head(3).values.tolist()}")
    mt_pred = pd.read_csv(mt_pred_csv)
    mt_pred = mt_pred[mt_pred["setup"] == setup]
    st_pred = pd.concat([
        pd.read_csv(args.stress_predictions).query("mode == @args.mode")[["clip_id", "fold", "seed"]].assign(dataset="mas"),
        pd.read_csv(args.deception_preds).query("mode == @args.mode")[["clip_id", "fold", "seed", "dataset"]]])
    clips = lambda d: {k: frozenset(g["clip_id"]) for k, g in d.groupby(["dataset", "fold", "seed"])}
    c_mt, c_st = clips(mt_pred), clips(st_pred)
    diff_sets = [k for k in c_mt if k in c_st and c_mt[k] != c_st[k]]
    if diff_sets and not smoke:
        ok = False
        print(f"  !! test clip sets differ in {len(diff_sets)} (dataset, fold, seed) cells, e.g. {diff_sets[:3]}")
    if smoke:
        print(f"  (smoke) test clip sets differ from the single-task ones in {len(diff_sets)} of {len(c_mt)} cells, "
              "as expected for subsampled clips")
    elif ok:
        print(f"  OK: same folds, seeds, n_test and test clip ids in all {len(c_mt)} (dataset, fold, seed) cells; "
              f"matched on [fold, seed, test_dataset] (+ mode={args.mode}, dropout={args.dropout:g}, "
              f"weight_decay={args.weight_decay:g})")

    rows = []
    for ds in ("mas", *DECEPTION_TEST_SETS):
        g = m[m["test_dataset"] == ds]
        for metric in METRICS:
            d = (g[metric + "_mt"] - g[metric + "_st"]).dropna()
            rows.append({"test_dataset": ds, "head": "stress" if ds == "mas" else "deception", "metric": metric,
                         "n_pairs": len(d), "multitask_mean": g.loc[d.index, metric + "_mt"].mean(),
                         "single_mean": g.loc[d.index, metric + "_st"].mean(),
                         "diff_mean": d.mean() if len(d) else np.nan,
                         "diff_std": d.std(ddof=1) if len(d) > 1 else np.nan,
                         "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())})
    table = pd.DataFrame(rows)
    print(f"  {'test set':8s} {'metric':9s} {'n':>3s} {'single':>7s} {'multi':>7s}   diff mean ± std   (better/worse runs)")
    for r in table.itertuples():
        print(f"  {r.test_dataset:8s} {r.metric:9s} {r.n_pairs:3d} {r.single_mean:7.3f} {r.multitask_mean:7.3f}   "
              f"{r.diff_mean:+.3f} ± {r.diff_std:.3f}      ({r.n_better}/{r.n_worse})")
    print("  diff = multitask - single-task over paired (fold, seed) runs; rlt AUC pairs are fewer where a "
          "test fold has one class.")
    return table


# ------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Multi-task (stress + deception) training of MultimodalModel.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="concat", choices=["audio", "video", "concat", "late", "xattn"],
                    help="fusion mode (also selects the single-task rows to compare with)")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lambda-s", type=float, default=1.0, help="weight of the stress BCE")
    ap.add_argument("--lambda-d", type=float, default=1.0, help="weight of the deception BCE")
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
    ap.add_argument("--out", default="results/integration/multitask.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--stress-runs", default="results/stress/deep_runs.csv")
    ap.add_argument("--stress-predictions", default="results/stress/deep_predictions.csv")
    ap.add_argument("--deception-runs", default="results/deception/runs/pooled.csv")
    ap.add_argument("--deception-preds", default="results/deception/runs/pooled_predictions.csv")
    ap.add_argument("--compare-only", action="store_true", help="skip training, rebuild the comparison")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a few clips per dataset, first 2 folds and 2 seeds; writes *_smoke files")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_csv, summary_csv, pred_csv, stem = output_paths(args.out, args.smoke)
    vs_csv = out_csv.with_name(stem + "_vs_single.csv")
    setup = setup_id(args)
    if args.compare_only:
        compare(out_csv, pred_csv, setup, args, args.smoke).to_csv(vs_csv, index=False)
        print(f"Wrote {vs_csv}")
        return

    folds, seeds = (args.folds[:2], args.seeds[:2]) if args.smoke else (args.folds, args.seeds)
    max_epochs = 2 if args.smoke else args.max_epochs
    log_every = 1 if args.smoke else 10
    ckpt_dir = Path(args.ckpt_dir) / stem
    if args.smoke:
        for p in (out_csv, summary_csv, pred_csv, vs_csv):
            p.unlink(missing_ok=True)

    device = choose_device()
    df = load_metadata(args.features_root, DATASETS, missing="drop" if args.drop_missing else "error")
    task_of_dataset = df.groupby("dataset")["task"].agg(lambda s: s.unique()[0]).to_dict()
    assert df.groupby("dataset")["task"].nunique().max() == 1, "a dataset has more than one task"
    print(f"Loaded {len(df)} clips: " + ", ".join(f"{d} ({task_of_dataset[d]}) {n}"
                                                  for d, n in df["dataset"].value_counts().sort_index().items()))
    cache = {} if args.preload else None
    finished = set() if args.smoke else load_finished_runs(out_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Setup: {setup}\nResults: {out_csv}  ({len(finished)} finished runs found)")
    score_fn = make_score_fn(task_of_dataset)

    for fold in folds:
        for seed in seeds:
            if run_key(setup, args.mode, fold, seed) in finished:
                print(f"[{args.mode} fold {fold} seed {seed}] already finished, skipping")
                continue
            train_df, val_df, test_df = make_fold_splits(df, fold, args.val_frac, seed)
            if args.smoke:
                train_df, val_df, test_df = smoke_subsample(train_df, val_df, test_df, seed)
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                da, db = {"train": train_df, "val": val_df, "test": test_df}[a], {"train": train_df, "val": val_df, "test": test_df}[b]
                assert not (set(da["group_key"]) & set(db["group_key"])), f"{a}/{b} share a group"
            print(f"\n== fold {fold}, seed {seed}: train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
                  f"val {len(val_df)}, test {len(test_df)} ("
                  + ", ".join(f"{d} {n}" for d, n in test_df['dataset'].value_counts().sort_index().items()) + ")")

            stats = compute_stats(train_df, args.features_root, cache)  # training rows only
            pw_s, pw_d = task_pos_weight(train_df, "stress"), task_pos_weight(train_df, "deception")
            val_loader = build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache,
                                      workers=args.workers, device=device)
            test_loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache,
                                       workers=args.workers, device=device)
            print(f"[{args.mode} fold {fold} seed {seed}] pos_weight stress {pw_s:.3f}, deception {pw_d:.3f}")
            t0 = time.perf_counter()
            seed_everything(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_loader = build_loader(train_df, args.features_root, stats, args.batch_size, train=True,
                                        balance_by=BALANCE_BY, seed=seed, workers=args.workers, cache=cache,
                                        device=device)
            model = build_multitask_model(args.mode, args.dropout)
            info = train_model(model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
                               max_epochs=max_epochs, patience=args.patience, accum_steps=args.accum_steps,
                               forward=forward_multitask, log_every=log_every,
                               loss_fn=make_masked_loss(args.lambda_s, args.lambda_d, pw_s, pw_d), score_fn=score_fn)
            preds = predict(model, test_loader, device, forward_multitask)
            preds["task"] = preds["dataset"].map(task_of_dataset)

            run_rows = []
            for name, members in EVAL_SETS:
                d = preds[preds["dataset"].isin(members)]
                if d.empty:
                    continue
                run_rows.append({
                    "setup": setup, "mode": args.mode, "fold": fold, "seed": seed, "test_dataset": name,
                    "task": task_of_dataset[members[0]], "n_test": len(d), **compute_metrics(d["label"], d["prob"])})

            ckpt_path = ckpt_dir / f"{args.mode}_fold{fold}_seed{seed}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "stats": stats.to_dict(),
                        "setup": setup, "mode": args.mode, "fold": fold, "seed": seed,
                        "best_epoch": info["best_epoch"], "args": vars(args)}, ckpt_path)

            elapsed = time.perf_counter() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
            run_info = {"lambda_s": args.lambda_s, "lambda_d": args.lambda_d, "dropout": args.dropout,
                        "weight_decay": args.weight_decay, "n_train": len(train_df), "n_val": len(val_df),
                        "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                        "val_score": info["best_val_macro_f1"], "time_s": round(elapsed, 1),
                        "peak_gpu_mb": round(peak_mb, 1), "device": device.type}
            # Predictions first, the results rows last: they mark the run as finished.
            append_rows(pred_csv, preds.assign(mode=args.mode, fold=fold, seed=seed, setup=setup))
            append_rows(out_csv, [{**r, **run_info} for r in run_rows])
            finished.add(run_key(setup, args.mode, fold, seed))
            print(f"[{args.mode} fold {fold} seed {seed}] "
                  + "  ".join(f"{r['test_dataset']}: acc {r['accuracy']:.3f} F1 {r['macro_f1']:.3f} AUC {r['roc_auc']:.3f}"
                              for r in run_rows)
                  + f"\n    best epoch {info['best_epoch']}/{info['epochs_run']}  | time per run {elapsed:.1f} s  | "
                  + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
            del model, train_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not out_csv.is_file():
        print("No results were produced.")
        return
    results = pd.read_csv(out_csv)
    results = results[results["setup"] == setup]
    summary = (results.groupby(["test_dataset"], sort=False)
               .agg(n_runs=("fold", "size"), **{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")},
                    time_s_mean=("time_s", "mean")).reset_index())
    summary.to_csv(summary_csv, index=False)
    print("\nMultitask summary (mean ± std over runs)")
    for r in summary.itertuples():
        print(f"  {r.test_dataset:8s} (n={r.n_runs})  " + "  ".join(
            f"{m} {getattr(r, m + '_mean'):.3f} ± {getattr(r, m + '_std'):.3f}" for m in METRICS))
    compare(out_csv, pred_csv, setup, args, args.smoke).to_csv(vs_csv, index=False)
    print(f"\nWrote {out_csv}, {summary_csv}, {pred_csv}, {vs_csv}; checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
