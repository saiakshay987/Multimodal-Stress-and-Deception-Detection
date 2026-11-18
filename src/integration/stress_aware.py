#!/usr/bin/env python3
"""Stress-conditioned deception: the deception head sees [shared embedding | stress probability].

Examples (run from the repository root):
    python src/integration/stress_aware.py --smoke      # 2 epochs on a few clips, writes *_smoke files
    python src/integration/stress_aware.py --preload    # 5 test folds x seeds 0 1 2, concat fusion

Model: the two-head MultimodalModel of train_multitask.py (shared encoders + fusion + dropout) with a stress
head Linear(d, 1) as before, and a deception head Linear(d + 1, 1) whose input is the shared embedding
concatenated with sigmoid(stress logit). That probability is DETACHED, so the deception loss cannot train the
stress head through it (it still trains the shared embedding, exactly as in the plain two-head model). The stress
logit of a deception clip is the stress head's output on that clip; it is not supervised there.

Everything else is imported from train_multitask.py and used unchanged: same splits (fold column, val = 15 % of
the remaining groups drawn with the run seed), balanced task+dataset batches, masked loss, early stopping on the
mean validation macro-F1 of the two tasks, AdamW lr 1e-3, fp16 autocast on CUDA, forward_multitask.

Comparison: run by run with the plain two-head model from results/integration/multitask.csv (same mode, dropout,
weight decay, lambda, fold, seed). Run `python src/integration/train_multitask.py --mode concat --seeds 0 1 2` first.

Outputs (OUT = --out, default results/integration/stress_aware.csv)
    OUT                          one row per (fold, seed, test_dataset): mas, bol, rlt and bol+rlt
    OUT stem + _summary.csv      mean / std over runs per test_dataset
    OUT stem + _predictions.csv  every test clip
    OUT stem + _vs_plain.csv     paired difference (stress-aware - plain two-head)
    <ckpt-dir>/<OUT stem>/<mode>_fold<k>_seed<s>.pt
Runs already present in OUT are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import compute_stats, load_metadata  # noqa: E402
from models import MultimodalModel  # noqa: E402
from train import (append_rows, build_loader, choose_device, drop_unfinished_predictions,  # noqa: E402
                   load_finished_runs, make_fold_splits, output_paths, predict, run_key, seed_everything,
                   train_model)
import train_multitask as tm  # noqa: E402
from exp_utils import assert_disjoint, eval_rows, paired_table, summarize  # noqa: E402


class StressConditionedModel(nn.Module):
    """Shared MultimodalModel (stress head only) + a deception head on [embedding | detached stress prob]."""

    def __init__(self, mode, dropout):
        super().__init__()
        if mode == "late":
            raise ValueError("late fusion has no shared embedding to condition on")
        self.base = MultimodalModel(fusion=mode, heads=("stress",), dropout=dropout, return_dict=True)
        self.dec_head = nn.Linear(self.base.heads["stress"].in_features + 1, 1)
        self._emb = None
        # the tensor that feeds the heads is the output of base.drop: capture it instead of re-implementing forward
        self.base.drop.register_forward_hook(lambda module, inputs, output: setattr(self, "_emb", output))

    def forward(self, audio, video, audio_mask, video_mask, video_valid=None, return_dict=True):
        self._emb = None
        stress = self.base(audio, video, audio_mask, video_mask, video_valid=video_valid, return_dict=True)["stress"]
        emb = self._emb
        p = torch.sigmoid(stress.float()).detach().unsqueeze(-1).to(emb.dtype)  # no gradient into the stress head
        deception = self.dec_head(torch.cat([emb, p], dim=-1)).squeeze(-1)
        return {"stress": stress, "deception": deception}


def check_detach(mode, dropout, device):
    """The deception loss must reach the shared encoders but not the stress head."""
    torch.manual_seed(0)
    model = StressConditionedModel(mode, dropout).to(device).train()
    b, t = 4, 12
    audio, video = torch.randn(b, t, 771, device=device), torch.randn(b, t, 770, device=device)
    mask = torch.zeros(b, t, dtype=torch.bool, device=device)
    out = model(audio, video, mask, mask)
    out["deception"].sum().backward()
    stress_grads = [p.grad for p in model.base.heads["stress"].parameters()]
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in stress_grads), "deception loss reaches the stress head"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.base.audio_enc.parameters()), \
        "deception loss does not reach the shared encoder"
    model.zero_grad()
    out = model(audio, video, mask, mask)
    out["stress"].sum().backward()
    assert any(g.grad is not None and float(g.grad.abs().sum()) > 0 for g in model.base.heads["stress"].parameters())
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.dec_head.parameters())
    print("check: deception loss -> shared encoder yes, stress head no (detach works); stress loss -> deception head no")


def setup_id(args):
    return tm.setup_id(args) + ";cond=stress_prob_detached"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stress-conditioned deception head.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="concat", choices=["audio", "video", "concat", "xattn"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lambda-s", type=float, default=1.0)
    ap.add_argument("--lambda-d", type=float, default=1.0)
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
    ap.add_argument("--out", default="results/integration/stress_aware.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--plain", default="results/integration/multitask.csv", help="results of train_multitask.py")
    ap.add_argument("--compare-only", action="store_true", help="skip training, rebuild the comparison")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a few clips, 2 folds x 2 seeds; *_smoke files")
    return ap.parse_args(argv)


def compare(args, out_csv, pred_csv, vs_csv, smoke):
    """Paired comparison with the plain two-head run of the same configuration."""
    plain_args = SimpleNamespace(dropout=args.dropout, weight_decay=args.weight_decay,
                                 lambda_s=args.lambda_s, lambda_d=args.lambda_d)
    plain_setup = tm.setup_id(plain_args)
    plain_csv = Path(args.plain)
    plain_pred = plain_csv.with_name(plain_csv.stem + "_predictions.csv")
    if not plain_csv.is_file():
        print(f"\nNo plain two-head results at {plain_csv}: run train_multitask.py first; comparison skipped.")
        return
    plain = pd.read_csv(plain_csv)
    plain = plain[(plain["setup"] == plain_setup) & (plain["mode"] == args.mode)]
    new = pd.read_csv(out_csv)
    new = new[(new["setup"] == setup_id(args)) & (new["mode"] == args.mode)]
    print(f"\n== Paired comparison with the plain two-head model (mode {args.mode}); plain setup: {plain_setup}")
    p_pred = pd.read_csv(plain_pred).query("setup == @plain_setup and mode == @args.mode")
    n_pred = pd.read_csv(pred_csv).query("mode == @args.mode")
    table = paired_table(new, plain, [n for n, _ in tm.EVAL_SETS], "stress_aware", "plain", n_pred, p_pred, smoke)
    table.to_csv(vs_csv, index=False)
    print(f"Wrote {vs_csv}")


def main(argv=None):
    args = parse_args(argv)
    out_csv, summary_csv, pred_csv, stem = output_paths(args.out, args.smoke)
    vs_csv = out_csv.with_name(stem + "_vs_plain.csv")
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
    check_detach(args.mode, args.dropout, device)
    df = load_metadata(args.features_root, tm.DATASETS, missing="drop" if args.drop_missing else "error")
    task_of_dataset = df.groupby("dataset")["task"].agg(lambda s: s.unique()[0]).to_dict()
    assert df.groupby("dataset")["task"].nunique().max() == 1, "a dataset has more than one task"
    cache = {} if args.preload else None
    finished = set() if args.smoke else load_finished_runs(out_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Loaded {len(df)} clips. Setup: {setup}\nResults: {out_csv}  ({len(finished)} finished runs found)")
    score_fn = tm.make_score_fn(task_of_dataset)

    for fold in folds:
        for seed in seeds:
            if run_key(setup, args.mode, fold, seed) in finished:
                print(f"[{args.mode} fold {fold} seed {seed}] already finished, skipping")
                continue
            train_df, val_df, test_df = make_fold_splits(df, fold, args.val_frac, seed)
            if args.smoke:
                train_df, val_df, test_df = tm.smoke_subsample(train_df, val_df, test_df, seed)
            assert_disjoint(train_df, val_df, test_df)
            print(f"\n== fold {fold}, seed {seed}: train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
                  f"val {len(val_df)}, test {len(test_df)}")
            stats = compute_stats(train_df, args.features_root, cache)  # training rows only
            pw_s, pw_d = tm.task_pos_weight(train_df, "stress"), tm.task_pos_weight(train_df, "deception")
            val_loader = build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache,
                                      workers=args.workers, device=device)
            test_loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache,
                                       workers=args.workers, device=device)
            t0 = time.perf_counter()
            seed_everything(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_loader = build_loader(train_df, args.features_root, stats, args.batch_size, train=True,
                                        balance_by=tm.BALANCE_BY, seed=seed, workers=args.workers, cache=cache,
                                        device=device)
            model = StressConditionedModel(args.mode, args.dropout)
            info = train_model(model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
                               max_epochs=max_epochs, patience=args.patience, accum_steps=args.accum_steps,
                               forward=tm.forward_multitask, log_every=log_every,
                               loss_fn=tm.make_masked_loss(args.lambda_s, args.lambda_d, pw_s, pw_d), score_fn=score_fn)
            preds = predict(model, test_loader, device, tm.forward_multitask)
            preds["task"] = preds["dataset"].map(task_of_dataset)
            run_rows = eval_rows(preds, tm.EVAL_SETS, task_of_dataset, setup, args.mode, fold, seed)

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
    print("\nStress-aware summary (mean ± std over runs)")
    summarize(results, summary_csv)
    compare(args, out_csv, pred_csv, vs_csv, args.smoke)
    print(f"\nWrote {out_csv}, {summary_csv}, {pred_csv}; checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
