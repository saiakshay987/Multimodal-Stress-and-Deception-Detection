#!/usr/bin/env python3
"""Train the chosen deception model on ALL Bag-of-Lies + Real-Life Trial clips for a fixed number of epochs.

    python src/deception/train_final.py --out checkpoints/deception_final.pt

Chosen setup: frozen-feature MultimodalModel, late fusion (audio head + video head, logits averaged),
trained on both datasets pooled with dataset-balanced batches (data.BalancedBatchSampler, same as
``train.py --balance dataset``). No validation split: the epoch count is the median ``best_epoch`` of
that mode over the cross-validation runs in --runs-csv (results/deception/runs/pooled.csv, 5 folds x
3 seeds). Feature z-scoring statistics come from all training clips. Loss, optimiser and batch size
are those of train.py (AdamW lr 1e-3, BCE with pos_weight = n_neg / n_pos, batch 16, fp16 autocast
on CUDA). Uses src/common unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import compute_stats, load_metadata  # noqa: E402
from train import (BALANCE_CHOICES, _autocast, build_loader, build_model, choose_device,  # noqa: E402
                   forward_multimodal, seed_everything)

DATASETS = ["bol", "rlt"]


def median_best_epoch(runs_csv, mode, train_datasets):
    """(median best_epoch, list of best epochs) of ``mode`` runs trained on ``train_datasets``."""
    runs = pd.read_csv(runs_csv)
    runs = runs[(runs["mode"] == mode) & (runs["train_datasets"] == "+".join(train_datasets))]
    if runs.empty:
        raise SystemExit(f"no {mode!r} runs trained on {train_datasets} in {runs_csv}")
    return int(round(runs["best_epoch"].median())), runs["best_epoch"].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--runs-csv", default="results/deception/runs/pooled.csv",
                    help="cross-validation runs (needs columns mode, train_datasets, best_epoch)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=["bol", "rlt"])
    ap.add_argument("--mode", default="late", choices=["audio", "video", "concat", "late", "xattn"])
    ap.add_argument("--balance", default="dataset", choices=list(BALANCE_CHOICES))
    ap.add_argument("--epochs", type=int, default=None, help="default: median best_epoch of --mode in --runs-csv")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/deception_final.pt")
    a = ap.parse_args()

    seed_everything(a.seed)
    device = choose_device()
    cv_epochs = None
    if a.epochs is None:
        a.epochs, cv_epochs = median_best_epoch(a.runs_csv, a.mode, a.datasets)
        print(f"cross-validation best epochs ({len(cv_epochs)} runs): {sorted(cv_epochs)}")
    print(f"mode {a.mode}, {a.epochs} epochs, balance={a.balance}")

    df = load_metadata(a.features_root, a.datasets, task="deception")
    stats = compute_stats(df, a.features_root)
    loader = build_loader(df, a.features_root, stats, a.batch_size, train=True,
                          balance_by=BALANCE_CHOICES[a.balance], seed=a.seed, device=device)
    n_pos = int(df["label"].sum())
    pos_weight = (len(df) - n_pos) / max(n_pos, 1)
    print(f"{len(df)} clips ({', '.join(f'{d} {n}' for d, n in df['dataset'].value_counts().sort_index().items())}), "
          f"{df['group_key'].nunique()} groups, {n_pos} deceptive, pos_weight {pos_weight:.3f}")

    model = build_model(a.mode, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total, n = 0.0, 0
        for i, batch in enumerate(loader):
            with _autocast(device):
                logits = forward_multimodal(model, batch, device)
            y = batch["label"].to(device)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / a.accum_steps).backward()
            if (i + 1) % a.accum_steps == 0 or i + 1 == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * len(y)
            n += len(y)
        print(f"  epoch {epoch:2d}  train_loss {total / n:.4f}")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    else:
        print("peak GPU memory: n/a (cpu)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "stats": stats.to_dict(),
                "mode": a.mode, "dropout": a.dropout, "weight_decay": a.weight_decay, "lr": a.lr,
                "batch_size": a.batch_size, "epochs": a.epochs, "seed": a.seed, "n_train": len(df),
                "train_datasets": a.datasets, "balance": a.balance, "cv_best_epochs": cv_epochs,
                "pos_weight": float(pos_weight)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
