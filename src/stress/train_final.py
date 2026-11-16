#!/usr/bin/env python3
"""Train the chosen stress model (audio-only MultimodalModel) on ALL MAS clips for a fixed number of epochs.

    python src/stress/train_final.py --epochs 7 --out checkpoints/stress_final.pt

No validation split: the epoch count is the median best epoch found in cross-validation
(results/stress/deep_runs.csv). Feature z-scoring statistics come from all MAS clips.
Uses src/common unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import compute_stats, load_metadata  # noqa: E402
from train import (_autocast, build_loader, build_model, choose_device, forward_multimodal,  # noqa: E402
                   seed_everything)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--runs-csv", default="results/stress/deep_runs.csv")
    ap.add_argument("--mode", default="audio")
    ap.add_argument("--epochs", type=int, default=None, help="default: median best_epoch of --mode in --runs-csv")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/stress_final.pt")
    a = ap.parse_args()

    seed_everything(a.seed)
    device = choose_device()
    if a.epochs is None:
        runs = pd.read_csv(a.runs_csv)
        a.epochs = int(round(runs.loc[runs["mode"] == a.mode, "best_epoch"].median()))
    print(f"mode {a.mode}, {a.epochs} epochs")

    df = load_metadata(a.features_root, ["mas"], task="stress")
    stats = compute_stats(df, a.features_root)
    loader = build_loader(df, a.features_root, stats, a.batch_size, train=True, seed=a.seed, device=device)
    pos_weight = (df["label"] == 0).sum() / max((df["label"] == 1).sum(), 1)
    print(f"{len(df)} MAS clips, {df['group_id'].nunique()} groups, pos_weight {pos_weight:.3f}")

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

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "stats": stats.to_dict(),
                "mode": a.mode, "dropout": a.dropout, "weight_decay": a.weight_decay, "epochs": a.epochs,
                "seed": a.seed, "n_train": len(df), "train_datasets": ["mas"]}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
