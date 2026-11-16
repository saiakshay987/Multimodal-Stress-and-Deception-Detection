#!/usr/bin/env python3
"""Video-only deception model trained from random initialisation on video_crops64 (no pretrained weights).

Model (about 0.1 M parameters)
    per frame   4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2] with 16, 32, 64, 64 channels
                (64x64 -> 4x4), then global average pooling -> 64 numbers per frame
    over frames one-layer GRU, hidden size 64 (last hidden state) -> dropout 0.3 -> Linear -> 1 logit

Input: a window of WINDOW = 32 consecutive frames (5 frames/s, about 6.4 s) of the (T, 64, 64) uint8
grayscale face crops, scaled to [0, 1]. Clips shorter than 32 frames are tiled (frame i -> i mod T).
    training    a random window every time a clip is drawn, random horizontal flip of the whole
                window (p = 0.5), brightness jitter (one factor in [0.7, 1.3] per window, clipped to [0, 1])
    validation / test   the first 32 frames, no augmentation
video_valid is not used: the model sees every frame of the window, face found or not.

Protocol = src/common/train.py, whose helpers are imported and used unchanged: the folds come from the
fold column of features/<dataset>/metadata.csv, validation = 15 % of the remaining groups drawn with the
run seed, AdamW (lr 1e-3, weight decay 1e-2), BCE with pos_weight = n_neg / n_pos, batch size 16, fp16
autocast on CUDA, early stopping on validation macro-F1 (patience 10, at most 100 epochs), the best
epoch is evaluated on the test fold. Within Bag-of-Lies and within Real-Life Trial, 5 folds x seeds 0 1 2.

Files (under --results-dir; runs already in the runs file are skipped, so the script can be restarted)
    runs/scratch_video_runs.csv         one row per (dataset, fold, seed), same columns as train.py
    runs/scratch_video_predictions.csv  every test clip of every run
    scratch_video.csv                   aggregate in the format of deep.csv (setting, test_dataset, mode, ...)
    <ckpt-dir>/scratch_video/<dataset>_fold<k>_seed<s>.pt
Then the video-only rows of deep.csv are printed next to scratch_video.csv.

    python src/deception/scratch_video.py                 # from the repository root
    python src/deception/scratch_video.py --datasets rlt --folds 0 --seeds 0 --max-epochs 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import load_metadata, npz_path  # noqa: E402  (src/common/data.py, unchanged)
from train import (append_rows, choose_device, compute_metrics, drop_unfinished_predictions,  # noqa: E402
                   load_finished_runs, make_fold_splits, predict, run_key, seed_everything, setup_id,
                   train_model)

MODE = "scratch_video"
WINDOW = 32
CHANNELS = (16, 32, 64, 64)
HIDDEN = 64
DATASETS = {"bol": "within_bol", "rlt": "within_rlt"}  # dataset -> setting name used in deep.csv
FLIP_P = 0.5
BRIGHTNESS = (0.7, 1.3)


# ------------------------------------------------------------------------------------- data
def load_crops(rows, features_root):
    """{clip_id: (T, 64, 64) uint8} for the clips of ``rows`` (kept in RAM; about 0.5 GB for both datasets)."""
    out = {}
    for ds, cid in zip(rows["dataset"], rows["clip_id"]):
        with np.load(npz_path(features_root, ds, cid)) as z:
            x = z["video_crops64"]
        if x.ndim != 3 or x.shape[1:] != (64, 64) or len(x) == 0:
            raise ValueError(f"{cid}: video_crops64 has shape {x.shape}, expected (T>0, 64, 64)")
        out[cid] = x
    return out


class CropWindows(Dataset):
    """One (WINDOW, 64, 64) float window per clip: random + augmented if ``train`` else the first frames."""

    def __init__(self, rows, crops, train):
        self.rows, self.crops, self.train = rows.reset_index(drop=True), crops, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.crops[r["clip_id"]]
        t = len(x)
        start = int(torch.randint(0, t - WINDOW + 1, (1,))) if self.train and t > WINDOW else 0
        idx = (start + np.arange(WINDOW)) % t  # tiles clips shorter than WINDOW
        w = torch.from_numpy(x[idx]).float() / 255.0
        if self.train:
            if torch.rand(1).item() < FLIP_P:
                w = w.flip(-1)
            w = (w * torch.empty(1).uniform_(*BRIGHTNESS).item()).clamp_(0.0, 1.0)
        return w, float(r["label"]), r["clip_id"], r["dataset"]


def collate(batch):
    w, y, cid, ds = zip(*batch)
    return {"video": torch.stack(w), "label": torch.tensor(y, dtype=torch.float32),
            "clip_id": list(cid), "dataset": list(ds)}


def make_loader(rows, crops, batch_size, train, seed, device):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(CropWindows(rows, crops, train), batch_size=batch_size, shuffle=train,
                      generator=gen if train else None, collate_fn=collate, num_workers=0,
                      pin_memory=device.type == "cuda")


# ------------------------------------------------------------------------------------ model
def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class ScratchVideoNet(nn.Module):
    """Per-frame CNN (16-32-64-64) + global average pooling, one-layer GRU, dropout, one logit."""

    def __init__(self, dropout=0.3):
        super().__init__()
        chans = (1, *CHANNELS)
        self.cnn = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])],
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru = nn.GRU(CHANNELS[-1], HIDDEN, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN, 1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """x (B, T, 64, 64) in [0, 1] -> logits (B,)."""
        b, t = x.shape[:2]
        f = self.cnn(x.reshape(b * t, 1, *x.shape[2:])).reshape(b, t, -1)
        _, h = self.gru(f)  # h (1, B, HIDDEN): state after the last frame
        return self.head(self.drop(h[-1])).squeeze(-1)


def forward_video(model, batch, device):
    return model(batch["video"].to(device, non_blocking=True))


# ------------------------------------------------------------------------------ aggregate
def aggregate(runs):
    """Same columns as deep.csv; ``runs`` is the runs CSV (one row per dataset x fold x seed)."""
    rows = []
    for ds, setting in DATASETS.items():
        g = runs[runs["test_datasets"] == ds]
        if g.empty:
            continue
        g = g.assign(correct=(g["accuracy"] * g["n_test"]).round().astype(int))
        by_seed = g.groupby("seed")[["correct", "n_test"]].sum()  # all folds of a seed = every clip once
        n_clips = int(by_seed["n_test"].iloc[0])
        p_worst = max(binomtest(int(k), int(n), 0.5).pvalue for k, n in by_seed.itertuples(index=False))
        row = {"setting": setting, "test_dataset": ds, "mode": MODE, "n_runs": len(g),
               "n_seeds": len(by_seed), "n_clips": n_clips}
        for m in ("accuracy", "macro_f1", "roc_auc"):
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        row["p_binom"] = binomtest(int(round(by_seed["correct"].mean())), n_clips, 0.5).pvalue
        row["p_binom_worst_seed"] = p_worst
        rows.append(row)
    return pd.DataFrame(rows)


def show_table(df):
    s = df.copy()
    for m in ("accuracy", "macro_f1", "roc_auc"):
        s[m] = [f"{a:.3f}±{b:.3f}" for a, b in zip(s[f"{m}_mean"], s[f"{m}_std"])]
    s["p_binom"] = s["p_binom"].map(lambda p: f"{p:.4f}")
    s["p_worst_seed"] = s["p_binom_worst_seed"].map(lambda p: f"{p:.4f}")
    cols = ["setting", "test_dataset", "mode", "n_runs", "n_clips", "accuracy", "macro_f1", "roc_auc",
            "p_binom", "p_worst_seed"]
    return s[cols].to_string(index=False)


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--results-dir", default="results/deception")
    ap.add_argument("--ckpt-dir", default="checkpoints/deception")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    rdir = Path(args.results_dir)
    runs_csv, pred_csv = rdir / "runs" / "scratch_video_runs.csv", rdir / "runs" / "scratch_video_predictions.csv"
    ckpt_dir = Path(args.ckpt_dir) / MODE
    device = choose_device()
    print(f"Model parameters: {ScratchVideoNet(args.dropout).num_parameters():,}")

    finished = load_finished_runs(runs_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Runs file: {runs_csv}  ({len(finished)} finished runs found)")

    t_all = time.perf_counter()
    for ds in args.datasets:
        df = load_metadata(args.features_root, [ds], task="deception")
        crops = load_crops(df, args.features_root)
        setup = setup_id("deception", [ds], [ds], args.dropout, args.weight_decay)
        print(f"\n######## within {ds}: {len(df)} clips, setup {setup}")
        for fold in args.folds:
            for seed in args.seeds:
                tag = f"[{MODE} {ds} fold {fold} seed {seed}]"
                if run_key(setup, MODE, fold, seed) in finished:
                    print(f"{tag} already finished, skipping")
                    continue
                train_df, val_df, test_df = make_fold_splits(df, fold, args.val_frac, seed)
                parts = {"train": train_df, "val": val_df, "test": test_df}
                for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                    assert not (set(parts[a]["group_key"]) & set(parts[b]["group_key"])), f"{a}/{b} share a group"
                n_pos = int(train_df["label"].sum())
                pos_weight = (len(train_df) - n_pos) / n_pos if n_pos else 1.0
                print(f"\n{tag} train {len(train_df)} / val {len(val_df)} / test {len(test_df)} clips, "
                      f"pos_weight {pos_weight:.3f}")

                t0 = time.perf_counter()
                seed_everything(seed)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                train_loader = make_loader(train_df, crops, args.batch_size, True, seed, device)
                val_loader = make_loader(val_df, crops, args.batch_size, False, seed, device)
                test_loader = make_loader(test_df, crops, args.batch_size, False, seed, device)
                model = ScratchVideoNet(args.dropout)
                info = train_model(model, train_loader, val_loader, device, pos_weight=pos_weight, lr=args.lr,
                                   weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                                   patience=args.patience, forward=forward_video)
                preds = predict(model, test_loader, device, forward_video)  # best epoch's weights
                metrics = compute_metrics(preds["label"], preds["prob"])

                ckpt = ckpt_dir / f"{ds}_fold{fold}_seed{seed}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "setup": setup,
                            "mode": MODE, "fold": fold, "seed": seed, "best_epoch": info["best_epoch"],
                            "best_val_macro_f1": info["best_val_macro_f1"], "args": vars(args)}, ckpt)

                elapsed = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                # predictions first, the runs row last: the row marks the run as finished
                append_rows(pred_csv, preds.assign(mode=MODE, fold=fold, seed=seed, setup=setup))
                append_rows(runs_csv, [{
                    "setup": setup, "task": "deception", "train_datasets": ds, "test_datasets": ds, "mode": MODE,
                    "fold": fold, "seed": seed, "dropout": args.dropout, "weight_decay": args.weight_decay,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                    "val_macro_f1": info["best_val_macro_f1"], **metrics, "time_s": round(elapsed, 1),
                    "peak_gpu_mb": round(peak_mb, 1), "device": device.type}])
                finished.add(run_key(setup, MODE, fold, seed))
                print(f"{tag} test acc {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
                      f"AUC {metrics['roc_auc']:.3f}  | best epoch {info['best_epoch']}/{info['epochs_run']}  | "
                      f"{elapsed:.1f} s per run  | "
                      + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
                del model, train_loader, val_loader, test_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del crops

    print(f"\nFinished in {(time.perf_counter() - t_all) / 60:.1f} min")
    if not runs_csv.is_file():
        print("No results were produced.")
        return 1
    runs = pd.read_csv(runs_csv)
    agg = aggregate(runs)
    out = rdir / "scratch_video.csv"
    agg.to_csv(out, index=False)
    print(f"Peak GPU memory over all runs: {runs['peak_gpu_mb'].max():.0f} MB; "
          f"time per run: mean {runs['time_s'].mean():.1f} s, max {runs['time_s'].max():.1f} s")

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(f"\n=== {out} (from scratch, video only) ===")
    print(show_table(agg))
    deep_path = rdir / "deep.csv"
    print(f"\n=== video-only rows of {deep_path} (pretrained ViT features + transformer) ===")
    if deep_path.is_file():
        deep = pd.read_csv(deep_path)
        print(show_table(deep[(deep["mode"] == "video") & deep["setting"].isin(DATASETS.values())]))
    else:
        print("not found")
    print(f"\nWrote {out}, {runs_csv}, {pred_csv}; checkpoints in {ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
