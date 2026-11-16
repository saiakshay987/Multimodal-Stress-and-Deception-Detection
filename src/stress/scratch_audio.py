#!/usr/bin/env python3
"""Audio-only stress model trained from random initialisation on audio_logmel (no pretrained weights).

Model (about 0.06 M parameters)
    4 x [Conv2d 3x3 -> BatchNorm -> ReLU -> MaxPool 2] with 16, 32, 64, 64 channels on the
    (1, 64 mel bands, 1000 frames) input, global average pooling -> dropout 0.3 -> Linear -> 1 logit

Input: audio_logmel (T_m, 64) of the .npz files (100 frames/s), standardised per mel band with the mean
and std of the TRAINING rows of the run (all frames of every training clip), then cropped or zero-padded
(= padded with the training mean) to 10 s = 1000 frames.
    training    a random 10 s crop every time a clip is drawn (clips of 10 s or less are padded at the end);
                time masking: 0-2 masks, each of 1-50 frames; frequency masking: 0-2 masks, each of 1-8
                mel bands (number and width drawn uniformly, masked cells set to 0 = the training mean)
    validation / test   the first 10 s, no augmentation

Protocol = src/common/train.py, whose helpers are imported and used unchanged: the folds come from the
fold column of features/mas/metadata.csv, validation = 15 % of the remaining groups drawn with the run
seed, AdamW (lr 1e-3, weight decay 1e-2), BCE with pos_weight = n_neg / n_pos, batch size 16, fp16
autocast only on CUDA, early stopping on validation macro-F1 (patience 10, at most --max-epochs = 100
epochs), the best epoch is evaluated on the test fold. 5 folds x seeds 0 1 2.

Files (under --results-dir; runs already in the runs file are skipped, so the script can be restarted)
    runs/scratch_audio_runs.csv         one row per (fold, seed): columns of train.py plus train_acc, val_acc
    runs/scratch_audio_predictions.csv  every test clip of every run
    scratch_audio.csv                   aggregate in the format of deep.csv (mode "scratch_audio")
    <ckpt-dir>/scratch_audio/mas_fold<k>_seed<s>.pt
With --max-epochs below 100 (a smoke test) every file and the checkpoint folder get a _smoke suffix, so
a real run never resumes from smoke results. Then the audio row of deep.csv is printed next to
scratch_audio.csv.

    python src/stress/scratch_audio.py                    # from the repository root
    python src/stress/scratch_audio.py --folds 0 --seeds 0 --max-epochs 2 --threads 4   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import STD_FLOOR, load_metadata, npz_path  # noqa: E402  (src/common/data.py, unchanged)
from train import (append_rows, choose_device, compute_metrics, drop_unfinished_predictions,  # noqa: E402
                   load_finished_runs, make_fold_splits, predict, run_key, seed_everything, setup_id,
                   train_model)

MODE = "scratch_audio"
DATASET = "mas"
N_MELS = 64
FRAMES = 1000  # 10 s at 100 frames/s
CHANNELS = (16, 32, 64, 64)
TIME_MASKS, TIME_WIDTH = 2, 50
FREQ_MASKS, FREQ_WIDTH = 2, 8
SMOKE_BELOW = 100  # --max-epochs below this is a smoke test
METRICS = ("accuracy", "macro_f1", "roc_auc")


# ------------------------------------------------------------------------------------- data
def load_logmel(rows, features_root):
    """{clip_id: (T_m, 64) float16} for the clips of ``rows`` (kept in RAM; about 130 KB per 10 s clip)."""
    out = {}
    for ds, cid in zip(rows["dataset"], rows["clip_id"]):
        with np.load(npz_path(features_root, ds, cid)) as z:
            x = z["audio_logmel"]
        if x.ndim != 2 or x.shape[1] != N_MELS or len(x) == 0:
            raise ValueError(f"{cid}: audio_logmel has shape {x.shape}, expected (T>0, {N_MELS})")
        out[cid] = np.nan_to_num(x)
    return out


def logmel_stats(rows, logmel):
    """Per-mel-band mean and std over every frame of the clips in ``rows`` (pass TRAINING rows only)."""
    s, sq, n = np.zeros(N_MELS), np.zeros(N_MELS), 0
    for cid in rows["clip_id"]:
        x = logmel[cid].astype(np.float64)
        s += x.sum(0)
        sq += (x ** 2).sum(0)
        n += len(x)
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 0.0))
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def mask_axis(x, axis, max_masks, max_width):
    """Zero 0..max_masks random bands of 1..max_width entries along ``axis`` (1 = mel, 2 = time) of (1, M, F)."""
    size = x.shape[axis]
    for _ in range(int(torch.randint(0, max_masks + 1, (1,)))):
        w = int(torch.randint(1, max_width + 1, (1,)))
        start = int(torch.randint(0, max(size - w, 0) + 1, (1,)))
        if axis == 1:
            x[:, start:start + w, :] = 0.0
        else:
            x[:, :, start:start + w] = 0.0
    return x


class LogMelCrops(Dataset):
    """One standardised (1, 64, FRAMES) float tensor per clip: random crop + masking if ``train``,
    else the first 10 s. Clips shorter than 10 s are zero-padded at the end."""

    def __init__(self, rows, logmel, stats, train):
        self.rows, self.logmel, self.train = rows.reset_index(drop=True), logmel, train
        self.mean, self.std = stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.logmel[r["clip_id"]]
        t = len(x)
        start = int(torch.randint(0, t - FRAMES + 1, (1,))) if self.train and t > FRAMES else 0
        seg = (x[start:start + FRAMES].astype(np.float32) - self.mean) / self.std  # (<=FRAMES, 64)
        out = torch.zeros(1, N_MELS, FRAMES)
        out[0, :, :len(seg)] = torch.from_numpy(seg.T.copy())
        if self.train:
            mask_axis(out, 2, TIME_MASKS, TIME_WIDTH)
            mask_axis(out, 1, FREQ_MASKS, FREQ_WIDTH)
        return out, float(r["label"]), r["clip_id"], r["dataset"]


def collate(batch):
    x, y, cid, ds = zip(*batch)
    return {"audio": torch.stack(x), "label": torch.tensor(y, dtype=torch.float32),
            "clip_id": list(cid), "dataset": list(ds)}


def make_loader(rows, logmel, stats, batch_size, train, seed, device):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(LogMelCrops(rows, logmel, stats, train), batch_size=batch_size, shuffle=train,
                      generator=gen if train else None, collate_fn=collate, num_workers=0,
                      pin_memory=device.type == "cuda")


# ------------------------------------------------------------------------------------ model
def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class ScratchAudioNet(nn.Module):
    """Four conv blocks (16-32-64-64) + global average pooling, dropout, one logit."""

    def __init__(self, dropout=0.3):
        super().__init__()
        chans = (1, *CHANNELS)
        self.cnn = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])],
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(CHANNELS[-1], 1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """x (B, 1, 64, FRAMES) standardised log-mel -> logits (B,)."""
        return self.head(self.drop(self.cnn(x))).squeeze(-1)


def forward_audio(model, batch, device):
    return model(batch["audio"].to(device, non_blocking=True))


# ------------------------------------------------------------------------------ aggregate
def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def aggregate(runs):
    """One row in the format of results/stress/deep.csv; ``runs`` is the runs CSV (one row per fold x seed)."""
    row = {"mode": MODE, "n_runs": len(runs)}
    for m in METRICS + ("train_acc", "val_acc"):
        row[f"{m}_mean"], row[f"{m}_std"] = runs[m].mean(), runs[m].std()
    for m in METRICS:
        row[m] = fmt(row[f"{m}_mean"], row[f"{m}_std"])
    return pd.DataFrame([row])


def show_table(df):
    cols = ["mode", "n_runs", "accuracy", "macro_f1", "roc_auc", "train_acc_mean", "val_acc_mean"]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--results-dir", default="results/stress")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=100,
                    help=f"below {SMOKE_BELOW} this is a smoke test and every output gets a _smoke suffix")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be at least 1")

    smoke = args.max_epochs < SMOKE_BELOW
    sfx = "_smoke" if smoke else ""
    rdir = Path(args.results_dir)
    runs_csv = rdir / "runs" / f"scratch_audio_runs{sfx}.csv"
    pred_csv = rdir / "runs" / f"scratch_audio_predictions{sfx}.csv"
    out = rdir / f"scratch_audio{sfx}.csv"
    ckpt_dir = Path(args.ckpt_dir) / f"{MODE}{sfx}"
    if smoke:
        print(f"SMOKE TEST (--max-epochs {args.max_epochs} < {SMOKE_BELOW}): writing *_smoke files only")

    device = choose_device()
    if args.threads is not None:
        torch.set_num_threads(args.threads)  # choose_device() set it to all cores
    print(f"Threads in use: {torch.get_num_threads()}")
    print(f"Model parameters: {ScratchAudioNet(args.dropout).num_parameters():,}")

    finished = load_finished_runs(runs_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Runs file: {runs_csv}  ({len(finished)} finished runs found)")

    df = load_metadata(args.features_root, [DATASET], task="stress")
    logmel = load_logmel(df, args.features_root)
    setup = setup_id("stress", [DATASET], [DATASET], args.dropout, args.weight_decay)
    print(f"\n######## stress on {DATASET}: {len(df)} clips, {df['group_key'].nunique()} groups, setup {setup}")

    t_all = time.perf_counter()
    for fold in args.folds:
        for seed in args.seeds:
            tag = f"[{MODE} fold {fold} seed {seed}]"
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
            stats = logmel_stats(train_df, logmel)  # training rows only
            mk = lambda rows, train=False: make_loader(  # noqa: E731
                rows, logmel, stats, args.batch_size, train, seed, device)
            train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df), mk(test_df)
            model = ScratchAudioNet(args.dropout)
            info = train_model(model, train_loader, val_loader, device, pos_weight=pos_weight, lr=args.lr,
                               weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                               patience=args.patience, forward=forward_audio)
            preds = predict(model, test_loader, device, forward_audio)  # best epoch's weights
            metrics = compute_metrics(preds["label"], preds["prob"])
            tr = predict(model, mk(train_df), device, forward_audio)  # first 10 s, no augmentation
            va = predict(model, val_loader, device, forward_audio)
            train_acc = compute_metrics(tr["label"], tr["prob"])["accuracy"]
            val_acc = compute_metrics(va["label"], va["prob"])["accuracy"]

            ckpt = ckpt_dir / f"{DATASET}_fold{fold}_seed{seed}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "stats": {"mean": stats[0], "std": stats[1]}, "setup": setup, "mode": MODE,
                        "fold": fold, "seed": seed, "best_epoch": info["best_epoch"],
                        "best_val_macro_f1": info["best_val_macro_f1"], "args": vars(args)}, ckpt)

            elapsed = time.perf_counter() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
            # predictions first, the runs row last: the row marks the run as finished
            append_rows(pred_csv, preds.assign(mode=MODE, fold=fold, seed=seed, setup=setup))
            append_rows(runs_csv, [{
                "setup": setup, "task": "stress", "train_datasets": DATASET, "test_datasets": DATASET,
                "mode": MODE, "fold": fold, "seed": seed, "dropout": args.dropout,
                "weight_decay": args.weight_decay, "n_train": len(train_df), "n_val": len(val_df),
                "n_test": len(test_df), "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                "train_acc": train_acc, "val_acc": val_acc, "val_macro_f1": info["best_val_macro_f1"],
                **metrics, "time_s": round(elapsed, 1), "peak_gpu_mb": round(peak_mb, 1),
                "device": device.type}])
            finished.add(run_key(setup, MODE, fold, seed))
            print(f"{tag} train acc {train_acc:.3f}  val acc {val_acc:.3f}  test acc {metrics['accuracy']:.3f}  "
                  f"macro-F1 {metrics['macro_f1']:.3f}  AUC {metrics['roc_auc']:.3f}  | best epoch "
                  f"{info['best_epoch']}/{info['epochs_run']}  | {elapsed:.1f} s per run  | "
                  + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
            del model, train_loader, val_loader, test_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nFinished in {(time.perf_counter() - t_all) / 60:.1f} min")
    if not runs_csv.is_file():
        print("No results were produced.")
        return 1
    runs = pd.read_csv(runs_csv)
    runs = runs[(runs["setup"] == setup) & (runs["mode"] == MODE)]
    agg = aggregate(runs)
    agg.to_csv(out, index=False)
    print(f"Peak GPU memory over all runs: {runs['peak_gpu_mb'].max():.0f} MB; "
          f"time per run: mean {runs['time_s'].mean():.1f} s, max {runs['time_s'].max():.1f} s")

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(f"\n=== {out} (from scratch, audio only) next to the audio row of deep.csv "
          f"(pretrained wav2vec2 features + transformer) ===")
    deep_path = rdir / "deep.csv"
    table = agg
    if deep_path.is_file():
        deep = pd.read_csv(deep_path)
        table = pd.concat([deep[deep["mode"] == "audio"], agg], ignore_index=True)
    else:
        print(f"{deep_path} not found")
    print(show_table(table))
    print(f"\nWrote {out}, {runs_csv}, {pred_csv}; checkpoints in {ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
