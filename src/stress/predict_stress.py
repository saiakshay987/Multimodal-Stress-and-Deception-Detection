#!/usr/bin/env python3
"""P(stress) from saved features, using checkpoints/stress_final.pt (works on CPU).

Python:
    from predict_stress import load_model, predict_clip
    model, stats = load_model("checkpoints/stress_final.pt")
    p = predict_clip("features/bol/User_0_run_0.npz", model, stats)

Command line:
    python src/stress/predict_stress.py --clips features/bol/User_0_run_0.npz
    python src/stress/predict_stress.py --datasets bol rlt --out results/stress/stress_scores_on_deception.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import FeatureStats, collate_clips, load_clip, load_metadata, npz_path  # noqa: E402
from train import build_model, forward_multimodal, seed_everything  # noqa: E402


def load_model(ckpt_path="checkpoints/stress_final.pt", device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["mode"], ck["dropout"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), FeatureStats.from_dict(ck["stats"])


@torch.no_grad()
def predict_clip(npz_file, model, stats, device="cpu"):
    """P(stress) in [0, 1] for one clip given the path of its .npz feature file."""
    audio, video, valid = load_clip(npz_file)
    s = stats
    audio = (audio - s.audio_mean) / s.audio_std
    video = np.where(valid[:, None], (video - s.video_mean) / s.video_std, 0.0)
    batch = collate_clips([(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                            torch.from_numpy(np.ascontiguousarray(video, dtype=np.float32)),
                            torch.from_numpy(valid.copy()), 0, 0.0, "clip", "x")])
    return float(torch.sigmoid(forward_multimodal(model, batch, torch.device(device)).float())[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stress_final.pt")
    ap.add_argument("--clips", nargs="*", default=[], help=".npz files to score (printed)")
    ap.add_argument("--datasets", nargs="*", default=[], help="score every clip of these datasets")
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out", default="results/stress/stress_scores_on_deception.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_everything(a.seed)
    model, stats = load_model(a.ckpt, "cpu")
    for f in a.clips:
        print(f"{f}\t{predict_clip(f, model, stats):.4f}")
    if a.datasets:
        df = load_metadata(a.features_root, a.datasets)
        probs = [predict_clip(npz_path(a.features_root, d, c), model, stats)
                 for d, c in zip(df["dataset"], df["clip_id"])]
        out = pd.DataFrame({"clip_id": df["clip_id"], "dataset": df["dataset"], "stress_prob": probs})
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
