#!/usr/bin/env python3
"""P(deceptive) and per-modality probabilities from saved features, using checkpoints/deception_final.pt.

The model is the late-fusion MultimodalModel: an audio classifier and a video classifier whose logits
are averaged. So for one clip:
    p_audio       sigmoid(audio logit)   - audio alone
    p_video       sigmoid(video logit)   - video alone (None if no face was found in any frame)
    p_deceptive   sigmoid(mean of the available modality logits) - the model's output
Note p_deceptive is the average in logit space, not the mean of p_audio and p_video.
If a modality is absent for a clip, its probability is None and p_deceptive uses the other one only.

Python (works on CPU):
    from predict_deception import load_model, predict_clip
    model, stats = load_model("checkpoints/deception_final.pt")
    r = predict_clip("features/bol/User_0_run_0.npz", model, stats)
    r["p_deceptive"], r["p_audio"], r["p_video"]

Command line:
    python src/deception/predict_deception.py --clips features/bol/User_0_run_0.npz
    python src/deception/predict_deception.py --datasets bol rlt --out results/deception/deception_scores.csv
    python src/deception/predict_deception.py --clips a.npz --device cpu      # force the CPU
(--datasets scores every clip of the dataset; those are the model's TRAINING clips, so the numbers are
 optimistic and say nothing about generalisation.)
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
from train import build_model, seed_everything  # noqa: E402

DECEPTION = 1  # task id used by data.TASK_TO_ID


def resolve_device(name="auto"):
    """'auto' = CUDA when available, otherwise the CPU; or 'cpu' / 'cuda'."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_model(ckpt_path="checkpoints/deception_final.pt", device="cpu"):
    """(model in eval mode on ``device``, FeatureStats) from a train_final.py checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["mode"], ck["dropout"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), FeatureStats.from_dict(ck["stats"])


def _prob(logit):
    return float(torch.sigmoid(logit.float())[0])


@torch.no_grad()
def predict_clip(npz_file, model, stats, device="cpu"):
    """Score one clip from the path of its .npz feature file.

    Returns a dict ``{"p_deceptive": float, "p_audio": float | None, "p_video": float | None}``.
    A modality that has no usable steps (no audio frames, or no frame with a face) gives None.
    """
    device = torch.device(device)
    audio, video, valid = load_clip(npz_file)
    audio = (audio - stats.audio_mean) / stats.audio_std
    video = np.where(valid[:, None], (video - stats.video_mean) / stats.video_std, 0.0)
    batch = collate_clips([(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                            torch.from_numpy(np.ascontiguousarray(video, dtype=np.float32)),
                            torch.from_numpy(valid.copy()), DECEPTION, 0.0, "clip", "x")])
    out = model(*(batch[k].to(device) for k in ("audio", "video", "audio_mask", "video_mask")), return_dict=True)
    fused = out["late"] if "late" in out else next(v for k, v in out.items() if k not in ("audio", "video"))
    has_audio, has_video = len(audio) > 0, bool(valid.any())
    return {"p_deceptive": _prob(fused),
            "p_audio": _prob(out["audio"]) if "audio" in out and has_audio else None,
            "p_video": _prob(out["video"]) if "video" in out and has_video else None}


def _fmt(p):
    return "n/a" if p is None else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/deception_final.pt")
    ap.add_argument("--clips", nargs="*", default=[], help=".npz feature files to score (printed)")
    ap.add_argument("--datasets", nargs="*", default=[], choices=["bol", "rlt"],
                    help="score every clip of these datasets and write --out")
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out", default="results/deception/deception_scores.csv")
    ap.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu or cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not a.clips and not a.datasets:
        ap.error("give --clips and/or --datasets")

    seed_everything(a.seed)
    device = resolve_device(a.device)
    print(f"device: {device}")
    model, stats = load_model(a.ckpt, device)

    if a.clips:
        print("clip\tp_deceptive\tp_audio\tp_video")
    for f in a.clips:
        r = predict_clip(f, model, stats, device)
        print(f"{f}\t{_fmt(r['p_deceptive'])}\t{_fmt(r['p_audio'])}\t{_fmt(r['p_video'])}")
    if a.datasets:
        df = load_metadata(a.features_root, a.datasets)
        rows = [predict_clip(npz_path(a.features_root, d, c), model, stats, device)
                for d, c in zip(df["dataset"], df["clip_id"])]
        res = pd.DataFrame(rows).rename(columns={"p_deceptive": "deception_prob", "p_audio": "audio_prob",
                                                 "p_video": "video_prob"})
        res.insert(0, "dataset", df["dataset"].to_numpy())
        res.insert(0, "clip_id", df["clip_id"].to_numpy())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out, index=False)
        print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
