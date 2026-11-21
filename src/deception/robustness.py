#!/usr/bin/env python3
"""Lighting robustness of the Bag-of-Lies deception models (one test fold).

For the test clips of one fold, the video features are re-extracted with extract_video.py
(--perturb brightness --level X --clip-ids FILE) at several brightness factors; the audio
features are the clean ones (the perturbed .npz files only hold video keys). The video-only
checkpoint and the best fused checkpoint of that fold are evaluated at every factor.

"Best fused" = the fused mode (concat / late / xattn) with the highest mean validation
macro-F1 over seeds, read from the checkpoints (never the test score). All seeds found for
the two modes are evaluated and averaged.

    python src/deception/robustness.py --ckpt_dir ../checkpoints/deception/within_bol    # from the repo root
    python src/deception/robustness.py --ckpt_dir CKPTS --skip_extract                   # reuse extracted features

Outputs: <out_dir>/lighting.csv (per seed rows + a mean row per mode and factor) and
<out_dir>/lighting.png (accuracy and % of frames with a face found vs brightness).
Nothing under features/ is written; extracted features go to <perturbed_root>/brightness_<X>/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import FeatureStats, load_clip, load_metadata, npz_path  # noqa: E402
from train import (build_loader, build_model, choose_device, compute_metrics, predict,  # noqa: E402
                   seed_everything)

FUSED = ("concat", "late", "xattn")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", required=True, help="folder with <mode>_fold<k>_seed<s>.pt (e.g. .../within_bol)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dataset", default="bol")
    ap.add_argument("--task", default="deception")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0, 1.3, 1.6])
    ap.add_argument("--features_root", default="features")
    ap.add_argument("--perturbed_root", default="features_perturbed")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--extract_script", default="src/common/extract_video.py")
    ap.add_argument("--out_dir", default="results/deception")
    ap.add_argument("--skip_extract", action="store_true", help="use already extracted perturbed features")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def find_checkpoints(ckpt_dir, fold):
    """{mode: [(seed, path, ckpt)]} for video and the fused modes."""
    found = {}
    for p in sorted(Path(ckpt_dir).glob(f"*_fold{fold}_seed*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        found.setdefault(ck["mode"], []).append((ck["seed"], p, ck))
    return found


def pick_models(found):
    if "video" not in found:
        sys.exit("no video checkpoints found")
    fused = {m: float(np.mean([c["best_val_macro_f1"] for _, _, c in found[m]])) for m in FUSED if m in found}
    if not fused:
        sys.exit("no fused checkpoints found")
    for m, v in fused.items():
        print(f"fused mode {m}: mean val macro-F1 {v:.4f} over {len(found[m])} seeds")
    best = max(fused, key=fused.get)
    print(f"best fused mode (by validation macro-F1): {best}")
    return ["video", best]


def extract(args, level, ids_file):
    cmd = [sys.executable, args.extract_script, args.dataset, "--perturb", "brightness", "--level", str(level),
           "--clip-ids", str(ids_file), "--features_root", args.features_root,
           "--perturbed_root", args.perturbed_root, "--data_root", args.data_root, "--seed", str(args.seed)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_video(path):
    with np.load(path) as z:
        return (np.nan_to_num(z["video_face"].astype(np.float32)), np.nan_to_num(z["video_flow"].astype(np.float32)),
                z["video_valid"].astype(bool))


def merged_cache(args, test_df, video_root):
    """cache for ClipDataset: clean audio + video from video_root (None = clean features)."""
    cache, valid_all = {}, []
    for ds, cid in zip(test_df["dataset"], test_df["clip_id"]):
        key = str(npz_path(args.features_root, ds, cid))
        audio, video, valid = load_clip(key)
        if video_root is not None:
            face, flow, valid = load_video(Path(video_root) / ds / f"{cid}.npz")
            video = np.concatenate([face, flow], axis=1)
        cache[key] = (audio, video, valid)
        valid_all.append(valid)
    return cache, valid_all


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.features_root, [args.dataset], task=args.task)
    test_df = df[df["fold"] == args.fold].reset_index(drop=True)
    print(f"{args.dataset} fold {args.fold}: {len(test_df)} test clips, {test_df['group_key'].nunique()} groups")

    found = find_checkpoints(args.ckpt_dir, args.fold)
    modes = pick_models(found)

    tmp = tempfile.TemporaryDirectory()
    ids_file = Path(tmp.name) / "test_clip_ids.txt"
    ids_file.write_text("\n".join(test_df["clip_id"]) + "\n")

    rows = []
    # (label, level, video_root): clean features first as a sanity reference at factor 1.0
    sources = [("clean_features", 1.0, None)]
    for lv in args.levels:
        if not args.skip_extract:
            extract(args, lv, ids_file)
        sources.append(("reextracted", lv, Path(args.perturbed_root) / f"brightness_{lv:g}"))

    for source, lv, vroot in sources:
        cache, valid_all = merged_cache(args, test_df, vroot)
        n_frames = sum(len(v) for v in valid_all)
        face_pct = 100.0 * sum(int(v.sum()) for v in valid_all) / n_frames
        no_face = sum(1 for v in valid_all if not v.any())
        print(f"[{source} brightness {lv:g}] face found in {face_pct:.1f}% of {n_frames} frames; "
              f"{no_face} clips without any face")
        for mode in modes:
            for seed, _, ck in found[mode]:
                stats = FeatureStats.from_dict(ck["stats"])
                model = build_model(ck["mode"], ck["args"]["dropout"])
                model.load_state_dict(ck["model"])
                model.to(device)
                loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache, device=device)
                preds = predict(model, loader, device)
                m = compute_metrics(preds["label"], preds["prob"])
                rows.append({"source": source, "brightness": lv, "mode": mode, "seed": seed, "n_clips": len(preds),
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "roc_auc": m["roc_auc"],
                             "pct_frames_face": face_pct, "n_clips_no_face": no_face})
                del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    res = pd.DataFrame(rows)
    mean = (res.groupby(["source", "brightness", "mode"], as_index=False)
               .agg(n_clips=("n_clips", "first"), accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                    roc_auc=("roc_auc", "mean"), pct_frames_face=("pct_frames_face", "first"),
                    n_clips_no_face=("n_clips_no_face", "first")))
    mean["seed"] = "mean"
    out = pd.concat([res.assign(seed=res["seed"].astype(str)), mean], ignore_index=True)
    out.insert(0, "fold", args.fold)
    out.to_csv(out_dir / "lighting.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    re = res[res["source"] == "reextracted"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in modes:
        g = re[re["mode"] == mode].groupby("brightness")["accuracy"]
        mu, lo, hi = g.mean(), g.min(), g.max()
        label = f"{mode} (mean of {len(found[mode])} seeds)"
        a1.plot(mu.index, mu.values, marker="o", label=label)
        a1.fill_between(mu.index, lo.values, hi.values, alpha=0.2)
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set(xlabel="brightness factor", ylabel="accuracy", title=f"Accuracy, {args.dataset} fold {args.fold} test clips")
    a1.legend()
    f = re.groupby("brightness")["pct_frames_face"].first()
    a2.plot(f.index, f.values, marker="o", color="tab:green")
    a2.set(xlabel="brightness factor", ylabel="% frames with a face found", ylim=(0, 105),
           title="Face detection")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.savefig(out_dir / "lighting.png", dpi=150)
    print(f"wrote {out_dir / 'lighting.csv'} and {out_dir / 'lighting.png'}")
    tmp.cleanup()


if __name__ == "__main__":
    main()
