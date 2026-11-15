#!/usr/bin/env python3
"""Check extracted features against docs/feature_contract.md.

Usage:
    python src/common/check_features.py bol rlt
    python src/common/check_features.py mas --features_root features

For every row of features/<dataset>/metadata.csv this checks that

* features/<dataset>/<clip_id>.npz exists and holds the keys audio_w2v, audio_prosody,
  audio_logmel, video_face, video_valid, video_flow and video_crops64 with the dtypes and
  trailing shapes of the contract;
* the audio keys share one T_a and the video keys share one T_v (the contract writes
  them with a common length);
* T_a and T_v are within --tol (default 10 %) of min(duration_s, 60) * 5, the 5 steps per
  second of the contract (audio_logmel runs at 100 frames/s and is not length-checked);
* video_valid is true for at least --min_valid (default 0.5) of the frames;
* features/<dataset>/wav/<clip_id>.wav and features/<dataset>/crops224/<clip_id>.npy exist.

It prints a summary table per dataset and lists every failing clip with its reasons.
The exit status is 1 if any clip fails, 0 otherwise. Nothing is written anywhere.

File names are built as f"{clip_id}.npz" because clip ids can contain dots.
"""
import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_SECOND = 5
MAX_SECONDS = 60.0

# key -> (dtype, trailing shape, rate group). Group "a" = audio at 5/s, "v" = video at 5/s,
# None = not length-checked (audio_logmel is at 100 frames/s).
CONTRACT = {
    "audio_w2v": (np.float16, (768,), "a"),
    "audio_prosody": (np.float32, (3,), "a"),
    "audio_logmel": (np.float16, (64,), None),
    "video_face": (np.float16, (768,), "v"),
    "video_valid": (np.bool_, (), "v"),
    "video_flow": (np.float32, (2,), "v"),
    "video_crops64": (np.uint8, (64, 64), "v"),
}


def read_headers(npz_path):
    """Return {key: (shape, dtype)} read from the .npy headers, without loading the arrays."""
    headers = {}
    with zipfile.ZipFile(npz_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            headers[name[:-4]] = (shape, dtype)
    return headers


def check_clip(clip_id, duration_s, ds_dir, tol, min_valid):
    """Return a list of (category, message) problems for one clip; empty if it passes."""
    problems = []

    npz_path = ds_dir / f"{clip_id}.npz"
    if not npz_path.is_file():
        problems.append(("npz missing", f"{npz_path.name} not found"))
    else:
        try:
            headers = read_headers(npz_path)
        except Exception as exc:  # corrupt or truncated file
            headers = None
            problems.append(("npz unreadable", f"{type(exc).__name__}: {exc}"))

        if headers is not None:
            lengths = {"a": {}, "v": {}}
            for key, (dtype, tail, group) in CONTRACT.items():
                if key not in headers:
                    problems.append(("key missing", key))
                    continue
                shape, got_dtype = headers[key]
                if got_dtype != np.dtype(dtype):
                    problems.append(("bad dtype", f"{key} is {got_dtype}, expected {np.dtype(dtype)}"))
                if len(shape) != 1 + len(tail) or tuple(shape[1:]) != tail:
                    problems.append(("bad shape", f"{key} is {shape}, expected (T,{','.join(map(str, tail))})"
                                     if tail else f"{key} is {shape}, expected (T,)"))
                elif group:
                    lengths[group][key] = shape[0]

            expected = min(float(duration_s), MAX_SECONDS) * STEPS_PER_SECOND
            for group, name in (("a", "T_a"), ("v", "T_v")):
                per_key = lengths[group]
                if not per_key:
                    continue
                if len(set(per_key.values())) > 1:
                    detail = ", ".join(f"{k}={v}" for k, v in per_key.items())
                    problems.append(("T mismatch", f"{name} differs between keys: {detail}"))
                for key, t in per_key.items():
                    if abs(t - expected) > tol * expected:
                        problems.append(("bad length", f"{key} T={t}, expected {expected:.1f} +/- {tol:.0%}"))
                        break  # one message per stream is enough

            if "video_valid" in headers and headers["video_valid"][1] == np.dtype(np.bool_):
                try:
                    with np.load(npz_path, allow_pickle=False) as z:
                        valid = z["video_valid"]
                    frac = float(valid.mean()) if valid.size else 0.0
                    if frac < min_valid:
                        problems.append(("low valid", f"video_valid true for {frac:.0%} of {valid.size} frames"))
                except Exception as exc:
                    problems.append(("npz unreadable", f"video_valid: {type(exc).__name__}: {exc}"))

    if not (ds_dir / "wav" / f"{clip_id}.wav").is_file():
        problems.append(("wav missing", f"wav/{clip_id}.wav"))
    if not (ds_dir / "crops224" / f"{clip_id}.npy").is_file():
        problems.append(("crops224 missing", f"crops224/{clip_id}.npy"))
    return problems


def check_dataset(dataset, features_root, tol, min_valid):
    """Check one dataset. Returns (n_clips, {clip_id: problems}, Counter of categories)."""
    ds_dir = features_root / dataset
    meta_path = ds_dir / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found")
    meta = pd.read_csv(meta_path, dtype={"clip_id": str})

    failing = {}
    categories = Counter()
    for clip_id, duration_s in zip(meta["clip_id"], meta["duration_s"]):
        problems = check_clip(clip_id, duration_s, ds_dir, tol, min_valid)
        if problems:
            failing[clip_id] = problems
            categories.update({cat for cat, _ in problems})
    return len(meta), failing, categories


def main():
    parser = argparse.ArgumentParser(description="Check features against docs/feature_contract.md.")
    parser.add_argument("datasets", nargs="+", help="dataset names, e.g. mas bol rlt")
    parser.add_argument("--features_root", default="features", help="folder holding <dataset>/metadata.csv")
    parser.add_argument("--tol", type=float, default=0.10, help="allowed relative deviation of T (default 0.10)")
    parser.add_argument("--min_valid", type=float, default=0.5,
                        help="minimum fraction of frames with video_valid true (default 0.5)")
    args = parser.parse_args()

    features_root = Path(args.features_root)
    any_failure = False

    for dataset in args.datasets:
        print(f"\n=== {dataset} ({features_root / dataset}) ===")
        try:
            n_clips, failing, categories = check_dataset(dataset, features_root, args.tol, args.min_valid)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            any_failure = True
            continue

        n_fail = len(failing)
        any_failure |= n_fail > 0
        print(f"{'clips in metadata':<22}{n_clips:>8}")
        print(f"{'passing':<22}{n_clips - n_fail:>8}")
        print(f"{'failing':<22}{n_fail:>8}")
        if categories:
            print("failing clips per check:")
            for cat, count in sorted(categories.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {cat:<20}{count:>8}")

        if failing:
            print("failing clips:")
            for clip_id, problems in failing.items():
                print(f"  {clip_id}")
                for cat, msg in problems:
                    print(f"      [{cat}] {msg}")

    print("\nRESULT:", "FAIL" if any_failure else "all clips pass")
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
