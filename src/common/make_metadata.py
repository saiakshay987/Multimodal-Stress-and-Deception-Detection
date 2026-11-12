#!/usr/bin/env python3
"""Build features/<dataset>/metadata.csv as specified in docs/feature_contract.md.

Usage:
    python src/common/make_metadata.py mas data/mas
    python src/common/make_metadata.py mas data/mas --out_root features

Columns written (in this order):
    clip_id, dataset, video_path, group_id, task, label, fold, duration_s, mas_labels

Notes
-----
* clip_id is the clip file name without the final ".mp4" only. clip ids can contain
  dots, so never use Path.with_suffix / Path.stem on them. Build feature file names
  as f"{clip_id}.npz".
* video_path is relative to the data/ folder (the parent of the dataset folder), e.g.
  "mas/clips/<clip>.mp4".
* Folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on group_id,
  stratified by label.
* Only the mas dataset is implemented. build_bol() and build_rlt() are placeholders
  for Person 3.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

COLUMNS = [
    "clip_id", "dataset", "video_path", "group_id", "task",
    "label", "fold", "duration_s", "mas_labels",
]
N_SPLITS = 5
VIDEO_SUFFIX = ".mp4"


# --------------------------------------------------------------------------- helpers
def ffprobe_duration(path):
    """Return the container duration in seconds read with ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        value = float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, 3)


def assign_folds(df, seed):
    """Add a 'fold' column (0..4) with StratifiedGroupKFold on group_id, stratified by label."""
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(df, df["label"], groups=df["group_id"])):
        fold[test_idx] = k
    assert (fold >= 0).all(), "some clips did not get a fold"
    df["fold"] = fold
    # A group must live in exactly one fold.
    per_group = df.groupby("group_id")["fold"].nunique()
    assert (per_group == 1).all(), "a group_id appears in more than one fold"
    return df


def print_fold_summary(df):
    """Per-fold counts of clips, groups and labels."""
    has_mas = df["mas_labels"].astype(str).str.len().gt(0).any()
    tags = []
    if has_mas:
        tags = sorted({t for s in df["mas_labels"] for t in str(s).split("|") if t})
    rows = []
    for k in range(N_SPLITS):
        sub = df[df["fold"] == k]
        row = {
            "fold": k,
            "clips": len(sub),
            "groups": sub["group_id"].nunique(),
            "label=0": int((sub["label"] == 0).sum()),
            "label=1": int((sub["label"] == 1).sum()),
        }
        for t in tags:
            row[t] = int(sub["mas_labels"].apply(lambda s, t=t: t in str(s).split("|")).sum())
        rows.append(row)
    total = {
        "fold": "all",
        "clips": len(df),
        "groups": df["group_id"].nunique(),
        "label=0": int((df["label"] == 0).sum()),
        "label=1": int((df["label"] == 1).sum()),
    }
    for t in tags:
        total[t] = int(df["mas_labels"].apply(lambda s, t=t: t in str(s).split("|")).sum())
    rows.append(total)
    print("\nPer-fold counts (the mas_labels columns count clips carrying that tag):")
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------------------- mas
def build_mas(data_dir):
    """Return a DataFrame (without fold) for the MAS stress dataset.

    data_dir is the dataset folder, e.g. data/mas. It must contain metadata.csv and clips/.
    """
    data_dir = Path(data_dir)
    data_root = data_dir.parent  # the data/ folder; video_path is relative to it
    csv_path = data_dir / "metadata.csv"
    clips_dir = data_dir / "clips"
    if not csv_path.is_file():
        sys.exit(f"error: {csv_path} not found")
    if not clips_dir.is_dir():
        sys.exit(f"error: {clips_dir} not found")

    src = pd.read_csv(csv_path, dtype={"clip": str, "source_video": str, "labels": str})
    expected = {"clip", "source_video", "url", "start_s", "duration_s", "labels", "stress", "downloaded"}
    missing = expected - set(src.columns)
    if missing:
        sys.exit(f"error: {csv_path} is missing columns {sorted(missing)}")
    print(f"read {len(src)} rows from {csv_path}")

    # Keep only downloaded == 1 rows whose clip file exists and is non-empty.
    downloaded = pd.to_numeric(src["downloaded"], errors="coerce").fillna(0).astype(int) == 1
    print(f"  downloaded == 1: {int(downloaded.sum())}")
    src = src[downloaded].copy()

    keep = []
    for clip in src["clip"]:
        f = clips_dir / clip
        keep.append(f.is_file() and f.stat().st_size > 0)
    keep = np.array(keep, dtype=bool)
    if (~keep).any():
        print(f"  dropping {int((~keep).sum())} rows whose clip file is missing or empty:")
        for clip in src["clip"][~keep]:
            print(f"    {clip}")
    src = src[keep].copy()

    records = []
    failed = []
    for _, r in src.iterrows():
        clip = r["clip"]
        if not clip.endswith(VIDEO_SUFFIX):
            print(f"  warning: skipping {clip!r}, it does not end with {VIDEO_SUFFIX}")
            continue
        clip_id = clip[: -len(VIDEO_SUFFIX)]  # strip only the final ".mp4"
        path = clips_dir / clip
        dur = ffprobe_duration(path)
        if dur is None:
            failed.append(clip)
            continue
        labels = "|".join(p.strip() for p in str(r["labels"]).split("|") if p.strip())
        records.append({
            "clip_id": clip_id,
            "dataset": "mas",
            "video_path": path.relative_to(data_root).as_posix(),
            "group_id": r["source_video"],
            "task": "stress",
            "label": int(r["stress"]),
            "duration_s": dur,
            "mas_labels": labels,
        })
    if failed:
        print(f"  dropping {len(failed)} clips that ffprobe could not read:")
        for clip in failed:
            print(f"    {clip}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        sys.exit("error: no usable mas clips found")
    if df["clip_id"].duplicated().any():
        sys.exit("error: duplicate clip_id values")
    if not df["label"].isin([0, 1]).all():
        sys.exit("error: stress column must be 0/1")
    if df["group_id"].isna().any():
        sys.exit("error: missing source_video (group_id)")
    return df


# ---------------------------------------------------------------- bol / rlt (Person 3)
def build_bol(data_dir):
    """TODO (Person 3): build the Box of Lies (bol) metadata.

    Must return a DataFrame with columns clip_id, dataset ("bol"), video_path (relative to
    data/), group_id (person), task ("deception"), label (1 = deceptive), duration_s
    (from ffprobe) and mas_labels (empty string). The fold column is added by the caller.
    """
    raise NotImplementedError("bol is not implemented yet (Person 3)")


def build_rlt(data_dir):
    """TODO (Person 3): build the Real-Life Trial (rlt) metadata.

    Same return contract as build_bol(), with dataset "rlt".
    """
    raise NotImplementedError("rlt is not implemented yet (Person 3)")


BUILDERS = {"mas": build_mas, "bol": build_bol, "rlt": build_rlt}


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(BUILDERS), help="dataset name")
    ap.add_argument("data_dir", help="the dataset folder under data/, e.g. data/mas")
    ap.add_argument("--out_root", default="features",
                    help="output root; writes <out_root>/<dataset>/metadata.csv (default: features)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = BUILDERS[args.dataset](args.data_dir)
    df = df.sort_values("clip_id").reset_index(drop=True)
    df = assign_folds(df, args.seed)
    df = df[COLUMNS]

    out_path = Path(args.out_root) / args.dataset / "metadata.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} clips, {df['group_id'].nunique()} groups to {out_path}")
    print_fold_summary(df)


if __name__ == "__main__":
    main()
