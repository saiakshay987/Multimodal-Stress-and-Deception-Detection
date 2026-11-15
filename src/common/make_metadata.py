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
* bol (Bag of Lies) and rlt (Real-Life Trial) are deception datasets: label 1 = deceptive.
  bol uses the participant id as group_id. rlt has no speaker identities, so group_id is
  read from a hand-filled <out_root>/rlt/speaker_ids.csv (see --speaker_ids).
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


# ------------------------------------------------------- bol / rlt (deception datasets)
# bol: data/bol/Annotations.csv, column "truth". The dataset ships no description of this
# column. 1 = truthful is inferred (163 rows have truth == 1 and 162 have truth == 0, and
# the Bag-of-Lies paper reports 163 truthful / 162 deceptive samples). Change this constant
# if the owners of the dataset say otherwise; labels are derived from it only.
BOL_TRUTH_VALUE_MEANS_TRUTHFUL = 1


def _deception_frame(records, dataset):
    """Shared checks and DataFrame construction for the deception datasets."""
    df = pd.DataFrame.from_records(records)
    if df.empty:
        sys.exit(f"error: no usable {dataset} clips found")
    if df["clip_id"].duplicated().any():
        sys.exit("error: duplicate clip_id values")
    if not df["label"].isin([0, 1]).all():
        sys.exit("error: label must be 0/1")
    if df["group_id"].isna().any() or (df["group_id"].astype(str).str.strip() == "").any():
        sys.exit("error: empty group_id")
    return df


def build_bol(data_dir):
    """Bag of Lies (bol). data_dir is e.g. data/bol.

    Layout: Annotations.csv, Finalised/User_<n>/run_<m>/{video.mp4, Gaze.csv, EEG.csv,
    image.jpg}. The label comes from Annotations.csv (column "truth", see
    BOL_TRUTH_VALUE_MEANS_TRUTHFUL) and the participant id from its "usernum" column; the
    video path in the same row is checked against both. EEG/Gaze/image are not used.
    """
    data_dir = Path(data_dir)
    data_root = data_dir.parent
    csv_path = data_dir / "Annotations.csv"
    if not csv_path.is_file():
        sys.exit(f"error: {csv_path} not found")
    src = pd.read_csv(csv_path)
    missing = {"run", "truth", "usernum", "video"} - set(src.columns)
    if missing:
        sys.exit(f"error: {csv_path} is missing columns {sorted(missing)}")
    print(f"read {len(src)} rows from {csv_path}")
    if not src["truth"].isin([0, 1]).all():
        sys.exit("error: Annotations.csv column 'truth' must be 0/1")

    records, dropped, failed = [], [], []
    for _, r in src.iterrows():
        user, run = int(r["usernum"]), int(r["run"])
        rel = Path(str(r["video"]))  # "./Finalised/User_0/run_0/video.mp4"
        expected = Path("Finalised") / f"User_{user}" / f"run_{run}" / "video.mp4"
        if rel != expected:
            sys.exit(f"error: video path {r['video']!r} disagrees with usernum={user}, run={run}")
        path = data_dir / rel
        if not (path.is_file() and path.stat().st_size > 0):
            dropped.append(str(path))
            continue
        dur = ffprobe_duration(path)
        if dur is None:
            failed.append(str(path))
            continue
        truthful = int(r["truth"]) == BOL_TRUTH_VALUE_MEANS_TRUTHFUL
        records.append({
            "clip_id": f"User_{user}_run_{run}",
            "dataset": "bol",
            "video_path": path.relative_to(data_root).as_posix(),
            "group_id": f"User_{user}",
            "task": "deception",
            "label": 0 if truthful else 1,  # 1 = deceptive
            "duration_s": dur,
            "mas_labels": "",
        })
    for name, items in (("missing or empty video file", dropped), ("ffprobe could not read", failed)):
        if items:
            print(f"  dropping {len(items)} rows ({name}):")
            for p in items:
                print(f"    {p}")
    return _deception_frame(records, "bol")


def _write_speaker_template(clip_ids, path):
    """Write an empty speaker_ids.csv template (columns clip_id, speaker_id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"clip_id": clip_ids, "speaker_id": ""}).to_csv(path, index=False)


def _read_speaker_ids(path, clip_ids):
    """Return {clip_id: speaker_id}; refuse to continue if the file is incomplete."""
    sp = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(sp.columns) != ["clip_id", "speaker_id"]:
        sys.exit(f"error: {path} must have exactly the columns clip_id,speaker_id (got {list(sp.columns)})")
    sp["speaker_id"] = sp["speaker_id"].str.strip()
    if sp["clip_id"].duplicated().any():
        sys.exit(f"error: {path} has duplicate clip_id rows: {sorted(sp['clip_id'][sp['clip_id'].duplicated()])}")
    empty = sp.loc[sp["speaker_id"] == "", "clip_id"].tolist()
    if empty:
        sys.exit(f"error: {path} has {len(empty)} row(s) with an empty speaker_id, fill them in first: "
                 f"{empty[:10]}{' ...' if len(empty) > 10 else ''}")
    extra, absent = set(sp["clip_id"]) - set(clip_ids), set(clip_ids) - set(sp["clip_id"])
    if extra or absent:
        sys.exit(f"error: {path} does not match the clips on disk; unknown clip_ids: {sorted(extra)}, "
                 f"clips without a row: {sorted(absent)}")
    return dict(zip(sp["clip_id"], sp["speaker_id"]))


def build_rlt(data_dir, speaker_ids_path):
    """Real-Life Trial (rlt). data_dir is e.g. data/rlt.

    Layout: Clips/{Deceptive,Truthful}/trial_{lie,truth}_NNN.mp4, Transcription/..., and
    Annotation/All_Gestures_Deceptive and Truthful.csv. The label comes from the Clips
    sub-folder, cross-checked against the file-name prefix and the annotation "class"
    column. The dataset provides NO speaker identity, so group_id is read from the
    hand-filled speaker_ids_path (columns clip_id, speaker_id). If that file does not
    exist an empty template is written and the script stops; an existing file is never
    overwritten. Any empty speaker_id makes the script refuse to run.
    """
    data_dir = Path(data_dir)
    data_root = data_dir.parent
    speaker_ids_path = Path(speaker_ids_path)
    folders = {"Deceptive": 1, "Truthful": 0}  # folder -> label (1 = deceptive)
    prefixes = {"Deceptive": "trial_lie_", "Truthful": "trial_truth_"}
    clips = {}
    for folder in folders:
        d = data_dir / "Clips" / folder
        if not d.is_dir():
            sys.exit(f"error: {d} not found")
        for p in sorted(d.glob(f"*{VIDEO_SUFFIX}")):
            if not p.name.startswith(prefixes[folder]):
                sys.exit(f"error: {p} is in {folder}/ but does not start with {prefixes[folder]!r}")
            clips[p.name[: -len(VIDEO_SUFFIX)]] = (folder, p)
    if not clips:
        sys.exit(f"error: no clips under {data_dir / 'Clips'}")
    print(f"found {len(clips)} rlt clips")

    ann_path = data_dir / "Annotation" / "All_Gestures_Deceptive and Truthful.csv"
    if ann_path.is_file():
        ann = pd.read_csv(ann_path, usecols=["id", "class"])
        cls = dict(zip(ann["id"], ann["class"].str.lower()))
        for cid, (folder, _) in clips.items():
            if cls.get(cid + VIDEO_SUFFIX) != folder.lower():
                sys.exit(f"error: {cid}: folder {folder} disagrees with annotation class "
                         f"{cls.get(cid + VIDEO_SUFFIX)!r}")
    else:
        print(f"  warning: {ann_path} not found, skipping the label cross-check")

    clip_ids = sorted(clips)
    if not speaker_ids_path.is_file():
        _write_speaker_template(clip_ids, speaker_ids_path)
        sys.exit(
            "rlt does not provide speaker identities (the README only gives a role / trial name,\n"
            "which is not a speaker id). Wrote an empty template with one row per clip to:\n"
            f"  {speaker_ids_path}\n"
            "Fill in the speaker_id column by hand (same id = same person), then run this again."
        )
    speaker = _read_speaker_ids(speaker_ids_path, clip_ids)
    print(f"read {len(speaker)} speaker ids from {speaker_ids_path}")

    records, failed = [], []
    for cid in clip_ids:
        folder, path = clips[cid]
        if path.stat().st_size == 0:
            failed.append(str(path))
            continue
        dur = ffprobe_duration(path)
        if dur is None:
            failed.append(str(path))
            continue
        records.append({
            "clip_id": cid,
            "dataset": "rlt",
            "video_path": path.relative_to(data_root).as_posix(),
            "group_id": speaker[cid],
            "task": "deception",
            "label": folders[folder],  # 1 = deceptive
            "duration_s": dur,
            "mas_labels": "",
        })
    if failed:
        print(f"  dropping {len(failed)} clips that are empty or that ffprobe could not read:")
        for p in failed:
            print(f"    {p}")
    return _deception_frame(records, "rlt")


def print_single_label_groups(df):
    """List every person (group_id) whose clips all carry the same label."""
    nun = df.groupby("group_id")["label"].nunique()
    single = nun[nun == 1].index
    print(f"\nPeople with only one label: {len(single)} of {df['group_id'].nunique()}")
    for g in single:
        sub = df[df["group_id"] == g]
        name = "deceptive (1)" if sub["label"].iloc[0] == 1 else "truthful (0)"
        print(f"  {g}: {len(sub)} clip(s), all {name}, fold {sorted(int(f) for f in sub['fold'].unique())}")


BUILDERS = {"mas": build_mas, "bol": build_bol, "rlt": build_rlt}


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(BUILDERS), help="dataset name")
    ap.add_argument("data_dir", help="the dataset folder under data/, e.g. data/mas")
    ap.add_argument("--out_root", default="features",
                    help="output root; writes <out_root>/<dataset>/metadata.csv (default: features)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--speaker_ids", default=None,
                    help="rlt only: hand-filled csv with columns clip_id,speaker_id used as group_id "
                         "(default: <out_root>/rlt/speaker_ids.csv; an empty template is written "
                         "there if it does not exist)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "rlt":
        speaker_ids = args.speaker_ids or Path(args.out_root) / "rlt" / "speaker_ids.csv"
        df = build_rlt(args.data_dir, speaker_ids)
    else:
        df = BUILDERS[args.dataset](args.data_dir)
    df = df.sort_values("clip_id").reset_index(drop=True)
    df = assign_folds(df, args.seed)
    df = df[COLUMNS]

    out_path = Path(args.out_root) / args.dataset / "metadata.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} clips, {df['group_id'].nunique()} groups to {out_path}")
    print_fold_summary(df)
    if (df["task"] == "deception").all():
        print_single_label_groups(df)


if __name__ == "__main__":
    main()
