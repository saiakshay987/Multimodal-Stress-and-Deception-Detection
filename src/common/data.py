"""Shared data loading for the multimodal stress / deception experiments.

Everything here follows docs/feature_contract.md and is independent of any model:

* ``load_metadata``      read features/<dataset>/metadata.csv for one or more datasets.
* ``compute_stats``      per-dimension mean/std of the audio and video inputs, from the
                         training rows only (``FeatureStats``).
* ``ClipDataset``        loads one .npz per clip and returns
                         ``(audio, video, video_valid, task, label, clip_id, dataset)``.
* ``collate_clips``      pads a batch to its longest clip and builds boolean masks.
* ``BalancedBatchSampler`` draws batches that are balanced across tasks and/or datasets.

Model inputs
------------
audio = concatenate(audio_w2v (768), audio_prosody (3))  -> 771 dims, 5 steps/s
video = concatenate(video_face (768), video_flow (2))    -> 770 dims, 5 steps/s
Audio and video are padded separately, because the two streams of a clip can differ by a
step. Only the keys listed above are read from the .npz files (audio_logmel, video_crops64
are never touched), so loading stays cheap.

Masks
-----
Every mask is a ``bool`` tensor that is **True where a step must be ignored**
(the convention of ``key_padding_mask`` in ``torch.nn``):

* ``audio_mask`` (B, T_a): True at padded steps.
* ``video_mask`` (B, T_v): True at padded steps *and* at frames where ``video_valid`` is False.

A clip in which no face was found at all has an all-True video mask row; models that attend
over video must cope with that (for example by ignoring the video branch for that clip).

Normalisation
-------------
Statistics are computed over every time step of the training clips. For video, only frames
with ``video_valid`` True enter the statistics, and after normalising, the frames with
``video_valid`` False are set to 0 (the training mean) so that no garbage reaches the model
even where a mask is not applied. Standard deviations below ``STD_FLOOR`` are replaced by 1.

Splits are not made here (see ``make_splits`` in train.py): always split by group_id and use
the ``fold`` column of metadata.csv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# ------------------------------------------------------------------------------ constants
TASKS = ("stress", "deception")
TASK_TO_ID = {t: i for i, t in enumerate(TASKS)}
DATASETS = ("mas", "bol", "rlt")
REQUIRED_COLUMNS = ("clip_id", "dataset", "group_id", "task", "label", "fold")

AUDIO_KEYS = ("audio_w2v", "audio_prosody")  # concatenated in this order
VIDEO_KEYS = ("video_face", "video_flow")    # concatenated in this order
AUDIO_DIM = 768 + 3
VIDEO_DIM = 768 + 2
STD_FLOOR = 1e-6


# ------------------------------------------------------------------------------- metadata
def npz_path(features_root, dataset, clip_id):
    """Path of a clip's feature file. Clip ids may contain dots, so never use with_suffix."""
    return Path(features_root) / dataset / f"{clip_id}.npz"


def load_metadata(features_root, datasets, task=None, missing="error"):
    """Read and concatenate features/<dataset>/metadata.csv for the given datasets.

    Args:
        features_root: folder that contains one sub-folder per dataset.
        datasets: iterable of dataset names, e.g. ["bol", "rlt"].
        task: if given ("stress" or "deception"), keep only rows of that task.
        missing: what to do about rows whose .npz file does not exist: "error" raises,
            "drop" prints a warning and removes them, "ignore" keeps them.

    Returns:
        DataFrame with the contract columns plus ``group_key`` = "<dataset>:<group_id>".
        Use ``group_key`` (not ``group_id``) whenever datasets are combined: person ids of
        different datasets can coincide, and they are different people.
    """
    if missing not in ("error", "drop", "ignore"):
        raise ValueError(f"missing must be error|drop|ignore, got {missing!r}")
    frames = []
    for name in datasets:
        path = Path(features_root) / name / "metadata.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path, dtype={"clip_id": str, "group_id": str})
        absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if absent:
            raise ValueError(f"{path} is missing columns {absent}")
        if (df["dataset"] != name).any():
            raise ValueError(f"{path}: 'dataset' column does not equal {name!r} everywhere")
        if df["clip_id"].duplicated().any():
            raise ValueError(f"{path}: clip_id is not unique")
        if not df["label"].isin([0, 1]).all():
            raise ValueError(f"{path}: label must be 0/1")
        frames.append(df)
    if not frames:
        raise ValueError("no datasets given")
    df = pd.concat(frames, ignore_index=True)
    unknown = sorted(set(df["task"]) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown task values {unknown}")
    if task is not None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        df = df[df["task"] == task].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows with task={task!r} in datasets {list(datasets)}")
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["group_key"] = df["dataset"] + ":" + df["group_id"].astype(str)

    if missing != "ignore":
        exists = np.array([npz_path(features_root, d, c).is_file()
                           for d, c in zip(df["dataset"], df["clip_id"])])
        if not exists.all():
            n_bad = int((~exists).sum())
            examples = df.loc[~exists, ["dataset", "clip_id"]].head(5).values.tolist()
            if missing == "error":
                raise FileNotFoundError(
                    f"{n_bad} of {len(df)} clips have no .npz under {features_root}, e.g. {examples}. "
                    "Extract the features first or pass missing='drop'.")
            print(f"warning: dropping {n_bad} of {len(df)} clips without an .npz file, e.g. {examples}")
            df = df[exists].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- loading clips
def load_clip(path):
    """Read one .npz and return raw ``(audio (T_a,771), video (T_v,770), valid (T_v,))``.

    audio and video are float32 and have NOT been z-scored. NaN/Inf are replaced by 0.
    """
    with np.load(path) as z:
        audio = np.concatenate([z[k].astype(np.float32) for k in AUDIO_KEYS], axis=1)
        video = np.concatenate([z[k].astype(np.float32) for k in VIDEO_KEYS], axis=1)
        valid = z["video_valid"].astype(bool)
    if len(valid) != len(video):
        raise ValueError(f"{path}: video_valid has {len(valid)} steps but video has {len(video)}")
    return np.nan_to_num(audio), np.nan_to_num(video), valid


def _get_clip(path, cache):
    if cache is None:
        return load_clip(path)
    key = str(path)
    if key not in cache:
        cache[key] = load_clip(path)
    return cache[key]


# ----------------------------------------------------------------------------- statistics
@dataclass
class FeatureStats:
    """Per-dimension mean and std of the audio (771) and video (770) inputs."""

    audio_mean: np.ndarray
    audio_std: np.ndarray
    video_mean: np.ndarray
    video_std: np.ndarray

    def to_dict(self):
        """Plain dict of numpy arrays, convenient to store in a checkpoint."""
        return {k: getattr(self, k) for k in ("audio_mean", "audio_std", "video_mean", "video_std")}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: np.asarray(d[k], dtype=np.float32)
                      for k in ("audio_mean", "audio_std", "video_mean", "video_std")})


def _mean_std(total, total_sq, n):
    if n == 0:
        raise ValueError("no time steps to compute statistics from")
    mean = total / n
    var = np.maximum(total_sq / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_stats(rows, features_root, cache=None):
    """Mean/std per feature dimension over the clips in ``rows`` (pass TRAINING rows only).

    Audio uses every time step, video uses only frames with video_valid True. Sums are
    accumulated in float64 so that float16 inputs do not lose precision.

    Args:
        rows: DataFrame with columns clip_id and dataset.
        features_root: folder with one sub-folder per dataset.
        cache: optional dict shared with ClipDataset to keep loaded clips in RAM.
    """
    a_sum = np.zeros(AUDIO_DIM); a_sq = np.zeros(AUDIO_DIM); a_n = 0
    v_sum = np.zeros(VIDEO_DIM); v_sq = np.zeros(VIDEO_DIM); v_n = 0
    for dataset, clip_id in zip(rows["dataset"], rows["clip_id"]):
        audio, video, valid = _get_clip(npz_path(features_root, dataset, clip_id), cache)
        a = audio.astype(np.float64)
        a_sum += a.sum(0); a_sq += (a ** 2).sum(0); a_n += len(a)
        v = video[valid].astype(np.float64)
        v_sum += v.sum(0); v_sq += (v ** 2).sum(0); v_n += len(v)
    a_mean, a_std = _mean_std(a_sum, a_sq, a_n)
    v_mean, v_std = _mean_std(v_sum, v_sq, v_n)
    return FeatureStats(a_mean, a_std, v_mean, v_std)


# ------------------------------------------------------------------------------- dataset
class ClipDataset(Dataset):
    """One item per metadata row.

    ``__getitem__`` returns ``(audio, video, video_valid, task, label, clip_id, dataset)``:

    * audio (T_a, 771) float32, z-scored          = [audio_w2v | audio_prosody]
    * video (T_v, 770) float32, z-scored          = [video_face | video_flow]
    * video_valid (T_v,) bool
    * task int (0 = stress, 1 = deception, see TASK_TO_ID)
    * label float (0.0 / 1.0)
    * clip_id str, dataset str

    Args:
        rows: DataFrame with columns clip_id, dataset, task, label (e.g. from load_metadata).
        features_root: folder with one sub-folder per dataset.
        stats: FeatureStats from ``compute_stats`` on the TRAINING rows. The same object must
            be used for the train, validation and test datasets of a run. If None, features
            are returned un-normalised.
        cache: optional dict; when given, loaded clips are kept in RAM (about 1 MB per
            60 s clip) and can be shared between datasets and runs.
    """

    def __init__(self, rows, features_root, stats=None, cache=None):
        self.rows = rows.reset_index(drop=True)
        self.features_root = Path(features_root)
        self.stats = stats
        self.cache = cache
        self.clip_ids = self.rows["clip_id"].tolist()
        self.datasets = self.rows["dataset"].tolist()
        self.tasks = [TASK_TO_ID[t] for t in self.rows["task"]]
        self.labels = self.rows["label"].astype(float).tolist()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        audio, video, valid = _get_clip(npz_path(self.features_root, self.datasets[i], self.clip_ids[i]),
                                        self.cache)
        if self.stats is not None:
            s = self.stats
            audio = (audio - s.audio_mean) / s.audio_std
            video = (video - s.video_mean) / s.video_std
            video = np.where(valid[:, None], video, 0.0)
        return (torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(video, dtype=np.float32)),
                torch.from_numpy(valid.copy()),
                self.tasks[i], self.labels[i], self.clip_ids[i], self.datasets[i])


def collate_clips(batch):
    """Collate ``ClipDataset`` items; pad audio and video to the longest clip in the batch.

    Returns a dict:
        audio       (B, T_a, 771) float32, zero padded
        video       (B, T_v, 770) float32, zero padded
        audio_mask  (B, T_a) bool, True at padded steps
        video_mask  (B, T_v) bool, True at padded steps and where video_valid is False
        video_valid (B, T_v) bool, False at padded steps and at frames without a face
        task        (B,) long   (0 = stress, 1 = deception)
        label       (B,) float32
        clip_id     list[str]
        dataset     list[str]
    """
    audios, videos, valids, tasks, labels, clip_ids, datasets = zip(*batch)
    b = len(batch)
    t_a = max(1, max(len(a) for a in audios))
    t_v = max(1, max(len(v) for v in videos))
    audio = torch.zeros(b, t_a, audios[0].shape[1])
    video = torch.zeros(b, t_v, videos[0].shape[1])
    audio_mask = torch.ones(b, t_a, dtype=torch.bool)
    video_valid = torch.zeros(b, t_v, dtype=torch.bool)
    for i in range(b):
        n_a, n_v = len(audios[i]), len(videos[i])
        audio[i, :n_a] = audios[i]
        audio_mask[i, :n_a] = False
        video[i, :n_v] = videos[i]
        video_valid[i, :n_v] = valids[i]
    return {
        "audio": audio, "video": video,
        "audio_mask": audio_mask, "video_mask": ~video_valid, "video_valid": video_valid,
        "task": torch.tensor(tasks, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "clip_id": list(clip_ids), "dataset": list(datasets),
    }


# ------------------------------------------------------------------------------- sampler
class _Pool:
    """Endless supply of row indices: shuffled passes over a fixed set (with reshuffling)."""

    def __init__(self, idx, rng):
        self.idx, self.rng = np.asarray(idx), rng
        self.order, self.pos = self.rng.permutation(self.idx), 0

    def take(self, n):
        out = []
        while n > 0:
            if self.pos >= len(self.order):
                self.order, self.pos = self.rng.permutation(self.idx), 0
            k = min(n, len(self.order) - self.pos)
            out.append(self.order[self.pos:self.pos + k])
            self.pos += k
            n -= k
        return np.concatenate(out) if out else np.empty(0, dtype=int)


class BalancedBatchSampler(Sampler):
    """Batch sampler whose batches are balanced across the values of one or more columns.

    Use it as ``DataLoader(ds, batch_sampler=BalancedBatchSampler(rows, 16, ("task", "dataset")),
    collate_fn=collate_clips)``.

    ``balance_by`` is an ordered tuple of metadata columns, applied hierarchically: a batch is
    first split equally among the values of the first column, then each share is split equally
    among the values of the second column that occur within it, and so on. With
    ``("task", "dataset")`` and the datasets mas (stress), bol and rlt (deception), half of
    every batch is stress (mas) and the other half deception, split equally between bol and
    rlt. With ``("dataset",)`` on bol + rlt every batch is half bol, half rlt. Remainders (when
    the batch size does not divide evenly) go to randomly chosen groups. A group with fewer
    clips than its share is passed over again within the epoch, so small groups are
    over-sampled; that is the point of balancing.

    Args:
        rows: DataFrame with one row per dataset item, in the same order as the Dataset.
        batch_size: clips per batch.
        balance_by: column names, e.g. ("dataset",), ("task",) or ("task", "dataset").
        num_batches: batches per epoch (default ceil(len(rows) / batch_size)).
        seed: seed of the sampler's own random generator; successive epochs differ but the
            whole sequence is reproducible.
    """

    def __init__(self, rows, batch_size, balance_by=("task", "dataset"), num_batches=None, seed=42):
        rows = rows.reset_index(drop=True)
        for c in balance_by:
            if c not in rows.columns:
                raise ValueError(f"balance column {c!r} not in rows")
        if len(rows) == 0:
            raise ValueError("no rows to sample from")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches) if num_batches else math.ceil(len(rows) / self.batch_size)
        self.rng = np.random.default_rng(seed)
        self.tree = self._build(rows, tuple(balance_by), np.arange(len(rows)))

    def _build(self, rows, cols, idx):
        if not cols:
            return _Pool(idx, self.rng)
        values = rows.iloc[idx][cols[0]].to_numpy()
        return [self._build(rows, cols[1:], idx[values == v]) for v in sorted(set(values))]

    def _draw(self, node, n):
        if isinstance(node, _Pool):
            return node.take(n)
        base, extra = divmod(n, len(node))
        counts = np.full(len(node), base)
        counts[self.rng.choice(len(node), extra, replace=False)] += 1
        parts = [self._draw(child, int(c)) for child, c in zip(node, counts) if c > 0]
        return np.concatenate(parts)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = self._draw(self.tree, self.batch_size)
            self.rng.shuffle(batch)
            yield batch.tolist()
