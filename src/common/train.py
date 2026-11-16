#!/usr/bin/env python3
"""Cross-validated training / evaluation of MultimodalModel on precomputed features.

Examples (run from the repository root):
    # stress on MAS, 5 folds x 1 seed x 5 modes
    python src/common/train.py --task stress --datasets mas --out results/stress_mas.csv
    # deception on Bag-of-Lies + Real-Life Trial pooled, three seeds, two modes
    python src/common/train.py --task deception --datasets bol rlt --modes concat xattn \
        --seeds 42 43 44 --out results/deception.csv
    # cross-dataset: train on bol, test on rlt (one run per mode and seed, fold = -1)
    python src/common/train.py --task deception --train-datasets bol --test-datasets rlt \
        --out results/bol_to_rlt.csv
    # quick check that everything runs: 2 epochs on 20 clips, writes *_smoke.csv files
    python src/common/train.py --task stress --datasets mas --smoke

Protocol (per mode, test fold, seed)
    * test   = clips whose ``fold`` column equals the test fold (cross-dataset: every clip of
               --test-datasets);
    * val    = 15 % of the remaining GROUPS (group_key = dataset:group_id), chosen with the run
               seed, so train / val / test never share a person or source video;
    * train  = the rest. Feature z-scoring statistics come from these rows only.
    * AdamW (lr 1e-3, weight decay --weight-decay), BCE with pos_weight = n_neg / n_pos of the
      training rows, batch size 16, fp16 autocast on CUDA, early stopping on validation
      macro-F1 (patience 10, at most 100 epochs); the best epoch is evaluated on the test fold.

Outputs (OUT = --out, e.g. results/x.csv)
    OUT                       one row per (setup, mode, fold, seed): accuracy, macro_f1, roc_auc, ...
    OUT stem + _summary.csv   mean and std over the rows of each (setup, mode)
    OUT stem + _predictions.csv  every test clip: clip_id, dataset, label, prob, mode, fold, seed
    <ckpt-dir>/<OUT stem>/<mode>_fold<k>_seed<s>.pt   best model state + normalisation stats
Runs already present in OUT are skipped, so a stopped job resumes where it stopped. A "setup"
column (task, train/test datasets, dropout, weight decay) is part of the key so that results of
different configurations written to the same file cannot be mistaken for each other.

Reusable helpers (import them from finetune.py, the from-scratch scripts, the two-head model)
    seed_everything, choose_device, make_fold_splits, make_cross_splits, split_groups,
    EarlyStopper, compute_metrics, train_model, predict, build_loader, run_key,
    load_finished_runs, drop_unfinished_predictions, append_rows, summarize.
``train_model`` and ``predict`` take a ``forward(model, batch, device) -> logits`` callable, so a
model with another signature (two heads, raw waveforms ...) only needs its own ``forward``.

Assumed interface of src/common/models.py (adjust ``build_model`` / ``forward_multimodal`` if the
real one differs):
    MultimodalModel(mode=..., dropout=..., audio_dim=771, video_dim=770)
    model(audio, video, audio_mask, video_mask) -> logits of shape (B,) or (B, 1)
    where mode is one of audio | video | concat | late | xattn and the masks are True at steps to
    ignore (see data.py).
"""
from __future__ import annotations

import argparse
import inspect
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

try:  # imported as part of a package (finetune.py, notebooks) ...
    from .data import (AUDIO_DIM, VIDEO_DIM, BalancedBatchSampler, ClipDataset, collate_clips,
                       compute_stats, load_metadata)
except ImportError:  # ... or run as a script from src/common
    from data import (AUDIO_DIM, VIDEO_DIM, BalancedBatchSampler, ClipDataset, collate_clips,
                      compute_stats, load_metadata)

MODES = ("audio", "video", "concat", "late", "xattn")
DEFAULT_DATASETS = {"stress": ["mas"], "deception": ["bol", "rlt"]}
BALANCE_CHOICES = {"none": None, "dataset": ("dataset",), "task": ("task",),
                   "task+dataset": ("task", "dataset")}
METRICS = ("accuracy", "macro_f1", "roc_auc")
SMOKE_SIZES = {"train": 12, "val": 4, "test": 4}  # 20 clips in all
CROSS_FOLD = -1  # value of the fold column for cross-dataset runs


# -------------------------------------------------------------------- seeds and device
def seed_everything(seed=42):
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def choose_device():
    """CUDA when available, otherwise the CPU using all cores. Prints the choice."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"Device: cuda ({props.name}, {props.total_memory / 2**30:.1f} GB)")
    else:
        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        device = torch.device("cpu")
        print(f"Device: cpu ({n} threads)")
    return device


# -------------------------------------------------------------------------------- splits
def split_groups(df, frac=0.15, seed=42, group_col="group_key", max_tries=200):
    """Split ``df`` into (rest, held_out) by group: ``frac`` of the groups are held out.

    Whole groups (persons / source videos) go to one side, never single clips. The groups are
    drawn with ``seed``; draws are repeated (up to ``max_tries``) until both sides contain both
    labels, and a warning is printed if that never happens.
    """
    groups = np.array(sorted(df[group_col].unique()))
    if len(groups) < 2:
        raise ValueError("need at least 2 groups to split")
    n_out = min(max(1, int(round(frac * len(groups)))), len(groups) - 1)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        out_groups = set(rng.choice(groups, n_out, replace=False))
        mask = df[group_col].isin(out_groups).to_numpy()
        if df.loc[mask, "label"].nunique() == 2 and df.loc[~mask, "label"].nunique() == 2:
            break
    else:
        print(f"warning: could not find a group split with both labels on both sides "
              f"(seed {seed}); using the last draw")
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def make_fold_splits(df, test_fold, val_frac=0.15, seed=42):
    """(train, val, test) for one cross-validation fold of ``df``.

    test = rows with fold == test_fold (the fold column of metadata.csv, never re-drawn);
    val = ``val_frac`` of the remaining groups; train = everything else.
    """
    test = df[df["fold"] == test_fold].reset_index(drop=True)
    rest = df[df["fold"] != test_fold]
    if test.empty or rest.empty:
        raise ValueError(f"fold {test_fold} leaves an empty test or training set")
    train, val = split_groups(rest, val_frac, seed)
    return train, val, test


def make_cross_splits(train_df, test_df, val_frac=0.15, seed=42):
    """(train, val, test) for a cross-dataset run: ``val_frac`` of train_df's groups are val."""
    overlap = set(train_df["dataset"]) & set(test_df["dataset"])
    if overlap:
        raise ValueError(f"datasets {sorted(overlap)} are in both the training and test side")
    train, val = split_groups(train_df, val_frac, seed)
    return train, val, test_df.reset_index(drop=True)


def subsample_for_smoke(train, val, test, seed=42):
    """Keep 12 / 4 / 4 random rows of train / val / test (20 clips). Splits stay disjoint."""
    return tuple(df.sample(n=min(len(df), SMOKE_SIZES[name]), random_state=seed).reset_index(drop=True)
                 for name, df in (("train", train), ("val", val), ("test", test)))


# ------------------------------------------------------------------- metrics and stopping
def compute_metrics(labels, probs, threshold=0.5):
    """accuracy, macro_f1 and roc_auc of probabilities against 0/1 labels.

    roc_auc is NaN when only one class is present in ``labels``.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "roc_auc": float(auc)}


class EarlyStopper:
    """Keeps the best model state by a score to maximise and signals when to stop.

    Usage: ``improved = stopper.update(score, epoch, model)`` once per epoch, stop the loop when
    ``stopper.should_stop``, afterwards ``model.load_state_dict(stopper.best_state)``.
    A new best must beat the old one by more than ``min_delta``.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_epoch, self.best_state = -float("inf"), -1, None
        self.bad_epochs = 0

    def update(self, score, epoch, model):
        if score > self.best_score + self.min_delta:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


# ----------------------------------------------------------------------------- the model
def build_model(mode, dropout, audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM):
    """Instantiate MultimodalModel from src/common/models.py (see the module docstring)."""
    try:
        from .models import MultimodalModel
    except ImportError:
        from models import MultimodalModel
    params = inspect.signature(MultimodalModel.__init__).parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {"mode": mode, "dropout": dropout}
    for name, value in (("audio_dim", audio_dim), ("video_dim", video_dim)):
        if takes_kwargs or name in params:
            kwargs[name] = value
    return MultimodalModel(**kwargs)


def forward_multimodal(model, batch, device):
    """Default ``forward`` for train_model / predict: logits of shape (B,)."""
    out = model(batch["audio"].to(device, non_blocking=True), batch["video"].to(device, non_blocking=True),
                batch["audio_mask"].to(device, non_blocking=True), batch["video_mask"].to(device, non_blocking=True))
    return out.reshape(-1)


def _autocast(device):
    """fp16 autocast on CUDA, a no-op on the CPU."""
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


# ----------------------------------------------------------------- loaders, train, predict
def build_loader(rows, features_root, stats, batch_size=16, train=False, balance_by=None, seed=42,
                 workers=0, cache=None, device=None):
    """DataLoader over ``rows``.

    train=True shuffles (or, with ``balance_by`` such as ("task", "dataset"), draws balanced
    batches via BalancedBatchSampler); train=False keeps the row order for evaluation.
    """
    ds = ClipDataset(rows, features_root, stats=stats, cache=cache)
    common = dict(collate_fn=collate_clips, num_workers=workers, persistent_workers=workers > 0,
                  pin_memory=device is not None and device.type == "cuda")
    if train and balance_by:
        return DataLoader(ds, batch_sampler=BalancedBatchSampler(ds.rows, batch_size, balance_by, seed=seed),
                          **common)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=gen if train else None, **common)


@torch.no_grad()
def predict(model, loader, device, forward=forward_multimodal):
    """Run ``model`` over ``loader``; DataFrame with clip_id, dataset, label, prob (sigmoid)."""
    model.eval()
    ids, datasets, labels, probs = [], [], [], []
    for batch in loader:
        with _autocast(device):
            logits = forward(model, batch, device)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(batch["label"].numpy())
        ids += batch["clip_id"]
        datasets += batch["dataset"]
    return pd.DataFrame({"clip_id": ids, "dataset": datasets, "label": np.concatenate(labels).astype(int),
                         "prob": np.concatenate(probs)})


def train_model(model, train_loader, val_loader, device, *, pos_weight=1.0, lr=1e-3, weight_decay=1e-2,
                max_epochs=100, patience=10, accum_steps=1, forward=forward_multimodal, log_every=10,
                loss_fn=None, score_fn=None):
    """Train with AdamW + BCE(pos_weight), fp16 autocast on CUDA and early stopping.

    The score for early stopping is the validation macro-F1 at threshold 0.5. On return, ``model``
    holds the best weights.

    Args:
        accum_steps: gradient accumulation steps (effective batch = batch size * accum_steps).
        forward: callable (model, batch, device) -> logits (B,).
        log_every: print a line every this many epochs (and at the last one).
        loss_fn: optional callable (logits, batch, device) -> scalar loss; default is BCE(pos_weight).
        score_fn: optional callable (validation predictions DataFrame) -> float to maximise for early
            stopping; default is macro-F1 over all validation clips.

    Returns:
        dict with best_epoch (1-based), best_val_macro_f1, epochs_run, history (list of dicts).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
        _bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
        loss_fn = lambda logits, batch, dev: _bce(logits.float(), batch["label"].to(dev))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopper(patience)
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total, n_seen = 0.0, 0
        n_batches = len(train_loader)
        for i, batch in enumerate(train_loader):
            with _autocast(device):
                logits = forward(model, batch, device)
            n_b = len(batch["label"])
            loss = loss_fn(logits, batch, device)
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or i + 1 == n_batches:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * n_b
            n_seen += n_b
        val = predict(model, val_loader, device, forward)
        val_f1 = score_fn(val) if score_fn else compute_metrics(val["label"], val["prob"])["macro_f1"]
        improved = stopper.update(val_f1, epoch, model)
        history.append({"epoch": epoch, "train_loss": total / max(n_seen, 1), "val_macro_f1": val_f1})
        last = stopper.should_stop or epoch == max_epochs
        if epoch % log_every == 0 or last or log_every == 1:
            print(f"    epoch {epoch:3d}  train_loss {history[-1]['train_loss']:.4f}  "
                  f"val_macro_f1 {val_f1:.4f}{'  *' if improved else ''}")
        if stopper.should_stop:
            break
    model.load_state_dict(stopper.best_state)
    return {"best_epoch": stopper.best_epoch, "best_val_macro_f1": stopper.best_score,
            "epochs_run": len(history), "history": history}


# ------------------------------------------------------------- results files and resuming
def setup_id(task, train_datasets, test_datasets, dropout, weight_decay):
    """String that identifies a configuration; part of every run key."""
    return (f"task={task};train={'+'.join(train_datasets)};test={'+'.join(test_datasets)};"
            f"dropout={dropout:g};wd={weight_decay:g}")


def run_key(setup, mode, fold, seed):
    """Hashable identity of one run."""
    return (str(setup), str(mode), int(fold), int(seed))


def load_finished_runs(path):
    """Set of run keys (setup, mode, fold, seed) already present in the results CSV."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    return {run_key(r.setup, r.mode, r.fold, r.seed) for r in df.itertuples()}


def append_rows(path, rows):
    """Append a list of dicts / a DataFrame to a CSV, writing the header only for a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def drop_unfinished_predictions(pred_path, finished):
    """Remove prediction rows of runs that never reached the results CSV (a run that was killed
    after writing its predictions would otherwise be duplicated when it is repeated)."""
    pred_path = Path(pred_path)
    if not pred_path.is_file() or pred_path.stat().st_size == 0:
        return
    df = pd.read_csv(pred_path)
    keep = [run_key(r.setup, r.mode, r.fold, r.seed) in finished for r in df.itertuples()]
    if not all(keep):
        print(f"Removing {len(keep) - sum(keep)} prediction rows of unfinished runs from {pred_path}")
        df[keep].to_csv(pred_path, index=False)


def summarize(results):
    """Mean and std (over folds and seeds) of the metrics for every (setup, mode)."""
    rows = []
    for (setup, mode), g in results.groupby(["setup", "mode"], sort=False):
        row = {"setup": setup, "mode": mode, "n_runs": len(g)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary):
    """Print the summary as 'mean ± std' per mode."""
    for setup, g in summary.groupby("setup", sort=False):
        print(f"\nSummary for {setup}")
        for r in g.itertuples():
            cells = "  ".join(f"{m} {getattr(r, m + '_mean'):.3f} ± {getattr(r, m + '_std'):.3f}" for m in METRICS)
            print(f"  {r.mode:7s} (n={r.n_runs})  {cells}")


# ------------------------------------------------------------------------------- the CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cross-validated training of MultimodalModel.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["stress", "deception"])
    ap.add_argument("--datasets", nargs="+", choices=["mas", "bol", "rlt"], default=None,
                    help="datasets for k-fold runs, pooled (default: mas for stress, bol rlt for deception)")
    ap.add_argument("--train-datasets", nargs="+", choices=["mas", "bol", "rlt"], default=None,
                    help="cross-dataset run: train (and validate) on these ...")
    ap.add_argument("--test-datasets", nargs="+", choices=["mas", "bol", "rlt"], default=None,
                    help="... and test on these (all their clips, fold = -1)")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="test folds to run (ignored for cross-dataset runs)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--out", default=None, help="results CSV (default results/<task>.csv)")
    ap.add_argument("--features-root", default="features", help="folder with <dataset>/metadata.csv and .npz files")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="best checkpoints go to <ckpt-dir>/<out stem>/")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of training groups used for validation")
    ap.add_argument("--balance", choices=list(BALANCE_CHOICES), default="none",
                    help="draw training batches balanced across tasks and/or datasets")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--preload", action="store_true", help="keep loaded clips in RAM (about 1 MB per clip)")
    ap.add_argument("--drop-missing", action="store_true", help="drop metadata rows without an .npz instead of failing")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 20 clips; writes <out stem>_smoke files, never resumes")
    return ap.parse_args(argv)


def output_paths(out, smoke):
    """(results csv, summary csv, predictions csv); smoke runs get a separate _smoke stem."""
    out = Path(out)
    stem = out.stem + ("_smoke" if smoke else "")
    return (out.with_name(stem + out.suffix), out.with_name(stem + "_summary.csv"),
            out.with_name(stem + "_predictions.csv"), stem)


def main(argv=None):
    args = parse_args(argv)
    cross = args.train_datasets is not None or args.test_datasets is not None
    if cross:
        if not (args.train_datasets and args.test_datasets):
            raise SystemExit("error: --train-datasets and --test-datasets must be given together")
        if args.datasets:
            raise SystemExit("error: use either --datasets or --train-datasets/--test-datasets")
        if set(args.train_datasets) & set(args.test_datasets):
            raise SystemExit("error: --train-datasets and --test-datasets must not overlap")
        train_names, test_names = args.train_datasets, args.test_datasets
        folds = [CROSS_FOLD]
    else:
        train_names = test_names = args.datasets or DEFAULT_DATASETS[args.task]
        folds = args.folds
    max_epochs = 2 if args.smoke else args.max_epochs
    log_every = 1 if args.smoke else 10

    out_csv, summary_csv, pred_csv, stem = output_paths(args.out or f"results/{args.task}.csv", args.smoke)
    ckpt_dir = Path(args.ckpt_dir) / stem
    if args.smoke:  # smoke outputs are throw-away: start them afresh
        for p in (out_csv, summary_csv, pred_csv):
            p.unlink(missing_ok=True)

    device = choose_device()
    missing = "drop" if args.drop_missing else "error"
    all_names = sorted(set(train_names) | set(test_names))
    df = load_metadata(args.features_root, all_names, task=args.task, missing=missing)
    print(f"Loaded {len(df)} {args.task} clips from {all_names}: "
          + ", ".join(f"{d} {n}" for d, n in df["dataset"].value_counts().sort_index().items()))
    setup = setup_id(args.task, train_names, test_names, args.dropout, args.weight_decay)
    balance_by = BALANCE_CHOICES[args.balance]
    cache = {} if args.preload else None

    finished = set() if args.smoke else load_finished_runs(out_csv)
    drop_unfinished_predictions(pred_csv, finished)
    print(f"Setup: {setup}\nResults: {out_csv}  ({len(finished)} finished runs found)")

    for fold in folds:
        for seed in args.seeds:
            todo = [m for m in args.modes if run_key(setup, m, fold, seed) not in finished]
            for m in args.modes:
                if m not in todo:
                    print(f"[{m} fold {fold} seed {seed}] already finished, skipping")
            if not todo:
                continue
            if cross:
                splits = make_cross_splits(df[df["dataset"].isin(train_names)], df[df["dataset"].isin(test_names)],
                                           args.val_frac, seed)
            else:
                if not (df["fold"] == fold).any():
                    print(f"warning: no clips in fold {fold}, skipping")
                    continue
                splits = make_fold_splits(df, fold, args.val_frac, seed)
            if args.smoke:
                splits = subsample_for_smoke(*splits, seed=seed)
            train_df, val_df, test_df = splits
            print(f"\n== fold {fold}, seed {seed}: train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
                  f"val {len(val_df)} / {val_df['group_key'].nunique()}, test {len(test_df)} / {test_df['group_key'].nunique()}")
            assert not (set(train_df["group_key"]) & set(val_df["group_key"])), "train/val share a group"
            assert not (set(train_df["group_key"]) & set(test_df["group_key"])), "train/test share a group"
            assert not (set(val_df["group_key"]) & set(test_df["group_key"])), "val/test share a group"

            stats = compute_stats(train_df, args.features_root, cache)  # training rows only
            n_pos = int(train_df["label"].sum())
            pos_weight = (len(train_df) - n_pos) / n_pos if n_pos else 1.0
            val_loader = build_loader(val_df, args.features_root, stats, args.batch_size, cache=cache,
                                      workers=args.workers, device=device)
            test_loader = build_loader(test_df, args.features_root, stats, args.batch_size, cache=cache,
                                       workers=args.workers, device=device)

            for mode in todo:
                print(f"[{mode} fold {fold} seed {seed}] pos_weight {pos_weight:.3f}")
                t0 = time.perf_counter()
                seed_everything(seed)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                train_loader = build_loader(train_df, args.features_root, stats, args.batch_size, train=True,
                                            balance_by=balance_by, seed=seed, workers=args.workers,
                                            cache=cache, device=device)
                model = build_model(mode, args.dropout)
                info = train_model(model, train_loader, val_loader, device, pos_weight=pos_weight, lr=args.lr,
                                   weight_decay=args.weight_decay, max_epochs=max_epochs, patience=args.patience,
                                   accum_steps=args.accum_steps, log_every=log_every)
                preds = predict(model, test_loader, device)  # model holds the best epoch's weights
                metrics = compute_metrics(preds["label"], preds["prob"])

                ckpt_path = ckpt_dir / f"{mode}_fold{'all' if fold == CROSS_FOLD else fold}_seed{seed}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                            "stats": stats.to_dict(), "setup": setup, "mode": mode, "fold": fold, "seed": seed,
                            "best_epoch": info["best_epoch"], "best_val_macro_f1": info["best_val_macro_f1"],
                            "args": vars(args)}, ckpt_path)

                elapsed = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                # Predictions first, the results row last: the row marks the run as finished.
                append_rows(pred_csv, preds.assign(mode=mode, fold=fold, seed=seed, setup=setup))
                append_rows(out_csv, [{
                    "setup": setup, "task": args.task, "train_datasets": "+".join(train_names),
                    "test_datasets": "+".join(test_names), "mode": mode, "fold": fold, "seed": seed,
                    "dropout": args.dropout, "weight_decay": args.weight_decay,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                    "val_macro_f1": info["best_val_macro_f1"], **metrics,
                    "time_s": round(elapsed, 1), "peak_gpu_mb": round(peak_mb, 1), "device": device.type}])
                finished.add(run_key(setup, mode, fold, seed))
                print(f"[{mode} fold {fold} seed {seed}] test acc {metrics['accuracy']:.3f}  "
                      f"macro-F1 {metrics['macro_f1']:.3f}  AUC {metrics['roc_auc']:.3f}  | best epoch "
                      f"{info['best_epoch']}/{info['epochs_run']}  | {elapsed:.1f} s  | "
                      + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
                del model, train_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if out_csv.is_file():
        results = pd.read_csv(out_csv)
        summary = summarize(results)
        summary.to_csv(summary_csv, index=False)
        print_summary(summary[summary["setup"] == setup])
        print(f"\nWrote {out_csv}, {summary_csv}, {pred_csv}; checkpoints in {ckpt_dir}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    main()
