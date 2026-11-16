#!/usr/bin/env python3
"""Pooled-feature logistic-regression baseline for deception detection (Bag-of-Lies, Real-Life Trial).

Each clip becomes ONE fixed-length vector (mean and std over time):
    audio = [audio_w2v | audio_prosody]            -> 771 dims x (mean, std) = 1542
    video = [video_face | video_flow]              -> 770 dims x (mean, std) = 1540
            pooled over frames with video_valid True only (both face and flow). A clip with
            no valid frame gets NaN for the whole video vector; the pipeline imputes it with
            the TRAINING mean (SimpleImputer is fitted on training rows only).
    av    = concatenation of the two.
Classifier: Pipeline(SimpleImputer, StandardScaler, LogisticRegression(class_weight="balanced")),
refitted from scratch in every split, so the imputer and the scaler only ever see training clips.

Evaluations (for audio, video and av):
    within_bol    5-fold CV with the fold column of features/bol/metadata.csv
    within_rlt    5-fold CV with the fold column of features/rlt/metadata.csv
    bol_to_rlt    train on all of Bag-of-Lies, test on all of Real-Life Trial
    rlt_to_bol    train on all of Real-Life Trial, test on all of Bag-of-Lies

Reported: accuracy, macro-F1, ROC-AUC (mean +- std over folds where there are folds; std is
blank for the single-split cross-dataset runs), the majority-class accuracy of the test data
(mean over folds), and a two-sided scipy.stats.binomtest p-value of the accuracy against 50 %,
computed on all out-of-fold predictions together (n = number of test clips; for the within
runs every clip is tested exactly once). The script stops if a group_id is in more than one fold.

    python src/deception/baseline.py                       # from the repository root
    python src/deception/baseline.py --features-root features --out results/deception/baseline.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from data import load_metadata, npz_path  # noqa: E402  (src/common/data.py, unchanged)

MODALITIES = ("audio", "video", "av")
SEED = 42


# --------------------------------------------------------------------------- pooling
def _pool(x):
    """(T, D) -> (2D,) [mean, std over time]; NaN if there are no steps."""
    if len(x) == 0:
        return np.full(2 * x.shape[1], np.nan, dtype=np.float32)
    x = x.astype(np.float64)
    return np.concatenate([x.mean(0), x.std(0)]).astype(np.float32)


def pool_clip(path):
    """Return (audio_vec (1542,), video_vec (1540,)) for one .npz file."""
    with np.load(path) as z:
        audio = np.concatenate([z["audio_w2v"].astype(np.float32), z["audio_prosody"].astype(np.float32)], axis=1)
        video = np.concatenate([z["video_face"].astype(np.float32), z["video_flow"].astype(np.float32)], axis=1)
        valid = z["video_valid"].astype(bool)
    if len(valid) != len(video):
        raise ValueError(f"{path}: video_valid has {len(valid)} steps but video has {len(video)}")
    audio, video = np.nan_to_num(audio), np.nan_to_num(video)
    return _pool(audio), _pool(video[valid])


def build_matrix(features_root, dataset):
    """Metadata (sorted as in the csv) plus one pooled vector per clip for each modality."""
    meta = load_metadata(features_root, [dataset])  # raises if an .npz is missing
    a, v = zip(*[pool_clip(npz_path(features_root, dataset, c)) for c in meta["clip_id"]])
    a, v = np.stack(a), np.stack(v)
    return meta, {"audio": a, "video": v, "av": np.concatenate([a, v], axis=1)}


def check_folds(meta, name):
    """Stop if any group_id is in more than one fold."""
    n_folds = meta.groupby("group_id")["fold"].nunique()
    bad = n_folds[n_folds > 1]
    if len(bad):
        sys.exit(f"STOP: {name}: {len(bad)} group_id(s) appear in more than one fold, e.g. {bad.index[:5].tolist()}")
    print(f"{name}: {len(meta)} clips, {meta['group_id'].nunique()} groups, folds {sorted(meta['fold'].unique())}, "
          f"no group_id spans two folds")


# ---------------------------------------------------------------------------- model
def make_model():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def fit_predict(x_train, y_train, x_test):
    model = make_model().fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)[:, 1]


def split_metrics(y, pred, prob):
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "roc_auc": auc, "majority_acc": max(y.mean(), 1 - y.mean())}


# ----------------------------------------------------------------------- evaluations
def run_within(name, meta, mats):
    rows = []
    y = meta["label"].to_numpy()
    for mod in MODALITIES:
        x = mats[mod]
        per_fold, correct, total = [], 0, 0
        for k in sorted(meta["fold"].unique()):
            te = (meta["fold"] == k).to_numpy()
            pred, prob = fit_predict(x[~te], y[~te], x[te])
            per_fold.append(split_metrics(y[te], pred, prob))
            correct += int((pred == y[te]).sum())
            total += int(te.sum())
        rows.append(summarise(name, mod, len(meta), per_fold, correct, total))
    return rows


def run_cross(name, meta_tr, mats_tr, meta_te, mats_te):
    rows = []
    y_tr, y_te = meta_tr["label"].to_numpy(), meta_te["label"].to_numpy()
    for mod in MODALITIES:
        pred, prob = fit_predict(mats_tr[mod], y_tr, mats_te[mod])
        m = split_metrics(y_te, pred, prob)
        rows.append(summarise(name, mod, len(meta_te), [m], int((pred == y_te).sum()), len(y_te)))
    return rows


def summarise(setting, modality, n_clips, per_fold, correct, total):
    df = pd.DataFrame(per_fold)
    row = {"setting": setting, "modality": modality, "n_test_clips": n_clips, "n_splits": len(df)}
    for col in ("accuracy", "macro_f1", "roc_auc"):
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else float("nan")
    row["majority_acc"] = df["majority_acc"].mean()
    row["n_correct"], row["n_tested"] = correct, total
    row["p_binom_vs_50"] = binomtest(correct, total, 0.5).pvalue
    return row


# ------------------------------------------------------------------------------ main
def fmt(m, s):
    return f"{m:.3f}" if np.isnan(s) else f"{m:.3f}±{s:.3f}"


def print_table(res):
    cols = ["setting", "modality", "n", "accuracy", "macro-F1", "ROC-AUC", "majority", "p (binom)"]
    lines = [[r.setting, r.modality, str(r.n_test_clips), fmt(r.accuracy_mean, r.accuracy_std),
              fmt(r.macro_f1_mean, r.macro_f1_std), fmt(r.roc_auc_mean, r.roc_auc_std),
              f"{r.majority_acc:.3f}", f"{r.p_binom_vs_50:.4f}"] for r in res.itertuples()]
    w = [max(len(c), *(len(l[i]) for l in lines)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print("  ".join("-" * x for x in w))
    for l in lines:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(l)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--out", default="results/deception/baseline.csv")
    args = ap.parse_args()
    np.random.seed(SEED)

    meta_b, mats_b = build_matrix(args.features_root, "bol")
    meta_r, mats_r = build_matrix(args.features_root, "rlt")
    check_folds(meta_b, "bol")
    check_folds(meta_r, "rlt")
    for n, meta, mats in (("bol", meta_b, mats_b), ("rlt", meta_r, mats_r)):
        nv = int(np.isnan(mats["video"]).any(1).sum())
        print(f"{n}: {nv} clip(s) without any valid face frame (video vector imputed with the training mean)")

    rows = []
    rows += run_within("within_bol", meta_b, mats_b)
    rows += run_within("within_rlt", meta_r, mats_r)
    rows += run_cross("bol_to_rlt", meta_b, mats_b, meta_r, mats_r)
    rows += run_cross("rlt_to_bol", meta_r, mats_r, meta_b, mats_b)
    res = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print()
    print_table(res)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
