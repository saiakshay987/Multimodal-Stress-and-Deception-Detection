#!/usr/bin/env python3
"""Pooled-feature logistic-regression baseline for stress detection (no deep model).

One fixed-length vector per clip (mean and std over time, see docs/feature_contract.md):
    audio = [mean, std of audio_w2v (768)] + [mean, std of audio_prosody (3)]            -> 1542 dims
    video = [mean, std of video_face over frames with video_valid] + [mean, std of video_flow]
            -> 1540 dims; all zeros when the clip has no valid frame or empty video arrays
    both  = audio ++ video

Protocol: for each test fold k of the fold column of features/<dataset>/metadata.csv, train on the other
four folds with StandardScaler (fitted on the training folds only) + LogisticRegression(class_weight=
"balanced", C=1.0, max_iter=2000). Nothing is tuned on the test folds. Reported per mode: per-fold and
mean +- std (sample std, ddof=1) accuracy, macro-F1, ROC-AUC (predict_proba), two majority-class baselines
(accuracy and macro-F1): "majority_overall" (overall majority class of the whole dataset, label 0, for every
test clip; also pooled accuracy and macro-F1 over all clips) and "majority_train_fold" (training-fold
majority class), and recall of the stress clips per tag over all
out-of-fold predictions together:
    sentiment_only  mas_labels is exactly "audio stress"
    nonverbal       any other stress clip (label 1)

    python src/stress/baseline.py
    python src/stress/baseline.py --modes audio --out results/stress/baseline_audio.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODES = ("audio", "video", "both")
KEYS = {
    "audio": ("audio_w2v", "audio_prosody"),
    "video": ("video_face", "video_valid", "video_flow"),
}
DIMS = {"audio_w2v": 768, "audio_prosody": 3, "video_face": 768, "video_flow": 2}
AUDIO_DIM = 2 * (768 + 3)
VIDEO_DIM = 2 * (768 + 2)


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def pool(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([x.mean(axis=0), x.std(axis=0)])


def clip_vectors(npz_path: Path, need_audio: bool, need_video: bool):
    """Return (audio_vec or None, video_vec or None, video_missing flag)."""
    if not npz_path.is_file():
        fail(f"missing feature file {npz_path}")
    keys = (KEYS["audio"] if need_audio else ()) + (KEYS["video"] if need_video else ())
    audio = video = None
    no_video = False
    with np.load(npz_path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            fail(f"{npz_path} has no key(s) {missing} (found {list(z.files)})")
        if need_audio:
            w2v, pros = z["audio_w2v"], z["audio_prosody"]
            if w2v.ndim != 2 or w2v.shape[1] != 768 or w2v.shape[0] == 0:
                fail(f"{npz_path}: audio_w2v has shape {w2v.shape}, expected (T>0, 768)")
            if pros.ndim != 2 or pros.shape[1] != 3 or pros.shape[0] == 0:
                fail(f"{npz_path}: audio_prosody has shape {pros.shape}, expected (T>0, 3)")
            audio = np.concatenate([pool(w2v), pool(pros)])
        if need_video:
            face, valid, flow = z["video_face"], z["video_valid"].astype(bool), z["video_flow"]
            if face.shape[0] == 0 or valid.shape[0] == 0 or flow.shape[0] == 0 or not valid.any():
                no_video = True
                video = np.zeros(VIDEO_DIM, dtype=np.float32)
            else:
                if face.shape[0] != valid.shape[0] or face.shape[1:] != (768,):
                    fail(f"{npz_path}: video_face {face.shape} vs video_valid {valid.shape} mismatch")
                if flow.ndim != 2 or flow.shape[1] != 2:
                    fail(f"{npz_path}: video_flow has shape {flow.shape}, expected (T, 2)")
                video = np.concatenate([pool(face[valid]), pool(flow)])
    return audio, video, no_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mas")
    ap.add_argument("--features_root", default="features")
    ap.add_argument("--modes", default="audio,video,both", help="comma list from audio,video,both")
    ap.add_argument("--out", default="results/stress/baseline.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if not modes or bad:
        fail(f"--modes must be a comma list from {MODES}, got {args.modes!r}")
    modes = list(dict.fromkeys(modes))
    need_audio = any(m in ("audio", "both") for m in modes)
    need_video = any(m in ("video", "both") for m in modes)

    root = Path(args.features_root) / args.dataset
    meta_path = root / "metadata.csv"
    if not meta_path.is_file():
        fail(f"missing {meta_path}")
    df = pd.read_csv(meta_path, keep_default_na=False, dtype={"mas_labels": str})
    for col in ("clip_id", "label", "fold", "mas_labels"):
        if col not in df.columns:
            fail(f"{meta_path} has no column {col!r}")
    if df["clip_id"].duplicated().any():
        fail("duplicate clip_id in metadata")
    if "task" in df.columns and not (df["task"] == "stress").all():
        fail(f"{meta_path} contains non-stress rows; this baseline is for the stress task")
    y = df["label"].to_numpy().astype(int)
    folds = df["fold"].to_numpy().astype(int)
    if set(np.unique(y)) - {0, 1}:
        fail("label must be 0/1")
    if sorted(np.unique(folds)) != [0, 1, 2, 3, 4]:
        fail(f"fold column must contain 0-4, got {sorted(np.unique(folds))}")

    # ---- features ----
    a_rows, v_rows, n_no_video = [], [], 0
    for cid in df["clip_id"]:
        a, v, nv = clip_vectors(root / f"{cid}.npz", need_audio, need_video)
        a_rows.append(a)
        v_rows.append(v)
        n_no_video += nv
    feats = {}
    if need_audio:
        feats["audio"] = np.stack(a_rows)
    if need_video:
        feats["video"] = np.stack(v_rows)
        print(f"Clips with no usable video (no valid frame or empty arrays): {n_no_video} of {len(df)} "
              f"-> video vector set to zeros")
    if "both" in modes:
        feats["both"] = np.concatenate([feats["audio"], feats["video"]], axis=1)
    for m in modes:
        if not np.isfinite(feats[m]).all():
            fail(f"non-finite values in the {m} features")

    # ---- tags ----
    tag = np.full(len(df), "", dtype=object)
    stress = y == 1
    sent = stress & (df["mas_labels"].to_numpy() == "audio stress")
    tag[stress] = "nonverbal"
    tag[sent] = "sentiment_only"
    n_sent, n_nonv = int(sent.sum()), int((tag == "nonverbal").sum())
    print(f"Stress clips: {int(stress.sum())} = {n_sent} sentiment_only + {n_nonv} nonverbal")

    # ---- cross-validation ----
    cols = ["accuracy", "macro_f1", "roc_auc",
            "majority_overall_accuracy", "majority_overall_macro_f1",
            "majority_train_fold_accuracy", "majority_train_fold_macro_f1"]
    overall_major = int(np.bincount(y).argmax())
    if overall_major != 0:
        fail(f"overall majority class is {overall_major}, expected 0 (no stress)")
    maj_all = np.full(len(df), overall_major)
    pooled_acc = accuracy_score(y, maj_all)
    pooled_f1 = f1_score(y, maj_all, average="macro")
    print(f"majority_overall (label {overall_major} for every clip), pooled over all {len(df)} clips: "
          f"acc {pooled_acc:.3f}, macro-F1 {pooled_f1:.3f}")
    rows = []
    for m in modes:
        X = feats[m]
        oof = np.full(len(df), -1, dtype=int)
        per_fold = []
        for k in range(5):
            te, tr = folds == k, folds != k
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=args.seed)
            clf.fit(scaler.transform(X[tr]), y[tr])
            Xte = scaler.transform(X[te])
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            oof[te] = pred
            maj = np.full(te.sum(), np.bincount(y[tr]).argmax())
            maj_ov = np.full(te.sum(), overall_major)
            if len(np.unique(y[te])) < 2:
                fail(f"test fold {k} has a single class; ROC-AUC undefined")
            per_fold.append({
                "mode": m, "fold": str(k), "n_test": int(te.sum()),
                "accuracy": accuracy_score(y[te], pred),
                "macro_f1": f1_score(y[te], pred, average="macro"),
                "roc_auc": roc_auc_score(y[te], proba),
                "majority_overall_accuracy": accuracy_score(y[te], maj_ov),
                "majority_overall_macro_f1": f1_score(y[te], maj_ov, average="macro"),
                "majority_train_fold_accuracy": accuracy_score(y[te], maj),
                "majority_train_fold_macro_f1": f1_score(y[te], maj, average="macro"),
            })
        assert (oof >= 0).all()
        rows += per_fold
        fd = pd.DataFrame(per_fold)
        mean_row = {"mode": m, "fold": "mean", "n_test": len(df)}
        std_row = {"mode": m, "fold": "std", "n_test": np.nan}
        for c in cols:
            mean_row[c] = fd[c].mean()
            std_row[c] = fd[c].std(ddof=1)
        # tag recall over all out-of-fold predictions together (stored on the mean row)
        mean_row["oof_recall_sentiment_only"] = float((oof[sent] == 1).mean()) if n_sent else np.nan
        mean_row["oof_recall_nonverbal"] = float((oof[tag == "nonverbal"] == 1).mean()) if n_nonv else np.nan
        mean_row["majority_overall_pooled_accuracy"] = pooled_acc
        mean_row["majority_overall_pooled_macro_f1"] = pooled_f1
        mean_row["n_sentiment_only"] = n_sent
        mean_row["n_nonverbal"] = n_nonv
        rows += [mean_row, std_row]

    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    # ---- print ----
    f = lambda v: "   -  " if pd.isna(v) else f"{v:.3f}"
    print(f"\nDataset {args.dataset}, {len(df)} clips, features: "
          + ", ".join(f"{m}={feats[m].shape[1]}d" for m in modes))
    print("majority_overall = label 0 (overall majority) for every test clip; "
          "majority_train_fold = training-fold majority class")
    hdr = (f"{'mode':<6}{'fold':<6}{'acc':>7}{'F1':>7}{'AUC':>7}"
           f"{'overall_acc':>13}{'overall_F1':>12}{'trainfold_acc':>15}{'trainfold_F1':>14}")
    print(hdr + "\n" + "-" * len(hdr))
    for _, r in res.iterrows():
        print(f"{r['mode']:<6}{r['fold']:<6}{f(r['accuracy']):>7}{f(r['macro_f1']):>7}{f(r['roc_auc']):>7}"
              f"{f(r['majority_overall_accuracy']):>13}{f(r['majority_overall_macro_f1']):>12}"
              f"{f(r['majority_train_fold_accuracy']):>15}{f(r['majority_train_fold_macro_f1']):>14}")
    print("\nMean +- std over folds, and out-of-fold stress recall by tag:")
    for m in modes:
        mr = res[(res["mode"] == m) & (res["fold"] == "mean")].iloc[0]
        sr = res[(res["mode"] == m) & (res["fold"] == "std")].iloc[0]
        print(f"  {m:<6} acc {mr['accuracy']:.3f}+-{sr['accuracy']:.3f}  F1 {mr['macro_f1']:.3f}+-{sr['macro_f1']:.3f}  "
              f"AUC {mr['roc_auc']:.3f}+-{sr['roc_auc']:.3f}  "
              f"| majority_overall acc {mr['majority_overall_accuracy']:.3f}+-{sr['majority_overall_accuracy']:.3f} "
              f"F1 {mr['majority_overall_macro_f1']:.3f}+-{sr['majority_overall_macro_f1']:.3f} "
              f"(pooled acc {mr['majority_overall_pooled_accuracy']:.3f} F1 {mr['majority_overall_pooled_macro_f1']:.3f})  "
              f"| majority_train_fold acc {mr['majority_train_fold_accuracy']:.3f}+-{sr['majority_train_fold_accuracy']:.3f} "
              f"F1 {mr['majority_train_fold_macro_f1']:.3f}+-{sr['majority_train_fold_macro_f1']:.3f}  "
              f"| recall sentiment_only {mr['oof_recall_sentiment_only']:.3f} "
              f"(n={n_sent}), nonverbal {mr['oof_recall_nonverbal']:.3f} (n={n_nonv})")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
