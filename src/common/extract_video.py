#!/usr/bin/env python3
"""Extract the video (face) features defined in docs/feature_contract.md.

Usage:
    python src/common/extract_video.py bol --limit 5 --verbose --examples_dir results/face_check
    python src/common/extract_video.py bol
    python src/common/extract_video.py rlt --metadata some/other/metadata.csv
    python src/common/extract_video.py bol --data_root data --features_root features
    python src/common/extract_video.py bol --perturb brightness --level 0.5 --clip-ids ids.txt

--perturb {brightness,gamma} --level X: every sampled frame is perturbed before face detection
(brightness: pixel * X, saturating, X >= 0; gamma: 255 * (pixel / 255) ** (1 / X), X > 0, so
X > 1 brightens) and the output goes to <perturbed_root>/<perturb>_<X>/<dataset>/ (default
perturbed_root: features_perturbed) instead of <features_root>/<dataset>/. The metadata is still
read from <features_root>/<dataset>/metadata.csv; nothing under <features_root> is written.
--clip-ids FILE: process only the clip_ids listed in FILE (one per line, blank lines ignored).

Reads <features_root>/<dataset>/metadata.csv (or --metadata). For every row, the clip
<data_root>/<video_path> is read with OpenCV, frames are sampled at 5 fps for at most the
first 60 s (frame k is the one nearest to t = k / 5 s), the largest face is found with
MediaPipe face detection and cropped, and these keys are written (added or replaced; other
keys already in the file are kept) to <features_root>/<dataset>/<clip_id>.npz:

    video_face    (T, 768)      float16  CLS embedding (last hidden state, after the final
                                         layer norm) of the 224x224 RGB face crop from
                                         trpakov/vit-face-expression; zeros where no face
    video_valid   (T,)          bool     a face was found in the frame
    video_flow    (T, 2)        float32  mean and std of the Farneback optical-flow magnitude
                                         between the previous and the current sampled face
                                         crop (both as 112x112 grayscale)
    video_crops64 (T, 64, 64)   uint8    grayscale face crops (zeros where no face)

and, for the GPU fine-tuning stage, 16 face crops (224x224 RGB, evenly spaced over the frames
with a face) to <features_root>/<dataset>/crops224/<clip_id>.npy as uint8 (16, 224, 224, 3).

Details
-------
* Face: MediaPipe Tasks FaceDetector (BlazeFace short-range, min confidence 0.5); the box with
  the largest area is used. The model file is downloaded once to --face_model if missing.
* Crop: the box is made square (side = max(w, h)), grown by a 20% margin on every side
  (side * 1.4) around the box centre, and clipped to the image. Crops are resized to 224x224
  (ViT / crops224) and, from the same crop converted to gray, to 112x112 (flow) and 64x64.
* Embedding model: --vit_model (default trpakov/vit-face-expression). If it cannot be loaded,
  google/vit-base-patch16-224-in21k is used instead and a loud warning is printed. Input
  normalisation uses the mean/std of the model's own preprocessor config. Inference is under
  torch.no_grad with fp16 autocast on CUDA.
* Flow: video_flow[t] compares sampled face crops t-1 and t. It is (0, 0) for t = 0 and
  wherever the frame t-1 or t has no face.
* If fewer than 16 frames have a face, crops224 repeats frames (evenly spaced indices with
  repeats). If no frame has a face the clip is still written to the .npz (video_valid all
  false) but no crops224 file is written; it is listed in low_face.csv with n_face = 0.
* Clips where fewer than half of the frames have a face are logged to
  <features_root>/<dataset>/low_face.csv (clip_id, n_frames, n_face, frac) and printed.
* Clips are skipped when the .npz already has all four keys and crops224/<clip_id>.npy exists
  (or the clip is listed in low_face.csv with n_face = 0).
* File names are always f"{clip_id}.npz" etc.: clip ids can contain dots, so never use
  Path.with_suffix / Path.stem on them.
"""
import argparse
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

FPS_OUT = 5
MAX_SECONDS = 60
MARGIN = 0.20
FLOW_SIZE = 112
SMALL_SIZE = 64
VIT_SIZE = 224
N_CROPS224 = 16
EMB_DIM = 768
KEYS = ("video_face", "video_valid", "video_flow", "video_crops64")
VIT_DEFAULT = "trpakov/vit-face-expression"
VIT_FALLBACK = "google/vit-base-patch16-224-in21k"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
LOW_FACE_COLUMNS = ["clip_id", "n_frames", "n_face", "frac"]


# --------------------------------------------------------------------------- setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save_npy(path, array):
    tmp = Path(f"{path}.tmp")
    try:
        with open(tmp, "wb") as f:  # file object: np.save must not append ".npy"
            np.save(f, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_update_npz(path, new_arrays):
    """Load path if it exists, add/replace new_arrays, write to a temp file, then rename."""
    arrays = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    arrays.update(new_arrays)
    tmp = Path(f"{path}.tmp")
    try:
        with open(tmp, "wb") as f:  # file object: np.savez must not append ".npz"
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def npz_keys(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return set(z.files)
    except Exception:
        return None


def read_low_face(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"clip_id": str})
    return {r.clip_id: (int(r.n_frames), int(r.n_face)) for r in df.itertuples(index=False)}


def write_low_face(path, entries):
    rows = [(cid, n, k, k / n if n else 0.0) for cid, (n, k) in entries.items()]
    pd.DataFrame(rows, columns=LOW_FACE_COLUMNS).to_csv(path, index=False, float_format="%.4f")


# --------------------------------------------------------------------------- models
def load_face_detector(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe face detector model\n  {FACE_MODEL_URL}\n  -> {model_path}")
        tmp = Path(f"{model_path}.tmp")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        os.replace(tmp, model_path)
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5)
    return mp, vision.FaceDetector.create_from_options(options)


def hide_broken_packages(names=("torchvision", "torchaudio")):
    """transformers imports torchvision / torchaudio when they are installed; builds made for
    another torch version raise at import and take transformers down with them. This script
    needs neither, so a package that fails to import is made invisible to transformers
    (sys.modules[x] = None makes importlib.util.find_spec return None). The venv is not touched."""
    for pkg in names:
        try:
            __import__(pkg)
        except Exception as e:
            for m in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
                del sys.modules[m]
            sys.modules[pkg] = None
            print(f"note: {pkg} is broken in this environment ({type(e).__name__}: {str(e)[:90]}); "
                  f"ignoring it (not needed here). Consider reinstalling a build that matches torch.")


def load_vit(name, device):
    """Return (model, mean, std, name_used). Falls back to the plain ImageNet-21k ViT."""
    hide_broken_packages()
    from transformers import ViTModel

    used = name
    try:
        model = ViTModel.from_pretrained(name, add_pooling_layer=False)
    except Exception as e:
        print("!" * 78)
        print(f"WARNING: could not load '{name}' ({type(e).__name__}: {str(e)[:200]})")
        print(f"WARNING: falling back to '{VIT_FALLBACK}' -> video_face is NOT a face-expression embedding")
        print("!" * 78)
        used = VIT_FALLBACK
        model = ViTModel.from_pretrained(used, add_pooling_layer=False)
    if model.config.hidden_size != EMB_DIM:
        sys.exit(f"{used}: hidden size {model.config.hidden_size}, expected {EMB_DIM}")
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(used)
        mean, std = tuple(proc.image_mean), tuple(proc.image_std)
    except Exception as e:
        print(f"note: could not read the image processor of '{used}' ({type(e).__name__}); "
              f"using mean/std 0.5")
    model = model.to(device).eval()
    return model, mean, std, used


# --------------------------------------------------------------------------- video
def sample_frames(path):
    """Yield BGR frames at FPS_OUT per second for at most MAX_SECONDS (frame nearest to k / FPS_OUT)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            raise RuntimeError(f"unusable fps {fps} in {path}")
        last = -1  # index of the last grabbed frame
        k = 0
        while k / FPS_OUT < MAX_SECONDS:
            target = int(round(k * fps / FPS_OUT))
            ok = True
            while last < target:
                ok = cap.grab()
                last += 1
                if not ok:
                    break
            if not ok:
                return
            ok, frame = cap.retrieve()
            if not ok:
                return
            yield frame
            k += 1
    finally:
        cap.release()


def make_perturbation(kind, level):
    """Return a function BGR uint8 frame -> perturbed BGR uint8 frame."""
    if kind == "brightness":
        return lambda frame: cv2.convertScaleAbs(frame, alpha=level)
    if kind == "gamma":
        lut = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** (1.0 / level)), 0, 255).astype(np.uint8)
        return lambda frame: cv2.LUT(frame, lut)
    raise ValueError(f"unknown perturbation: {kind}")


def largest_face_box(detector, mp, rgb):
    """(x0, y0, x1, y1) integer crop window (square, 20% margin, clipped) or None."""
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.detections:
        return None
    bb = max((d.bounding_box for d in res.detections), key=lambda b: b.width * b.height)
    h_img, w_img = rgb.shape[:2]
    side = max(bb.width, bb.height) * (1.0 + 2.0 * MARGIN)
    cx, cy = bb.origin_x + bb.width / 2.0, bb.origin_y + bb.height / 2.0
    x0 = int(max(0, round(cx - side / 2.0)))
    y0 = int(max(0, round(cy - side / 2.0)))
    x1 = int(min(w_img, round(cx + side / 2.0)))
    y1 = int(min(h_img, round(cy + side / 2.0)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def resize(img, size):
    interp = cv2.INTER_AREA if min(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def flow_stats(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float(mag.mean()), float(mag.std())


def process_clip(path, detector, mp, perturb=None):
    """Run detection / cropping / flow on one clip. Returns dict of per-frame arrays.
    perturb: optional function applied to each BGR frame before face detection."""
    valid, flow, crops64, crops224 = [], [], [], []
    prev_gray = None
    n = 0
    for frame in sample_frames(path):
        n += 1
        if perturb is not None:
            frame = perturb(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = largest_face_box(detector, mp, rgb)
        if box is None:
            valid.append(False)
            flow.append((0.0, 0.0))
            crops64.append(np.zeros((SMALL_SIZE, SMALL_SIZE), np.uint8))
            crops224.append(None)
            prev_gray = None
            continue
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        g112 = resize(gray, FLOW_SIZE)
        flow.append(flow_stats(prev_gray, g112) if prev_gray is not None else (0.0, 0.0))
        prev_gray = g112
        valid.append(True)
        crops64.append(resize(gray, SMALL_SIZE))
        crops224.append(resize(crop, VIT_SIZE))
    if n == 0:
        raise RuntimeError("no frames could be read")
    return {
        "valid": np.asarray(valid, dtype=bool),
        "flow": np.asarray(flow, dtype=np.float32),
        "crops64": np.stack(crops64).astype(np.uint8),
        "crops224": crops224,  # list of (224, 224, 3) uint8 or None
    }


def embed_crops(model, mean, std, crops, device, batch_size):
    """(N, 224, 224, 3) uint8 RGB list -> (N, 768) float16 CLS embeddings."""
    if not crops:
        return np.zeros((0, EMB_DIM), np.float16)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            x = torch.from_numpy(np.stack(crops[i:i + batch_size])).to(device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean_t) / std_t
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                h = model(pixel_values=x).last_hidden_state[:, 0]
            out.append(h.float().cpu())
    emb = torch.cat(out).numpy()
    if not np.isfinite(emb).all():
        raise RuntimeError("non-finite values in the ViT embedding")
    return emb.astype(np.float16)


def pick_16(valid):
    """Indices of 16 frames evenly spaced over the frames with a face (repeats if fewer)."""
    idx = np.flatnonzero(valid)
    return idx[np.round(np.linspace(0, len(idx) - 1, N_CROPS224)).astype(int)]


def save_examples(df, feat_dir, out_dir, dataset, n):
    """Save the middle crop of crops224 of the first n clips that have one as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in df.itertuples(index=False):
        p = feat_dir / "crops224" / f"{row.clip_id}.npy"
        if not p.exists():
            continue
        crop = np.load(p)[N_CROPS224 // 2]
        dst = out_dir / f"{dataset}_{row.clip_id}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"example crop -> {dst}")
        saved += 1
        if saved >= n:
            break


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name, e.g. bol (reads <features_root>/<dataset>/metadata.csv)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N metadata rows")
    ap.add_argument("--metadata", default=None,
                    help="metadata csv to use instead of <features_root>/<dataset>/metadata.csv "
                         "(needs columns clip_id, video_path)")
    ap.add_argument("--features_root", default="features", help="folder that holds <dataset>/ (default: features)")
    ap.add_argument("--data_root", default="data", help="folder that video_path is relative to (default: data)")
    ap.add_argument("--vit_model", default=VIT_DEFAULT, help=f"face-expression ViT (default: {VIT_DEFAULT})")
    ap.add_argument("--face_model", default="checkpoints/mediapipe/blaze_face_short_range.tflite",
                    help="MediaPipe face detector .tflite; downloaded there if missing")
    ap.add_argument("--batch_size", type=int, default=32, help="ViT batch size (default: 32)")
    ap.add_argument("--examples_dir", default=None,
                    help="if set, save example face crops (PNG) there after the run")
    ap.add_argument("--n_examples", type=int, default=3, help="number of example crops (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="print the shapes of all keys for every clip")
    ap.add_argument("--perturb", choices=("brightness", "gamma"), default=None,
                    help="perturb every frame before face detection (needs --level); output goes to "
                         "<perturbed_root>/<perturb>_<level>/<dataset>/ instead of <features_root>/<dataset>/")
    ap.add_argument("--level", type=float, default=None,
                    help="strength of --perturb: brightness multiplier (>= 0) or gamma (> 0)")
    ap.add_argument("--perturbed_root", default="features_perturbed",
                    help="folder for perturbed features (default: features_perturbed)")
    ap.add_argument("--clip-ids", "--clip_ids", dest="clip_ids", default=None,
                    help="text file with one clip_id per line; only these clips are processed")
    args = ap.parse_args()
    if (args.perturb is None) != (args.level is None):
        ap.error("--perturb and --level must be given together")
    if args.perturb == "brightness" and not args.level >= 0:
        ap.error("--level for brightness must be >= 0")
    if args.perturb == "gamma" and not args.level > 0:
        ap.error("--level for gamma must be > 0")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dir = Path(args.features_root) / args.dataset
    meta_path = Path(args.metadata) if args.metadata else feat_dir / "metadata.csv"
    perturb = None
    if args.perturb:  # write somewhere else; the clean feat_dir is only read (metadata)
        perturb = make_perturbation(args.perturb, args.level)
        feat_dir = Path(args.perturbed_root) / f"{args.perturb}_{args.level:g}" / args.dataset
        print(f"perturbation: {args.perturb} level {args.level:g} -> {feat_dir}")
    if not meta_path.exists():
        sys.exit(f"metadata not found: {meta_path}")
    df = pd.read_csv(meta_path, dtype={"clip_id": str})
    for col in ("clip_id", "video_path"):
        if col not in df.columns:
            sys.exit(f"{meta_path} has no column '{col}'")
    if args.clip_ids:
        with open(args.clip_ids) as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        missing = sorted(set(wanted) - set(df["clip_id"]))
        if missing:
            print(f"warning: {len(missing)} clip_ids from {args.clip_ids} are not in {meta_path}: "
                  f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
        df = df[df["clip_id"].isin(set(wanted))]
        if df.empty:
            sys.exit(f"none of the clip_ids in {args.clip_ids} are in {meta_path}")
    if args.limit is not None:
        df = df.head(args.limit)

    crops_dir = feat_dir / "crops224"
    crops_dir.mkdir(parents=True, exist_ok=True)
    low_face_path = feat_dir / "low_face.csv"
    low_face = read_low_face(low_face_path)

    # Skip check first so a fully done run does not load any model.
    todo = []
    n_skipped = 0
    for row in df.itertuples(index=False):
        npz_path = feat_dir / f"{row.clip_id}.npz"
        keys = npz_keys(npz_path) if npz_path.exists() else set()
        crops_ok = (crops_dir / f"{row.clip_id}.npy").exists() or low_face.get(row.clip_id, (0, -1))[1] == 0
        if keys is not None and set(KEYS) <= keys and crops_ok:
            n_skipped += 1
        else:
            todo.append(row)
    print(f"device={device} dataset={args.dataset} rows={len(df)} to_process={len(todo)} already_done={n_skipped}")

    mp = detector = model = None
    mean = std = None
    vit_used = args.vit_model
    if todo:
        mp, detector = load_face_detector(args.face_model)
        model, mean, std, vit_used = load_vit(args.vit_model, device)
        print(f"face embedding model: {vit_used} (mean={mean}, std={std})")

    n_done = n_failed = n_low = 0
    elapsed = []
    for row in tqdm(todo, desc=f"{args.dataset} video", unit="clip"):
        clip_id = row.clip_id
        video = Path(args.data_root) / row.video_path
        t0 = time.time()
        try:
            if not video.exists():
                raise FileNotFoundError(f"video not found: {video}")
            existing = feat_dir / f"{clip_id}.npz"
            if existing.exists() and npz_keys(existing) is None:
                raise RuntimeError(f"existing {existing.name} is unreadable; not overwriting it")
            r = process_clip(video, detector, mp, perturb)
            valid = r["valid"]
            t = len(valid)
            n_face = int(valid.sum())

            face = np.zeros((t, EMB_DIM), np.float16)
            if n_face:
                face[valid] = embed_crops(model, mean, std, [c for c in r["crops224"] if c is not None],
                                          device, args.batch_size)
                sel = pick_16(valid)
                atomic_save_npy(crops_dir / f"{clip_id}.npy", np.stack([r["crops224"][i] for i in sel]))
            atomic_update_npz(existing, {
                "video_face": face, "video_valid": valid,
                "video_flow": r["flow"], "video_crops64": r["crops64"]})

            if n_face / t < 0.5:
                n_low += 1
                low_face[clip_id] = (t, n_face)
                write_low_face(low_face_path, low_face)
                tqdm.write(f"[low face] {clip_id}: {n_face}/{t} frames have a face")
            elif clip_id in low_face:  # was low before, fine now
                del low_face[clip_id]
                write_low_face(low_face_path, low_face)
        except Exception as e:  # keep going; report at the end
            n_failed += 1
            tqdm.write(f"[FAILED] {clip_id}: {type(e).__name__}: {e}")
            continue
        n_done += 1
        elapsed.append(time.time() - t0)
        if args.verbose:
            c224 = "none (no face)" if not n_face else "(16, 224, 224, 3) uint8"
            tqdm.write(f"{clip_id}: video_face {face.shape} {face.dtype} | video_valid {valid.shape} "
                       f"{valid.dtype} ({n_face}/{t} with face) | video_flow {r['flow'].shape} "
                       f"{r['flow'].dtype} | video_crops64 {r['crops64'].shape} {r['crops64'].dtype} | "
                       f"crops224 {c224} | {elapsed[-1]:.2f}s")

    print(f"done={n_done} skipped={n_skipped} low_face={n_low} failed={n_failed}")
    if elapsed:
        print(f"seconds per clip: mean {np.mean(elapsed):.2f}, median {np.median(elapsed):.2f}, "
              f"max {np.max(elapsed):.2f} (first clip includes warm-up)")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    if args.examples_dir:
        save_examples(df, feat_dir, Path(args.examples_dir), args.dataset, args.n_examples)
    if detector is not None:
        detector.close()  # avoids a noisy __del__ traceback at interpreter shutdown
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
