#!/usr/bin/env python3
"""Extract the audio features defined in docs/feature_contract.md.

Usage:
    python src/common/extract_audio.py mas --limit 5 --verbose
    python src/common/extract_audio.py mas
    python src/common/extract_audio.py mas --data_root data --features_root features

Reads <features_root>/<dataset>/metadata.csv. For every row the first 60 s of audio of
<data_root>/<video_path> are decoded with ffmpeg (16 kHz, mono) and these keys are written
(added or replaced) in <features_root>/<dataset>/<clip_id>.npz:

    audio_w2v      (T, 768)  float16  wav2vec2-base last hidden state, mean over 0.2 s (10 frames)
    audio_prosody  (T, 3)    float32  pitch Hz (0 if unvoiced), RMS energy, voiced flag
    audio_logmel   (T_m, 64) float16  log(mel + 1e-6), 100 frames per second

and the audio itself is written to <features_root>/<dataset>/wav/<clip_id>.wav.

Details
-------
* Waveform for wav2vec2 is normalised to zero mean / unit variance (as the Hugging Face
  feature extractor does). Prosody and log-mel use the un-normalised waveform in [-1, 1].
* wav2vec2 gives 50 frames/s; non-overlapping groups of 10 frames are averaged and an
  incomplete last group is dropped. audio_prosody is trimmed / zero-padded to the same T.
* Prosody: Praat pitch (praat-parselmouth), 10 ms steps, floor 75 Hz, ceiling 500 Hz.
  Per 0.2 s window: pitch = mean pitch of voiced frames (0 if none); RMS = sqrt(mean(x^2))
  over the window's 3200 samples; voiced flag = 1 if more than half of the window's 20
  nominal 10 ms frames are voiced (frames Praat does not produce count as unvoiced).
  A Praat frame belongs to the window that contains its centre time.
* Clips are skipped when the .npz already has all three keys and the wav exists.
* Clips without an audio track (or with less than one 0.2 s window of audio) are listed in
  <features_root>/<dataset>/no_audio.txt and skipped.
* File names are always f"{clip_id}.npz" etc.: clip ids contain dots, so never use
  Path.with_suffix / Path.stem on them.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

SR = 16000
MAX_SECONDS = 60
W2V_GROUP = 10                      # 10 wav2vec2 frames (20 ms each) = 0.2 s
WINDOW_S = 0.2
WINDOW_SAMPLES = int(SR * WINDOW_S)  # 3200
PITCH_STEP = 0.01
FRAMES_PER_WINDOW = int(round(WINDOW_S / PITCH_STEP))  # 20
PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
N_MELS = 64
N_FFT = 400
HOP = 160
KEYS = ("audio_w2v", "audio_prosody", "audio_logmel")


# --------------------------------------------------------------------------- setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoAudio(Exception):
    """The clip has no audio track, or too little audio to form one 0.2 s window."""


# --------------------------------------------------------------------------- audio io
def has_audio_stream(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
           "stream=index", "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    return bool(out.stdout.strip())


def load_audio(path):
    """Decode the first MAX_SECONDS of audio as 16 kHz mono float32 in [-1, 1]."""
    if not has_audio_stream(path):
        raise NoAudio("no audio stream")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
           "-vn", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SECONDS),
           "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {out.stderr.decode(errors='replace').strip()[:300]}")
    wave = np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
    wave = np.nan_to_num(wave)
    return wave


def atomic_write_wav(path, wave):
    tmp = Path(f"{path}.tmp")
    try:
        sf.write(str(tmp), wave, SR, subtype="PCM_16", format="WAV")
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


# --------------------------------------------------------------------------- features
def w2v_features(model, wave, device):
    x = torch.from_numpy(wave)
    x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)  # HF feature extractor
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            h = model(x).last_hidden_state[0]
    h = h.float().cpu()
    t = h.shape[0] // W2V_GROUP
    if t == 0:
        raise NoAudio("shorter than one 0.2 s window")
    h = h[: t * W2V_GROUP].reshape(t, W2V_GROUP, -1).mean(dim=1)
    return h.numpy().astype(np.float16)


def prosody_features(wave, t):
    """(t, 3) float32: pitch Hz (0 if unvoiced), RMS energy, voiced flag per 0.2 s window."""
    import parselmouth

    out = np.zeros((t, 3), dtype=np.float32)

    # RMS over the window's samples (zero-pad the tail so the last window is complete)
    n = t * WINDOW_SAMPLES
    x = wave[:n].astype(np.float64)
    if x.shape[0] < n:
        x = np.pad(x, (0, n - x.shape[0]))
    out[:, 1] = np.sqrt((x.reshape(t, WINDOW_SAMPLES) ** 2).mean(axis=1))

    # Pitch
    snd = parselmouth.Sound(wave.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    times = np.asarray(pitch.xs())
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    voiced = np.isfinite(f0) & (f0 > 0)
    win = np.floor(times / WINDOW_S + 1e-9).astype(int)
    keep = (win >= 0) & (win < t)
    win, f0, voiced = win[keep], f0[keep], voiced[keep]
    n_voiced = np.bincount(win, weights=voiced.astype(np.float64), minlength=t)
    f0_sum = np.bincount(win, weights=np.where(voiced, f0, 0.0), minlength=t)
    out[:, 0] = np.where(n_voiced > 0, f0_sum / np.maximum(n_voiced, 1), 0.0)
    out[:, 2] = (n_voiced > FRAMES_PER_WINDOW / 2).astype(np.float32)
    return out


def logmel_features(mel_tf, wave, device):
    x = torch.from_numpy(wave).unsqueeze(0).to(device)
    with torch.no_grad():
        mel = mel_tf(x)[0]  # (64, frames), float32 (outside autocast)
        feat = torch.log(mel + 1e-6).T.contiguous()
    return feat.cpu().numpy().astype(np.float16)


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name, e.g. mas (reads <features_root>/<dataset>/metadata.csv)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N metadata rows")
    ap.add_argument("--features_root", default="features", help="folder that holds <dataset>/ (default: features)")
    ap.add_argument("--data_root", default="data", help="folder that video_path is relative to (default: data)")
    ap.add_argument("--model", default="facebook/wav2vec2-base", help="wav2vec2 checkpoint (default: facebook/wav2vec2-base)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="print the shapes of all three keys for every clip")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dir = Path(args.features_root) / args.dataset
    meta_path = feat_dir / "metadata.csv"
    if not meta_path.exists():
        sys.exit(f"metadata not found: {meta_path}")
    df = pd.read_csv(meta_path, dtype={"clip_id": str})
    for col in ("clip_id", "video_path"):
        if col not in df.columns:
            sys.exit(f"{meta_path} has no column '{col}'")
    if args.limit is not None:
        df = df.head(args.limit)

    wav_dir = feat_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    no_audio_path = feat_dir / "no_audio.txt"
    no_audio = set(no_audio_path.read_text().split()) if no_audio_path.exists() else set()

    # Skip check first so a fully done run does not load the model.
    todo = []
    n_skipped = 0
    for row in df.itertuples(index=False):
        npz_path = feat_dir / f"{row.clip_id}.npz"
        wav_path = wav_dir / f"{row.clip_id}.wav"
        keys = npz_keys(npz_path) if npz_path.exists() else set()
        if keys is not None and set(KEYS) <= keys and wav_path.exists():
            n_skipped += 1
        else:
            todo.append(row)
    print(f"device={device} dataset={args.dataset} rows={len(df)} to_process={len(todo)} already_done={n_skipped}")

    model = mel_tf = None
    if todo:
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(args.model).to(device).eval()
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS).to(device)

    n_done = n_failed = n_noaudio = 0
    elapsed = []
    for row in tqdm(todo, desc=f"{args.dataset} audio", unit="clip"):
        clip_id = row.clip_id
        video = Path(args.data_root) / row.video_path
        t0 = time.time()
        try:
            if not video.exists():
                raise FileNotFoundError(f"video not found: {video}")
            existing = feat_dir / f"{clip_id}.npz"
            if existing.exists() and npz_keys(existing) is None:
                raise RuntimeError(f"existing {existing.name} is unreadable; not overwriting it")
            wave = load_audio(video)
            if wave.shape[0] < WINDOW_SAMPLES:
                raise NoAudio("shorter than one 0.2 s window")
            w2v = w2v_features(model, wave, device)
            t = w2v.shape[0]
            prosody = prosody_features(wave, t)
            logmel = logmel_features(mel_tf, wave, device)
            atomic_write_wav(wav_dir / f"{clip_id}.wav", wave)
            atomic_update_npz(feat_dir / f"{clip_id}.npz", {
                "audio_w2v": w2v, "audio_prosody": prosody, "audio_logmel": logmel})
        except NoAudio as e:
            n_noaudio += 1
            if clip_id not in no_audio:
                no_audio.add(clip_id)
                with open(no_audio_path, "a") as f:
                    f.write(f"{clip_id}\n")
            tqdm.write(f"[no audio] {clip_id}: {e}")
            continue
        except Exception as e:  # keep going; report at the end
            n_failed += 1
            tqdm.write(f"[FAILED] {clip_id}: {type(e).__name__}: {e}")
            continue
        n_done += 1
        elapsed.append(time.time() - t0)
        if args.verbose:
            tqdm.write(f"{clip_id}: audio_w2v {w2v.shape} {w2v.dtype} | audio_prosody {prosody.shape} "
                       f"{prosody.dtype} | audio_logmel {logmel.shape} {logmel.dtype} | "
                       f"{len(wave) / SR:.2f}s audio, {elapsed[-1]:.2f}s")

    print(f"done={n_done} skipped={n_skipped} no_audio={n_noaudio} failed={n_failed}")
    if elapsed:
        print(f"seconds per clip: mean {np.mean(elapsed):.2f}, median {np.median(elapsed):.2f}, "
              f"max {np.max(elapsed):.2f} (first clip includes warm-up)")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
