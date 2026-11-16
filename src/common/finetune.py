#!/usr/bin/env python3
"""LoRA fine-tuning of wav2vec2-base (audio) + a face-expression ViT (video) for either task.

Examples (run from the repository root):
    # stress on MAS, all five fusion modes, 5 folds, seed 42 (needs a CUDA GPU)
    python src/common/finetune.py --task stress --datasets mas   # -> results/stress/finetune.csv
    # deception on BoL + RLT, two modes, three seeds
    python src/common/finetune.py --task deception --datasets bol rlt --modes concat xattn --seeds 42 43 44
    # quick check (works on CPU): 1 epoch on 6 clips, first fold / first seed, writes *_smoke files
    python src/common/finetune.py --task stress --datasets mas --smoke

Inputs (docs/feature_contract.md, "extra files for GPU fine-tuning")
    features/<dataset>/wav/<clip_id>.wav          16 kHz mono waveform
    features/<dataset>/crops224/<clip_id>.npy     (16, 224, 224, 3) uint8 face crops
A clip without a crops224 file (no face found, no video) is kept: its 16 frames are zero images and
the whole video mask row is True, exactly the all-masked-video case of models.py (the video branch
contributes a zero vector, cross-attention to it is off, late fusion averages the other modality).
Frames of an existing crops file that are all zero are masked individually. A clip without a wav
file is treated the same way on the audio side (all-masked audio, 1 s of zeros as waveform).

Model
    * wav2vec2-base (facebook/wav2vec2-base) on the raw waveform and ViT (trpakov/vit-face-expression)
      on every crop; both frozen, LoRA (r=8, alpha=16, dropout 0.1) on q_proj and v_proj, gradient
      checkpointing on. Audio hidden states (768) and the per-frame ViT CLS embedding (768) are
      projected to 128 (LayerNorm -> Linear -> GELU -> Dropout) and fused with the real fusion
      modules of models.py (ConcatFusion, LateFusion, CrossAttentionFusion; "audio" / "video" use
      AttentionPool only), then one linear head for the task. The encoder a mode does not use
      ("video" has no wav2vec2, "audio" has no ViT) is not loaded.
    * Audio is a random 10 s crop per epoch in training and the first 10 s at evaluation (shorter
      clips are zero-padded to the batch; padded frames are masked).

Training (same protocol as train.py: fold column, 15 % group validation split, metrics, outputs)
    batch 2 x accumulation 8 = 16, AdamW (lr 1e-4 LoRA, 1e-3 everything else), BCE with pos_weight,
    fp16 autocast + GradScaler on CUDA, <= 30 epochs, early stopping on validation macro-F1
    (patience 5). Finished (setup, mode, fold, seed) runs found in the output CSV are skipped.
    After the first epoch the peak GPU memory is printed and the job stops above --max-gpu-gb (5.5).
    Full runs need a CUDA GPU; --smoke also runs on the CPU.

Outputs: like train.py (OUT, OUT_summary.csv, OUT_predictions.csv, <ckpt-dir>/<task>/<OUT stem>/*.pt). The
checkpoints hold only the trainable weights (LoRA + projections + fusion + head); the frozen
encoders come from the Hugging Face cache. The default OUT is results/<task>/finetune.csv (smoke runs: finetune_smoke*.csv beside it) and the
setup string ends in ";ft=lora8", so these rows can never be mixed up with train.py rows.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:  # imported as part of a package ...
    from . import train as T
    from .models import AttentionPool, ConcatFusion, CrossAttentionFusion, LateFusion
    from .data import load_metadata
except ImportError:  # ... or run as a script from src/common
    import train as T
    from models import AttentionPool, ConcatFusion, CrossAttentionFusion, LateFusion
    from data import load_metadata

SR = 16000
CROP_S = 10
N_FRAMES, IMG = 16, 224
D = 128
LORA_TARGETS = ["q_proj", "v_proj"]  # attention query / value projections (wav2vec2 and ViT)
W2V_NAME, VIT_NAME = "facebook/wav2vec2-base", "trpakov/vit-face-expression"
VIT_MEAN, VIT_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # preprocessor_config.json of the ViT
MIN_SAMPLES = 1600  # below 0.1 s of audio a clip counts as having no audio


# ----------------------------------------------------------------------------- dataset
def wav_path(root, dataset, clip_id):
    return Path(root) / dataset / "wav" / f"{clip_id}.wav"


def crops_path(root, dataset, clip_id):
    return Path(root) / dataset / "crops224" / f"{clip_id}.npy"


def has_both_files(root, df):
    """Boolean array: rows that have a wav file and a crops224 file."""
    return np.array([wav_path(root, d, c).is_file() and crops_path(root, d, c).is_file()
                     for d, c in zip(df["dataset"], df["clip_id"])], dtype=bool)


class RawClipDataset(Dataset):
    """Waveform crop + face crops of one clip.

    Item: dict with wave (S,) float32 (zero mean / unit variance over the real samples), wave_len
    (0 = no audio), crops (16,224,224,3) uint8 (zeros when missing), frame_valid (16,) bool,
    label, clip_id, dataset. ``train=True`` takes a random 10 s window that depends on
    (seed, epoch, row), so it differs per epoch but is reproducible; ``train=False`` takes the
    first 10 s.
    """

    def __init__(self, rows, root, train=False, seed=42):
        self.rows = rows.reset_index(drop=True)
        self.root, self.train, self.seed, self.epoch = root, train, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def _wave(self, ds, clip_id, idx):
        path = wav_path(self.root, ds, clip_id)
        if not path.is_file():
            return np.zeros(0, np.float32)
        info = sf.info(str(path))
        if info.samplerate != SR:
            raise ValueError(f"{path}: sample rate {info.samplerate}, expected {SR}")
        n, want = info.frames, CROP_S * SR
        start = 0
        if self.train and n > want:
            start = int(np.random.default_rng([self.seed, self.epoch, idx]).integers(0, n - want + 1))
        x, _ = sf.read(str(path), start=start, frames=want, dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if len(x) < MIN_SAMPLES:
            return np.zeros(0, np.float32)
        return ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)

    def _crops(self, ds, clip_id):
        path = crops_path(self.root, ds, clip_id)
        if not path.is_file():
            return np.zeros((N_FRAMES, IMG, IMG, 3), np.uint8), np.zeros(N_FRAMES, bool)
        a = np.load(path)
        if a.shape != (N_FRAMES, IMG, IMG, 3) or a.dtype != np.uint8:
            raise ValueError(f"{path}: expected ({N_FRAMES},{IMG},{IMG},3) uint8, got {a.shape} {a.dtype}")
        return a, a.reshape(N_FRAMES, -1).any(axis=1)  # an all-zero frame is "no face" for that frame

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        wave = self._wave(r.dataset, r.clip_id, idx)
        crops, valid = self._crops(r.dataset, r.clip_id)
        return {"wave": wave, "crops": crops, "frame_valid": valid, "label": float(r.label),
                "clip_id": r.clip_id, "dataset": r.dataset}


def collate_raw(items):
    """Pad waves with zeros to the longest real wave (at least 1 s); wave_len 0 = no audio."""
    lens = [len(it["wave"]) for it in items]
    wave = torch.zeros(len(items), max(max(lens), SR))
    for i, it in enumerate(items):
        wave[i, :lens[i]] = torch.from_numpy(it["wave"])
    return {"wave": wave, "wave_len": torch.tensor(lens, dtype=torch.long),
            "crops": torch.from_numpy(np.stack([it["crops"] for it in items])),
            "frame_valid": torch.from_numpy(np.stack([it["frame_valid"] for it in items])),
            "label": torch.tensor([it["label"] for it in items], dtype=torch.float32),
            "clip_id": [it["clip_id"] for it in items], "dataset": [it["dataset"] for it in items]}


def build_raw_loader(rows, root, batch_size, train, seed, workers=0, device=None):
    ds = RawClipDataset(rows, root, train=train, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=gen if train else None,
                      collate_fn=collate_raw, num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=device is not None and device.type == "cuda")


# ------------------------------------------------------------------------------ model
def _lora(module, r, alpha, dropout):
    """Freeze ``module``, enable gradient checkpointing, add LoRA on q_proj / v_proj."""
    from peft import LoraConfig, get_peft_model
    for p in module.parameters():
        p.requires_grad = False
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none")
    return get_peft_model(module, cfg)


def _projection(in_dim, dropout):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, D), nn.GELU(), nn.Dropout(dropout))


class FinetuneModel(nn.Module):
    """LoRA wav2vec2 + LoRA ViT -> 128-d sequences -> fusion of models.py -> head of the task.

    mode: audio | video | concat | late | xattn (as in train.py / models.py).
    forward(wave, wave_len, crops, frame_valid) -> logits (B,).
    """

    def __init__(self, mode, task, dropout=0.3, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 w2v_name=W2V_NAME, vit_name=VIT_NAME, local_only=True):
        super().__init__()
        from transformers import ViTModel, Wav2Vec2Model
        if mode not in T.MODES:
            raise ValueError(f"mode must be one of {T.MODES}, got {mode!r}")
        self.mode, self.task = mode, task
        self.w2v = self.vit = self.proj_a = self.proj_v = None
        if mode != "video":
            w2v = Wav2Vec2Model.from_pretrained(w2v_name, local_files_only=local_only)
            self._kernels, self._strides = list(w2v.config.conv_kernel), list(w2v.config.conv_stride)
            self.w2v = _lora(w2v, lora_r, lora_alpha, lora_dropout)
            self.proj_a = _projection(w2v.config.hidden_size, dropout)
        if mode != "audio":
            vit = ViTModel.from_pretrained(vit_name, local_files_only=local_only, add_pooling_layer=False)
            self.vit = _lora(vit, lora_r, lora_alpha, lora_dropout)
            self.proj_v = _projection(vit.config.hidden_size, dropout)
        self.register_buffer("img_mean", torch.tensor(VIT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(VIT_STD).view(1, 3, 1, 1), persistent=False)
        if mode == "concat":
            self.fusion = ConcatFusion(D, D, D)
        elif mode == "late":
            self.fusion = LateFusion(D, D, n_out=1, dropout=dropout)
        elif mode == "xattn":
            self.fusion = CrossAttentionFusion(D, D, D, dropout=dropout)
        else:
            self.fusion = None
            self.pool = AttentionPool(D)
        if mode != "late":  # late fusion's per-modality classifiers are its heads
            self.drop = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({task: nn.Linear(D, 1)})

    # -- helpers
    def _out_frames(self, n):
        """wav2vec2 output length for n input samples (conv stack of the feature encoder)."""
        for k, s in zip(self._kernels, self._strides):
            n = torch.div(n - k, s, rounding_mode="floor") + 1
        return n.clamp_min(0)

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, state):
        missing = self.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- forward
    def forward(self, wave, wave_len, crops, frame_valid):
        a_seq = a_mask = v_seq = v_mask = None
        if self.w2v is not None:
            h = self.w2v(input_values=wave).last_hidden_state  # (B, T, 768); no attention mask (group norm)
            n_fr = torch.where(wave_len >= MIN_SAMPLES, self._out_frames(wave_len), torch.zeros_like(wave_len))
            a_mask = torch.arange(h.shape[1], device=h.device)[None] >= n_fr[:, None]
            a_seq = self.proj_a(h).masked_fill(a_mask[..., None], 0.0)
        if self.vit is not None:
            b, f = crops.shape[:2]
            x = crops.reshape(b * f, IMG, IMG, 3).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self.img_mean) / self.img_std
            cls = self.vit(pixel_values=x).last_hidden_state[:, 0].reshape(b, f, -1)
            v_mask = ~frame_valid
            v_seq = self.proj_v(cls).masked_fill(v_mask[..., None], 0.0)
        if self.mode == "late":
            return self.fusion(a_seq, a_mask, v_seq, v_mask)[0].reshape(-1)
        if self.mode == "audio":
            emb = self.pool(a_seq, a_mask)
        elif self.mode == "video":
            emb = self.pool(v_seq, v_mask)
        else:
            emb = self.fusion(a_seq, a_mask, v_seq, v_mask)
        return self.heads[self.task](self.drop(emb)).reshape(-1)


def forward_finetune(model, batch, device):
    """forward callable for train.predict: logits (B,)."""
    return model(batch["wave"].to(device, non_blocking=True), batch["wave_len"].to(device),
                 batch["crops"].to(device, non_blocking=True), batch["frame_valid"].to(device)).reshape(-1)


class _TrainableView:
    """Lets train.EarlyStopper keep only the trainable weights instead of the whole 180M-parameter state."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        return self.model.trainable_state_dict()


# ---------------------------------------------------------------------------- training
def train_finetune(model, train_loader, val_loader, device, *, pos_weight, lr_lora, lr_head, weight_decay,
                   max_epochs, patience, accum_steps, max_gpu_gb):
    """Two-learning-rate AdamW loop (train.train_model has a single lr); helpers come from train.py."""
    model.to(device)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora, "lr": lr_lora}, {"params": rest, "lr": lr_head}],
                            weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper, history = T.EarlyStopper(patience), []
    view = _TrainableView(model)
    print(f"    trainable parameters: {model.num_trainable():,} (LoRA {sum(p.numel() for p in lora):,})")
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        train_loader.dataset.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        total, n_seen, n_batches = 0.0, 0, len(train_loader)
        for i, batch in enumerate(train_loader):
            with T._autocast(device):
                logits = forward_finetune(model, batch, device)
            y = batch["label"].to(device)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or i + 1 == n_batches:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * len(y)
            n_seen += len(y)
        val = T.predict(model, val_loader, device, forward_finetune)
        val_f1 = T.compute_metrics(val["label"], val["prob"])["macro_f1"]
        improved = stopper.update(val_f1, epoch, view)
        history.append({"epoch": epoch, "train_loss": total / max(n_seen, 1), "val_macro_f1": val_f1})
        print(f"    epoch {epoch:2d}  train_loss {history[-1]['train_loss']:.4f}  val_macro_f1 {val_f1:.4f}"
              f"{'  *' if improved else ''}  ({time.perf_counter() - t0:.0f} s)")
        if device.type == "cuda" and epoch == 1:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"    peak GPU memory after first epoch: {peak:.2f} GB (limit {max_gpu_gb:g} GB)")
            if peak > max_gpu_gb:
                raise SystemExit(f"error: peak GPU memory {peak:.2f} GB exceeds {max_gpu_gb:g} GB after the first "
                                 f"epoch. Use --batch-size 1 with --accum-steps 16 (same effective batch) or "
                                 f"close other programs using the GPU.")
        if stopper.should_stop:
            break
    model.load_trainable_state_dict(stopper.best_state)
    return {"best_epoch": stopper.best_epoch, "best_val_macro_f1": stopper.best_score,
            "epochs_run": len(history), "history": history}


# ------------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LoRA fine-tuning of wav2vec2 + face ViT.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["stress", "deception"])
    ap.add_argument("--datasets", nargs="+", choices=["mas", "bol", "rlt"], default=None,
                    help="datasets, pooled (default: mas for stress, bol rlt for deception)")
    ap.add_argument("--modes", nargs="+", choices=T.MODES, default=list(T.MODES))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", default=None, help="results CSV (default results/<task>/finetune.csv); summary and predictions go beside it")
    ap.add_argument("--features-root", default="features")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3, help="lr of projections, fusion and head")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-steps", type=int, default=8, help="batch 2 x 8 = effective 16")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-gpu-gb", type=float, default=5.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--w2v-model", default=W2V_NAME)
    ap.add_argument("--vit-model", default=VIT_NAME)
    ap.add_argument("--allow-download", action="store_true",
                    help="let Hugging Face download the encoders if they are not cached (default: cache only)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 6 clips with wav and crops224, first fold and seed only; *_smoke files")
    return ap.parse_args(argv)


def smoke_subsample(root, train, val, test, seed):
    """3 / 1 / 2 clips that have both a wav and a crops224 file (splits stay disjoint)."""
    out = []
    for df, n in ((train, 3), (val, 1), (test, 2)):
        ok = df[has_both_files(root, df)]
        if len(ok) < n:
            raise SystemExit(f"error: only {len(ok)} clips with wav and crops224 in a split, need {n}")
        out.append(ok.sample(n=n, random_state=seed).reset_index(drop=True))
    return tuple(out)


def main(argv=None):
    args = parse_args(argv)
    names = args.datasets or T.DEFAULT_DATASETS[args.task]
    folds, seeds = args.folds, args.seeds
    max_epochs = 1 if args.smoke else args.max_epochs
    if args.smoke:
        folds, seeds = folds[:1], seeds[:1]
    if not args.smoke and not torch.cuda.is_available():
        raise SystemExit("error: no CUDA GPU found. Full fine-tuning runs need one (RTX 4050, 6 GB); "
                         "use --smoke to test on the CPU.")

    out_csv, summary_csv, pred_csv, stem = T.output_paths(args.out or f"results/{args.task}/finetune.csv", args.smoke)
    ckpt_dir = Path(args.ckpt_dir) / args.task / stem
    if args.smoke:
        for p in (out_csv, summary_csv, pred_csv):
            p.unlink(missing_ok=True)

    device = T.choose_device()
    df = load_metadata(args.features_root, names, task=args.task, missing="ignore")  # .npz files are not used here
    n_wav = sum(wav_path(args.features_root, d, c).is_file() for d, c in zip(df.dataset, df.clip_id))
    n_crop = sum(crops_path(args.features_root, d, c).is_file() for d, c in zip(df.dataset, df.clip_id))
    print(f"Loaded {len(df)} {args.task} clips from {names}: {n_wav} with wav, {n_crop} with crops224 "
          f"({len(df) - n_crop} clips run with a masked video branch, {len(df) - n_wav} with masked audio)")
    setup = T.setup_id(args.task, names, names, args.dropout, args.weight_decay) + ";ft=lora8"
    finished = set() if args.smoke else T.load_finished_runs(out_csv)
    T.drop_unfinished_predictions(pred_csv, finished)
    print(f"Setup: {setup}\nResults: {out_csv}  ({len(finished)} finished runs found)")

    for fold in folds:
        for seed in seeds:
            todo = [m for m in args.modes if T.run_key(setup, m, fold, seed) not in finished]
            for m in args.modes:
                if m not in todo:
                    print(f"[{m} fold {fold} seed {seed}] already finished, skipping")
            if not todo:
                continue
            if not (df["fold"] == fold).any():
                print(f"warning: no clips in fold {fold}, skipping")
                continue
            train_df, val_df, test_df = T.make_fold_splits(df, fold, args.val_frac, seed)
            if args.smoke:
                train_df, val_df, test_df = smoke_subsample(args.features_root, train_df, val_df, test_df, seed)
            print(f"\n== fold {fold}, seed {seed}: train {len(train_df)} clips / {train_df['group_key'].nunique()} groups, "
                  f"val {len(val_df)} / {val_df['group_key'].nunique()}, test {len(test_df)} / {test_df['group_key'].nunique()}")
            assert not (set(train_df["group_key"]) & set(val_df["group_key"])), "train/val share a group"
            assert not (set(train_df["group_key"]) & set(test_df["group_key"])), "train/test share a group"
            assert not (set(val_df["group_key"]) & set(test_df["group_key"])), "val/test share a group"
            n_pos = int(train_df["label"].sum())
            pos_weight = (len(train_df) - n_pos) / n_pos if n_pos else 1.0
            val_loader = build_raw_loader(val_df, args.features_root, args.eval_batch_size, False, seed, args.workers, device)
            test_loader = build_raw_loader(test_df, args.features_root, args.eval_batch_size, False, seed, args.workers, device)

            for mode in todo:
                print(f"[{mode} fold {fold} seed {seed}] pos_weight {pos_weight:.3f}")
                t0 = time.perf_counter()
                T.seed_everything(seed)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                train_loader = build_raw_loader(train_df, args.features_root, args.batch_size, True, seed,
                                                args.workers, device)
                model = FinetuneModel(mode, args.task, args.dropout, w2v_name=args.w2v_model,
                                      vit_name=args.vit_model, local_only=not args.allow_download)
                info = train_finetune(model, train_loader, val_loader, device, pos_weight=pos_weight,
                                      lr_lora=args.lr_lora, lr_head=args.lr_head, weight_decay=args.weight_decay,
                                      max_epochs=max_epochs, patience=args.patience, accum_steps=args.accum_steps,
                                      max_gpu_gb=args.max_gpu_gb)
                preds = T.predict(model, test_loader, device, forward_finetune)
                metrics = T.compute_metrics(preds["label"], preds["prob"])

                ckpt_path = ckpt_dir / f"{mode}_fold{fold}_seed{seed}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"trainable": model.trainable_state_dict(), "setup": setup, "mode": mode, "fold": fold,
                            "seed": seed, "best_epoch": info["best_epoch"],
                            "best_val_macro_f1": info["best_val_macro_f1"], "args": vars(args)}, ckpt_path)

                elapsed = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                T.append_rows(pred_csv, preds.assign(mode=mode, fold=fold, seed=seed, setup=setup))
                T.append_rows(out_csv, [{
                    "setup": setup, "task": args.task, "train_datasets": "+".join(names),
                    "test_datasets": "+".join(names), "mode": mode, "fold": fold, "seed": seed,
                    "dropout": args.dropout, "weight_decay": args.weight_decay,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                    "val_macro_f1": info["best_val_macro_f1"], **metrics,
                    "time_s": round(elapsed, 1), "peak_gpu_mb": round(peak_mb, 1), "device": device.type}])
                finished.add(T.run_key(setup, mode, fold, seed))
                print(f"[{mode} fold {fold} seed {seed}] test acc {metrics['accuracy']:.3f}  "
                      f"macro-F1 {metrics['macro_f1']:.3f}  AUC {metrics['roc_auc']:.3f}  | best epoch "
                      f"{info['best_epoch']}/{info['epochs_run']}  | {elapsed:.1f} s  | "
                      + (f"peak GPU memory {peak_mb:.0f} MB" if device.type == "cuda" else "peak GPU memory n/a (cpu)"))
                del model, train_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if out_csv.is_file():
        summary = T.summarize(pd.read_csv(out_csv))
        summary.to_csv(summary_csv, index=False)
        T.print_summary(summary[summary["setup"] == setup])
        print(f"\nWrote {out_csv}, {summary_csv}, {pred_csv}; checkpoints in {ckpt_dir}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    main()
