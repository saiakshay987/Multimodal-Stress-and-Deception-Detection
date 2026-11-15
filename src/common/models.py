"""Shared models for the multimodal stress / deception experiments.

Contents
--------
* ``AttentionPool``        masked attention pooling of a (B, T, D) sequence to (B, D).
* ``ModalityEncoder``      LayerNorm -> Linear -> GELU -> dropout -> one Transformer layer.
* ``ConcatFusion``, ``LateFusion``, ``CrossAttentionFusion``
                           fusion of an audio and a video sequence. They only need the two
                           sequences and their masks, so they work with any feature width
                           (finetune.py can reuse them with the widths of its own encoders).
* ``MultimodalModel``      encoders + one fusion + one linear head per task.

Mask convention (same as data.py / ``key_padding_mask`` in torch.nn): masks are bool tensors that
are **True at steps to ignore** (padding, and video frames without a face).

Clips without any usable step (for example a clip in which no face was found: its video mask row
is all True) never produce NaN. Such a row is encoded as a zero vector, cross-attention to it is
switched off, and late fusion averages only the modalities that are present.

Interface expected by train.py: ``MultimodalModel(mode=..., dropout=..., audio_dim=..., video_dim=...)``
called as ``model(audio, video, audio_mask, video_mask)``; with a single head the result is a plain
(B,) logit tensor. ``mode`` is an alias of ``fusion``.
"""
from __future__ import annotations

import torch
from torch import nn

FUSIONS = ("audio", "video", "concat", "late", "xattn")
HEADS = ("stress", "deception")
_NEG = -1e9


def _safe_mask(mask):
    """Copy of ``mask`` (B, T) in which rows that are entirely True keep step 0 visible.

    Attention over a fully masked row gives NaN; callers use this mask for the attention call and
    zero the result of the empty rows afterwards. Also returns the (B,) bool vector of empty rows.
    """
    empty = mask.all(dim=1)
    safe = mask.clone()
    safe[:, 0] &= ~empty
    return safe, empty


class AttentionPool(nn.Module):
    """Masked attention pooling: softmax over the valid steps of a learned score, weighted sum.

    A row with no valid step is pooled to the zero vector (no NaN).
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """x (B, T, D), mask (B, T) True = ignore -> (B, D)."""
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        s = self.score(x).squeeze(-1).float().masked_fill(mask, _NEG)
        w = torch.softmax(s, dim=1) * (~mask)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return torch.einsum("bt,btd->bd", w.to(x.dtype), x)


class ModalityEncoder(nn.Module):
    """LayerNorm -> Linear(in_dim, d) -> GELU -> dropout -> TransformerEncoderLayer (4 heads).

    ``forward(x, mask)`` with x (B, T, in_dim) and mask (B, T) True = ignore returns
    ``(seq, pooled)``: seq (B, T, d) with padded steps set to 0, and pooled (B, d) from masked
    attention pooling. A fully masked row gives zeros in both.
    """

    def __init__(self, in_dim, d=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        self.layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=dropout,
                                                batch_first=True)
        self.pool = AttentionPool(d)
        self.d = d

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        safe, _ = _safe_mask(mask)
        h = self.layer(self.proj(x), src_key_padding_mask=safe)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)  # padded steps and empty rows -> 0
        return h, self.pool(h, mask)


class ConcatFusion(nn.Module):
    """Pool each modality, concatenate, Linear to ``d``.

    forward(audio_seq, audio_mask, video_seq, video_mask) -> (B, d). ``video_dim`` defaults to
    ``audio_dim``; a missing (fully masked) modality contributes a zero vector.
    """

    def __init__(self, audio_dim, video_dim=None, d=128):
        super().__init__()
        video_dim = audio_dim if video_dim is None else video_dim
        self.pool_a, self.pool_v = AttentionPool(audio_dim), AttentionPool(video_dim)
        self.out = nn.Linear(audio_dim + video_dim, d)

    def forward(self, audio_seq, audio_mask, video_seq, video_mask):
        return self.out(torch.cat([self.pool_a(audio_seq, audio_mask), self.pool_v(video_seq, video_mask)], dim=-1))


class LateFusion(nn.Module):
    """One classifier per modality (pool -> dropout -> Linear to ``n_out`` logits); logits averaged.

    Unlike the other fusions this returns logits, not an embedding:
    ``forward(...) -> (logits (B, n_out), {"audio": (B, n_out), "video": (B, n_out)})``.
    A modality whose mask row is entirely True is left out of the average for that row.
    """

    def __init__(self, audio_dim, video_dim=None, n_out=1, dropout=0.3):
        super().__init__()
        video_dim = audio_dim if video_dim is None else video_dim
        self.pool_a, self.pool_v = AttentionPool(audio_dim), AttentionPool(video_dim)
        self.clf_a = nn.Sequential(nn.Dropout(dropout), nn.Linear(audio_dim, n_out))
        self.clf_v = nn.Sequential(nn.Dropout(dropout), nn.Linear(video_dim, n_out))

    def forward(self, audio_seq, audio_mask, video_seq, video_mask):
        la = self.clf_a(self.pool_a(audio_seq, audio_mask))
        lv = self.clf_v(self.pool_v(video_seq, video_mask))
        wa = (~audio_mask.all(dim=1)).to(la.dtype).unsqueeze(-1)
        wv = (~video_mask.all(dim=1)).to(lv.dtype).unsqueeze(-1)
        fused = (la * wa + lv * wv) / (wa + wv).clamp_min(1.0)
        return fused, {"audio": la, "video": lv}


class CrossAttentionFusion(nn.Module):
    """Audio attends to video and video attends to audio (nn.MultiheadAttention).

    Each stream: ``LayerNorm(x + dropout(attention(x, other)))``; then both are pooled,
    concatenated and mapped by Linear to ``d``. forward(...) -> (B, d). If the other modality is
    absent for a clip (all-True mask row) the attention output is 0, so the stream is just
    ``LayerNorm(x)``. ``audio_dim`` and ``video_dim`` must be divisible by ``nhead``.
    """

    def __init__(self, audio_dim, video_dim=None, d=128, nhead=4, dropout=0.1):
        super().__init__()
        video_dim = audio_dim if video_dim is None else video_dim
        self.a2v = nn.MultiheadAttention(audio_dim, nhead, dropout=dropout, kdim=video_dim, vdim=video_dim,
                                         batch_first=True)
        self.v2a = nn.MultiheadAttention(video_dim, nhead, dropout=dropout, kdim=audio_dim, vdim=audio_dim,
                                         batch_first=True)
        self.norm_a, self.norm_v = nn.LayerNorm(audio_dim), nn.LayerNorm(video_dim)
        self.drop = nn.Dropout(dropout)
        self.pool_a, self.pool_v = AttentionPool(audio_dim), AttentionPool(video_dim)
        self.out = nn.Linear(audio_dim + video_dim, d)

    @staticmethod
    def _attend(mha, query, other, other_mask):
        safe, empty = _safe_mask(other_mask)
        out, _ = mha(query, other, other, key_padding_mask=safe, need_weights=False)
        return out.masked_fill(empty[:, None, None], 0.0)

    def forward(self, audio_seq, audio_mask, video_seq, video_mask):
        a = self.norm_a(audio_seq + self.drop(self._attend(self.a2v, audio_seq, video_seq, video_mask)))
        v = self.norm_v(video_seq + self.drop(self._attend(self.v2a, video_seq, audio_seq, audio_mask)))
        return self.out(torch.cat([self.pool_a(a, audio_mask), self.pool_v(v, video_mask)], dim=-1))


class MultimodalModel(nn.Module):
    """Encoders + fusion + one linear head per task.

    Args:
        audio_dim, video_dim: input widths (771 = audio_w2v + audio_prosody, 770 = video_face + video_flow).
        fusion: "audio" | "video" | "concat" | "late" | "xattn". ``mode`` is an alias (train.py uses it).
        heads: tuple of tasks, e.g. ("stress",), ("deception",) or ("stress", "deception").
        d: model width. dropout: used everywhere: input projection, Transformer layer, cross-attention
            (attention and residual), late classifiers and before the heads.
        return_dict: default for ``forward``; a multi-head model always returns the dict.

    ``forward(audio, video, audio_mask, video_mask, video_valid=None, return_dict=None)``:
        audio (B, T_a, audio_dim), video (B, T_v, video_dim), masks (B, T) bool, True = ignore.
        ``video_valid`` (B, T_v) optional; frames where it is False are masked as well (data.py
        already folds it into video_mask). A modality that the fusion does not use may be None.
    Returns:
        one head and return_dict False: logits (B,).
        otherwise a dict: one (B,) logit tensor per head, plus modality logits where they exist:
        "audio" (fusion audio, late), "video" (video, late) and "late" (the averaged logits, late).
        Modality logits have shape (B,) for one head and (B, n_heads) for several (column order =
        ``heads``).
    """

    def __init__(self, audio_dim=771, video_dim=770, fusion="concat", heads=("stress",), d=128, dropout=0.3,
                 return_dict=False, mode=None):
        super().__init__()
        if mode is not None:
            fusion = mode
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        heads = (heads,) if isinstance(heads, str) else tuple(heads)
        if not heads or len(set(heads)) != len(heads) or any(h not in HEADS for h in heads):
            raise ValueError(f"heads must be a non-empty tuple of distinct values from {HEADS}, got {heads}")
        self.fusion_name, self.heads_names, self.return_dict = fusion, heads, return_dict
        self.audio_enc = ModalityEncoder(audio_dim, d, dropout) if fusion != "video" else None
        self.video_enc = ModalityEncoder(video_dim, d, dropout) if fusion != "audio" else None
        if fusion == "concat":
            self.fusion = ConcatFusion(d, d, d)
        elif fusion == "late":
            self.fusion = LateFusion(d, d, n_out=len(heads), dropout=dropout)
        elif fusion == "xattn":
            self.fusion = CrossAttentionFusion(d, d, d, dropout=dropout)
        else:
            self.fusion = None
        if fusion != "late":
            self.drop = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({h: nn.Linear(d, 1) for h in heads})

    def num_parameters(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, audio=None, video=None, audio_mask=None, video_mask=None, video_valid=None, return_dict=None):
        f = self.fusion_name
        if video_valid is not None and video_mask is not None:
            video_mask = video_mask | ~video_valid
        a_seq = a_vec = v_seq = v_vec = None
        if self.audio_enc is not None:
            a_seq, a_vec = self.audio_enc(audio, audio_mask)
        if self.video_enc is not None:
            v_seq, v_vec = self.video_enc(video, video_mask)

        n = len(self.heads_names)
        squeeze = (lambda t: t.squeeze(-1)) if n == 1 else (lambda t: t)
        extra = {}
        if f == "late":
            logits, per = self.fusion(a_seq, audio_mask, v_seq, video_mask)
            extra = {"audio": squeeze(per["audio"]), "video": squeeze(per["video"]), "late": squeeze(logits)}
        else:
            emb = {"audio": a_vec, "video": v_vec}.get(f)
            if emb is None:
                emb = self.fusion(a_seq, audio_mask, v_seq, video_mask)
            emb = self.drop(emb)
            logits = torch.cat([self.heads[h](emb) for h in self.heads_names], dim=-1)
            if f in ("audio", "video"):
                extra = {f: squeeze(logits)}
        want_dict = self.return_dict if return_dict is None else return_dict
        if n == 1 and not want_dict:
            return logits.squeeze(-1)
        out = {h: logits[:, i] for i, h in enumerate(self.heads_names)}
        out.update(extra)
        return out


# ------------------------------------------------------------------------------------ test
def _random_batch(device, lengths_a=(40, 25, 7, 1), lengths_v=(38, 25, 0, 1), no_face=(2,), d_a=771, d_v=770):
    """Padded random batch built like data.collate_clips (mask True = ignore). Clips in ``no_face``
    have no valid video frame at all."""
    b, ta, tv = len(lengths_a), max(lengths_a), max(max(lengths_v), 1)
    audio, video = torch.randn(b, ta, d_a), torch.randn(b, tv, d_v)
    a_mask = torch.arange(ta)[None] >= torch.tensor(lengths_a)[:, None]
    valid = torch.arange(tv)[None] < torch.tensor(lengths_v)[:, None]
    for i in no_face:
        valid[i] = False
    video = video.masked_fill(~valid[..., None], 0.0)
    audio = audio.masked_fill(a_mask[..., None], 0.0)
    return audio.to(device), video.to(device), a_mask.to(device), (~valid).to(device)


def _test():
    import sys
    from pathlib import Path

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    audio, video, am, vm = _random_batch(device)
    b = audio.shape[0]
    head_sets = [("stress",), ("deception",), ("stress", "deception")]

    # 1) every fusion x head setting, with one clip that has no face at all
    for fusion in FUSIONS:
        for heads in head_sets:
            model = MultimodalModel(fusion=fusion, heads=heads).to(device).train()
            out = model(audio, video, am, vm)
            if len(heads) == 1:
                assert torch.is_tensor(out) and out.shape == (b,), (fusion, heads, out.shape)
                out = model(audio, video, am, vm, return_dict=True)
            assert isinstance(out, dict) and all(out[h].shape == (b,) for h in heads), (fusion, heads)
            expect = {"audio": {"audio"}, "video": {"video"}, "late": {"audio", "video", "late"}}.get(fusion, set())
            assert set(out) - set(heads) == expect, (fusion, set(out))
            for k, v in out.items():
                assert torch.isfinite(v).all(), f"non-finite {k} in {fusion} {heads}"
            sum(v.float().sum() for v in out.values()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), (fusion, heads)
            print(f"  {fusion:6s} heads={'+'.join(heads):17s} params={model.num_parameters():7,d}  keys={sorted(out)}")
            assert model.num_parameters() < 1_000_000

    # 2) all clips without a face / without audio, and the video_valid argument
    for fusion in FUSIONS:
        model = MultimodalModel(fusion=fusion, heads=HEADS).to(device).eval()
        a2, v2, am2, vm2 = _random_batch(device, no_face=(0, 1, 2, 3))
        out = model(a2, v2, am2, vm2)
        assert all(torch.isfinite(v).all() for v in out.values()), f"NaN with no faces in {fusion}"
        out = model(a2, v2, torch.ones_like(am2), vm2)
        assert all(torch.isfinite(v).all() for v in out.values()), f"NaN with no audio in {fusion}"
        o_a = model(audio, video, am, vm)
        o_b = model(audio, video, am, torch.zeros_like(vm), video_valid=~vm)  # same masks via video_valid
        assert all(torch.allclose(o_a[k], o_b[k], atol=1e-5) for k in o_a), fusion

    # 3) padding must not change the result (eval): append extra padded steps
    for fusion in FUSIONS:
        model = MultimodalModel(fusion=fusion, heads=HEADS).to(device).eval()
        pad = lambda x, m: (torch.cat([x, torch.randn(b, 5, x.shape[2], device=device)], 1),
                            torch.cat([m, torch.ones(b, 5, dtype=torch.bool, device=device)], 1))
        (ap, amp), (vp, vmp) = pad(audio, am), pad(video, vm)
        o1, o2 = model(audio, video, am, vm), model(ap, vp, amp, vmp)
        assert all(torch.allclose(o1[k], o2[k], atol=1e-4) for k in o1), f"padding changes output in {fusion}"
    print("fusion x heads, no-face / no-audio, video_valid and padding-invariance tests passed")

    # 4) fusion modules on their own with other widths
    a = torch.randn(3, 11, 64, device=device); v = torch.randn(3, 9, 32, device=device)
    ma = torch.zeros(3, 11, dtype=torch.bool, device=device); mv = torch.zeros(3, 9, dtype=torch.bool, device=device)
    mv[1] = True
    assert ConcatFusion(64, 32, 48).to(device)(a, ma, v, mv).shape == (3, 48)
    assert CrossAttentionFusion(64, 32, 48).to(device)(a, ma, v, mv).shape == (3, 48)
    fused, per = LateFusion(64, 32, n_out=2).to(device)(a, ma, v, mv)
    assert fused.shape == (3, 2) and torch.isfinite(fused).all() and set(per) == {"audio", "video"}
    print("standalone fusion modules (audio width 64, video width 32) passed")

    # 5) the interface used by train.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import build_model, forward_multimodal
    from data import collate_clips

    items = []
    for n_a, n_v, face in ((30, 30, True), (12, 11, True), (9, 9, False)):
        valid = torch.ones(n_v, dtype=torch.bool) if face else torch.zeros(n_v, dtype=torch.bool)
        items.append((torch.randn(n_a, 771), torch.randn(n_v, 770).masked_fill(~valid[:, None], 0.0), valid,
                      0, float(len(items) % 2), f"c{len(items)}", "mas"))
    batch = collate_clips(items)
    bs = len(items)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5, device=device))
    for mode in ("audio", "video", "concat", "late", "xattn"):
        model = build_model(mode, 0.3).to(device)
        assert model.fusion_name == mode
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = forward_multimodal(model, batch, device)
        assert logits.shape == (bs,) and torch.isfinite(logits.float()).all(), mode
        loss_fn(logits.float(), batch["label"].to(device)).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), mode
        print(f"  train.py build_model/forward_multimodal OK: {mode:6s} params={model.num_parameters():,}")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    print("all tests passed")


if __name__ == "__main__":
    _test()
