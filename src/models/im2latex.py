"""Image-to-LaTeX transformer: CNN encoder + autoregressive transformer
decoder cross-attending over image features.

The encoder is the trained symbol classifier's conv trunk with its
classification head AND its global pool removed. The pool matters: in
SymbolClassifier the AdaptiveAvgPool2d(1) lives at the end of `features`,
not in `classifier`, so dropping only the head would collapse every image to
a single 1x1 feature and destroy the spatial layout the decoder needs.
`features[:-1]` keeps a (128, H/4, W/4) grid.

Warm start is mandatory: `build_from_classifier` refuses to run without the
trained checkpoint rather than silently falling back to a random encoder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from src.data.latex_tokenizer import BOS, EOS, PAD, TOKEN_TO_ID, VOCAB
from src.models.classifier import SymbolClassifier

TRUNK_CHANNELS = 128
TRUNK_STRIDE = 4


def _sinusoid(positions: int, dim: int, device=None) -> torch.Tensor:
    """(positions, dim) standard sinusoidal table; dim must be even."""
    pos = torch.arange(positions, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(positions, dim, device=device)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class PositionalEncoding2D(nn.Module):
    """Separate sinusoidal encodings for row and column, each d_model/2,
    concatenated -- NOT a 1D encoding over the flattened grid. Superscript
    vs subscript is purely a vertical offset, so row position has to be an
    explicit, independent signal rather than something the decoder would
    have to recover from a raster-scan index."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 4:
            raise ValueError("d_model must be divisible by 4 (two sin/cos halves)")
        self.half = d_model // 2

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, d_model, H, W) -> same shape with PE added."""
        _, _, h, w = feats.shape
        rows = _sinusoid(h, self.half, feats.device)  # (H, d/2)
        cols = _sinusoid(w, self.half, feats.device)  # (W, d/2)
        pe = torch.cat(
            [rows[:, None, :].expand(h, w, self.half), cols[None, :, :].expand(h, w, self.half)], dim=-1
        )  # (H, W, d)
        return feats + pe.permute(2, 0, 1).unsqueeze(0)


class Im2LatexModel(nn.Module):
    def __init__(
        self,
        trunk: nn.Sequential,
        vocab_size: int = len(VOCAB),
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        max_len: int = 96,
        freeze_trunk_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.freeze_trunk_batchnorm = freeze_trunk_batchnorm
        self.proj = nn.Conv2d(TRUNK_CHANNELS, d_model, kernel_size=1)
        self.pos2d = PositionalEncoding2D(d_model)
        self.enc_dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=TOKEN_TO_ID[PAD])
        self.register_buffer("pos1d", _sinusoid(max_len, d_model), persistent=False)
        self.dec_dropout = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size)

    def train(self, mode: bool = True):
        """The trunk's BatchNorm running stats come from tightly-cropped
        single symbols; full expressions are mostly whitespace. Updating
        those stats on expression batches would silently shift the warm-
        started features, so BN stays in eval mode (weights still train)."""
        super().train(mode)
        if self.freeze_trunk_batchnorm:
            for m in self.trunk.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def encode(self, images: torch.Tensor, image_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images: (B, 1, H, W) normalized; image_mask: (B, H, W) True = real pixel.
        Returns memory (B, H'*W', d) and memory_key_padding_mask (B, H'*W'),
        True = padding (PyTorch's convention)."""
        feats = self.proj(self.trunk(images))  # (B, d, H', W')
        feats = self.pos2d(feats)
        b, d, h, w = feats.shape
        memory = self.enc_dropout(feats.flatten(2).transpose(1, 2))
        valid = nn.functional.max_pool2d(image_mask[:, None].float(), TRUNK_STRIDE)[:, 0] > 0
        valid = valid[:, :h, :w]
        return memory, ~valid.reshape(b, h * w)

    def _decode_step(self, tgt: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor) -> torch.Tensor:
        seq = tgt.shape[1]
        x = self.embed(tgt) * math.sqrt(self.d_model) + self.pos1d[:seq]
        causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=tgt.device), diagonal=1)
        x = self.decoder(
            self.dec_dropout(x),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt == TOKEN_TO_ID[PAD],
            memory_key_padding_mask=memory_pad,
        )
        return self.out(x)

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits (B, T, vocab) for input tokens tgt_in (B, T)."""
        memory, memory_pad = self.encode(images, image_mask)
        return self._decode_step(tgt_in, memory, memory_pad)

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, image_mask: torch.Tensor, max_len: int | None = None) -> list[list[int]]:
        max_len = min(max_len or self.max_len, self.max_len)
        memory, memory_pad = self.encode(images, image_mask)
        b = images.shape[0]
        seq = torch.full((b, 1), TOKEN_TO_ID[BOS], dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            next_tok = self._decode_step(seq, memory, memory_pad)[:, -1].argmax(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, TOKEN_TO_ID[PAD]), next_tok)
            seq = torch.cat([seq, next_tok[:, None]], dim=1)
            done |= next_tok == TOKEN_TO_ID[EOS]
            if done.all():
                break
        return seq[:, 1:].tolist()


def load_classifier_trunk(classifier_run_dir: Path) -> nn.Sequential:
    """The trained classifier's conv trunk minus its global pool. Raises if
    the checkpoint is missing -- no random-init fallback, by design."""
    checkpoint = classifier_run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No classifier checkpoint at {checkpoint}. The im2latex encoder must be warm-started "
            "from the trained symbol classifier; train it first (src.training.train_classifier)."
        )
    with open(classifier_run_dir / "class_mapping.json") as f:
        num_classes = len(json.load(f)["class_names"])
    classifier = SymbolClassifier(num_classes=num_classes)
    classifier.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    trunk = classifier.features[:-1]
    if isinstance(trunk[-1], nn.AdaptiveAvgPool2d):  # defensive: never hand the decoder a 1x1 grid
        raise RuntimeError("trunk still ends in a global pool")
    return trunk


def build_from_classifier(classifier_run_dir: Path, **model_kwargs) -> Im2LatexModel:
    return Im2LatexModel(load_classifier_trunk(classifier_run_dir), **model_kwargs)
