"""Inference wrapper for the image-to-LaTeX transformer. Used by the API
and by the expression benchmark, so both exercise exactly the same
preprocessing and decoding."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.expression_dataset import array_to_tensor
from src.data.expression_images import normalize_with_geometry
from src.data.latex_tokenizer import EOS, TOKEN_TO_ID, decode, to_latex
from src.models.classifier import SymbolClassifier
from src.models.im2latex import TRUNK_STRIDE, Im2LatexModel


@dataclass
class AttentionMaps:
    """Per-token cross-attention over the encoder grid, placed in the
    coordinates of the image passed to `recognize`.

    `maps[t]` is token t's (grid_h x grid_w) map, row-major, scaled so its
    peak is 255 and packed as base64 uint8. `box` = (x0, y0, x1, y1) is the
    region of the source image the grid covers."""

    grid_h: int
    grid_w: int
    box: tuple[float, float, float, float]
    maps: list[str]


@dataclass
class TransformerResult:
    latex: str
    tokens: list[str]
    attention: AttentionMaps | None = None


class TransformerRecognizer:
    def __init__(self, model: Im2LatexModel, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, run_dir: str | Path, device: torch.device | None = None) -> "TransformerRecognizer":
        run_dir = Path(run_dir)
        with open(run_dir / "model_config.json") as f:
            model_cfg = json.load(f)
        # Architecture-only trunk; its trained weights come from the
        # im2latex checkpoint below (which already contains the fine-tuned
        # trunk), not from a fresh init.
        trunk = SymbolClassifier(num_classes=1).features[:-1]
        model = Im2LatexModel(trunk, **model_cfg)
        device = device or torch.device("cpu")
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        return cls(model, device)

    @torch.no_grad()
    def recognize(self, image: Image.Image, explain: bool = False) -> TransformerResult | None:
        """Returns None for a blank image. `explain=True` also returns where
        the decoder attended while emitting each token."""
        normalized = normalize_with_geometry(image)
        if normalized is None:
            return None
        norm_image, geometry = normalized
        images = array_to_tensor(np.asarray(norm_image, dtype=np.uint8)).unsqueeze(0).to(self.device)
        mask = torch.ones(images.shape[0], images.shape[2], images.shape[3], dtype=torch.bool, device=self.device)
        ids = self.model.greedy_decode(images, mask)[0]
        tokens = decode(ids)
        result = TransformerResult(latex=to_latex(tokens), tokens=tokens)
        if explain and tokens:
            eos = TOKEN_TO_ID[EOS]
            emitted = ids[: ids.index(eos)] if eos in ids else ids
            attn = self.model.cross_attention(images, mask, emitted[: len(tokens)]).cpu().numpy()
            grid_h, grid_w = attn.shape[1:]
            # Each grid cell covers TRUNK_STRIDE x TRUNK_STRIDE normalized pixels.
            x0, y0 = geometry.to_source(0, 0)
            x1, y1 = geometry.to_source(grid_w * TRUNK_STRIDE, grid_h * TRUNK_STRIDE)
            peak = attn.reshape(len(attn), -1).max(axis=1).clip(min=1e-8)[:, None, None]
            packed = [
                base64.b64encode((m * 255).round().astype(np.uint8).tobytes()).decode("ascii")
                for m in attn / peak
            ]
            result.attention = AttentionMaps(grid_h, grid_w, (x0, y0, x1, y1), packed)
        return result
