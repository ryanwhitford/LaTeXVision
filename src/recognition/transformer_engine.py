"""Inference wrapper for the image-to-LaTeX transformer. Used by the API
and by the expression benchmark, so both exercise exactly the same
preprocessing and decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data.expression_dataset import image_to_tensor
from src.data.latex_tokenizer import decode, to_latex
from src.models.classifier import SymbolClassifier
from src.models.im2latex import Im2LatexModel


@dataclass
class TransformerResult:
    latex: str
    tokens: list[str]


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
    def recognize(self, image: Image.Image) -> TransformerResult | None:
        """Returns None for a blank image."""
        tensor = image_to_tensor(image)
        if tensor is None:
            return None
        images = tensor.unsqueeze(0).to(self.device)
        mask = torch.ones(images.shape[0], images.shape[2], images.shape[3], dtype=torch.bool, device=self.device)
        tokens = decode(self.model.greedy_decode(images, mask)[0])
        return TransformerResult(latex=to_latex(tokens), tokens=tokens)
