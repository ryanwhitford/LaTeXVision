"""Loads a trained SymbolClassifier checkpoint and runs single-symbol inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data.preprocessing import get_transforms, has_ink
from src.models.classifier import SymbolClassifier


@dataclass
class SymbolPrediction:
    class_name: str
    latex: str
    confidence: float


class SymbolPredictor:
    """Wraps a trained checkpoint + class mapping for image -> prediction inference."""

    def __init__(
        self,
        model: SymbolClassifier,
        class_names: list[str],
        latex_by_name: dict[str, str],
        image_size: int,
        device: torch.device,
        content_crop: bool = True,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.latex_by_name = latex_by_name
        self.image_size = image_size
        self.device = device
        # Must match whatever this specific checkpoint was trained with --
        # read from class_mapping.json (see `load` below) rather than
        # hardcoded, so an older checkpoint trained without content_crop
        # doesn't silently get served with a mismatched pipeline. See
        # get_transforms' docstring / experiments/audit/AUDIT_REPORT.md.
        self.transform = get_transforms(image_size, train=False, content_crop=content_crop)

    @classmethod
    def load(cls, run_dir: str | Path, device: torch.device | None = None) -> "SymbolPredictor":
        run_dir = Path(run_dir)
        with open(run_dir / "class_mapping.json") as f:
            class_mapping = json.load(f)

        resolved_device = device or torch.device("cpu")
        model = SymbolClassifier(num_classes=len(class_mapping["class_names"]))
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=resolved_device))
        model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            class_names=class_mapping["class_names"],
            latex_by_name=class_mapping["latex"],
            image_size=class_mapping["image_size"],
            device=resolved_device,
            # Older checkpoints (pre-audit) predate this key; they were, in
            # fact, always served with content_crop applied (main.py called
            # crop_to_content unconditionally) -- so True is the historically
            # accurate default, not just a fallback guess.
            content_crop=class_mapping.get("content_crop", True),
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[SymbolPrediction] | None:
        """Predict the symbol drawn in `image`, ranked by confidence.

        Returns None if no content is detected (e.g. a blank canvas).
        """
        if not has_ink(image):
            return None

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(top_k, len(self.class_names))
        top_probs, top_indices = probs.topk(k)

        return [
            SymbolPrediction(
                class_name=self.class_names[idx],
                latex=self.latex_by_name[self.class_names[idx]],
                confidence=float(prob),
            )
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]
