"""Loads a trained SymbolClassifier checkpoint and runs single-symbol inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data.preprocessing import crop_to_content, get_transforms
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
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.latex_by_name = latex_by_name
        self.image_size = image_size
        self.device = device
        self.transform = get_transforms(image_size, train=False)

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
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[SymbolPrediction] | None:
        """Predict the symbol drawn in `image`, ranked by confidence.

        Returns None if no content is detected (e.g. a blank canvas).
        """
        cropped = crop_to_content(image)
        if cropped is None:
            return None

        tensor = self.transform(cropped).unsqueeze(0).to(self.device)
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
