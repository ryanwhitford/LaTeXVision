"""Robustness benchmark: how does a trained classifier hold up on conditions
that resemble what an imperfect symbol-localization crop or a real browser
drawing will produce, rather than HASYv2's clean, tightly-cropped images?

HASYv2 images have the ink bounding box filling ~100% of the 32x32 frame
(see experiments/audit/AUDIT_REPORT.md, Finding 1) -- there's no room to
express "off-center" or "extra margin" within a native image. So every
condition here first pastes the native symbol onto a larger blank canvas at
a controlled scale/position, THEN perturbs it, THEN runs it through the
model's real, deployed inference path (SymbolPredictor.predict, i.e.
including crop_to_content) -- this measures the actual production pipeline,
not just a raw transform.

Usage:
    python -m src.evaluation.robustness --run-dir models/symbol_classifier_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src.api.inference import SymbolPredictor
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)

CANVAS_SIZE = 128  # large enough to express generous margins/off-center placement


def _paste(symbol: Image.Image, scale: float, offset_x: float, offset_y: float) -> Image.Image:
    """Paste a native HASYv2 symbol onto a blank CANVAS_SIZE white canvas.

    `scale` is the symbol's side length as a fraction of the canvas.
    `offset_x`/`offset_y` are in [0, 1]: 0 = flush against the top/left of
    the available margin, 0.5 = centered, 1 = flush bottom/right.
    """
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    side = max(1, int(CANVAS_SIZE * scale))
    resized = symbol.convert("L").resize((side, side), Image.LANCZOS)
    max_offset = CANVAS_SIZE - side
    x = int(offset_x * max_offset)
    y = int(offset_y * max_offset)
    canvas.paste(resized, (x, y))
    return canvas


def _condition_clean(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return _paste(image, scale=0.7, offset_x=0.5, offset_y=0.5)


def _condition_translation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    ox, oy = rng.uniform(0.1, 0.9, size=2)
    return _paste(image, scale=0.7, offset_x=ox, offset_y=oy)


def _condition_rotation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    angle = rng.uniform(-25, 25)
    rotated = image.convert("L").rotate(angle, expand=True, fillcolor=255)
    return _paste(rotated, scale=0.7, offset_x=0.5, offset_y=0.5)


def _condition_scale_zoomed_out(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    # Simulates a generous/loose detector box: symbol occupies a small
    # fraction of the crop.
    return _paste(image, scale=0.3, offset_x=0.5, offset_y=0.5)


def _condition_scale_zoomed_in(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    # Simulates a tight detector box that leaves almost no margin.
    return _paste(image, scale=0.98, offset_x=0.5, offset_y=0.5)


def _condition_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    canvas = _condition_clean(image, rng)
    arr = np.array(canvas).astype(np.float32)
    noise = rng.normal(0, 20, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


def _condition_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    canvas = _condition_clean(image, rng)
    return canvas.filter(ImageFilter.GaussianBlur(radius=1.5))


def _condition_thin_stroke(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    eroded = image.convert("L").filter(ImageFilter.MaxFilter(3))  # max = erodes dark ink on light bg
    return _paste(eroded, scale=0.7, offset_x=0.5, offset_y=0.5)


def _condition_thick_stroke(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    dilated = image.convert("L").filter(ImageFilter.MinFilter(3))  # min = dilates dark ink on light bg
    return _paste(dilated, scale=0.7, offset_x=0.5, offset_y=0.5)


def _condition_off_center(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    corner = rng.choice([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    return _paste(image, scale=0.55, offset_x=corner[0], offset_y=corner[1])


def _condition_imperfect_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    # A detector box that clips into the symbol itself on 1-2 sides.
    canvas = _paste(image, scale=0.75, offset_x=0.5, offset_y=0.5)
    arr = np.array(canvas)
    clip = int(CANVAS_SIZE * rng.uniform(0.08, 0.18))
    side = rng.choice(["left", "right", "top", "bottom"])
    if side == "left":
        arr[:, :clip] = 255
    elif side == "right":
        arr[:, -clip:] = 255
    elif side == "top":
        arr[:clip, :] = 255
    else:
        arr[-clip:, :] = 255
    return Image.fromarray(arr, mode="L")


CONDITIONS: dict[str, Callable[[Image.Image, np.random.Generator], Image.Image]] = {
    "clean": _condition_clean,
    "translation": _condition_translation,
    "rotation": _condition_rotation,
    "scale_zoomed_out": _condition_scale_zoomed_out,
    "scale_zoomed_in": _condition_scale_zoomed_in,
    "noise": _condition_noise,
    "blur": _condition_blur,
    "thin_stroke": _condition_thin_stroke,
    "thick_stroke": _condition_thick_stroke,
    "off_center": _condition_off_center,
    "imperfect_crop": _condition_imperfect_crop,
}


@dataclass
class RobustnessResult:
    condition: str
    accuracy: float
    n: int
    n_no_prediction: int  # crop_to_content found no ink (shouldn't happen but tracked)


def run_robustness_benchmark(
    predictor: SymbolPredictor,
    df: pd.DataFrame,
    conditions: list[str] | None = None,
    seed: int = 42,
) -> list[RobustnessResult]:
    """Evaluate `predictor` (the real inference path) on every condition.

    `df` must have `path` and `label` columns (e.g. a test_split.csv), with
    `label` as an index into `predictor.class_names`.
    """
    conditions = conditions or list(CONDITIONS)
    rng = np.random.default_rng(seed)
    results = []

    for condition_name in conditions:
        condition_fn = CONDITIONS[condition_name]
        correct = 0
        no_prediction = 0
        for _, row in df.iterrows():
            image = Image.open(row["path"])
            perturbed = condition_fn(image, rng)
            predictions = predictor.predict(perturbed, top_k=1)
            if predictions is None:
                no_prediction += 1
                continue
            if predictions[0].class_name == predictor.class_names[row["label"]]:
                correct += 1
        n = len(df)
        results.append(
            RobustnessResult(
                condition=condition_name,
                accuracy=correct / n,
                n=n,
                n_no_prediction=no_prediction,
            )
        )
        logger.info("%-18s accuracy=%.4f  (no-prediction: %d/%d)", condition_name, correct / n, no_prediction, n)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the detector-crop / handwriting-variation robustness benchmark")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=400, help="Subsample the split for speed; 0 = use all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir)

    device = resolve_device(args.device)
    predictor = SymbolPredictor.load(run_dir, device=device)

    df = pd.read_csv(run_dir / f"{args.split}_split.csv")
    if args.n_samples and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    logger.info("Running robustness benchmark on %d images from %s", len(df), run_dir)
    results = run_robustness_benchmark(predictor, df, seed=args.seed)

    out = {r.condition: {"accuracy": r.accuracy, "n": r.n, "n_no_prediction": r.n_no_prediction} for r in results}
    out_path = run_dir / f"robustness_benchmark_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved %s", out_path)

    print("\n| Condition | Accuracy |")
    print("|---|---:|")
    for r in results:
        print(f"| {r.condition} | {r.accuracy:.1%} |")


if __name__ == "__main__":
    main()
