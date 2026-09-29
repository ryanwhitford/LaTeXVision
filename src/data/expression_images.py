"""Full-expression image normalization for the transformer recognizer.

`normalize_expression_image` is THE preprocessing step between a raw
expression image (a canvas PNG, a rasterized MathWriting/CROHME ink, or a
synthetic HASYv2 composition) and the transformer's encoder. The training
Dataset and the serving path both call this one function -- the Phase-1
audit (experiments/audit/AUDIT_REPORT.md) found that letting training and
serving preprocess independently cost ~55 points of real-world accuracy, so
there is deliberately no second implementation anywhere.

Why rescale by glyph size rather than by image height: the encoder is the
symbol classifier's conv trunk, warm-started from weights trained on
32x32 crops in which a glyph's long side is ~23px (crop_to_content pads 20%
per side: 32 / 1.4). The trunk's receptive field is only ~18px, so its
features only mean what they were trained to mean if each glyph arrives at
roughly that size. Scaling an expression to a fixed *height* would shrink a
tall fraction's glyphs and inflate a flat expression's; scaling so the
*median glyph* is ~23px keeps every expression in the trunk's native scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

TARGET_GLYPH_EXTENT = 23.0
MAX_HEIGHT = 128
MAX_WIDTH = 512
MARGIN = 6
INK_THRESHOLD = 200


def estimate_glyph_extent(gray: np.ndarray, ink_threshold: int = INK_THRESHOLD) -> float | None:
    """Median long-side extent of connected ink components -- a proxy for
    typical glyph size that works on any raster, with no stroke data.
    Components far smaller than the largest (specks, dots) are ignored."""
    binary = (gray < ink_threshold).astype(np.uint8)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None
    extents = np.array([max(w, h) for _x, _y, w, h, _area in stats[1:]], dtype=np.float32)
    extents = extents[extents >= 0.1 * extents.max()]
    return float(np.median(extents)) if len(extents) else None


@dataclass(frozen=True)
class NormalizationGeometry:
    """Where the normalized image came from in the source image, so model
    outputs over the normalized image (e.g. attention maps) can be drawn
    back onto the original canvas: source x = crop_x + (u - margin) * scale_x."""

    crop_x: int
    crop_y: int
    scale_x: float  # source pixels per normalized pixel
    scale_y: float
    margin: int

    def to_source(self, u: float, v: float) -> tuple[float, float]:
        return self.crop_x + (u - self.margin) * self.scale_x, self.crop_y + (v - self.margin) * self.scale_y


def normalize_with_geometry(image: Image.Image) -> tuple[Image.Image, NormalizationGeometry] | None:
    """`normalize_expression_image` plus the crop/scale it applied.
    Returns None for a blank image."""
    gray = np.array(image.convert("L"))
    ink_rows, ink_cols = np.where(gray < 250)
    if len(ink_rows) == 0:
        return None
    top, left = int(ink_rows.min()), int(ink_cols.min())
    gray = gray[top : ink_rows.max() + 1, left : ink_cols.max() + 1]

    extent = estimate_glyph_extent(gray)
    scale = TARGET_GLYPH_EXTENT / extent if extent else 1.0
    height, width = gray.shape
    fit = min(1.0, MAX_HEIGHT / (height * scale + 2 * MARGIN), MAX_WIDTH / (width * scale + 2 * MARGIN))
    scale *= fit
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = Image.fromarray(gray).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (new_w + 2 * MARGIN, new_h + 2 * MARGIN), 255)
    canvas.paste(resized, (MARGIN, MARGIN))
    return canvas, NormalizationGeometry(left, top, width / new_w, height / new_h, MARGIN)


def normalize_expression_image(image: Image.Image) -> Image.Image | None:
    """Crop to ink, rescale so the median glyph is ~TARGET_GLYPH_EXTENT px,
    cap at MAX_HEIGHT x MAX_WIDTH, and pad a small white margin.
    Returns None for a blank image."""
    result = normalize_with_geometry(image)
    return None if result is None else result[0]
