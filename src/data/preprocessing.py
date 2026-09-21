"""Image transforms for HASYv2 symbols.

Normalization uses the empirical mean/std of HASYv2 grayscale pixel
intensities (symbols are black strokes on a white background, so the
distribution is heavily skewed toward white). Train-time augmentation is
intentionally mild: handwritten symbols are already low-resolution (32x32)
and semantically fragile -- e.g. a '6' rotated too far becomes a '9', and
heavy distortion destroys the stroke shape a human reader relies on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

# Computed once over a 2000-image sample of the selected-class HASYv2 subset.
HASY_MEAN = 0.8542
HASY_STD = 0.3529


class ContentCrop:
    """torchvision-transform-compatible wrapper around `crop_to_content`.

    Falls back to the original image if no ink is found (shouldn't happen
    for real HASYv2/drawn-symbol images, but avoids a hard crash on a
    pathological all-white input).
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        cropped = crop_to_content(image)
        return cropped if cropped is not None else image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_transforms(image_size: int, train: bool, content_crop: bool = False) -> transforms.Compose:
    """Return the preprocessing pipeline for the given split.

    `content_crop`: whether to run `ContentCrop` (ink-bbox crop + padding)
    before resizing. This MUST be the same for whatever pipeline trained the
    model and whatever pipeline serves it -- see
    experiments/audit/AUDIT_REPORT.md, Finding 1, for what happens when it
    isn't (a systematic, ~55-point accuracy drop in production despite a
    healthy-looking offline test score). `src/api/inference.py` and
    `train_classifier.py`/`evaluate_classifier.py` both call this function
    rather than applying `ContentCrop` independently, specifically so they
    cannot drift apart again.

    Augmentation (train only) is a small random affine (rotation, translation,
    scale) plus mild erasing, chosen to preserve the identity of visually
    similar symbol pairs (6/9, x/times, o/0).
    """
    steps: list = [ContentCrop()] if content_crop else []
    steps.append(transforms.Resize((image_size, image_size)))

    if train:
        steps += [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=5,
                fill=255,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[HASY_MEAN], std=[HASY_STD]),
            # Small occlusion patches -- extra regularization, useful now that
            # minority classes are oversampled (same image seen repeatedly/epoch).
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), value=0.0),
        ]
    else:
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[HASY_MEAN], std=[HASY_STD]),
        ]

    return transforms.Compose(steps)


def has_ink(image: Image.Image, ink_threshold: int = 250) -> bool:
    """True if `image` has any non-background pixel (i.e. isn't a blank canvas)."""
    arr = np.array(image.convert("L"))
    return bool((arr < ink_threshold).any())


def crop_to_content(
    image: Image.Image, ink_threshold: int = 250, padding_fraction: float = 0.2
) -> Image.Image | None:
    """Crop a grayscale image to its drawn content, padded to a square.

    HASYv2 training images are tightly cropped around the glyph. A raw
    canvas drawing is not -- the symbol is small within a mostly-blank
    square -- so classifying it unmodified would present the model with a
    very different scale distribution than it was trained on. This finds
    the bounding box of "ink" (non-background) pixels, pads it, and pastes
    it onto a square white canvas so the aspect ratio is preserved under
    the caller's subsequent `Resize`.

    Returns None if no ink is found (a blank canvas).
    """
    arr = np.array(image.convert("L"))
    ink_rows, ink_cols = np.where(arr < ink_threshold)
    if len(ink_rows) == 0:
        return None

    top, bottom = ink_rows.min(), ink_rows.max()
    left, right = ink_cols.min(), ink_cols.max()

    height, width = bottom - top + 1, right - left + 1
    side = max(height, width)
    pad = int(side * padding_fraction)
    side += 2 * pad

    canvas = Image.new("L", (side, side), color=255)
    cropped = image.convert("L").crop((left, top, right + 1, bottom + 1))
    offset = ((side - width) // 2, (side - height) // 2)
    canvas.paste(cropped, offset)
    return canvas
