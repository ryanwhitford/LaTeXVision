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


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Return the preprocessing pipeline for the given split.

    Augmentation (train only) is a small random affine (rotation, translation,
    scale) plus mild erasing, chosen to preserve the identity of visually
    similar symbol pairs (6/9, x/times, o/0).
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
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
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[HASY_MEAN], std=[HASY_STD]),
        ]
    )


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
