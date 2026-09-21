"""Image transforms for HASYv2 symbols.

Normalization uses the empirical mean/std of HASYv2 grayscale pixel
intensities (symbols are black strokes on a white background, so the
distribution is heavily skewed toward white). Train-time augmentation is
intentionally mild: handwritten symbols are already low-resolution (32x32)
and semantically fragile -- e.g. a '6' rotated too far becomes a '9', and
heavy distortion destroys the stroke shape a human reader relies on.
"""

from __future__ import annotations

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
