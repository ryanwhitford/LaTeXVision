"""One-off audit script: does train-time preprocessing match inference-time preprocessing?

Training/evaluation (src/training/*.py) feeds raw HASYv2 images straight
into get_transforms() -- no crop_to_content(). Inference (src/api/inference.py)
always calls crop_to_content() first. This script quantifies whether that
matters: if HASYv2 images are already tightly cropped, crop_to_content on
them should be close to a no-op. If not, train and serve see systematically
different framing/scale for the same underlying image, which is exactly the
kind of skew the optimization brief calls "extremely important".

Not a pytest test -- a one-time measurement, run manually and the output
pasted into the audit report.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.hasy import load_dataset_config, load_labels  # noqa: E402
from src.data.preprocessing import crop_to_content, get_transforms  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
dataset_config = load_dataset_config(REPO_ROOT / "configs" / "dataset.yaml")
df = load_labels(dataset_config)

rng = np.random.default_rng(42)
sample = df.sample(n=500, random_state=42)

eval_transform = get_transforms(dataset_config.image_size, train=False)

bbox_fill_ratios = []
pixel_diffs = []
n_no_ink = 0

for path in sample["path"]:
    image = Image.open(path).convert("L")
    arr = np.array(image)

    ink_rows, ink_cols = np.where(arr < 250)
    if len(ink_rows) == 0:
        n_no_ink += 1
        continue

    h = ink_rows.max() - ink_rows.min() + 1
    w = ink_cols.max() - ink_cols.min() + 1
    bbox_fill_ratios.append(max(h, w) / arr.shape[0])

    # "train/eval path": raw image straight through get_transforms.
    train_path_tensor = eval_transform(image)

    # "inference path": crop_to_content first, then the same transform.
    cropped = crop_to_content(image)
    infer_path_tensor = eval_transform(cropped)

    diff = (train_path_tensor - infer_path_tensor).abs().mean().item()
    pixel_diffs.append(diff)

bbox_fill_ratios = np.array(bbox_fill_ratios)
pixel_diffs = np.array(pixel_diffs)

print(f"n images sampled: {len(sample)}  (no-ink: {n_no_ink})")
print()
print("Ink bounding-box fill ratio (max(h,w) / image_size=32), i.e. how much")
print("of the native HASYv2 image the symbol's bounding box already occupies:")
print(f"  mean={bbox_fill_ratios.mean():.3f}  median={np.median(bbox_fill_ratios):.3f}  "
      f"min={bbox_fill_ratios.min():.3f}  max={bbox_fill_ratios.max():.3f}")
print(f"  p10={np.percentile(bbox_fill_ratios, 10):.3f}  p90={np.percentile(bbox_fill_ratios, 90):.3f}")
print()
print("Mean absolute pixel difference (normalized tensor space) between")
print("'train/eval path' (raw resize) and 'inference path' (crop_to_content + resize)")
print("for the SAME underlying image:")
print(f"  mean={pixel_diffs.mean():.4f}  median={np.median(pixel_diffs):.4f}  "
      f"max={pixel_diffs.max():.4f}")
print(f"  fraction of images with diff > 0.05: {(pixel_diffs > 0.05).mean():.3f}")
print(f"  fraction of images with diff > 0.15: {(pixel_diffs > 0.15).mean():.3f}")
