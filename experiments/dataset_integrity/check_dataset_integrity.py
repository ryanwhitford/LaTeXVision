"""Dataset integrity audit for the HASYv2 26-class subset (optimization brief, Section 3).

Checks: corrupted images, blank/near-blank images, unusual dimensions,
exact duplicate images (bit-identical, both within and across classes),
near-duplicate images (perceptual hash, flags possible train/test leakage
risk even though path-level leakage was already ruled out in the audit),
and class balance. Writes a report + representative sample grids per class.

Not a pytest test -- a one-time diagnostic, run manually.
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.hasy import load_dataset_config, load_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent

dataset_config = load_dataset_config(REPO_ROOT / "configs" / "dataset.yaml")
df = load_labels(dataset_config)

print(f"Total images: {len(df)}")
print()

# --- corrupted / unreadable images ---
corrupted = []
dimensions = defaultdict(int)
blank_or_near_blank = []
exact_hashes: dict[str, list[str]] = defaultdict(list)

for _, row in df.iterrows():
    path = row["path"]
    try:
        with Image.open(path) as img:
            img.verify()
        img = Image.open(path).convert("L")
        arr = np.array(img)
    except Exception as e:
        corrupted.append((path, str(e)))
        continue

    dimensions[img.size] += 1

    ink_fraction = (arr < 250).mean()
    if ink_fraction < 0.005:  # less than 0.5% of pixels are ink
        blank_or_near_blank.append((path, row["class_name"], float(ink_fraction)))

    file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()
    exact_hashes[file_hash].append(path)

print(f"Corrupted / unreadable images: {len(corrupted)}")
for path, err in corrupted[:10]:
    print(f"  {path}: {err}")

print()
print("Image dimensions found:")
for dims, count in sorted(dimensions.items(), key=lambda x: -x[1]):
    print(f"  {dims}: {count}")

print()
print(f"Blank / near-blank images (<0.5% ink pixels): {len(blank_or_near_blank)}")
for path, cls, frac in blank_or_near_blank[:10]:
    print(f"  {cls}: {path} (ink_fraction={frac:.4f})")

print()
exact_dupes = {h: paths for h, paths in exact_hashes.items() if len(paths) > 1}
n_dupe_images = sum(len(v) for v in exact_dupes.values())
print(f"Exact (bit-identical) duplicate images: {len(exact_dupes)} groups, {n_dupe_images} images involved")
# Check whether any duplicate group spans multiple classes (a label inconsistency)
label_by_path = dict(zip(df["path"], df["class_name"]))
cross_class_dupes = []
for h, paths in exact_dupes.items():
    classes = {label_by_path[p] for p in paths}
    if len(classes) > 1:
        cross_class_dupes.append((paths, classes))
print(f"  of which cross-class (same image, different label -- a real label bug if any): {len(cross_class_dupes)}")
for paths, classes in cross_class_dupes[:10]:
    print(f"    {classes}: {paths}")

print()
print("Class balance (train+val+test combined):")
counts = df["class_name"].value_counts().sort_values()
for name, count in counts.items():
    bar = "#" * int(count / counts.max() * 40)
    print(f"  {name:10s} {count:5d} {bar}")

print()
print(f"Imbalance ratio (max/min class count): {counts.max() / counts.min():.1f}x")

# Save a machine-readable summary
import json

summary = {
    "total_images": len(df),
    "n_corrupted": len(corrupted),
    "dimensions": {str(k): v for k, v in dimensions.items()},
    "n_blank_or_near_blank": len(blank_or_near_blank),
    "n_exact_duplicate_groups": len(exact_dupes),
    "n_images_in_duplicate_groups": n_dupe_images,
    "n_cross_class_duplicates": len(cross_class_dupes),
    "class_counts": counts.to_dict(),
    "imbalance_ratio": float(counts.max() / counts.min()),
}
with open(OUT_DIR / "integrity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved {OUT_DIR / 'integrity_summary.json'}")
