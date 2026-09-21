"""HASYv2 dataset loading and class-vocabulary filtering.

HASYv2 (Thoma, 2017) ships as a flat CSV of 168k+ handwritten LaTeX symbol
images (32x32 PNG, drawn as black strokes on a white background) covering
369 classes. This module loads the CSV, restricts it to a configurable
subset of classes, and produces stratified train/val/test splits.

See configs/dataset.yaml for the class vocabulary and split fractions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Resolved configuration for a HASYv2 subset."""

    raw_dir: Path
    labels_csv: str
    image_size: int
    selected_classes: dict[str, str]  # class name -> HASYv2 latex label
    val_fraction: float
    test_fraction: float
    seed: int

    @property
    def labels_path(self) -> Path:
        return self.raw_dir / self.labels_csv

    @property
    def class_names(self) -> list[str]:
        """Class names sorted alphabetically, defining the label index order."""
        return sorted(self.selected_classes)

    @property
    def latex_to_name(self) -> dict[str, str]:
        return {latex: name for name, latex in self.selected_classes.items()}


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load a dataset YAML config (see configs/dataset.yaml) into a DatasetConfig."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # raw_dir is resolved relative to the repo root (config file's grandparent).
    repo_root = config_path.resolve().parent.parent
    raw_dir = Path(raw["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = repo_root / raw_dir

    split = raw["split"]
    return DatasetConfig(
        raw_dir=raw_dir,
        labels_csv=raw["labels_csv"],
        image_size=raw["image_size"],
        selected_classes=raw["selected_classes"],
        val_fraction=split["val_fraction"],
        test_fraction=split["test_fraction"],
        seed=split["seed"],
    )


def load_labels(config: DatasetConfig) -> pd.DataFrame:
    """Load hasy-data-labels.csv and filter to the configured class vocabulary.

    Returns a DataFrame with columns: path (absolute), latex, class_name, label.
    `label` is an integer index into `config.class_names`.
    """
    if not config.labels_path.exists():
        raise FileNotFoundError(
            f"HASYv2 labels CSV not found at {config.labels_path}. "
            "See README.md for download instructions."
        )

    df = pd.read_csv(config.labels_path)
    latex_to_name = config.latex_to_name
    df = df[df["latex"].isin(latex_to_name)].copy()
    if df.empty:
        raise ValueError(
            "No rows matched the configured selected_classes. Check that "
            "the latex labels in configs/dataset.yaml exist in symbols.csv."
        )

    df["class_name"] = df["latex"].map(latex_to_name)
    name_to_label = {name: i for i, name in enumerate(config.class_names)}
    df["label"] = df["class_name"].map(name_to_label)
    df["path"] = df["path"].apply(lambda p: str(config.raw_dir / p))

    counts = df["class_name"].value_counts()
    logger.info("Loaded %d images across %d classes", len(df), len(counts))
    for name in config.class_names:
        logger.info("  %-10s %5d images", name, int(counts.get(name, 0)))

    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test, preserving per-class proportions.

    Falls back to a non-stratified split with a warning for any class too
    small to stratify (HASYv2 classes range from ~50 to ~3000 examples).
    """
    val_and_test = config.val_fraction + config.test_fraction
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=val_and_test,
            stratify=df["label"],
            random_state=config.seed,
        )
    except ValueError as e:
        logger.warning("Stratified split failed (%s); falling back to random split", e)
        train_df, temp_df = train_test_split(
            df, test_size=val_and_test, random_state=config.seed
        )

    relative_test_fraction = config.test_fraction / val_and_test
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_fraction,
            stratify=temp_df["label"],
            random_state=config.seed,
        )
    except ValueError:
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_fraction, random_state=config.seed
        )

    logger.info(
        "Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class HASYSymbolDataset(Dataset):
    """PyTorch Dataset over a HASYv2 subset DataFrame (path, label columns)."""

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]
