from pathlib import Path

import pytest

from src.data.hasy import HASYSymbolDataset, load_dataset_config, load_labels, stratified_split
from src.data.preprocessing import get_transforms

DATASET_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml"

requires_hasy_data = pytest.mark.skipif(
    not load_dataset_config(DATASET_CONFIG_PATH).labels_path.exists(),
    reason="HASYv2 raw data not downloaded; see README.md",
)


def test_load_dataset_config_resolves_paths_and_vocab():
    config = load_dataset_config(DATASET_CONFIG_PATH)
    assert config.image_size == 32
    assert "x" in config.class_names
    assert config.selected_classes["times"] == "\\times"
    assert config.raw_dir.is_absolute()


@requires_hasy_data
def test_load_labels_only_keeps_selected_classes():
    config = load_dataset_config(DATASET_CONFIG_PATH)
    df = load_labels(config)
    assert set(df["class_name"]) <= set(config.class_names)
    assert df["label"].max() < len(config.class_names)
    assert Path(df["path"].iloc[0]).exists()


@requires_hasy_data
def test_stratified_split_is_disjoint_and_covers_all_rows():
    config = load_dataset_config(DATASET_CONFIG_PATH)
    df = load_labels(config)
    train_df, val_df, test_df = stratified_split(df, config)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # Every class should appear in the training split.
    assert set(train_df["class_name"]) == set(config.class_names)


@requires_hasy_data
def test_dataset_returns_correctly_shaped_tensor():
    config = load_dataset_config(DATASET_CONFIG_PATH)
    df = load_labels(config)
    transform = get_transforms(config.image_size, train=False)
    dataset = HASYSymbolDataset(df.head(4), transform=transform)

    image, label = dataset[0]
    assert image.shape == (1, config.image_size, config.image_size)
    assert isinstance(label, int)
