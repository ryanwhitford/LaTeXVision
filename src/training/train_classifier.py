"""Train the Phase 1 symbol classifier on a HASYv2 subset.

Usage:
    python -m src.training.train_classifier
    python -m src.training.train_classifier --epochs 5 --batch-size 32
    python -m src.training.train_classifier --max-samples-per-class 20  # smoke test

Config values come from configs/classifier.yaml by default; any value can
be overridden with a matching CLI flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.hasy import HASYSymbolDataset, load_dataset_config, load_labels, stratified_split
from src.data.preprocessing import get_transforms
from src.models.classifier import SymbolClassifier

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HASYv2 symbol classifier")
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--architecture", type=str, default=None, choices=["simple_cnn"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default=None,
        choices=["none", "class_weights", "weighted_sampler"],
        help="How to counter HASYv2's per-class imbalance during training.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Cap examples per class before splitting (fast smoke test).",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    overrides = {
        ("dataset_config",): args.dataset_config,
        ("data", "batch_size"): args.batch_size,
        ("data", "num_workers"): args.num_workers,
        ("model", "architecture"): args.architecture,
        ("model", "dropout"): args.dropout,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "weight_decay"): args.weight_decay,
        ("training", "epochs"): args.epochs,
        ("training", "balance_strategy"): args.balance_strategy,
        ("training", "device"): args.device,
        ("training", "seed"): args.seed,
        ("training", "early_stopping_patience"): args.patience,
        ("output", "output_dir"): args.output_dir,
        ("output", "run_name"): args.run_name,
    }
    for keys, value in overrides.items():
        if value is None:
            continue
        node = config
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value

    return config


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args)

    torch.manual_seed(config["training"]["seed"])
    device = resolve_device(config["training"]["device"])
    logger.info("Using device: %s", device)

    dataset_config = load_dataset_config(config["dataset_config"])
    df = load_labels(dataset_config)

    if args.max_samples_per_class:
        df = (
            df.sample(frac=1, random_state=42)
            .groupby("class_name", group_keys=False)
            .head(args.max_samples_per_class)
            .reset_index(drop=True)
        )
        logger.info("Capped to %d samples/class -> %d total", args.max_samples_per_class, len(df))

    train_df, val_df, test_df = stratified_split(df, dataset_config)

    output_dir = Path(config["output"]["output_dir"]) / config["output"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df[["path", "class_name", "label"]].to_csv(output_dir / f"{name}_split.csv", index=False)

    train_transform = get_transforms(dataset_config.image_size, train=True)
    eval_transform = get_transforms(dataset_config.image_size, train=False)

    train_ds = HASYSymbolDataset(train_df, transform=train_transform)
    val_ds = HASYSymbolDataset(val_df, transform=eval_transform)
    test_ds = HASYSymbolDataset(test_df, transform=eval_transform)

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    num_classes = len(dataset_config.class_names)

    balance_strategy = config["training"]["balance_strategy"]
    class_counts = train_df["label"].value_counts().reindex(range(num_classes)).fillna(0) + 1e-6
    inverse_freq = (1.0 / class_counts).values

    if balance_strategy == "weighted_sampler":
        # Oversample minority classes (e.g. 'x': 46 train examples vs 'infty':
        # ~2000) so each epoch shows the model roughly equal exposure per
        # class, with augmentation providing per-draw variety. This tends to
        # help small/rare classes more than loss reweighting alone, since the
        # model sees more distinct augmented views of them per epoch instead
        # of just a larger gradient on the same handful of examples.
        sample_weights = inverse_freq[train_df["label"].values]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()
    elif balance_strategy == "class_weights":
        weights = inverse_freq / inverse_freq.sum() * num_classes
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SymbolClassifier(
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
        use_batchnorm=config["model"]["use_batchnorm"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    history: list[dict] = []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        logger.info(
            "epoch %3d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f  (%.1fs)",
            epoch,
            config["training"]["epochs"],
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            elapsed,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
            logger.info("  new best val_acc=%.4f -> saved checkpoint", val_acc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping: no val_acc improvement for %d epochs", patience)
                break

    # Final test-set evaluation using the best checkpoint.
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    logger.info("Test: loss=%.4f acc=%.4f", test_loss, test_acc)

    class_mapping = {
        "class_names": dataset_config.class_names,
        "latex": {name: dataset_config.selected_classes[name] for name in dataset_config.class_names},
        "image_size": dataset_config.image_size,
    }
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "num_classes": num_classes,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved model, class mapping, and history to %s", output_dir)


if __name__ == "__main__":
    main()
