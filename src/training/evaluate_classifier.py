"""Detailed evaluation of a trained symbol classifier on its held-out test split.

Usage:
    python -m src.training.evaluate_classifier --run-dir models/symbol_classifier_v1

Loads the checkpoint, class mapping, and test_split.csv written by
train_classifier.py, then reports overall accuracy, macro F1, per-class
accuracy, and a confusion matrix (saved as PNG + CSV).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.hasy import HASYSymbolDataset
from src.data.preprocessing import get_transforms
from src.models.classifier import SymbolClassifier
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained symbol classifier")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def bootstrap_accuracy_ci(
    correct: np.ndarray, n_bootstrap: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """95% CI for accuracy via resampling `correct` (a 0/1 array) with replacement.

    HASYv2 test support for the rarest classes here is under 10 examples, so
    a single point-estimate accuracy (e.g. "40%") is easy to over-read: one
    flipped prediction swings it by 10 points. This makes that uncertainty
    explicit instead of reporting a false-precision percentage.
    """
    if len(correct) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(correct, size=len(correct), replace=True)
        boot_means[i] = sample.mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    all_preds, all_labels, all_confidences = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confidences.extend(confidences.cpu().tolist())
    return all_preds, all_labels, all_confidences


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir)

    with open(run_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)
    class_names = class_mapping["class_names"]
    image_size = class_mapping["image_size"]
    # Same historical default as SymbolPredictor.load -- see its comment.
    content_crop = class_mapping.get("content_crop", True)

    device = resolve_device(args.device)
    model = SymbolClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model = model.to(device)

    split_df = pd.read_csv(run_dir / f"{args.split}_split.csv")
    transform = get_transforms(image_size, train=False, content_crop=content_crop)
    dataset = HASYSymbolDataset(split_df, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds, labels, confidences = collect_predictions(model, loader, device)
    preds_arr, labels_arr = np.array(preds), np.array(labels)
    correct_arr = (preds_arr == labels_arr).astype(int)

    accuracy = float(correct_arr.mean())
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    acc_ci_lo, acc_ci_hi = bootstrap_accuracy_ci(correct_arr)
    logger.info(
        "%s accuracy=%.4f (95%% CI %.4f-%.4f)  macro_f1=%.4f  weighted_f1=%.4f  n=%d",
        args.split, accuracy, acc_ci_lo, acc_ci_hi, macro_f1, weighted_f1, len(labels),
    )

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    per_class_acc = {}
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        if support == 0:
            per_class_acc[name] = {
                "accuracy": None, "ci_low": None, "ci_high": None, "support": 0,
                "precision": None, "recall": None, "f1": None,
            }
            logger.info("  %-10s acc=None  support=0", name)
            continue
        class_correct = correct_arr[labels_arr == i]
        acc = float(class_correct.mean())
        ci_lo, ci_hi = bootstrap_accuracy_ci(class_correct)
        per_class_acc[name] = {
            "accuracy": acc, "ci_low": ci_lo, "ci_high": ci_hi, "support": support,
            "precision": float(precisions[i]), "recall": float(recalls[i]), "f1": float(f1s[i]),
        }
        flag = "  <- low support, wide CI" if support < 15 else ""
        logger.info(
            "  %-10s acc=%.4f (95%% CI %.4f-%.4f)  prec=%.3f rec=%.3f f1=%.3f  support=%3d%s",
            name, acc, ci_lo, ci_hi, precisions[i], recalls[i], f1s[i], support, flag,
        )

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / f"{args.split}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({args.split})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(run_dir / f"{args.split}_confusion_matrix.png", dpi=150)
    logger.info("Saved confusion matrix to %s", run_dir / f"{args.split}_confusion_matrix.png")

    report = {
        "split": args.split,
        "accuracy": accuracy,
        "accuracy_ci_95": [acc_ci_lo, acc_ci_hi],
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_acc,
        "n_examples": len(labels),
    }
    with open(run_dir / f"{args.split}_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", run_dir / f"{args.split}_evaluation.json")


if __name__ == "__main__":
    main()
