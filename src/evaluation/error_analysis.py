"""Error analysis for a trained symbol classifier (optimization brief, Sections 14-15).

Produces, from a run's test split:
  - the actual worst symbol-pair confusions, ranked by count (not assumed)
  - the most confident WRONG predictions (systematic mistakes worth caring about)
  - the most uncertain predictions (near-ties between two or more classes)
  - per-class false positive / false negative counts
  - a saved image grid of representative misclassified examples

Usage:
    python -m src.evaluation.error_analysis --run-dir models/symbol_classifier_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.hasy import HASYSymbolDataset
from src.data.preprocessing import get_transforms
from src.models.classifier import SymbolClassifier
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error analysis for a trained symbol classifier")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-n-confusions", type=int, default=10)
    parser.add_argument("--top-n-examples", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir

    with open(run_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)
    class_names = class_mapping["class_names"]
    image_size = class_mapping["image_size"]
    content_crop = class_mapping.get("content_crop", True)

    device = resolve_device(args.device)
    model = SymbolClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model = model.to(device).eval()

    split_df = pd.read_csv(run_dir / f"{args.split}_split.csv").reset_index(drop=True)
    transform = get_transforms(image_size, train=False, content_crop=content_crop)
    dataset = HASYSymbolDataset(split_df, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs)  # [N, num_classes]

    preds = all_probs.argmax(dim=1)
    confidences = all_probs.max(dim=1).values
    labels = torch.tensor(split_df["label"].values)

    is_wrong = preds != labels

    # --- worst confusions, ranked by actual count ---
    confusion_pairs = Counter()
    for true_idx, pred_idx in zip(labels[is_wrong].tolist(), preds[is_wrong].tolist()):
        confusion_pairs[(class_names[true_idx], class_names[pred_idx])] += 1
    worst_confusions = confusion_pairs.most_common(args.top_n_confusions)

    logger.info("Worst confusions (true -> predicted : count), ranked by actual frequency:")
    for (true_cls, pred_cls), count in worst_confusions:
        logger.info("  %-10s -> %-10s : %d", true_cls, pred_cls, count)

    # --- most confident wrong predictions ---
    wrong_indices = is_wrong.nonzero(as_tuple=True)[0]
    wrong_confidences = confidences[wrong_indices]
    order = wrong_confidences.argsort(descending=True)
    most_confident_wrong = []
    for i in order[: args.top_n_examples].tolist():
        idx = wrong_indices[i].item()
        most_confident_wrong.append(
            {
                "path": split_df.loc[idx, "path"],
                "true": class_names[labels[idx].item()],
                "predicted": class_names[preds[idx].item()],
                "confidence": float(confidences[idx]),
            }
        )
    logger.info("Most confident WRONG predictions:")
    for ex in most_confident_wrong[:10]:
        logger.info("  true=%-10s pred=%-10s conf=%.4f  %s", ex["true"], ex["predicted"], ex["confidence"], ex["path"])

    # --- most uncertain predictions (top1 vs top2 margin, regardless of correctness) ---
    top2 = all_probs.topk(2, dim=1)
    margins = top2.values[:, 0] - top2.values[:, 1]
    uncertain_order = margins.argsort()
    most_uncertain = []
    for idx in uncertain_order[: args.top_n_examples].tolist():
        top1_idx, top2_idx = top2.indices[idx].tolist()
        most_uncertain.append(
            {
                "path": split_df.loc[idx, "path"],
                "true": class_names[labels[idx].item()],
                "top1": class_names[top1_idx],
                "top1_prob": float(top2.values[idx, 0]),
                "top2": class_names[top2_idx],
                "top2_prob": float(top2.values[idx, 1]),
                "correct": bool(preds[idx] == labels[idx]),
            }
        )
    logger.info("Most uncertain predictions (smallest top1-top2 margin):")
    for ex in most_uncertain[:10]:
        logger.info(
            "  true=%-10s  %s=%.3f vs %s=%.3f  (correct=%s)",
            ex["true"], ex["top1"], ex["top1_prob"], ex["top2"], ex["top2_prob"], ex["correct"],
        )

    # --- per-class false positives / false negatives ---
    per_class = {}
    for i, name in enumerate(class_names):
        false_negatives = int(((labels == i) & is_wrong).sum())
        false_positives = int(((preds == i) & is_wrong & (labels != i)).sum())
        class_confidences = confidences[labels == i]
        per_class[name] = {
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "mean_confidence_when_true_class": float(class_confidences.mean()) if len(class_confidences) else None,
        }

    report = {
        "split": args.split,
        "n_examples": len(split_df),
        "n_wrong": int(is_wrong.sum()),
        "worst_confusions": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in worst_confusions
        ],
        "most_confident_wrong": most_confident_wrong,
        "most_uncertain": most_uncertain,
        "per_class_fp_fn": per_class,
    }
    with open(out_dir / f"{args.split}_error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved %s", out_dir / f"{args.split}_error_analysis.json")

    # --- image grid of most-confident-wrong examples ---
    n = min(len(most_confident_wrong), args.top_n_examples)
    if n > 0:
        cols = 6
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, ex in zip(axes, most_confident_wrong[:n]):
            img = Image.open(ex["path"]).convert("L")
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{ex['true']}→{ex['predicted']}\n{ex['confidence']:.2f}", fontsize=8)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"Most confident wrong predictions ({args.split})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.split}_most_confident_wrong.png", dpi=150)
        logger.info("Saved %s", out_dir / f"{args.split}_most_confident_wrong.png")


if __name__ == "__main__":
    main()
