"""Expression-level test benchmark for the image-to-LaTeX transformer, on
the stored test images, using exactly the preprocessing the API serves with.

    python -m src.evaluation.expression_benchmark --out experiments/expression_benchmark/im2latex_v1

Metrics are fixed here, before the model is scored, and stratified by
source and structure type:

- exact:           canonical token sequences identical
- structure_exact: identical after replacing every symbol with a
                   placeholder (layout right, regardless of classification)
- edit_rate:       token Levenshtein distance / ground-truth length
- error bucket (for wrong outputs):
    missing_symbols  -- fewer symbols than ground truth
    extra_symbols    -- more symbols than ground truth
    wrong_symbols    -- same count, different symbol identities (recognition)
    wrong_structure  -- exactly the right symbols, wrong layout/order (structure)
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics
import time
from pathlib import Path

from PIL import Image

from src.data.expression_dataset import load_manifest
from src.data.latex_tokenizer import (
    STRUCTURAL_TOKENS,
    canonicalize,
    structure_skeleton,
)
from src.training.evaluate_classifier import bootstrap_accuracy_ci
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _symbols(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STRUCTURAL_TOKENS]


def error_bucket(pred: list[str], gold: list[str]) -> str | None:
    if pred == gold:
        return None
    ps, gs = _symbols(pred), _symbols(gold)
    if len(ps) < len(gs):
        return "missing_symbols"
    if len(ps) > len(gs):
        return "extra_symbols"
    if collections.Counter(ps) != collections.Counter(gs):
        return "wrong_symbols"
    return "wrong_structure"


def score(pred: list[str], gold: list[str]) -> dict:
    return {
        "exact": pred == gold,
        "structure_exact": structure_skeleton(pred) == structure_skeleton(gold),
        "edit_rate": edit_distance(pred, gold) / max(len(gold), 1),
        "error": error_bucket(pred, gold),
    }


class TransformerEngine:
    def __init__(self, device, run_dir: Path) -> None:
        from src.recognition.transformer_engine import TransformerRecognizer

        self.recognizer = TransformerRecognizer.load(run_dir, device=device)

    def predict(self, image: Image.Image) -> list[str]:
        result = self.recognizer.recognize(image)
        return canonicalize(result.tokens) if result else []


def summarize(rows: list[dict]) -> dict:
    exact = [int(r["exact"]) for r in rows]
    lo, hi = bootstrap_accuracy_ci(__import__("numpy").array(exact)) if exact else (float("nan"),) * 2
    errors = collections.Counter(r["error"] for r in rows if r["error"])
    return {
        "n": len(rows),
        "exact": sum(exact) / len(rows) if rows else float("nan"),
        "exact_ci95": [lo, hi],
        "structure_exact": sum(r["structure_exact"] for r in rows) / len(rows) if rows else float("nan"),
        "edit_rate": statistics.mean(r["edit_rate"] for r in rows) if rows else float("nan"),
        "errors": dict(errors),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/processed/expressions/manifest.jsonl")
    parser.add_argument("--transformer-run", type=Path, default=REPO_ROOT / "models/im2latex_v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments/expression_benchmark/im2latex_v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = resolve_device(args.device)
    records = load_manifest(args.manifest, [args.split])
    root = args.manifest.parent
    model = TransformerEngine(device, args.transformer_run)

    per_sample, latencies = [], []
    for i, rec in enumerate(records):
        image = Image.open(root / rec["image"]).convert("L")
        t0 = time.perf_counter()
        pred = model.predict(image)
        latencies.append((time.perf_counter() - t0) * 1000)
        source = rec.get("subsource", "") and f"{rec['source']}_{rec['subsource']}" or rec["source"]
        per_sample.append({"id": rec["id"], "source": source, "structure": rec["structure"], "gold": rec["tokens"],
                           "pred": pred, **score(pred, rec["tokens"])})
        if (i + 1) % 500 == 0:
            logger.info("%d / %d", i + 1, len(records))

    # Groups: ALL, each source, and "human" (every source that isn't the
    # synthetic generator) -- each crossed with ALL and each structure type.
    by = collections.defaultdict(list)
    for row in per_sample:
        sources = ["ALL", row["source"]] + (["human"] if row["source"] != "hasy_synth" else [])
        for src in sources:
            by[(src, "ALL")].append(row)
            by[(src, row["structure"])].append(row)
    summary = {
        "split": args.split,
        "latency_ms_median": statistics.median(latencies),
        "groups": {f"{s}|{t}": summarize(v) for (s, t), v in sorted(by.items())},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.split}_per_sample.jsonl").write_text("\n".join(json.dumps(r) for r in per_sample) + "\n")
    (args.out / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {"n": v["n"], "exact": round(v["exact"], 4)} for k, v in summary["groups"].items()}, indent=1))
    print("latency_ms_median:", summary["latency_ms_median"])


if __name__ == "__main__":
    main()
