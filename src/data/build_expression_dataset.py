"""Build the combined expression dataset (image -> LaTeX) from three sources.

    python -m src.data.build_expression_dataset

Sources, all filtered to the 33-token vocabulary (src/data/latex_tokenizer.py):

- hasy_synth: synthetic compositions of HASYv2 glyphs, drawn per split
  from the pretrained encoder classifier's own split CSVs.
- mathwriting: Google's MathWriting (2024), human + synthetic inks.
  train+synthetic -> train, valid -> val, test -> test (their split is by
  writer and by expression).
- crohme: CROHME 2019 package. Train_2014 + the 2012/2013 test sets ->
  train; the CROHME 2014 test set (the package's "valid" folder, the
  standard benchmark split in the literature) -> test.

Images are stored at canvas-like scale (what the app actually receives),
NOT pre-normalized: training, evaluation and serving all apply the same
normalization (src/data/expression_images.py) to them at load time.

Output: data/processed/expressions/images/<source>/<split>/<id>.png and
data/processed/expressions/manifest.jsonl (one record per sample).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

from src.data.inkml import crohme_truth_to_latex, parse_inkml, render_traces
from src.data.latex_tokenizer import canonical_tokens, in_vocab, structure_type, to_latex
from src.data.synthetic_expressions import generate, load_config, load_glyph_pools

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CROHME_DIR = REPO_ROOT / "data/raw/crohme2019/Task1_and_Task2/Task1_and_Task2/Task1_onlineRec/MainTask_formula"
MATHWRITING_DIR = REPO_ROOT / "data/raw/mathwriting/filtered"
CLASSIFIER_RUN = REPO_ROOT / "models/symbol_classifier_v1"


def _record(out_root: Path, source: str, split: str, sample_id: str, image, tokens: list[str], **extra) -> dict:
    rel = Path("images") / source / split / f"{sample_id}.png"
    (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_root / rel)
    return {
        "id": f"{source}/{sample_id}",
        "source": source,
        "split": split,
        "image": str(rel),
        "latex": to_latex(tokens),
        "tokens": tokens,
        "structure": structure_type(tokens),
        **extra,
    }


def build_hasy_synth(out_root: Path, cfg_path: Path) -> list[dict]:
    cfg = load_config(cfg_path)
    records = []
    for split, seed in (("train", 0), ("val", 1), ("test", 2)):
        pools = load_glyph_pools(CLASSIFIER_RUN / f"{split}_split.csv", REPO_ROOT)
        n = cfg["samples"][split]
        for i, (img, tokens, _st) in enumerate(generate(pools, cfg, n, seed=seed)):
            records.append(_record(out_root, "hasy_synth", split, f"{split}_{i:06d}", img, tokens))
        logger.info("hasy_synth %s: %d", split, n)
    return records


def _ink_records(out_root: Path, source: str, files: list[tuple[Path, str]], label_of, subsource_of=None) -> list[dict]:
    records = []
    skipped = collections.Counter()
    for path, split in files:
        traces, ann = parse_inkml(str(path))
        tokens = canonical_tokens(label_of(ann))
        if not tokens or not in_vocab(tokens):
            skipped["out_of_vocab"] += 1
            continue
        img = render_traces(traces)
        if img is None:
            skipped["unrenderable"] += 1
            continue
        extra = {"subsource": subsource_of(path)} if subsource_of else {}
        records.append(_record(out_root, source, split, path.stem, img, tokens, **extra))
    logger.info("%s: kept %d, skipped %s", source, len(records), dict(skipped))
    return records


def build_mathwriting(out_root: Path) -> list[dict]:
    split_map = {"train": "train", "synthetic": "train", "valid": "val", "test": "test"}
    files = [(p, split_map[p.parent.name]) for p in sorted(MATHWRITING_DIR.glob("*/*.inkml"))]
    return _ink_records(
        out_root, "mathwriting", files,
        label_of=lambda ann: ann.get("normalizedLabel", ""),
        subsource_of=lambda p: "synthetic" if p.parent.name == "synthetic" else "human",
    )


def build_crohme(out_root: Path) -> list[dict]:
    files = [(p, "train") for p in sorted((CROHME_DIR / "train_extracted").rglob("*.inkml"))]
    files += [(p, "test") for p in sorted((CROHME_DIR / "valid_extracted").rglob("*.inkml"))]
    return _ink_records(out_root, "crohme", files, label_of=lambda ann: crohme_truth_to_latex(ann.get("truth", "")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/expressions")
    parser.add_argument("--synthetic-config", type=Path, default=REPO_ROOT / "configs/synthetic_expressions.yaml")
    parser.add_argument("--sources", nargs="+", default=["hasy_synth", "mathwriting", "crohme"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    builders = {
        "hasy_synth": lambda: build_hasy_synth(args.out, args.synthetic_config),
        "mathwriting": lambda: build_mathwriting(args.out),
        "crohme": lambda: build_crohme(args.out),
    }
    records = [r for source in args.sources for r in builders[source]()]
    with open(args.out / "manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    table = collections.Counter((r["source"], r["split"], r["structure"]) for r in records)
    summary = collections.defaultdict(dict)
    for (source, split, structure), n in sorted(table.items()):
        summary[f"{source}/{split}"][structure] = n
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
