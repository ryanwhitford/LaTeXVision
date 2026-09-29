"""MathWriting (Google, 2024) loader: stream-filter the 3.1GB archive down to
expressions the project's vocabulary can express.

The archive holds ~630k tiny InkML files; extracting all of them to find the
few percent that fit a 33-token vocabulary is wasteful, so this reads the
.tgz as a stream and writes out only the matches.

Usage:
    python -m src.data.mathwriting --archive data/raw/mathwriting/mathwriting-2024.tgz
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import tarfile
from pathlib import Path

from src.data.inkml import parse_inkml
from src.data.latex_tokenizer import canonical_tokens, in_vocab, structure_type

logger = logging.getLogger(__name__)

# MathWriting's own split semantics: train + synthetic for training, valid
# for tuning, test for the final evaluation (split by writer AND by
# expression). `symbols/` is single glyphs, not expressions -- skipped.
KEPT_SPLITS = {"train", "synthetic", "valid", "test"}


def filter_archive(archive: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = collections.Counter()
    seen = collections.Counter()
    types = collections.defaultdict(collections.Counter)

    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".inkml"):
                continue
            parts = Path(member.name).parts
            split = parts[-2] if len(parts) >= 2 else ""
            if split not in KEPT_SPLITS:
                continue
            seen[split] += 1
            content = tar.extractfile(member).read()
            _, annotations = parse_inkml(content)
            tokens = canonical_tokens(annotations.get("normalizedLabel", ""))
            if not tokens or not in_vocab(tokens):
                continue
            (out_dir / split).mkdir(exist_ok=True)
            (out_dir / split / parts[-1]).write_bytes(content)
            kept[split] += 1
            types[split][structure_type(tokens)] += 1
            if sum(seen.values()) % 50000 == 0:
                logger.info("scanned %d, kept %d", sum(seen.values()), sum(kept.values()))

    summary = {
        split: {"scanned": seen[split], "kept": kept[split], "structure_types": dict(types[split])}
        for split in sorted(seen)
    }
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/mathwriting/mathwriting-2024.tgz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/mathwriting/filtered"))
    args = parser.parse_args()
    summary = filter_archive(args.archive, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
