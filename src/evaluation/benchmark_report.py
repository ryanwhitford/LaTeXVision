"""Render the expression benchmark summary as a markdown report.

    python -m src.evaluation.benchmark_report \
        --summary experiments/expression_benchmark/im2latex_v1/test_summary.json

Numbers come straight from the benchmark's summary JSON -- nothing here
recomputes or selects metrics, so the report can't drift from the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STRUCTURES = [
    ("ALL", "All"),
    ("flat", "Flat"),
    ("script", "Scripts (single level)"),
    ("nested_script", "Nested scripts"),
    ("fraction", "Fractions"),
]
SOURCES = [
    ("ALL", "All test expressions"),
    ("human", "Human handwriting (MathWriting + CROHME)"),
    ("hasy_synth", "Synthetic (HASYv2 glyphs)"),
    ("mathwriting_human", "MathWriting (human)"),
    ("crohme", "CROHME 2014 test"),
]


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _row(label: str, g: dict) -> str:
    lo, hi = g["exact_ci95"]
    return (f"| {label} | {g['n']} | {_pct(g['exact'])} [{lo:.0%}–{hi:.0%}] | "
            f"{_pct(g['structure_exact'])} | {g['edit_rate']:.3f} |\n")


def render(summary: dict) -> str:
    groups = summary["groups"]
    head = "| | n | exact match [95% CI] | structure match | token edit rate |\n|---|---:|---:|---:|---:|\n"
    lines = ["## By source\n\n", head]
    lines += [_row(label, groups[f"{src}|ALL"]) for src, label in SOURCES if f"{src}|ALL" in groups]
    lines.append("\n## By structure (all sources)\n\n" + head)
    lines += [_row(label, groups[f"ALL|{st}"]) for st, label in STRUCTURES if f"ALL|{st}" in groups]
    lines.append("\n## By structure (human handwriting only)\n\n" + head)
    lines += [_row(label, groups[f"human|{st}"]) for st, label in STRUCTURES if f"human|{st}" in groups]
    err = groups["ALL|ALL"]["errors"]
    lines.append("\n## Error buckets (all test expressions)\n\n| missing symbols | extra symbols | wrong symbols | wrong structure |\n"
                 "|---:|---:|---:|---:|\n"
                 f"| {err.get('missing_symbols', 0)} | {err.get('extra_symbols', 0)} | "
                 f"{err.get('wrong_symbols', 0)} | {err.get('wrong_structure', 0)} |\n")
    lines.append(f"\nMedian latency per expression (CPU, batch 1): {summary['latency_ms_median']:.1f} ms\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    md = render(json.loads(args.summary.read_text()))
    if args.out:
        args.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
