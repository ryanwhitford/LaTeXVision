"""Deterministic, geometry-based spatial-relationship inference (Phase 4).

Given two classified, positioned symbols, decides whether one is a
superscript/subscript of the other or whether they simply share a baseline
(are reading-order neighbors). All thresholds live in configs/parser.yaml,
not in this code -- see that file's comments for what each one means and
why.

NUMERATOR_OF / DENOMINATOR_OF (fractions) and INSIDE (matched brackets) are
intentionally not implemented yet -- fractions are a deliberately deferred,
separate feature (they need fraction-bar detection, not just symbol
geometry), and bracket matching needs pairing logic this module doesn't do.
Both are natural extensions of `SpatialRelation`/`classify_relation` when
that work starts; nothing here needs to change shape to add them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class SpatialRelation(str, Enum):
    SUPERSCRIPT = "SUPERSCRIPT"
    SUBSCRIPT = "SUBSCRIPT"
    SAME_BASELINE = "SAME_BASELINE"
    UNRELATED = "UNRELATED"  # neither adjacent enough nor vertically offset in a meaningful way


@dataclass
class DetectedSymbol:
    """A classified, positioned symbol -- the unit spatial_relations.py and
    expression_tree.py operate on."""

    symbol: str
    latex: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    width: int
    height: int


@dataclass
class ParserConfig:
    superscript_min_vertical_offset_ratio: float
    superscript_max_size_ratio: float
    superscript_max_horizontal_gap_ratio: float
    subscript_min_vertical_offset_ratio: float
    subscript_max_size_ratio: float
    subscript_max_horizontal_gap_ratio: float
    same_baseline_max_vertical_center_diff_ratio: float
    same_baseline_max_horizontal_gap_ratio: float


def load_parser_config(config_path: str | Path) -> ParserConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ParserConfig(
        superscript_min_vertical_offset_ratio=raw["superscript"]["min_vertical_offset_ratio"],
        superscript_max_size_ratio=raw["superscript"]["max_size_ratio"],
        superscript_max_horizontal_gap_ratio=raw["superscript"]["max_horizontal_gap_ratio"],
        subscript_min_vertical_offset_ratio=raw["subscript"]["min_vertical_offset_ratio"],
        subscript_max_size_ratio=raw["subscript"]["max_size_ratio"],
        subscript_max_horizontal_gap_ratio=raw["subscript"]["max_horizontal_gap_ratio"],
        same_baseline_max_vertical_center_diff_ratio=raw["same_baseline"]["max_vertical_center_diff_ratio"],
        same_baseline_max_horizontal_gap_ratio=raw["same_baseline"]["max_horizontal_gap_ratio"],
    )


def classify_relation(base: DetectedSymbol, candidate: DetectedSymbol, config: ParserConfig) -> SpatialRelation:
    """Classify `candidate`'s relationship to `base`, assuming `candidate` is
    at or after `base` in left-to-right reading order.

    Mirrors the project brief's rules directly:
    - candidate significantly above base, smaller, and horizontally close
      -> SUPERSCRIPT
    - candidate significantly below base, smaller, and horizontally close
      -> SUBSCRIPT
    - candidate roughly on the same vertical center as base -> SAME_BASELINE
    - otherwise -> UNRELATED (too far away, or an ambiguous vertical offset
      that isn't confidently either a modifier or a baseline neighbor)
    """
    vertical_offset = base.center[1] - candidate.center[1]  # positive = candidate is above base
    size_ratio = candidate.height / base.height if base.height else 1.0
    horizontal_gap = candidate.bbox[0] - base.bbox[2]  # negative = horizontal overlap

    # Horizontal-gap thresholds scale off base.height, not base.width: many
    # symbols in this vocabulary have a degenerate width relative to their
    # height (e.g. '1', '-', 'l'-shaped strokes), which would otherwise make
    # the modifier-attachment window unrealistically tiny for exactly the
    # symbols most likely to have a real superscript/subscript. Height is a
    # much more stable proxy for a symbol's overall scale.
    vertical_center_diff_ratio = abs(vertical_offset) / ((base.height + candidate.height) / 2 or 1)
    horizontal_gap_ratio = horizontal_gap / (base.height or 1)
    if (
        vertical_center_diff_ratio <= config.same_baseline_max_vertical_center_diff_ratio
        and horizontal_gap_ratio <= config.same_baseline_max_horizontal_gap_ratio
    ):
        return SpatialRelation.SAME_BASELINE

    if (
        vertical_offset > config.superscript_min_vertical_offset_ratio * base.height
        and size_ratio <= config.superscript_max_size_ratio
        and horizontal_gap <= config.superscript_max_horizontal_gap_ratio * base.height
    ):
        return SpatialRelation.SUPERSCRIPT

    if (
        vertical_offset < -config.subscript_min_vertical_offset_ratio * base.height
        and size_ratio <= config.subscript_max_size_ratio
        and horizontal_gap <= config.subscript_max_horizontal_gap_ratio * base.height
    ):
        return SpatialRelation.SUBSCRIPT

    return SpatialRelation.UNRELATED
