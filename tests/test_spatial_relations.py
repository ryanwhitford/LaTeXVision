from pathlib import Path

import pytest

from src.recognition.spatial_relations import (
    DetectedSymbol,
    SpatialRelation,
    classify_relation,
    load_parser_config,
)

PARSER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "parser.yaml"


@pytest.fixture
def config():
    return load_parser_config(PARSER_CONFIG_PATH)


def _symbol(name: str, x1: int, y1: int, x2: int, y2: int) -> DetectedSymbol:
    return DetectedSymbol(
        symbol=name,
        latex=name,
        confidence=0.99,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2, (y1 + y2) / 2),
        width=x2 - x1,
        height=y2 - y1,
    )


def test_load_parser_config_reads_all_fields(config):
    assert config.superscript_min_vertical_offset_ratio > 0
    assert config.subscript_min_vertical_offset_ratio > 0
    assert config.same_baseline_max_vertical_center_diff_ratio > 0


def test_small_raised_symbol_to_the_right_is_superscript(config):
    # 'x' as a 40px-tall base; '2' small and raised, immediately to its right.
    base = _symbol("x", 0, 0, 30, 40)
    candidate = _symbol("2", 32, -25, 47, -5)  # smaller, well above, adjacent
    assert classify_relation(base, candidate, config) == SpatialRelation.SUPERSCRIPT


def test_small_lowered_symbol_to_the_right_is_subscript(config):
    base = _symbol("y", 0, 0, 30, 40)
    candidate = _symbol("1", 32, 30, 45, 55)  # smaller, well below, adjacent
    assert classify_relation(base, candidate, config) == SpatialRelation.SUBSCRIPT


def test_same_size_horizontally_adjacent_symbol_is_same_baseline(config):
    base = _symbol("a", 0, 0, 30, 40)
    candidate = _symbol("+", 35, 0, 65, 40)  # same height, same vertical position
    assert classify_relation(base, candidate, config) == SpatialRelation.SAME_BASELINE


def test_far_away_symbol_is_unrelated(config):
    base = _symbol("a", 0, 0, 30, 40)
    candidate = _symbol("b", 500, 0, 530, 40)  # same baseline vertically, but very far horizontally
    assert classify_relation(base, candidate, config) == SpatialRelation.UNRELATED


def test_large_raised_symbol_is_not_superscript(config):
    # Raised but NOT smaller -- shouldn't count as a superscript (e.g. two
    # separate terms stacked in an unusual layout, not a modifier).
    base = _symbol("x", 0, 0, 30, 40)
    candidate = _symbol("y", 32, -50, 62, -10)  # same size as base, well above
    assert classify_relation(base, candidate, config) != SpatialRelation.SUPERSCRIPT


def test_superscript_of_a_narrow_base_is_still_detected(config):
    # Regression test: a thin base (e.g. a '1'-like vertical stroke, width
    # << height) previously failed to pick up an adjacent superscript
    # because the horizontal-gap threshold scaled off base.width, making the
    # attachment window a few pixels wide regardless of the writing's actual
    # scale. Found via live testing in the browser (Phase 3/4 rollout);
    # thresholds now scale off base.height instead.
    base = _symbol("1", 0, 0, 10, 120)  # narrow, tall
    candidate = _symbol("2", 15, -40, 45, -10)  # small, raised, modestly offset to the right
    assert classify_relation(base, candidate, config) == SpatialRelation.SUPERSCRIPT
