from pathlib import Path

import pytest

from src.recognition.expression_tree import Operator, Term, build_expression
from src.recognition.spatial_relations import DetectedSymbol, load_parser_config

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


def test_x_squared_groups_into_one_term_with_superscript(config):
    # "x^2": x is the base, 2 is small and raised to its right.
    x = _symbol("x", 0, 0, 30, 40)
    two = _symbol("2", 32, -25, 47, -5)
    expression = build_expression([x, two], config)
    assert len(expression) == 1
    assert isinstance(expression[0], Term)
    assert expression[0].base.symbol == "x"
    assert expression[0].superscript.symbol == "2"
    assert expression[0].subscript is None


def test_y_subscript_1_groups_into_one_term_with_subscript(config):
    y = _symbol("y", 0, 0, 30, 40)
    one = _symbol("1", 32, 30, 45, 55)
    expression = build_expression([y, one], config)
    assert len(expression) == 1
    assert expression[0].base.symbol == "y"
    assert expression[0].subscript.symbol == "1"
    assert expression[0].superscript is None


def test_a_plus_b_produces_three_separate_elements(config):
    a = _symbol("a", 0, 0, 30, 40)
    plus = _symbol("plus", 35, 0, 65, 40)
    b = _symbol("b", 70, 0, 100, 40)
    expression = build_expression([a, plus, b], config)
    assert len(expression) == 3
    assert isinstance(expression[0], Term) and expression[0].base.symbol == "a"
    assert isinstance(expression[1], Operator) and expression[1].symbol.symbol == "plus"
    assert isinstance(expression[2], Term) and expression[2].base.symbol == "b"
    # plain baseline terms shouldn't pick up spurious modifiers
    assert expression[0].superscript is None and expression[0].subscript is None


def test_x_squared_plus_y_sub_1(config):
    # "x^2 + y_1" -- mixes a superscript term, an operator, and a subscript term.
    x = _symbol("x", 0, 0, 30, 40)
    two = _symbol("2", 32, -25, 47, -5)
    plus = _symbol("plus", 60, 0, 90, 40)
    y = _symbol("y", 100, 0, 130, 40)
    one = _symbol("1", 132, 30, 145, 55)
    expression = build_expression([x, two, plus, y, one], config)

    assert len(expression) == 3
    assert expression[0].base.symbol == "x" and expression[0].superscript.symbol == "2"
    assert isinstance(expression[1], Operator) and expression[1].symbol.symbol == "plus"
    assert expression[2].base.symbol == "y" and expression[2].subscript.symbol == "1"
