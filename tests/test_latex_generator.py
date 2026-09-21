from src.recognition.expression_tree import Operator, Term
from src.recognition.latex_generator import generate_latex
from src.recognition.spatial_relations import DetectedSymbol


def _symbol(name: str, latex: str | None = None) -> DetectedSymbol:
    return DetectedSymbol(
        symbol=name,
        latex=latex if latex is not None else name,
        confidence=0.99,
        bbox=(0, 0, 1, 1),
        center=(0.5, 0.5),
        width=1,
        height=1,
    )


def test_plain_term_renders_as_bare_symbol():
    expression = [Term(base=_symbol("a"))]
    assert generate_latex(expression) == "a"


def test_superscript_term_renders_with_caret_braces():
    expression = [Term(base=_symbol("x"), superscript=_symbol("2"))]
    assert generate_latex(expression) == "x^{2}"


def test_subscript_term_renders_with_underscore_braces():
    expression = [Term(base=_symbol("y"), subscript=_symbol("1"))]
    assert generate_latex(expression) == "y_{1}"


def test_superscript_and_subscript_together():
    expression = [Term(base=_symbol("x"), superscript=_symbol("2"), subscript=_symbol("i"))]
    assert generate_latex(expression) == "x^{2}_{i}"


def test_full_expression_x_squared_plus_y_sub_1():
    expression = [
        Term(base=_symbol("x"), superscript=_symbol("2")),
        Operator(symbol=_symbol("plus", latex="+")),
        Term(base=_symbol("y"), subscript=_symbol("1")),
    ]
    assert generate_latex(expression) == "x^{2} + y_{1}"


def test_uses_latex_field_not_class_name():
    # e.g. class name 'times' but latex '\times' -- the generator must emit
    # the latex, not the internal class name.
    expression = [Operator(symbol=_symbol("times", latex="\\times"))]
    assert generate_latex(expression) == "\\times"
