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


def test_adjacent_digit_terms_concatenate_without_a_space():
    # Regression test: "3" then "6" (two separate detected/classified Terms,
    # e.g. digits of one multi-digit number written close together) must
    # render as "36", not "3 6" -- see bug report on multi-digit numbers.
    expression = [Term(base=_symbol("3")), Term(base=_symbol("6"))]
    assert generate_latex(expression) == "36"


def test_three_adjacent_digits_concatenate():
    expression = [Term(base=_symbol("1")), Term(base=_symbol("2")), Term(base=_symbol("3"))]
    assert generate_latex(expression) == "123"


def test_multidigit_number_with_trailing_superscript():
    # "36^2" -- the exponent on the last digit applies to the whole number
    # once rendered; digits before it still concatenate without a space.
    expression = [Term(base=_symbol("3")), Term(base=_symbol("6"), superscript=_symbol("2"))]
    assert generate_latex(expression) == "36^{2}"


def test_digit_and_variable_do_not_concatenate():
    # A digit next to a variable (e.g. "3" then "x", as in "3x" meaning
    # multiplication by juxtaposition) is NOT the same case as two digits --
    # only digit-digit adjacency is merged; this project doesn't attempt
    # implicit-multiplication semantics.
    expression = [Term(base=_symbol("3")), Term(base=_symbol("x"))]
    assert generate_latex(expression) == "3 x"


def test_digit_after_modified_digit_starts_a_new_number():
    # "3^2" then "6" -- the "3" already has a modifier, so it's not a
    # "continuing" digit; "6" starts fresh rather than merging into "3^{2}".
    expression = [Term(base=_symbol("3"), superscript=_symbol("2")), Term(base=_symbol("6"))]
    assert generate_latex(expression) == "3^{2} 6"
