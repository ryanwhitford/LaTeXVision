"""Expression IR -> LaTeX string (Phase 5).

Deliberately the only place that knows about LaTeX syntax -- everything
upstream (components.py, spatial_relations.py, expression_tree.py) deals in
geometry and classified symbols, never strings to emit. Swapping the output
format (e.g. MathML) later means changing only this file.
"""

from __future__ import annotations

from src.recognition.expression_tree import DIGIT_CLASSES, Expression, Operator, Term


def _is_unmodified_digit_term(element: object) -> bool:
    """A bare digit with no superscript/subscript -- a "continuing" digit of
    a multi-digit number. A digit that already picked up a modifier ends the
    number (e.g. "3" then "6^2" is "36^2", the exponent applies to the whole
    number, not just the last digit in isolation -- but a hypothetical digit
    *after* a modified one would start a new, separate number)."""
    return (
        isinstance(element, Term)
        and element.base.symbol in DIGIT_CLASSES
        and element.superscript is None
        and element.subscript is None
    )


def generate_latex(expression: Expression) -> str:
    """Render an Expression as a LaTeX string, e.g. Term(x, superscript=2)
    -> 'x^{2}'. Elements are space-joined, EXCEPT consecutive digit Terms
    with no operator between them, which concatenate directly -- "3" then
    "6" is the number 36, not two separate single-digit terms ("3 6")."""
    parts = []
    for i, element in enumerate(expression):
        if isinstance(element, Operator):
            piece = element.symbol.latex
        elif isinstance(element, Term):
            piece = element.base.latex
            if element.superscript is not None:
                piece += f"^{{{element.superscript.latex}}}"
            if element.subscript is not None:
                piece += f"_{{{element.subscript.latex}}}"
        else:
            raise TypeError(f"Unknown expression element type: {type(element)}")

        if i == 0:
            parts.append(piece)
            continue

        joins_without_space = _is_unmodified_digit_term(expression[i - 1]) and (
            isinstance(element, Term) and element.base.symbol in DIGIT_CLASSES
        )
        parts.append(("" if joins_without_space else " ") + piece)

    return "".join(parts)
