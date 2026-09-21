"""Expression IR -> LaTeX string (Phase 5).

Deliberately the only place that knows about LaTeX syntax -- everything
upstream (components.py, spatial_relations.py, expression_tree.py) deals in
geometry and classified symbols, never strings to emit. Swapping the output
format (e.g. MathML) later means changing only this file.
"""

from __future__ import annotations

from src.recognition.expression_tree import Expression, Operator, Term


def generate_latex(expression: Expression) -> str:
    """Render an Expression as a LaTeX string, e.g. Term(x, superscript=2)
    -> 'x^{2}'; a sequence of elements is space-joined."""
    parts = []
    for element in expression:
        if isinstance(element, Operator):
            parts.append(element.symbol.latex)
        elif isinstance(element, Term):
            piece = element.base.latex
            if element.superscript is not None:
                piece += f"^{{{element.superscript.latex}}}"
            if element.subscript is not None:
                piece += f"_{{{element.subscript.latex}}}"
            parts.append(piece)
        else:
            raise TypeError(f"Unknown expression element type: {type(element)}")
    return " ".join(parts)
