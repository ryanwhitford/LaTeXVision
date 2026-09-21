"""Intermediate expression representation (Phase 5).

Groups a flat, left-to-right list of classified+positioned symbols into a
small tree: a sequence of `Term`s (a base symbol with an optional
superscript/subscript) and `Operator`s (a standalone symbol like `+`).
This exists specifically so `latex_generator.py` never has to reason about
geometry, and `spatial_relations.py` never has to reason about LaTeX --
each stage only knows about the stage adjacent to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.recognition.spatial_relations import DetectedSymbol, ParserConfig, SpatialRelation, classify_relation

# Classes in configs/dataset.yaml that are operators rather than things a
# superscript/subscript would attach to. Brackets, digits, variables, sqrt,
# and infty can all plausibly be a Term's base; operators can't meaningfully
# carry a superscript/subscript in this vocabulary, so they're never treated
# as a modifier-attachment target.
OPERATOR_CLASSES = {"plus", "minus", "times", "lt", "gt"}

# Digit class names -- consecutive unmodified digit Terms with no operator
# between them are one multi-digit number (e.g. "3" then "6" -> "36"), not
# two separate terms. Used by latex_generator.py to decide when to omit the
# space it otherwise puts between elements. Kept here rather than in
# latex_generator.py because it's a fact about the symbol vocabulary
# (expression_tree.py's domain), not about LaTeX syntax.
DIGIT_CLASSES = {str(d) for d in range(10)}


@dataclass
class Term:
    base: DetectedSymbol
    superscript: DetectedSymbol | None = None
    subscript: DetectedSymbol | None = None


@dataclass
class Operator:
    symbol: DetectedSymbol


ExpressionElement = Term | Operator
Expression = list[ExpressionElement]


def build_expression(symbols: list[DetectedSymbol], config: ParserConfig) -> Expression:
    """Group `symbols` (already left-to-right by x-center) into an Expression.

    Greedy, single left-to-right pass: for each not-yet-used symbol (a
    candidate "base"), scan forward for a superscript and/or subscript
    belonging to it, stopping the scan as soon as a same-baseline symbol is
    hit (that's the next base, not a modifier of this one). This is the
    project's documented "simplest appropriate approach" -- it handles the
    common cases (x^2, y_1, a+b+c) directly; chained modifiers (x^2^3) and
    modifiers separated from their base by noise aren't handled, and aren't
    expected to come up with this vocabulary.
    """
    n = len(symbols)
    used = [False] * n
    elements: Expression = []

    for i in range(n):
        if used[i]:
            continue
        base = symbols[i]
        superscript: DetectedSymbol | None = None
        subscript: DetectedSymbol | None = None

        for j in range(i + 1, n):
            if used[j]:
                continue
            relation = classify_relation(base, symbols[j], config)
            if relation == SpatialRelation.SUPERSCRIPT and superscript is None:
                superscript = symbols[j]
                used[j] = True
            elif relation == SpatialRelation.SUBSCRIPT and subscript is None:
                subscript = symbols[j]
                used[j] = True
            elif relation == SpatialRelation.SAME_BASELINE:
                break  # reached the next base; stop looking for this one's modifiers

        if base.symbol in OPERATOR_CLASSES:
            elements.append(Operator(symbol=base))
        else:
            elements.append(Term(base=base, superscript=superscript, subscript=subscript))

    return elements
