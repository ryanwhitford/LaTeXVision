"""End-to-end multi-symbol recognition: image -> LaTeX.

Wires together Phases 3-5: connected-component localization
(components.py), per-region classification (the existing Phase 1
`SymbolPredictor`), spatial-relationship-driven grouping
(expression_tree.py), and LaTeX generation (latex_generator.py). This is
the module the expression-writing API endpoint and frontend call; nothing
else needs to know how the pieces fit together.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.api.inference import SymbolPredictor
from src.recognition.components import detect_candidate_regions
from src.recognition.expression_tree import Expression, Operator, Term, build_expression
from src.recognition.latex_generator import generate_latex
from src.recognition.spatial_relations import DetectedSymbol, ParserConfig


@dataclass
class RecognizedExpression:
    latex: str
    symbols: list[DetectedSymbol]
    expression: Expression


def recognize_expression(
    image: Image.Image,
    predictor: SymbolPredictor,
    parser_config: ParserConfig,
    min_region_area: int = 12,
) -> RecognizedExpression:
    """Detect, classify, and assemble every symbol in `image` into LaTeX."""
    regions = detect_candidate_regions(image, min_area=min_region_area)

    detected_symbols = []
    for region in regions:
        predictions = predictor.predict(region.crop, top_k=5)
        if predictions is None:
            continue  # a region from real detected ink should always yield a prediction; skip defensively
        top = predictions[0]
        detected_symbols.append(
            DetectedSymbol(
                symbol=top.class_name,
                latex=top.latex,
                confidence=top.confidence,
                bbox=region.bbox,
                center=region.center,
                width=region.width,
                height=region.height,
            )
        )

    expression = build_expression(detected_symbols, parser_config)
    latex = generate_latex(expression)
    return RecognizedExpression(latex=latex, symbols=detected_symbols, expression=expression)


def element_roles(expression: Expression) -> list[dict]:
    """Flatten an Expression back into a per-symbol list annotated with its
    structural role, for debugging/visualization (e.g. drawing labeled boxes
    over the detected regions). Order matches reading order, not necessarily
    the original left-to-right detection order (superscripts/subscripts are
    nested under their base rather than listed as separate top-level items).
    """
    rows = []
    for element in expression:
        if isinstance(element, Operator):
            rows.append({"symbol": element.symbol, "role": "operator", "attached_to": None})
        elif isinstance(element, Term):
            rows.append({"symbol": element.base, "role": "base", "attached_to": None})
            if element.superscript is not None:
                rows.append({"symbol": element.superscript, "role": "superscript", "attached_to": element.base.symbol})
            if element.subscript is not None:
                rows.append({"symbol": element.subscript, "role": "subscript", "attached_to": element.base.symbol})
    return rows
