"""InkML parsing and rasterization, shared by the MathWriting and CROHME
loaders.

Both datasets store handwriting as *online* ink: pen trajectories, not
pixels. Rasterizing them with a round brush is structurally the same thing
the browser canvas does (`frontend/expression.js` draws polylines with a
round cap), which makes these sources a closer domain match to real app
input than HASYv2's scanned 32x32 glyphs.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

_NS = {"ink": "http://www.w3.org/2003/InkML"}


def parse_inkml(source: str | bytes) -> tuple[list[np.ndarray], dict[str, str]]:
    """Parse an InkML document (path, or raw bytes/str content).

    Returns (traces, annotations): each trace is an (N, 2) float array of
    x, y points (extra channels such as time are dropped); annotations maps
    `annotation type=...` to its text. Handles both MathWriting's
    "x y t, x y t" and CROHME's "x y, x y" point formats.
    """
    if isinstance(source, bytes) or (isinstance(source, str) and source.lstrip().startswith("<")):
        root = ET.fromstring(source)
    else:
        root = ET.parse(source).getroot()

    annotations = {}
    for ann in root.findall("ink:annotation", _NS):
        annotations[ann.get("type", "")] = (ann.text or "").strip()

    traces = []
    for trace in root.iter("{http://www.w3.org/2003/InkML}trace"):
        points = []
        for point in (trace.text or "").split(","):
            values = point.split()
            if len(values) >= 2:
                try:
                    points.append((float(values[0]), float(values[1])))
                except ValueError:
                    continue  # e.g. CROHME's occasional "'" velocity markers
        if points:
            traces.append(np.asarray(points, dtype=np.float32))
    return traces, annotations


def render_traces(
    traces: list[np.ndarray],
    target_stroke_extent: float = 100.0,
    stroke_width_ratio: float = 0.1,
    margin: int = 20,
) -> Image.Image | None:
    """Rasterize ink into a grayscale image with black strokes on white.

    Rendered at a canvas-like scale (median stroke extent ~100px, stroke
    width ~10% of that -- matching the expression canvas's 10px stroke on
    ~100px symbols) rather than at the final model resolution. The final
    resize is done by `normalize_expression_image`, which is applied
    identically to canvas drawings at serve time -- rendering straight to
    model resolution here would make training images differ from served
    ones in exactly the way the Phase-1 audit found to be catastrophic.
    Returns None for empty ink.
    """
    traces = [t for t in traces if len(t) > 0]
    if not traces:
        return None

    extents = [max(np.ptp(t[:, 0]), np.ptp(t[:, 1])) for t in traces]
    positive = [e for e in extents if e > 0]
    median_extent = float(np.median(positive)) if positive else 1.0
    scale = target_stroke_extent / median_extent

    all_points = np.concatenate(traces) * scale
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    width = int(np.ceil(x_max - x_min)) + 2 * margin
    height = int(np.ceil(y_max - y_min)) + 2 * margin
    # Guard against pathological inks (a single stray point far away).
    if width > 20000 or height > 20000:
        return None

    image = Image.new("L", (max(width, 1), max(height, 1)), 255)
    draw = ImageDraw.Draw(image)
    stroke = max(1, int(round(target_stroke_extent * stroke_width_ratio)))
    radius = stroke / 2
    for trace in traces:
        pts = [((x * scale) - x_min + margin, (y * scale) - y_min + margin) for x, y in trace]
        if len(pts) > 1:
            draw.line(pts, fill=0, width=stroke, joint="curve")
        for x, y in (pts[0], pts[-1]):  # round caps, like the canvas's lineCap="round"
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
    return image


_DOLLAR_RE = re.compile(r"^\$+|\$+$")


def crohme_truth_to_latex(truth: str) -> str:
    """CROHME's `truth` annotation wraps LaTeX in $...$ -- strip it."""
    return _DOLLAR_RE.sub("", truth.strip()).strip()
