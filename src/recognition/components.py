"""Symbol localization via connected-component analysis (Phase 3).

Deliberately the simplest appropriate approach, per the project's phased
plan: connected components, not a learned detector. Finds candidate symbol
regions in a full canvas image so each can be cropped and classified
independently by the existing `SymbolClassifier`. Swappable for a learned
detector later without touching anything downstream (spatial_relations.py
and expression_tree.py only depend on the `CandidateRegion` shape, not on
how it was produced).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class CandidateRegion:
    """A candidate symbol region found in a larger canvas, ready to classify."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in the source image
    center: tuple[float, float]
    width: int
    height: int
    crop: Image.Image  # cropped from the source image at `bbox`, grayscale


def _boxes_should_merge(a: list[int], b: list[int], gap_ratio: float, min_size_ratio: float) -> bool:
    """True if boxes `a` and `b` are close enough (both horizontally and
    vertically) that they're probably strokes of the same symbol rather than
    two different symbols -- e.g. a stroke drawn as two disconnected
    segments. The gap tolerance scales with the smaller box's size so it
    doesn't depend on absolute pixel scale.

    Requires the two boxes to be roughly SIMILAR IN SIZE too. This is what
    keeps a tightly-drawn superscript/subscript from being merged into its
    base: a genuine split-stroke fragment of one symbol is close in size to
    the rest of that symbol, but a modifier is -- definitionally --
    substantially smaller than its base. Without this check, "x" with a
    small "2" drawn close to its upper-right (completely normal handwriting)
    gets swallowed into one box before Phase 4 ever sees two symbols to
    relate; that's Phase 4's job, and this function shouldn't preempt it.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    h_gap = max(bx1 - ax2, ax1 - bx2, 0)
    v_gap = max(by1 - ay2, ay1 - by2, 0)
    min_size = min(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1)
    threshold = max(min_size * gap_ratio, 1)

    a_scale = max(ax2 - ax1, ay2 - ay1)
    b_scale = max(bx2 - bx1, by2 - by1)
    size_ratio = min(a_scale, b_scale) / max(a_scale, b_scale)

    return h_gap < threshold and v_gap < threshold and size_ratio >= min_size_ratio


def _merge_nearby_boxes(boxes: list[list[int]], gap_ratio: float, min_size_ratio: float) -> list[list[int]]:
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _boxes_should_merge(boxes[i], boxes[j], gap_ratio, min_size_ratio):
                    x1 = min(boxes[i][0], boxes[j][0])
                    y1 = min(boxes[i][1], boxes[j][1])
                    x2 = max(boxes[i][2], boxes[j][2])
                    y2 = max(boxes[i][3], boxes[j][3])
                    boxes[i] = [x1, y1, x2, y2]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def detect_candidate_regions(
    image: Image.Image,
    ink_threshold: int = 250,
    min_area: int = 12,
    merge_gap_ratio: float = 0.25,
    merge_min_size_ratio: float = 0.6,
) -> list[CandidateRegion]:
    """Find candidate symbol regions in a (possibly multi-symbol) canvas image.

    Pipeline: threshold to binary ink -> connected components -> drop
    tiny-area noise -> merge boxes that are close enough AND similar enough
    in size to plausibly be one symbol drawn as disconnected strokes (see
    `_boxes_should_merge`) -> return crops sorted left-to-right by
    horizontal center (a first-pass reading order; spatial_relations.py does
    the real work of grouping superscripts/subscripts with their base rather
    than treating them as separate reading-order items).

    `merge_gap_ratio` was previously 0.5 and merged adjacent digits of the
    same multi-digit number (e.g. "36") into one unclassifiable blob -- their
    natural handwriting gap was well within that tolerance. Lowered to 0.25.
    `merge_min_size_ratio` is new: without it, a tightly (and completely
    normally) drawn superscript/subscript gets absorbed into its base before
    Phase 4 ever sees two separate symbols to relate.
    """
    gray = np.array(image.convert("L"))
    binary = (gray < ink_threshold).astype(np.uint8)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    if not boxes:
        return []

    boxes = _merge_nearby_boxes(boxes, gap_ratio=merge_gap_ratio, min_size_ratio=merge_min_size_ratio)
    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)

    regions = []
    for x1, y1, x2, y2 in boxes:
        crop = image.convert("L").crop((x1, y1, x2, y2))
        regions.append(
            CandidateRegion(
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) / 2, (y1 + y2) / 2),
                width=x2 - x1,
                height=y2 - y1,
                crop=crop,
            )
        )
    return regions
