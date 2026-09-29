"""Synthetic multi-symbol expression generator.

Composes real HASYv2 handwritten glyphs into multi-symbol expressions with
exact LaTeX ground truth: flat sequences, superscripts/subscripts, NESTED
scripts, and fractions (no square roots -- out of scope). A grammar builds
an expression tree; a small box-layout engine places each glyph image.

Glyphs are drawn only from the production classifier's own
train/val/test split (models/symbol_classifier_v1/*_split.csv), so a
test-split expression never contains a glyph the classifier -- or the
transformer's warm-started trunk -- was trained on. Layout ranges live in
configs/synthetic_expressions.yaml; see the note there on why they're
deliberately wide.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter

from src.data.latex_tokenizer import canonicalize, structure_type

CLASS_TO_LATEX = {
    **{str(d): str(d) for d in range(10)},
    **{v: v for v in ["a", "b", "c", "x", "y", "z"]},
    "plus": "+", "minus": "-", "times": "\\times", "lt": "<", "gt": ">",
    "lbracket": "[", "rbracket": "]", "infty": "\\infty",
}
DIGITS = [str(d) for d in range(10)]
VARIABLES = ["a", "b", "c", "x", "y", "z"]
OPERATORS = ["plus", "minus", "times", "lt", "gt"]

# Glyph long side relative to the nominal glyph size, per class.
_CLASS_SCALE = {"plus": 0.75, "times": 0.7, "lt": 0.75, "gt": 0.75, "minus": 0.7, "infty": 0.9,
                "lbracket": 1.25, "rbracket": 1.25}


# ---------------------------------------------------------------- tree ----

@dataclass
class Glyph:
    cls: str


@dataclass
class Row:
    items: list


@dataclass
class Script:
    base: object
    sup: Row | None
    sub: Row | None


@dataclass
class Frac:
    num: Row
    den: Row


@dataclass
class Bracketed:
    inner: Row


def to_tokens(node) -> list[str]:
    if isinstance(node, Glyph):
        return [CLASS_TO_LATEX[node.cls]]
    if isinstance(node, Row):
        return [t for item in node.items for t in to_tokens(item)]
    if isinstance(node, Script):
        out = to_tokens(node.base)
        if node.sup is not None:
            out += ["^", "{", *to_tokens(node.sup), "}"]
        if node.sub is not None:
            out += ["_", "{", *to_tokens(node.sub), "}"]
        return out
    if isinstance(node, Frac):
        return ["\\frac", "{", *to_tokens(node.num), "}", "{", *to_tokens(node.den), "}"]
    if isinstance(node, Bracketed):
        return ["[", *to_tokens(node.inner), "]"]
    raise TypeError(node)


# ------------------------------------------------------------- grammar ----

class Grammar:
    """Random expression trees; `sample(target)` rejection-samples until the
    tree's canonical tokens fall in the requested structure bucket, so the
    dataset's structure mix is exact rather than emergent."""

    def __init__(self, cfg: dict, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng

    def sample(self, target: str) -> Row:
        for _ in range(1000):
            tree = self._row(0, self.cfg["max_top_level_terms"], target)
            tokens = canonicalize(to_tokens(tree))
            if structure_type(tokens) == target and len(tokens) <= self.cfg["max_tokens"]:
                return tree
        raise RuntimeError(f"could not sample a '{target}' expression")

    def _row(self, depth: int, max_terms: int, mode: str) -> Row:
        n = self.rng.randint(1, max_terms)
        items = []
        for k in range(n):
            if k > 0:
                items.append(Glyph(self.rng.choice(OPERATORS)))
            items.append(self._term(depth, mode))
        return Row(items)

    def _operand(self, depth: int, mode: str):
        r = self.rng.random()
        if r < self.cfg["p_infty"]:
            return Glyph("infty")
        if r < self.cfg["p_infty"] + self.cfg["p_bracket_group"] and depth < self.cfg["max_depth"]:
            return Bracketed(self._row(depth + 1, self.cfg["max_inner_terms"], "flat"))
        if self.rng.random() < 0.45:
            n_digits = self.rng.randint(1, self.cfg["max_number_digits"])
            return Row([Glyph(self.rng.choice(DIGITS)) for _ in range(n_digits)])
        return Glyph(self.rng.choice(VARIABLES))

    def _term(self, depth: int, mode: str):
        at_limit = depth >= self.cfg["max_depth"]
        if mode == "fraction" and not at_limit and self.rng.random() < 0.5:
            inner = "script" if self.rng.random() < 0.3 else "flat"
            return Frac(self._row(depth + 1, self.cfg["max_inner_terms"], inner),
                        self._row(depth + 1, self.cfg["max_inner_terms"], inner))
        p_script = {"flat": 0.0, "script": 0.6, "nested_script": 0.7, "fraction": 0.2}[mode]
        operand = self._operand(depth, mode)
        if at_limit or self.rng.random() >= p_script:
            return operand
        inner_mode = "script" if (mode == "nested_script" and self.rng.random() < 0.75) else "flat"
        which = self.rng.random()
        sup = self._row(depth + 1, self.cfg["max_inner_terms"], inner_mode) if which < 0.7 else None
        sub = self._row(depth + 1, self.cfg["max_inner_terms"], inner_mode) if which >= 0.5 else None
        return Script(operand, sup, sub)


# -------------------------------------------------------------- layout ----

@dataclass
class Box:
    img: np.ndarray  # uint8, 255 = background
    axis: float      # y of the box's horizontal center line (what neighbors align to)


def _blank(h: int, w: int) -> np.ndarray:
    return np.full((max(h, 1), max(w, 1)), 255, dtype=np.uint8)


def _paste(canvas: np.ndarray, img: np.ndarray, top: int, left: int) -> None:
    h, w = img.shape
    region = canvas[top : top + h, left : left + w]
    np.minimum(region, img[: region.shape[0], : region.shape[1]], out=region)


class Renderer:
    def __init__(self, pools: dict[str, list[np.ndarray]], cfg: dict, rng: random.Random) -> None:
        self.pools = pools
        self.cfg = cfg
        self.rng = rng

    def _u(self, key: str) -> float:
        lo, hi = self.cfg[key]
        return self.rng.uniform(lo, hi)

    def _glyph(self, cls: str, size: float, long_side: float | None = None) -> Box:
        crop = self.rng.choice(self.pools[cls])
        h, w = crop.shape
        target = long_side if long_side is not None else size * _CLASS_SCALE.get(cls, 1.0) * self._u("glyph_scale_jitter")
        scale = target / max(h, w)
        img = np.array(Image.fromarray(crop).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS))
        return Box(img, img.shape[0] / 2)

    def _hconcat(self, boxes: list[Box], size: float, gaps: list[float] | None = None) -> Box:
        if gaps is None:
            gaps = [self._u("gap") * size for _ in boxes[1:]]
        tops = [-b.axis + self.rng.uniform(-1, 1) * self.cfg["vertical_jitter"] * size for b in boxes]
        min_top = min(tops)
        height = int(round(max(t + b.img.shape[0] for t, b in zip(tops, boxes)) - min_top))
        width = int(round(sum(b.img.shape[1] for b in boxes) + sum(max(g, 0) for g in gaps)))
        canvas = _blank(height + 1, width + 1)
        x = 0.0
        for i, (t, b) in enumerate(zip(tops, boxes)):
            _paste(canvas, b.img, int(round(t - min_top)), int(round(x)))
            x += b.img.shape[1] + (max(gaps[i], 0) if i < len(gaps) else 0)
        return Box(canvas, -min_top)

    def render(self, node, size: float) -> Box:
        if isinstance(node, Glyph):
            return self._glyph(node.cls, size)
        if isinstance(node, Row):
            boxes = [self.render(item, size) for item in node.items]
            # Digits of one number sit closer together than separate terms.
            gaps = []
            for a, b in zip(node.items, node.items[1:]):
                both_digits = isinstance(a, Glyph) and isinstance(b, Glyph) and a.cls in DIGITS and b.cls in DIGITS
                gaps.append(self.rng.uniform(0.08, 0.22) * size if both_digits else self._u("gap") * size)
            return self._hconcat(boxes, size, gaps) if len(boxes) > 1 else boxes[0]
        if isinstance(node, Script):
            return self._script(node, size)
        if isinstance(node, Frac):
            return self._frac(node, size)
        if isinstance(node, Bracketed):
            inner = self.render(node.inner, size)
            h = inner.img.shape[0] * 1.15
            left = self._glyph("lbracket", size, long_side=max(h, size))
            right = self._glyph("rbracket", size, long_side=max(h, size))
            left.axis, right.axis = left.img.shape[0] / 2, right.img.shape[0] / 2
            inner_centered = Box(inner.img, inner.img.shape[0] / 2)
            return self._hconcat([left, inner_centered, right], size, [0.1 * size, 0.1 * size])
        raise TypeError(node)

    def _script(self, node: Script, size: float) -> Box:
        base = self.render(node.base, size)
        s = size * self._u("script_scale")
        sup = self.render(node.sup, s) if node.sup is not None else None
        sub = self.render(node.sub, s) if node.sub is not None else None
        base_h, base_w = base.img.shape
        x_script = base_w + self._u("script_gap") * size

        parts = [(base.img, 0.0, 0.0)]  # (img, top, left)
        sup_bottom = None
        if sup is not None:
            center = self._u("superscript_center") * base_h
            top = center - sup.img.shape[0] / 2
            parts.append((sup.img, top, x_script))
            sup_bottom = top + sup.img.shape[0]
        if sub is not None:
            center = base_h + self._u("subscript_center") * base_h
            top = center - sub.img.shape[0] / 2
            if sup_bottom is not None:
                # Keep a stacked sup/sub pair from colliding -- overlapping
                # ink would make the ground-truth label unreadable.
                top = max(top, sup_bottom + 0.1 * s)
            parts.append((sub.img, top, x_script))

        min_top = min(p[1] for p in parts)
        min_left = min(p[2] for p in parts)
        height = int(round(max(p[1] + p[0].shape[0] for p in parts) - min_top))
        width = int(round(max(p[2] + p[0].shape[1] for p in parts) - min_left))
        canvas = _blank(height + 1, width + 1)
        for img, top, left in parts:
            _paste(canvas, img, int(round(top - min_top)), int(round(left - min_left)))
        return Box(canvas, base.axis - min_top)

    def _frac(self, node: Frac, size: float) -> Box:
        p = size * self._u("fraction_part_scale")
        num, den = self.render(node.num, p), self.render(node.den, p)
        bar_w = max(num.img.shape[1], den.img.shape[1]) * self._u("fraction_bar_overhang")
        bar = self._glyph("minus", size, long_side=bar_w)
        bar_img = bar.img
        max_bar_h = max(2, int(0.15 * size))
        if bar_img.shape[0] > max_bar_h:
            bar_img = np.array(Image.fromarray(bar_img).resize((bar_img.shape[1], max_bar_h), Image.LANCZOS))
        gap = self._u("fraction_gap") * size
        width = int(round(max(bar_img.shape[1], num.img.shape[1], den.img.shape[1])))
        y_bar = num.img.shape[0] + gap
        y_den = y_bar + bar_img.shape[0] + gap
        canvas = _blank(int(round(y_den + den.img.shape[0])) + 1, width + 1)
        _paste(canvas, num.img, 0, (width - num.img.shape[1]) // 2)
        _paste(canvas, bar_img, int(round(y_bar)), (width - bar_img.shape[1]) // 2)
        _paste(canvas, den.img, int(round(y_den)), (width - den.img.shape[1]) // 2)
        return Box(canvas, y_bar + bar_img.shape[0] / 2)


# ---------------------------------------------------------- glyph pools ----

def _ink_crop(path: Path) -> np.ndarray | None:
    arr = np.array(Image.open(path).convert("L"))
    rows, cols = np.where(arr < 250)
    if len(rows) == 0:
        return None
    return arr[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def load_glyph_pools(split_csv: Path, repo_root: Path) -> dict[str, list[np.ndarray]]:
    """Ink-cropped glyph images per class from one classifier split CSV."""
    df = pd.read_csv(split_csv)
    pools: dict[str, list[np.ndarray]] = {}
    for cls, path in zip(df["class_name"], df["path"]):
        if cls not in CLASS_TO_LATEX:
            continue  # e.g. 'sqrt' -- out of scope for expressions
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        crop = _ink_crop(p)
        if crop is not None:
            pools.setdefault(str(cls), []).append(crop)
    missing = set(CLASS_TO_LATEX) - set(pools)
    if missing:
        raise ValueError(f"{split_csv} has no glyphs for classes {sorted(missing)}")
    return pools


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate(
    pools: dict[str, list[np.ndarray]], cfg: dict, n: int, seed: int
):
    """Yield (image, canonical_tokens, structure_type) for n samples,
    cycling through structure types in proportion to cfg['structure_mix']."""
    rng = random.Random(seed)
    grammar = Grammar(cfg["grammar"], rng)
    renderer = Renderer(pools, cfg["layout"], rng)
    layout = cfg["layout"]
    mix = cfg["structure_mix"]
    # Exact per-type counts, shuffled -- so any prefix of the stream is mixed
    # and the full split matches the configured proportions.
    names = list(mix)
    counts = [int(round(mix[t] * n)) for t in names]
    counts[0] += n - sum(counts)
    targets = [t for t, c in zip(names, counts) for _ in range(c)]
    rng.shuffle(targets)
    for target in targets:
        tree = grammar.sample(target)
        box = renderer.render(tree, layout["glyph_size"])
        margin = int(layout["glyph_size"] * 0.5)
        img = np.pad(box.img, margin, constant_values=255)
        pil = Image.fromarray(img)
        up = layout["output_upscale"]
        pil = pil.resize((pil.width * up, pil.height * up), Image.BILINEAR).filter(ImageFilter.MinFilter(3))
        # Resampling leaves faint gray halos around strokes that a real
        # canvas never produces; at an ink threshold of <250 they become
        # detached phantom marks ("a" -> "a-"). Real ink renders have zero
        # such pixels, so clear them to pure background.
        arr = np.array(pil)
        arr[arr >= 200] = 255
        pil = Image.fromarray(arr)
        tokens = canonicalize(to_tokens(tree))
        yield pil, tokens, structure_type(tokens)
