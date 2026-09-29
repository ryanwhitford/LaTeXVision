"""LaTeX-aware tokenizer shared by the expression datasets, the transformer
recognizer, and the expression benchmark.

The vocabulary is deliberately small: the 24 symbol classes the CNN
encoder was pretrained on (minus `\\sqrt`, which is out of scope for
expressions), plus the structural tokens for scripts and fractions. Real
datasets (MathWriting, CROHME) are filtered down to expressions that fit
rather than growing the vocabulary, so every symbol the decoder emits is one
the warm-started encoder already learned to see.

Canonicalization matters as much as tokenization: the three data sources
write the same expression differently (`x^2` vs `x^{2}`, `\\lt` vs `<`,
`\\left[` vs `[`, stray grouping braces). Every label and every model
output passes through `canonicalize` before comparison, so a match is a
match regardless of which source's conventions produced it.
"""

from __future__ import annotations

import re

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
STRUCTURAL_TOKENS = ["^", "_", "{", "}", "\\frac"]
SYMBOL_TOKENS = (
    [str(d) for d in range(10)]
    + ["a", "b", "c", "x", "y", "z"]
    + ["+", "-", "\\times", "<", ">"]
    + ["[", "]"]
    + ["\\infty"]
)
VOCAB = SPECIAL_TOKENS + STRUCTURAL_TOKENS + SYMBOL_TOKENS
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}

_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|[{}^_]|\S")

# Tokens that carry no meaning for handwriting (sizing/spacing/math-mode,
# and font-style wrappers like CROHME's `\mbox{z}` -- an upright z is
# handwritten exactly like an italic one). Dropping a wrapper leaves its
# `{...}` as a plain grouping brace, which `_canonical` then unwraps.
_DROP = {
    "\\left", "\\right", "$", "\\displaystyle", "\\,", "\\;", "\\!", "\\ ", "\\:", "\\quad", "\\qquad",
    "\\mbox", "\\mathrm", "\\text", "\\mathit", "\\rm",
}
_SYNONYMS = {"\\lt": "<", "\\gt": ">", "\\lbrack": "[", "\\rbrack": "]", "\\dfrac": "\\frac", "\\tfrac": "\\frac"}


def tokenize(latex: str) -> list[str]:
    """Split a LaTeX string into tokens (commands, structural chars, single
    characters). No canonicalization -- see `canonicalize`."""
    return _TOKEN_RE.findall(latex)


def _read_group(tokens: list[str], i: int) -> tuple[list[str], int]:
    """Read one argument starting at index i: a balanced `{...}` group
    (returned WITHOUT its outer braces), a `\\frac` with its two arguments,
    or a single token. Returns (group_tokens, next_index)."""
    if i >= len(tokens):
        return [], i
    if tokens[i] == "{":
        depth, j = 0, i
        while j < len(tokens):
            if tokens[j] == "{":
                depth += 1
            elif tokens[j] == "}":
                depth -= 1
                if depth == 0:
                    return tokens[i + 1 : j], j + 1
            j += 1
        return tokens[i + 1 :], len(tokens)  # unbalanced: take the rest
    if tokens[i] == "\\frac":
        num, j = _read_group(tokens, i + 1)
        den, k = _read_group(tokens, j)
        return ["\\frac", "{", *_canonical(num), "}", "{", *_canonical(den), "}"], k
    return [tokens[i]], i + 1


def _canonical(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("^", "_"):
            scripts = {}
            while i < len(tokens) and tokens[i] in ("^", "_") and tokens[i] not in scripts:
                marker = tokens[i]
                group, i = _read_group(tokens, i + 1)
                scripts[marker] = _canonical(group)
            # x_{1}^{2} and x^{2}_{1} are the same expression; fix one order
            # (superscript first).
            for marker in ("^", "_"):
                if marker in scripts:
                    out += [marker, "{", *scripts[marker], "}"]
        elif tok == "\\frac":
            group, i = _read_group(tokens, i)
            out += group
        elif tok == "{":
            # A brace group that isn't a script or \frac argument is just
            # grouping -- it has no rendered meaning, so unwrap it.
            group, i = _read_group(tokens, i)
            out += _canonical(group)
        elif tok == "}":
            i += 1  # stray closing brace
        else:
            out.append(tok)
            i += 1
    return out


def canonicalize(tokens: list[str]) -> list[str]:
    """Normalize a token list so equivalent LaTeX compares equal: drop
    spacing/sizing tokens, map synonyms, brace every script and \\frac
    argument, and unwrap purely-grouping braces."""
    cleaned = [_SYNONYMS.get(t, t) for t in tokens if t not in _DROP]
    return _canonical(cleaned)


def canonical_tokens(latex: str) -> list[str]:
    return canonicalize(tokenize(latex))


def in_vocab(tokens: list[str]) -> bool:
    return all(t in TOKEN_TO_ID and t not in SPECIAL_TOKENS for t in tokens)


def encode(tokens: list[str], add_bos_eos: bool = True) -> list[int]:
    ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID[UNK]) for t in tokens]
    if add_bos_eos:
        ids = [TOKEN_TO_ID[BOS], *ids, TOKEN_TO_ID[EOS]]
    return ids


def decode(ids: list[int]) -> list[str]:
    """Ids -> tokens, stopping at <eos> and dropping <bos>/<pad>."""
    out = []
    for i in ids:
        tok = VOCAB[i]
        if tok == EOS:
            break
        if tok in (BOS, PAD):
            continue
        out.append(tok)
    return out


def to_latex(tokens: list[str]) -> str:
    """Tokens -> a LaTeX string. Commands followed by a letter need a space
    (`\\times x`, not `\\timesx`); everything else is joined directly."""
    parts = []
    for tok in tokens:
        if parts and parts[-1].startswith("\\") and parts[-1][1:].isalpha() and tok[:1].isalpha():
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


STRUCTURE_TYPES = ["flat", "script", "nested_script", "fraction"]


def structure_type(tokens: list[str]) -> str:
    """Bucket a canonical token list by structural complexity, used to
    stratify the benchmark: fraction > nested_script > script > flat."""
    if "\\frac" in tokens:
        return "fraction"
    depth_stack: list[bool] = []  # True for each open brace that belongs to a script
    pending_script = False
    for tok in tokens:
        if tok in ("^", "_"):
            if any(depth_stack):
                return "nested_script"
            pending_script = True
        elif tok == "{":
            depth_stack.append(pending_script)
            pending_script = False
        elif tok == "}":
            if depth_stack:
                depth_stack.pop()
    return "script" if ("^" in tokens or "_" in tokens) else "flat"


def structure_skeleton(tokens: list[str]) -> list[str]:
    """Replace every symbol with a placeholder, keeping only layout. Two
    outputs with equal skeletons got the *structure* right even if some
    symbols were misclassified -- separates decoding errors from
    classification errors."""
    return [t if t in STRUCTURAL_TOKENS else "S" for t in tokens]
