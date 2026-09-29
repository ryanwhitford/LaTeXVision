import pytest

from src.data.latex_tokenizer import (
    VOCAB,
    canonical_tokens,
    decode,
    encode,
    in_vocab,
    structure_skeleton,
    structure_type,
    to_latex,
    tokenize,
)


def test_vocab_is_the_agreed_small_size():
    # ~35 tokens agreed: 24 symbols + 5 structural + 4 special.
    assert len(VOCAB) == 33
    assert "\\sqrt" not in VOCAB and "=" not in VOCAB


def test_tokenize_commands_and_single_chars():
    assert tokenize("x^{2}+\\frac{a}{b}") == ["x", "^", "{", "2", "}", "+", "\\frac", "{", "a", "}", "{", "b", "}"]


@pytest.mark.parametrize(
    "a, b",
    [
        ("x^2", "x^{2}"),                     # unbraced script
        ("x_{1}^{2}", "x^{2}_{1}"),           # script order
        ("\\frac12", "\\frac{1}{2}"),         # unbraced frac args
        ("a \\lt b", "a<b"),                  # synonym
        ("\\left[ x \\right]", "[x]"),        # sizing commands
        ("\\mbox{z}+1", "z+1"),               # CROHME font wrapper
        ("{x}+{y}", "x+y"),                   # purely-grouping braces
        ("$x ^ { 2 }$", "x^{2}"),             # math-mode dollars, spacing
    ],
)
def test_equivalent_latex_canonicalizes_equal(a, b):
    assert canonical_tokens(a) == canonical_tokens(b)


def test_in_vocab_rejects_unsupported_symbols():
    assert in_vocab(canonical_tokens("x^{2}+\\frac{a}{b}"))
    assert not in_vocab(canonical_tokens("x=1"))
    assert not in_vocab(canonical_tokens("\\sqrt{x}"))


def test_encode_decode_round_trip():
    toks = canonical_tokens("\\frac{x^{2}}{3}-\\infty")
    assert decode(encode(toks)) == toks


def test_to_latex_separates_commands_from_letters():
    assert to_latex(["\\times", "x"]) == "\\times x"
    assert to_latex(["x", "^", "{", "2", "}"]) == "x^{2}"


@pytest.mark.parametrize(
    "latex, expected",
    [
        ("a+b", "flat"),
        ("x^{2}+y_{1}", "script"),
        ("x^{2^{3}}", "nested_script"),
        ("a_{b_{c}}", "nested_script"),
        ("x^{2}_{1}", "script"),  # sup + sub on one base is not nesting
        ("\\frac{a}{b}", "fraction"),
        ("\\frac{x^{2}}{y}", "fraction"),
    ],
)
def test_structure_type(latex, expected):
    assert structure_type(canonical_tokens(latex)) == expected


def test_structure_skeleton_ignores_symbol_identity():
    assert structure_skeleton(canonical_tokens("x^{2}")) == structure_skeleton(canonical_tokens("y^{3}"))
    assert structure_skeleton(canonical_tokens("x^{2}")) != structure_skeleton(canonical_tokens("x_{2}"))
