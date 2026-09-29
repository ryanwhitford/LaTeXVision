from src.data.latex_tokenizer import canonical_tokens as ct
from src.evaluation.expression_benchmark import edit_distance, error_bucket, score


def test_edit_distance():
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a"], ["a", "b"]) == 1
    assert edit_distance(["a", "c"], ["a", "b"]) == 1


def test_error_buckets_follow_the_brief_taxonomy():
    gold = ct("x^{2}+y")
    assert error_bucket(gold, gold) is None
    assert error_bucket(ct("x+y"), gold) == "missing_symbols"
    assert error_bucket(ct("x^{2}+y+1"), gold) == "extra_symbols"
    assert error_bucket(ct("x^{3}+y"), gold) == "wrong_symbols"
    assert error_bucket(ct("x_{2}+y"), gold) == "wrong_structure"  # right symbols, wrong layout


def test_score_separates_structure_from_identity():
    s = score(ct("y^{3}"), ct("x^{2}"))
    assert not s["exact"] and s["structure_exact"]
    s = score(ct("x_{2}"), ct("x^{2}"))
    assert not s["structure_exact"]


def test_report_renders_every_group_present_in_the_summary():
    from src.evaluation.benchmark_report import render

    g = {"n": 2, "exact": 0.5, "exact_ci95": [0.0, 1.0], "structure_exact": 1.0, "edit_rate": 0.25,
         "errors": {"wrong_symbols": 1}}
    md = render({"split": "test", "latency_ms_median": 40.0,
                 "groups": {"ALL|ALL": g, "ALL|flat": g, "human|ALL": g, "human|flat": g}})
    assert "| All test expressions | 2 | 50.0% [0%–100%] | 100.0% | 0.250 |" in md
    assert "Human handwriting" in md and "Nested scripts" not in md
    assert "| 0 | 0 | 1 | 0 |" in md
