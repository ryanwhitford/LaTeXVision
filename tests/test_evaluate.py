import numpy as np

from src.training.evaluate_classifier import bootstrap_accuracy_ci


def test_all_correct_gives_degenerate_ci_at_one():
    correct = np.ones(20, dtype=int)
    lo, hi = bootstrap_accuracy_ci(correct)
    assert lo == 1.0
    assert hi == 1.0


def test_all_incorrect_gives_degenerate_ci_at_zero():
    correct = np.zeros(20, dtype=int)
    lo, hi = bootstrap_accuracy_ci(correct)
    assert lo == 0.0
    assert hi == 0.0


def test_ci_bounds_are_ordered_and_within_unit_interval():
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    lo, hi = bootstrap_accuracy_ci(correct)
    assert 0.0 <= lo <= hi <= 1.0


def test_small_sample_ci_is_wide():
    # With only 10 examples (roughly the 'x'/'y' test support in this
    # project's HASYv2 subset), the CI should be noticeably wider than a
    # large-sample CI at the same point estimate.
    small = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # 60% accuracy, n=10
    large = np.tile(small, 20)  # same 60% accuracy, n=200
    small_lo, small_hi = bootstrap_accuracy_ci(small)
    large_lo, large_hi = bootstrap_accuracy_ci(large)
    assert (small_hi - small_lo) > (large_hi - large_lo)
