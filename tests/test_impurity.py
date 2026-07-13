"""Tests for src.trees._impurity: gini, entropy, impurity dispatch."""

import numpy as np
import pytest

from src.trees._impurity import gini, entropy, impurity


def test_gini_pure_node_is_zero():
    y_pure = np.array([1, 1, 1, 1])
    assert gini(y_pure) == 0.0


def test_gini_balanced_binary_node_is_half():
    y_balanced = np.array([0, 0, 1, 1])
    assert abs(gini(y_balanced) - 0.5) < 1e-9  # 1 - (0.5^2 + 0.5^2) = 0.5


def test_entropy_pure_node_is_near_zero():
    y_pure = np.array([1, 1, 1, 1])
    assert entropy(y_pure) < 1e-9


def test_entropy_balanced_binary_node_is_one_bit():
    y_balanced = np.array([0, 0, 1, 1])
    assert abs(entropy(y_balanced) - 1.0) < 1e-6  # exactly 1 bit for 50/50


def test_weighted_gini_matches_manual_calculation():
    """Exact decimal match against a by-hand weighted-Gini derivation."""
    y = np.array([0, 0, 1, 1])
    w = np.array([3.0, 1.0, 1.0, 3.0])  # class 0 total=4, class 1 total=4, total=8
    assert abs(gini(y, w) - 0.5) < 1e-9

    w_skewed = np.array([100.0, 1.0, 1.0, 1.0])
    p0, p1 = 101 / 103, 2 / 103
    expected = 1 - (p0 ** 2 + p1 ** 2)
    assert abs(gini(y, w_skewed) - expected) < 1e-9


def test_weighted_gini_decreases_when_skewed_toward_pure_class():
    y_balanced = np.array([0, 0, 1, 1])
    w = np.array([10.0, 10.0, 1.0, 1.0])
    assert gini(y_balanced, w) < gini(y_balanced)


def test_impurity_dispatch_matches_direct_calls():
    y = np.array([0, 0, 1, 1])
    assert impurity(y, None, "gini") == gini(y)
    assert impurity(y, None, "entropy") == entropy(y)


def test_impurity_rejects_unknown_criterion():
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        impurity(y, None, "not_a_real_criterion")


def test_impurity_functions_do_not_mutate_inputs():
    """Callers (e.g. a boosting weight update) call these functions
    repeatedly with the same array -- they must never modify it."""
    y = np.array([0, 0, 1, 1])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    y_copy, w_copy = y.copy(), w.copy()
    gini(y, w)
    entropy(y, w)
    assert (y == y_copy).all()
    assert (w == w_copy).all()


def test_zero_total_weight_raises():
    y = np.array([0, 1])
    w = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        gini(y, w)
