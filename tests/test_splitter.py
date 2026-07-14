"""Tests for src.trees._splitter and src.trees._stopping."""

import numpy as np
import pytest

from src.trees._splitter import best_split_for_feature, find_best_split
from src.trees._stopping import should_stop


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_best_split_finds_clean_separation(criterion):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    w = np.ones(4)

    threshold, gain = best_split_for_feature(x, y, w, criterion)
    assert abs(threshold - 2.5) < 1e-9
    assert gain > 0


def test_best_split_returns_none_for_constant_feature():
    x_flat = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([0, 0, 1, 1])
    w = np.ones(4)
    assert best_split_for_feature(x_flat, y, w, "gini") is None


def test_find_best_split_picks_best_feature_across_columns():
    # Feature 0 is useless (constant); feature 1 cleanly separates the classes.
    X = np.array([[9.0, 1.0], [9.0, 2.0], [9.0, 3.0], [9.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    w = np.ones(4)

    result = find_best_split(X, y, w, "gini", feature_indices=np.array([0, 1]))
    assert result is not None
    assert result.feature_index == 1
    assert abs(result.threshold - 2.5) < 1e-9


def test_find_best_split_returns_none_when_no_feature_admits_a_split():
    X = np.array([[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]])
    y = np.array([0, 1, 0])
    w = np.ones(3)
    assert find_best_split(X, y, w, "gini", feature_indices=np.array([0, 1])) is None


def test_extreme_weight_changes_chosen_split():
    """Proves weighting affects split *choice*, not just leaf statistics.

    Uses a dataset with two isolated "class 1" points at both ends
    (indices 0 and 4) surrounded by class 0. Unweighted, gain at
    threshold=1.5 (isolating point 0) ties with threshold=4.5
    (isolating point 4); the sweep's left-to-right tie-break picks 1.5.
    Heavily up-weighting the last sample (index 4) breaks the tie in
    the other direction, shifting the optimal threshold to 4.5.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 0, 0, 0, 1])

    w_uniform = np.ones(5)
    uniform_result = find_best_split(X, y, w_uniform, "gini", feature_indices=np.array([0]))
    assert uniform_result is not None
    assert abs(uniform_result.threshold - 1.5) < 1e-9  # left-to-right tie-break

    w_skewed = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
    skewed_result = find_best_split(X, y, w_skewed, "gini", feature_indices=np.array([0]))
    assert skewed_result is not None
    assert abs(skewed_result.threshold - 4.5) < 1e-9  # now isolates the heavily-weighted point instead


def test_should_stop_at_max_depth():
    y = np.array([0, 0, 1, 1])
    assert should_stop(depth=3, n_samples=10, y=y, max_depth=3, min_samples_split=2) is True


def test_should_stop_below_min_samples_split():
    y = np.array([0, 0, 1, 1])
    assert should_stop(depth=0, n_samples=1, y=y, max_depth=None, min_samples_split=2) is True


def test_should_stop_on_pure_node():
    y_pure = np.array([1, 1, 1, 1])
    assert should_stop(depth=0, n_samples=4, y=y_pure, max_depth=None, min_samples_split=2) is True


def test_should_not_stop_when_no_criterion_met():
    y = np.array([0, 0, 1, 1])
    assert should_stop(depth=0, n_samples=4, y=y, max_depth=None, min_samples_split=2) is False
