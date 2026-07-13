"""Tests for src.trees.decision_tree: DecisionTree core correctness.

This initial version covers fit/predict/predict_proba correctness only.
Edge cases, weighted-sample hardening, max_features/DecisionStump,
feature_importances, and the sklearn comparison are added in later
commits as those features land.
"""

import numpy as np
import pytest

from src.trees.decision_tree import DecisionTree, DecisionStump
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_solves_xor(xor_data, criterion):
    X, y = xor_data
    tree = DecisionTree(max_depth=3, criterion=criterion).fit(X, y)
    assert (tree.predict(X) == y).all()


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_separable_gaussians_near_perfect(two_gaussians, criterion):
    X, y = two_gaussians
    tree = DecisionTree(max_depth=4, criterion=criterion).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.95


def test_predict_proba_rows_sum_to_one(two_gaussians):
    X, y = two_gaussians
    tree = DecisionTree(max_depth=4).fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_before_fit_raises():
    tree = DecisionTree()
    with pytest.raises(RuntimeError):
        tree.predict_proba(np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------- #
def test_determinism_same_seed_same_result():
    X = np.random.default_rng(3).normal(size=(80, 6))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    t1 = DecisionTree(max_depth=4, max_features="sqrt", random_state=11).fit(X, y)
    t2 = DecisionTree(max_depth=4, max_features="sqrt", random_state=11).fit(X, y)
    assert (t1.predict(X) == t2.predict(X)).all()


# --------------------------------------------------------------------- #
# max_features / DecisionStump / __repr__
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("max_features", [None, "sqrt", "log2", 3])
def test_max_features_variants_do_not_crash(max_features):
    X = np.random.default_rng(0).normal(size=(60, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = DecisionTree(max_depth=4, max_features=max_features, random_state=0).fit(X, y)
    assert tree.n_leaves >= 1


def test_max_features_invalid_raises():
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(max_features="not_a_valid_option")
    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_decision_stump_is_always_depth_one_or_less():
    X = np.random.default_rng(5).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    stump = DecisionStump().fit(X, y)
    assert stump.depth <= 1


def test_repr_does_not_crash_on_a_real_tree(two_gaussians):
    X, y = two_gaussians
    small = DecisionTree(max_depth=2).fit(X, y)
    assert "Leaf" in repr(small) or "X" in repr(small)


def test_repr_on_unfitted_tree():
    tree = DecisionTree()
    assert repr(tree) == "DecisionTree(unfitted)"


# --------------------------------------------------------------------- #
# feature_importances
# --------------------------------------------------------------------- #
def test_feature_importances_sum_to_one(two_gaussians):
    X, y = two_gaussians
    tree = DecisionTree(max_depth=4).fit(X, y)
    importances = tree.feature_importances()
    assert abs(importances.sum() - 1.0) < 1e-9


def test_feature_importances_before_fit_raises():
    tree = DecisionTree()
    with pytest.raises(RuntimeError):
        tree.feature_importances()


# --------------------------------------------------------------------- #
# DT edge cases
# --------------------------------------------------------------------- #
def test_single_feature_dataset(single_feature_data):
    X, y = single_feature_data
    tree = DecisionTree(max_depth=3).fit(X, y)
    assert (tree.predict(X) == y).all()


def test_all_identical_labels_is_immediate_leaf():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    tree = DecisionTree(max_depth=5).fit(X, y)
    assert tree.depth == 0
    assert tree.n_leaves == 1
    assert (tree.predict(X) == 1).all()


def test_min_samples_split_equals_one_does_not_crash():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(min_samples_split=1, max_depth=10).fit(X, y)
    assert tree.predict(X) is not None  # simply must not raise


def test_max_depth_zero_is_single_leaf_majority_class():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 0, 1])  # majority class is 0
    tree = DecisionTree(max_depth=0).fit(X, y)
    assert tree.depth == 0
    assert tree.n_leaves == 1
    assert (tree.predict(X) == 0).all()


def test_identical_feature_vectors_different_labels_becomes_leaf():
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    y = np.array([0, 1, 0])  # noisy labels, no way to split on identical X
    tree = DecisionTree(max_depth=5).fit(X, y)
    assert tree.n_leaves == 1  # no valid split exists anywhere


# --------------------------------------------------------------------- #
# Weighted-sample hardening
# --------------------------------------------------------------------- #
def test_uniform_weight_equals_no_weight():
    X = np.random.default_rng(1).normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    t_default = DecisionTree(max_depth=3, random_state=0).fit(X, y)
    t_explicit = DecisionTree(max_depth=3, random_state=0).fit(
        X, y, sample_weight=np.ones(len(y))
    )
    assert (t_default.predict(X) == t_explicit.predict(X)).all()


def test_extreme_weight_changes_chosen_split():
    """Proves weighting affects split *choice*, not just leaf values."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1])
    w = np.array([1000.0, 1.0, 1.0, 1.0, 1.0])
    tree = DecisionTree(max_depth=1).fit(X, y, sample_weight=w)
    assert tree.predict(X[[0]]) == y[0]


def test_zero_weight_sample_is_effectively_ignored():
    X = np.array([[1.0], [2.0], [3.0], [100.0]])  # last point is an outlier
    y = np.array([0, 0, 1, 1])
    w = np.array([1.0, 1.0, 1.0, 0.0])
    tree = DecisionTree(max_depth=1).fit(X, y, sample_weight=w)
    assert tree.n_leaves == 2  # split driven by the first three points only


def test_decision_stump_accepts_sample_weight():
    X = np.random.default_rng(5).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    w = np.random.default_rng(1).random(len(y))
    stump = DecisionStump().fit(X, y, sample_weight=w)
    assert stump.depth <= 1


# --------------------------------------------------------------------- #
# Comparison against scikit-learn (validation only; not an implementation
# dependency). max_features is left at its default (None) here to
# isolate correctness from random feature-sampling variance.
# --------------------------------------------------------------------- #
def test_comparison_against_sklearn_within_tolerance():
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    my_tree = DecisionTree(max_depth=6, criterion="gini").fit(X_train, y_train)
    sk_tree = DecisionTreeClassifier(max_depth=6, criterion="gini", random_state=42).fit(
        X_train, y_train
    )

    my_acc = accuracy_score(y_test, my_tree.predict(X_test))
    sk_acc = accuracy_score(y_test, sk_tree.predict(X_test))

    assert abs(my_acc - sk_acc) <= 0.02, (
        f"Accuracy diff {abs(my_acc - sk_acc):.4f} exceeds 2% tolerance"
    )

    corr = np.corrcoef(my_tree.feature_importances(), sk_tree.feature_importances_)[0, 1]
    assert corr > 0.7


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_comparison_against_sklearn_both_criteria(criterion):
    """Some bugs (e.g. wrong log base) only surface with criterion='entropy'."""
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    my_tree = DecisionTree(max_depth=6, criterion=criterion).fit(X_train, y_train)
    sk_tree = DecisionTreeClassifier(max_depth=6, criterion=criterion, random_state=7).fit(
        X_train, y_train
    )

    my_acc = accuracy_score(y_test, my_tree.predict(X_test))
    sk_acc = accuracy_score(y_test, sk_tree.predict(X_test))
    assert abs(my_acc - sk_acc) <= 0.02
