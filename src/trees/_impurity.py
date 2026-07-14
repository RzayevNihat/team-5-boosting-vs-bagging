"""Impurity criteria for CART-style decision trees.

All functions accept an optional `sample_weight` array so that the same
code path serves both plain decision trees and weighted decision stumps
(e.g. for AdaBoost) without duplication. Weighted class proportions are
used from day one:

    p_c = sum(w_i for i where y_i == c) / sum(w_i)

with w_i = 1 for all i by default, which recovers the standard
unweighted impurity as a special case.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def _class_proportions(y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    """Return weighted class proportions p_c, vectorized over classes.

    Args:
        y: Class labels, shape (n_samples,).
        sample_weight: Per-sample weights, shape (n_samples,).

    Returns:
        Array of weighted proportions, one per distinct class in y,
        summing to 1.0.

    Raises:
        ValueError: If the total weight is not strictly positive.
    """
    classes = np.unique(y)
    total_weight = sample_weight.sum()
    if total_weight <= 0:
        raise ValueError("sample_weight must sum to a positive value")
    proportions: np.ndarray = np.array(
        [sample_weight[y == c].sum() for c in classes]
    ) / total_weight
    return proportions


def gini(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Weighted Gini impurity: 1 - sum(p_c^2).

    Interpretation: the probability that two samples drawn at random
    (with replacement, weighted by sample_weight) from this node have
    different class labels. 0 for a pure node; 1 - 1/C for a uniform
    node over C classes.

    Args:
        y: Class labels, shape (n_samples,).
        sample_weight: Per-sample weights, or None for uniform weighting.

    Returns:
        Gini impurity in [0, 1 - 1/C].
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    p = _class_proportions(y, sample_weight)
    result: float = float(1.0 - np.sum(p ** 2))
    return result


def entropy(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Weighted Shannon entropy in bits: -sum(p_c * log2(p_c)).

    An epsilon is added inside the log to guard against log2(0) for
    classes with zero weighted mass in this node.

    Args:
        y: Class labels, shape (n_samples,).
        sample_weight: Per-sample weights, or None for uniform weighting.

    Returns:
        Entropy in bits, always >= 0. 0 for a pure node.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    p = _class_proportions(y, sample_weight)
    return float(-np.sum(p * np.log2(p + _EPSILON)))


_CRITERIA = {"gini": gini, "entropy": entropy}


def impurity(
    y: np.ndarray, sample_weight: np.ndarray | None, criterion: str
) -> float:
    """Dispatch to the requested impurity criterion.

    Args:
        y: Class labels, shape (n_samples,).
        sample_weight: Per-sample weights, or None for uniform weighting.
        criterion: One of "gini" or "entropy".

    Returns:
        The impurity value for the given criterion.

    Raises:
        ValueError: If criterion is not a recognized criterion name.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; use 'gini' or 'entropy'")
    return _CRITERIA[criterion](y, sample_weight)
