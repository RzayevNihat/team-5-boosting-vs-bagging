"""Best-split search for CART decision trees.

Finds the best (feature, threshold) split for a node by exhaustive
search over candidate midpoint thresholds, using a sort + incremental
weighted class-count sweep instead of naive O(N^2) rescanning.

Algorithm (per feature):
    1. Sort samples by the feature's values -- O(N log N).
    2. Sweep left-to-right, moving one sample at a time from the
       "right" bucket to the "left" bucket, updating running weighted
       class-count vectors incrementally in O(C) per step (C = number
       of classes).
    3. At each valid boundary (between two distinct sorted values),
       compute the impurity-reduction gain directly from the running
       counts (O(C), not O(N)).

Total per feature: O(N log N + N * C), instead of the naive O(N^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitResult:
    """The best split found for a node, across all searched features.

    Attributes:
        feature_index: Index of the feature to split on.
        threshold: Samples with feature value <= threshold go left.
        gain: Impurity reduction (delta I) achieved by this split.
    """

    feature_index: int
    threshold: float
    gain: float


def _impurity_from_counts(
    class_weights: np.ndarray, total_weight: float, criterion: str
) -> float:
    """Impurity computed directly from a class-weight vector.

    Avoids re-deriving y from counts, which is what makes the
    incremental sweep O(C) per step instead of O(N).

    Args:
        class_weights: Weighted count per class, shape (n_classes,).
        total_weight: Sum of class_weights.
        criterion: One of "gini" or "entropy".

    Returns:
        The impurity value; 0.0 if total_weight is not positive.
    """
    if total_weight <= 0:
        return 0.0
    p = class_weights / total_weight
    if criterion == "gini":
        return float(1.0 - np.sum(p ** 2))
    elif criterion == "entropy":
        return float(-np.sum(p * np.log2(p + 1e-12)))
    raise ValueError(f"Unknown criterion {criterion!r}")


def best_split_for_feature(
    x_col: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, criterion: str
) -> tuple[float, float] | None:
    """Find the best threshold on a single feature column.

    Only thresholds strictly between consecutive distinct sorted
    feature values are considered (midpoints); there is no benefit to
    considering thresholds elsewhere, since points sharing a value
    can't be separated and off-boundary thresholds are dominated by
    the midpoint.

    Args:
        x_col: Values of a single feature, shape (n_samples,).
        y: Class labels, shape (n_samples,).
        sample_weight: Per-sample weights, shape (n_samples,).
        criterion: One of "gini" or "entropy".

    Returns:
        (threshold, gain) for the best split found, or None if no
        valid split exists (e.g. all feature values are identical).
    """
    order = np.argsort(x_col, kind="mergesort")  # stable sort
    x_sorted = x_col[order]
    y_sorted = y[order]
    w_sorted = sample_weight[order]

    classes, y_encoded = np.unique(y_sorted, return_inverse=True)
    n_classes = len(classes)
    total_weight = w_sorted.sum()

    parent_impurity = _impurity_from_counts(
        np.array([w_sorted[y_encoded == c].sum() for c in range(n_classes)]),
        total_weight,
        criterion,
    )

    left_counts = np.zeros(n_classes)
    right_counts = np.array(
        [w_sorted[y_encoded == c].sum() for c in range(n_classes)]
    )
    left_w, right_w = 0.0, total_weight

    best_gain, best_threshold = -np.inf, None
    n = len(x_sorted)

    for i in range(n - 1):
        c = y_encoded[i]
        left_counts[c] += w_sorted[i]
        right_counts[c] -= w_sorted[i]
        left_w += w_sorted[i]
        right_w -= w_sorted[i]

        if x_sorted[i] == x_sorted[i + 1]:
            continue  # identical feature values: no valid boundary here

        left_imp = _impurity_from_counts(left_counts, left_w, criterion)
        right_imp = _impurity_from_counts(right_counts, right_w, criterion)

        gain = (
            parent_impurity
            - (left_w / total_weight) * left_imp
            - (right_w / total_weight) * right_imp
        )

        if gain > best_gain:
            best_gain = gain
            best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

    if best_threshold is None:
        return None
    return best_threshold, best_gain


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    criterion: str,
    feature_indices: np.ndarray,
) -> SplitResult | None:
    """Search the given subset of features for the globally best split.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        sample_weight: Per-sample weights, shape (n_samples,).
        criterion: "gini" or "entropy".
        feature_indices: Which columns of X to search over (supports
            max_features sub-sampling for Random-Forest-style use).

    Returns:
        The best SplitResult found, or None if no feature admits any
        valid split (e.g. every feature is constant on this node).
    """
    best: SplitResult | None = None
    for j in feature_indices:
        result = best_split_for_feature(X[:, j], y, sample_weight, criterion)
        if result is None:
            continue
        threshold, gain = result
        if best is None or gain > best.gain:
            best = SplitResult(feature_index=int(j), threshold=threshold, gain=gain)
    return best
