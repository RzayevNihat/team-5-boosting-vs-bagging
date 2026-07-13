"""Preprocessing utilities implemented from scratch (no sklearn).

Provides the minimum a baseline experiment needs: missing-value
imputation, feature standardization, and a stratifiable train/test
split. sklearn is used nowhere in this module -- only numpy.
"""

from __future__ import annotations

import numpy as np


def handle_missing_values(X: np.ndarray, strategy: str = "median_impute") -> np.ndarray:
    """Impute missing (NaN) values column-by-column.

    Args:
        X: Feature matrix, shape (n_samples, n_features), possibly
            containing NaNs.
        strategy: Currently only "median_impute" is supported: each
            NaN is replaced by the median of its own column, computed
            over the non-missing values in that column.

    Returns:
        A new array (X is not modified in place) with NaNs replaced.

    Raises:
        ValueError: If an unsupported strategy is requested, or if a
            column is entirely NaN (no median can be computed).
    """
    if strategy != "median_impute":
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    X = np.array(X, dtype=float, copy=True)  # never mutate the caller's array
    for col in range(X.shape[1]):
        column = X[:, col]
        missing_mask = np.isnan(column)
        if not missing_mask.any():
            continue
        observed = column[~missing_mask]
        if observed.size == 0:
            raise ValueError(f"Column {col} is entirely NaN; cannot impute a median")
        median = np.median(observed)
        column[missing_mask] = median
    return X


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit
    variance, fit on training data and reused (never refit) on test data.

    This mirrors the standard discipline of never calling fit_transform
    on held-out data: the scaler's mean_/scale_ are learned once from
    the training split and then applied identically everywhere else.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute per-column mean and standard deviation from X.

        Args:
            X: Training feature matrix, shape (n_samples, n_features).

        Returns:
            self, for chaining.
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against zero-variance columns (constant features) --
        # replace a zero std with 1.0 so those columns pass through as
        # (x - mean) = 0 rather than producing a division-by-zero NaN.
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted (mean_, scale_) standardization to X.

        Args:
            X: Feature matrix to transform.

        Returns:
            Standardized array, same shape as X.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform()")
        X = np.asarray(X, dtype=float)
        result: np.ndarray = (X - self.mean_) / self.scale_
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then transform X. Only ever call this on training data."""
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) into train/test partitions, optionally stratified.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        test_size: Fraction of samples to hold out for testing.
        random_state: Seed for reproducible shuffling.
        stratify: If provided (typically the same array as y), the
            split preserves each class's proportion in both partitions
            -- important for imbalanced datasets, where a plain random
            split can leave the test set with zero minority-class
            examples.

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(random_state)
    n = len(y)

    if stratify is None:
        indices = rng.permutation(n)
        n_test = int(round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        train_idx_parts: list[np.ndarray] = []
        test_idx_parts: list[np.ndarray] = []
        for c in np.unique(stratify):
            class_idx = np.where(stratify == c)[0]
            class_idx = rng.permutation(class_idx)
            n_test_c = int(round(len(class_idx) * test_size))
            test_idx_parts.append(class_idx[:n_test_c])
            train_idx_parts.append(class_idx[n_test_c:])
        train_idx = np.concatenate(train_idx_parts)
        test_idx = np.concatenate(test_idx_parts)
        # Shuffle again so classes are interleaved, not grouped.
        train_idx = rng.permutation(train_idx)
        test_idx = rng.permutation(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
