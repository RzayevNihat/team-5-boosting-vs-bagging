"""Tests for src.utils.preprocessing: StandardScaler, missing-value
imputation, and train_test_split.
"""

import numpy as np
import pytest

from src.utils.preprocessing import StandardScaler, handle_missing_values, train_test_split


def test_handle_missing_values_imputes_column_median():
    X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
    X_clean = handle_missing_values(X)
    assert not np.isnan(X_clean).any()
    assert X_clean[0, 1] == 5.0  # median of [4.0, 6.0]


def test_handle_missing_values_does_not_mutate_input():
    X = np.array([[1.0, np.nan], [2.0, 4.0]])
    X_copy = X.copy()
    handle_missing_values(X)
    # original array (with NaN) should be untouched
    assert np.isnan(X[0, 1])
    assert np.array_equal(X, X_copy, equal_nan=True)


def test_handle_missing_values_all_nan_column_raises():
    X = np.array([[np.nan], [np.nan]])
    with pytest.raises(ValueError):
        handle_missing_values(X)


def test_standard_scaler_produces_zero_mean_unit_variance():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-9)


def test_standard_scaler_handles_constant_column_without_nan():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])  # first column constant
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert not np.isnan(X_scaled).any()
    assert np.allclose(X_scaled[:, 0], 0.0)


def test_standard_scaler_transform_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_test_set_uses_train_statistics():
    """Reuse discipline: fitting only on train, then transforming test
    with those same statistics -- never refit on test data."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[100.0]])  # far outside training range
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Should reflect train mean/std, not be re-centered around X_test itself.
    expected = (100.0 - X_train.mean()) / X_train.std()
    assert abs(X_test_scaled[0, 0] - expected) < 1e-9


def test_train_test_split_sizes_and_no_overlap():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10
    # No sample should appear in both splits.
    train_rows = {tuple(row) for row in X_train}
    test_rows = {tuple(row) for row in X_test}
    assert train_rows.isdisjoint(test_rows)


def test_train_test_split_stratify_preserves_class_ratio():
    X = np.arange(200).reshape(100, 2).astype(float)
    y = np.array([0] * 90 + [1] * 10)  # imbalanced 90/10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    # The minority class must appear in the test set (a plain random
    # split on this ratio could easily miss it).
    assert (y_test == 1).sum() > 0
    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()
    assert abs(train_ratio - test_ratio) < 0.1


def test_train_test_split_is_reproducible_with_same_seed():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.array([0, 1] * 10)
    split1 = train_test_split(X, y, test_size=0.25, random_state=5)
    split2 = train_test_split(X, y, test_size=0.25, random_state=5)
    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)
