"""Tests for the from-scratch RandomForestClassifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.bagging.random_forest import RandomForestClassifier


def _binary_dataset(random_state: int = 42):
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        class_sep=1.2,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)


def _multiclass_dataset(random_state: int = 42):
    X, y = make_classification(
        n_samples=320,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)


def test_fit_predict_binary_accuracy_is_reasonable():
    X_train, X_test, y_train, y_test = _binary_dataset()
    forest = RandomForestClassifier(
        n_estimators=25,
        max_depth=None,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
    )
    forest.fit(X_train, y_train)

    predictions = forest.predict(X_test)
    assert predictions.shape == y_test.shape
    assert accuracy_score(y_test, predictions) >= 0.75
    assert 0.0 <= forest.oob_score_ <= 1.0


def test_predict_proba_has_correct_shape_and_sums_to_one():
    X_train, X_test, y_train, _ = _multiclass_dataset()
    forest = RandomForestClassifier(n_estimators=15, random_state=7)
    forest.fit(X_train, y_train)

    probabilities = forest.predict_proba(X_test[:11])
    assert probabilities.shape == (11, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(11), atol=1e-8)


def test_multiclass_prediction_returns_original_labels():
    X_train, X_test, y_train, _ = _multiclass_dataset()
    y_train_shifted = y_train + 10
    forest = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=8)
    forest.fit(X_train, y_train_shifted)

    predictions = forest.predict(X_test[:20])
    assert set(np.unique(predictions)).issubset(set(np.unique(y_train_shifted)))


def test_feature_importances_are_normalized():
    X_train, _, y_train, _ = _binary_dataset()
    forest = RandomForestClassifier(n_estimators=12, max_depth=6, random_state=123)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    assert importances.shape == (X_train.shape[1],)
    assert np.all(importances >= 0.0)
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-8)


def test_random_state_makes_results_reproducible():
    X_train, X_test, y_train, _ = _binary_dataset()
    first = RandomForestClassifier(n_estimators=14, random_state=99).fit(X_train, y_train)
    second = RandomForestClassifier(n_estimators=14, random_state=99).fit(X_train, y_train)

    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
    np.testing.assert_allclose(first.predict_proba(X_test), second.predict_proba(X_test))


def test_parallel_and_sequential_training_match_with_same_seed():
    X_train, X_test, y_train, _ = _binary_dataset()
    sequential = RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=2026).fit(
        X_train, y_train
    )
    parallel = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=2026).fit(
        X_train, y_train
    )

    np.testing.assert_array_equal(sequential.predict(X_test), parallel.predict(X_test))
    np.testing.assert_allclose(sequential.predict_proba(X_test), parallel.predict_proba(X_test))


def test_oob_score_requires_bootstrap():
    with pytest.raises(ValueError, match="oob_score=True requires bootstrap=True"):
        RandomForestClassifier(bootstrap=False, oob_score=True)


def test_oob_score_property_requires_oob_enabled():
    X_train, _, y_train, _ = _binary_dataset()
    forest = RandomForestClassifier(n_estimators=5, oob_score=False, random_state=42)
    forest.fit(X_train, y_train)

    with pytest.raises(AttributeError):
        _ = forest.oob_score_


@pytest.mark.parametrize("max_features", [1, 3, "sqrt", "log2", None])
def test_supported_max_features_values(max_features):
    X_train, X_test, y_train, _ = _binary_dataset()
    forest = RandomForestClassifier(
        n_estimators=7,
        max_depth=5,
        max_features=max_features,
        random_state=42,
    )
    forest.fit(X_train, y_train)
    assert forest.predict(X_test[:5]).shape == (5,)


def test_invalid_n_jobs_raises_during_fit():
    X_train, _, y_train, _ = _binary_dataset()
    forest = RandomForestClassifier(n_jobs=0)
    with pytest.raises(ValueError, match="n_jobs"):
        forest.fit(X_train, y_train)


def test_constructor_validates_immediate_parameters():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(max_depth=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(min_samples_split=1)
    with pytest.raises(ValueError):
        RandomForestClassifier(criterion="mse")


def test_invalid_max_features_raises_during_fit():
    X_train, _, y_train, _ = _binary_dataset()
    forest = RandomForestClassifier(max_features=100, random_state=42)
    with pytest.raises(ValueError, match="max_features"):
        forest.fit(X_train, y_train)


def test_predictions_before_fit_raise_runtime_error():
    forest = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        forest.predict(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        forest.predict_proba(np.zeros((3, 2)))


def test_breast_cancer_dataset_sanity_check():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    forest = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        max_features="sqrt",
        random_state=42,
        oob_score=True,
    )
    forest.fit(X_train, y_train)
    assert accuracy_score(y_test, forest.predict(X_test)) >= 0.88
    assert forest.oob_score_ >= 0.85
