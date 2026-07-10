from pathlib import Path

import numpy as np
import pytest

from src.boosting.adaboost import AdaBoostClassifier, DecisionStump
from src.utils.preprocessing import load_wdbc


def test_decision_stump_fits_simple_separable_split() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    stump = DecisionStump(random_state=0).fit(X, y)

    assert stump.feature_index_ == 0
    assert stump.threshold_ is not None
    np.testing.assert_array_equal(stump.predict(X), y)
    np.testing.assert_allclose(stump.predict_proba(X).sum(axis=1), 1.0)


def test_adaboost_tracks_round_level_introspection() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = AdaBoostClassifier(n_estimators=5, random_state=7).fit(X, y)

    assert len(model.estimators_) == 5
    assert model.estimator_weights.shape == (5,)
    assert model.estimator_errors.shape == (5,)
    assert len(model.sample_weights_history_) == 6
    assert len(list(model.staged_predict(X))) == 5
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_adaboost_handles_multiclass_samme_predictions() -> None:
    X = np.array([[0.0], [0.5], [1.5], [2.0], [3.0], [3.5], [4.5], [5.0]])
    y = np.array([0, 0, 1, 1, 2, 2, 2, 2])

    model = AdaBoostClassifier(n_estimators=10, random_state=11).fit(X, y)
    predictions = model.predict(X)

    assert set(np.unique(predictions)).issubset(set(np.unique(y)))
    assert model.predict_proba(X).shape == (X.shape[0], 3)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_adaboost_fits_downloaded_wdbc_subset() -> None:
    if not Path("data/wdbc.data").exists():
        pytest.skip("download_data.sh has not been run yet.")

    dataset = load_wdbc()
    indices = np.concatenate(
        [np.flatnonzero(dataset.y == label)[:40] for label in np.unique(dataset.y)]
    )
    X = dataset.X[indices]
    y = dataset.y[indices]

    model = AdaBoostClassifier(n_estimators=10, random_state=19).fit(X, y)

    assert len(model.estimators_) > 0
    assert model.score(X, y) >= 0.8
