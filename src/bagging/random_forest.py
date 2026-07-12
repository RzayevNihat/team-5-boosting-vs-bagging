"""Random Forest classifier implemented from scratch.

This module is owned by Person 3 in the final project work split. It implements
bootstrap aggregation, feature sub-sampling, optional out-of-bag scoring,
parallel tree fitting with ``multiprocessing``, averaged class probabilities,
and mean impurity-based feature importances.

The implementation intentionally does not use ``sklearn.tree.DecisionTreeClassifier``
or ``sklearn.ensemble.RandomForestClassifier``. A small private CART-style tree is
implemented inside this module so the Random Forest module can be tested even
while the team member responsible for ``src/trees/decision_tree.py`` is still
working. During final integration, this private tree can be replaced with the
team's shared ``DecisionTree`` class if its interface is compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

Criterion = Literal["gini", "entropy"]
MaxFeatures = int | Literal["sqrt", "log2"] | None


@dataclass(slots=True)
class _TreeNode:
    """A single node in the private CART classification tree."""

    prediction: int
    proba: np.ndarray
    n_samples: int
    impurity: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    impurity_decrease: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _CARTClassificationTree:
    """Small CART classifier used internally by ``RandomForestClassifier``.

    It supports continuous features, Gini/entropy split criteria, random feature
    sub-sampling at every node, class-probability prediction, and impurity-based
    feature importances.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: Criterion = "gini",
        max_features: MaxFeatures = None,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_: _TreeNode | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self._rng = np.random.default_rng(random_state)
        self._feature_importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CARTClassificationTree":
        X = _validate_2d_array(X, "X")
        y = _validate_1d_array(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a tree on an empty dataset.")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._feature_importances = np.zeros(self.n_features_in_, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build_node(X, y_encoded, depth=0)
        total = self._feature_importances.sum()
        if total > 0:
            self._feature_importances /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_2d_array(X, "X")
        encoded = np.array([self._predict_one(row).prediction for row in X], dtype=int)
        return self.classes_[encoded]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_2d_array(X, "X")
        return np.vstack([self._predict_one(row).proba for row in X])

    def feature_importances(self) -> np.ndarray:
        self._check_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def _build_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        proba = counts / counts.sum()
        impurity = self._impurity_from_counts(counts)
        prediction = int(np.argmax(counts))

        node = _TreeNode(
            prediction=prediction,
            proba=proba,
            n_samples=n_samples,
            impurity=impurity,
        )

        if self._should_stop(y, depth, n_samples):
            return node

        split = self._find_best_split(X, y, counts, impurity)
        if split is None:
            return node

        feature_index, threshold, gain = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return node

        node.feature_index = feature_index
        node.threshold = float(threshold)
        node.impurity_decrease = float(gain)
        assert self._feature_importances is not None
        self._feature_importances[feature_index] += gain * n_samples
        node.left = self._build_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_node(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, y: np.ndarray, depth: int, n_samples: int) -> bool:
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self.min_samples_split:
            return True
        if np.unique(y).size == 1:
            return True
        return False

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_counts: np.ndarray,
        parent_impurity: float,
    ) -> tuple[int, float, float] | None:
        n_samples, n_features = X.shape
        feature_indices = self._sample_feature_indices(n_features)

        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature_index in feature_indices:
            values = X[:, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_y = y[order]

            if sorted_values[0] == sorted_values[-1]:
                continue

            left_counts = np.zeros(self.n_classes_, dtype=float)
            right_counts = parent_counts.copy()

            for split_pos in range(1, n_samples):
                cls = sorted_y[split_pos - 1]
                left_counts[cls] += 1.0
                right_counts[cls] -= 1.0

                if sorted_values[split_pos] == sorted_values[split_pos - 1]:
                    continue

                n_left = split_pos
                n_right = n_samples - split_pos
                if n_left == 0 or n_right == 0:
                    continue

                left_impurity = self._impurity_from_counts(left_counts)
                right_impurity = self._impurity_from_counts(right_counts)
                weighted_child_impurity = (
                    (n_left / n_samples) * left_impurity
                    + (n_right / n_samples) * right_impurity
                )
                gain = parent_impurity - weighted_child_impurity

                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature_index)
                    best_threshold = float(
                        (sorted_values[split_pos - 1] + sorted_values[split_pos]) / 2.0
                    )

        if best_feature is None or best_threshold is None or best_gain <= 1e-12:
            return None
        return best_feature, best_threshold, best_gain

    def _sample_feature_indices(self, n_features: int) -> np.ndarray:
        max_features = _resolve_max_features(self.max_features, n_features)
        if max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=max_features, replace=False)

    def _impurity_from_counts(self, counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs**2))
        non_zero = probs[probs > 0]
        return float(-np.sum(non_zero * np.log2(non_zero)))

    def _predict_one(self, row: np.ndarray) -> _TreeNode:
        self._check_fitted()
        assert self.root_ is not None
        node = self.root_
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

    def _check_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("The tree must be fitted before prediction.")


def _fit_single_tree(task: tuple) -> tuple[_CARTClassificationTree, np.ndarray]:
    """Fit one tree for a Random Forest worker.

    This function is top-level so it can be pickled by ``multiprocessing`` on
    Windows and Linux.
    """

    (
        X,
        y,
        bootstrap,
        sample_size,
        max_depth,
        min_samples_split,
        max_features,
        criterion,
        seed,
    ) = task

    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    if bootstrap:
        bootstrap_indices = rng.integers(0, n_samples, size=sample_size)
    else:
        bootstrap_indices = np.arange(n_samples)

    in_bag = np.zeros(n_samples, dtype=bool)
    in_bag[bootstrap_indices] = True
    oob_indices = np.flatnonzero(~in_bag)

    tree = _CARTClassificationTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        random_state=int(seed),
    )
    tree.fit(X[bootstrap_indices], y[bootstrap_indices])
    return tree, oob_indices


class RandomForestClassifier:
    """Random Forest classifier using bootstrap aggregation over CART trees.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree. ``None`` means fully expanded trees.
    max_features:
        Number of candidate features considered at each split. Supports ``int``,
        ``"sqrt"``, ``"log2"``, or ``None``.
    min_samples_split:
        Minimum number of samples needed to split an internal tree node.
    bootstrap:
        Whether to train each tree on a bootstrap sample.
    oob_score:
        Whether to compute out-of-bag accuracy. Only meaningful when
        ``bootstrap=True``.
    n_jobs:
        Number of parallel worker processes used for tree fitting. Use ``1`` for
        sequential fitting. Use ``-1`` for all CPU cores.
    random_state:
        Seed used to make bootstrap samples and tree feature choices
        deterministic.
    criterion:
        Tree split criterion, either ``"gini"`` or ``"entropy"``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: MaxFeatures = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
        criterion: Criterion = "gini",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if not bootstrap and oob_score:
            raise ValueError("oob_score=True requires bootstrap=True.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.criterion = criterion

        self.estimators_: list[_CARTClassificationTree] = []
        self.oob_indices_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self._oob_score: float | None = None
        self._feature_importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the Random Forest on training data."""
        X = _validate_2d_array(X, "X")
        y = _validate_1d_array(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a forest on an empty dataset.")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        _resolve_max_features(self.max_features, self.n_features_in_)

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.n_estimators)
        sample_size = X.shape[0]

        tasks = [
            (
                X,
                y_encoded,
                self.bootstrap,
                sample_size,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                self.criterion,
                int(seed),
            )
            for seed in seeds
        ]

        n_jobs = self._effective_n_jobs()
        if n_jobs == 1:
            fitted = [_fit_single_tree(task) for task in tasks]
        else:
            with Pool(processes=n_jobs) as pool:
                fitted = pool.map(_fit_single_tree, tasks)

        self.estimators_ = [tree for tree, _ in fitted]
        self.oob_indices_ = [indices for _, indices in fitted]
        self._feature_importances = self._compute_feature_importances()
        self._oob_score = self._compute_oob_score(X, y_encoded) if self.oob_score else None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by majority vote."""
        proba = self.predict_proba(X)
        encoded = np.argmax(proba, axis=1)
        return self.classes_[encoded]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging tree probabilities."""
        self._check_fitted()
        X = _validate_2d_array(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the forest was fitted with "
                f"{self.n_features_in_} features."
            )
        probabilities = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree in self.estimators_:
            probabilities += tree.predict_proba(X)
        probabilities /= len(self.estimators_)
        return probabilities

    @property
    def oob_score_(self) -> float:
        """Out-of-bag accuracy computed during fitting."""
        if not self.oob_score:
            raise AttributeError("oob_score_ is available only when oob_score=True.")
        if self._oob_score is None:
            raise RuntimeError("The forest must be fitted before reading oob_score_.")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean impurity-based feature importances across all trees."""
        self._check_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def _effective_n_jobs(self) -> int:
        if self.n_jobs == -1:
            return max(1, cpu_count())
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer or -1.")
        return min(self.n_jobs, self.n_estimators)

    def _compute_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.estimators_:
            importances += tree.feature_importances()
        importances /= len(self.estimators_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _compute_oob_score(self, X: np.ndarray, y_encoded: np.ndarray) -> float:
        vote_counts = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree, oob_indices in zip(self.estimators_, self.oob_indices_):
            if oob_indices.size == 0:
                continue
            predicted = tree.predict(X[oob_indices]).astype(int)
            for sample_index, predicted_class in zip(oob_indices, predicted):
                vote_counts[sample_index, predicted_class] += 1.0

        valid = vote_counts.sum(axis=1) > 0
        if not np.any(valid):
            return float("nan")
        oob_predictions = np.argmax(vote_counts[valid], axis=1)
        return float(np.mean(oob_predictions == y_encoded[valid]))

    def _check_fitted(self) -> None:
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("The forest must be fitted before prediction.")


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    """Resolve Random Forest ``max_features`` into an integer count."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if max_features is None:
        return n_features
    if isinstance(max_features, int):
        if max_features < 1 or max_features > n_features:
            raise ValueError("Integer max_features must be in [1, n_features].")
        return max_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    raise ValueError("max_features must be int, 'sqrt', 'log2', or None.")


def _validate_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_1d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array
