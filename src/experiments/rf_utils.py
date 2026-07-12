"""Utility functions for Person 3 Random Forest experiments.

The helpers in this file keep the experiment scripts short and reproducible.
They intentionally use sklearn only for datasets, metrics, preprocessing, and
reference baselines. The project implementation of Random Forest remains the
from-scratch class in ``src.bagging.random_forest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class DatasetBundle:
    """Container used by the experiment scripts."""

    name: str
    X: np.ndarray
    y: np.ndarray
    description: str
    is_binary: bool


def ensure_output_dirs() -> None:
    """Create output folders for figures and CSV result tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_breast_cancer_bundle() -> DatasetBundle:
    """Load a small real-world binary classification dataset."""
    data = load_breast_cancer()
    return DatasetBundle(
        name="breast_cancer",
        X=data.data.astype(float),
        y=data.target.astype(int),
        description="Breast Cancer Wisconsin Diagnostic dataset from sklearn.datasets.",
        is_binary=True,
    )


def load_digits_binary_bundle() -> DatasetBundle:
    """Load a high-dimensional binary subset of the Digits dataset.

    Digits has 64 input features (>20), which is useful for testing how Random
    Forest behaves with high-dimensional input when ``max_features='sqrt'``.
    """
    data = load_digits()
    mask = np.isin(data.target, [3, 8])
    X = data.data[mask].astype(float)
    y = (data.target[mask] == 8).astype(int)
    return DatasetBundle(
        name="digits_3_vs_8",
        X=X,
        y=y,
        description="Binary high-dimensional subset of sklearn Digits: class 3 vs class 8.",
        is_binary=True,
    )


def load_imbalanced_bundle() -> DatasetBundle:
    """Load or create an imbalanced dataset for imbalance-treatment experiments.

    Final submission should prefer a real downloaded dataset such as Covertype
    when available. For offline smoke testing, this function falls back to a
    deterministic synthetic dataset with a 99:1 class ratio.
    """
    covtype_path = DATA_DIR / "covtype.data"
    if covtype_path.exists():
        columns = [f"feature_{i}" for i in range(54)] + ["target"]
        df = pd.read_csv(covtype_path, header=None, names=columns)
        # Keep a binary one-vs-rest task where a rare cover type becomes class 1.
        # Cover type 4 is rare in the full dataset, making it suitable for an
        # imbalance discussion. Limit rows to keep experiments fast.
        df = df.sample(n=min(5000, len(df)), random_state=RANDOM_STATE)
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = (df["target"].to_numpy() == 4).astype(int)
        return DatasetBundle(
            name="covertype_type4_imbalanced",
            X=X,
            y=y,
            description="Covertype one-vs-rest task: rare cover type 4 as positive class.",
            is_binary=True,
        )

    X, y = make_classification(
        n_samples=1200,
        n_features=24,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.99, 0.01],
        flip_y=0.0,
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )
    return DatasetBundle(
        name="synthetic_imbalanced_99_1",
        X=X.astype(float),
        y=y.astype(int),
        description=(
            "Offline fallback synthetic dataset with 99:1 imbalance. Replace with "
            "Covertype for the final report when downloaded data is available."
        ),
        is_binary=True,
    )


def load_default_bundles() -> list[DatasetBundle]:
    """Return the three dataset bundles used by Person 3 scripts."""
    return [
        load_breast_cancer_bundle(),
        load_digits_binary_bundle(),
        load_imbalanced_bundle(),
    ]


def train_test_scaled_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split with train-only standardization."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def random_oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling for severe class imbalance.

    This avoids adding an ``imbalanced-learn`` dependency. The minority class is
    sampled with replacement until all classes have the same count.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    X_parts = []
    y_parts = []
    for cls in classes:
        indices = np.flatnonzero(y == cls)
        if len(indices) < max_count:
            sampled = rng.choice(indices, size=max_count, replace=True)
        else:
            sampled = indices
        X_parts.append(X[sampled])
        y_parts.append(y[sampled])
    X_resampled = np.vstack(X_parts)
    y_resampled = np.concatenate(y_parts)
    permutation = rng.permutation(len(y_resampled))
    return X_resampled[permutation], y_resampled[permutation]


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute accuracy, macro F1, and ROC-AUC when possible."""
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        try:
            if proba.shape[1] == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_test, proba[:, 1]))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def add_label_noise(
    y: np.ndarray,
    noise_fraction: float,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Randomly flip a fraction of labels in a binary classification target."""
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between 0 and 1.")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("add_label_noise currently supports binary targets only.")

    noisy = y.copy()
    rng = np.random.default_rng(random_state)
    n_flip = int(round(noise_fraction * len(y)))
    if n_flip == 0:
        return noisy
    flip_indices = rng.choice(len(y), size=n_flip, replace=False)
    noisy[flip_indices] = np.where(noisy[flip_indices] == classes[0], classes[1], classes[0])
    return noisy


def save_results_table(rows: list[dict], filename: str) -> Path:
    """Save experiment rows as CSV and return the path."""
    ensure_output_dirs()
    output_path = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
