"""Experiment 1: Baseline -- unpruned single tree vs. decision stump vs.
a scikit-learn reference tree.

Establishes the foundation the rest of an ensemble-methods study stands
on: a single unpruned tree overfits (low bias, high variance), a stump
underfits (high bias, low variance), and both motivate why ensembling
is useful. All modeling here (scaling, splitting, the tree itself) is
implemented from scratch in this project; scikit-learn is used only as
a reference model for validation and for its metric functions
(accuracy_score, f1_score, roc_auc_score), consistent with the "no
sklearn implementation code" requirement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.trees.decision_tree import DecisionTree, DecisionStump
from src.utils.preprocessing import StandardScaler, handle_missing_values, train_test_split

RESULTS_DIR = Path("experiments/results")
RANDOM_STATE = 42


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weighting for imbalanced datasets.

    Routes through the sample_weight machinery already built into
    DecisionTree.fit(), so no new modeling code is needed to handle
    class imbalance.

    Args:
        y: Class labels, shape (n_samples,).

    Returns:
        Per-sample weights, shape (n_samples,), inversely proportional
        to each sample's class frequency.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_per_class = {c: total / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight_per_class[label] for label in y])


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, binary: bool) -> dict:
    """Compute accuracy, macro-F1, and AUC-ROC for a fitted model.

    Args:
        model: Any object exposing predict() and predict_proba().
        X_test: Held-out feature matrix.
        y_test: Held-out labels.
        binary: Whether this is a binary classification problem
            (controls how AUC-ROC is computed).

    Returns:
        Dict with keys "accuracy", "macro_f1", "auc_roc".
    """
    preds = model.predict(X_test)
    result = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    }
    if binary:
        proba = model.predict_proba(X_test)[:, 1]
        result["auc_roc"] = roc_auc_score(y_test, proba)
    else:
        proba = model.predict_proba(X_test)
        result["auc_roc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    return result


def run_baseline_experiment(
    X: np.ndarray, y: np.ndarray, dataset_name: str, imbalanced: bool = False
) -> pd.DataFrame:
    """Train MyTree / MyStump / a sklearn reference tree and report metrics.

    Args:
        X: Raw feature matrix (may contain NaNs).
        y: Class labels.
        dataset_name: Label used in the results table.
        imbalanced: If True, applies inverse-frequency class weighting
            (passed through sample_weight) as the imbalance treatment.

    Returns:
        A DataFrame with one row per model (MyTree, MyStump, sklearn),
        columns: accuracy, macro_f1, auc_roc, model, dataset.
    """
    X = handle_missing_values(X, strategy="median_impute")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    sample_weight = compute_class_weights(y_train) if imbalanced else None
    binary = len(np.unique(y)) == 2

    my_tree = DecisionTree(max_depth=None, random_state=RANDOM_STATE)
    my_tree.fit(X_train_s, y_train, sample_weight=sample_weight)

    my_stump = DecisionStump(random_state=RANDOM_STATE)
    my_stump.fit(X_train_s, y_train, sample_weight=sample_weight)

    sk_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    sk_tree.fit(X_train_s, y_train, sample_weight=sample_weight)

    rows = []
    for model, name in [(my_tree, "MyTree"), (my_stump, "MyStump"), (sk_tree, "sklearn")]:
        metrics = evaluate(model, X_test_s, y_test, binary)
        metrics["model"] = name
        metrics["dataset"] = dataset_name
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    """Run the baseline experiment on the breast cancer dataset and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    df = run_baseline_experiment(X, y, dataset_name="breast_cancer_wisconsin", imbalanced=False)

    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "baseline_breast_cancer.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR / 'baseline_breast_cancer.csv'}")


if __name__ == "__main__":
    main()
