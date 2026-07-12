"""Experiment 3: Random Forest scaling.

Runs two sweeps required by the project brief:
1. Vary ``n_estimators`` while keeping ``max_depth=None``.
2. Vary ``max_depth`` while keeping ``n_estimators=100``.

The script saves CSV tables to ``results/`` and plots to ``figures/``.
Run from the repository root:

    python experiments/rf_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.rf_utils import (
    FIGURES_DIR,
    RANDOM_STATE,
    ensure_output_dirs,
    evaluate_classifier,
    load_default_bundles,
    save_results_table,
    train_test_scaled_split,
)
from src.bagging.random_forest import RandomForestClassifier


def run_n_estimators_sweep(
    estimator_values: list[int] | None = None,
    n_jobs: int = 1,
    max_depth: int | None = None,
    bundles=None,
) -> list[dict]:
    """Evaluate RF accuracy/OOB accuracy as the number of trees increases."""
    if estimator_values is None:
        estimator_values = [1, 5, 10, 25, 50, 75, 100, 150, 200]

    rows: list[dict] = []
    if bundles is None:
        bundles = load_default_bundles()
    for bundle in bundles:
        X_train, X_test, y_train, y_test = train_test_scaled_split(bundle.X, bundle.y)
        for n_estimators in estimator_values:
            start = time.perf_counter()
            forest = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            )
            forest.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            metrics = evaluate_classifier(forest, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "sweep": "n_estimators",
                    "n_estimators": n_estimators,
                    "max_depth": "None",
                    "test_accuracy": metrics["accuracy"],
                    "test_f1_macro": metrics["f1_macro"],
                    "test_auc_roc": metrics["auc_roc"],
                    "oob_accuracy": forest.oob_score_,
                    "fit_seconds": elapsed,
                }
            )
    return rows


def run_max_depth_sweep(
    depth_values: list[int] | None = None,
    n_jobs: int = 1,
    bundles=None,
) -> list[dict]:
    """Evaluate RF accuracy/OOB accuracy as tree depth changes."""
    if depth_values is None:
        depth_values = list(range(1, 21))

    rows: list[dict] = []
    if bundles is None:
        bundles = load_default_bundles()
    for bundle in bundles:
        X_train, X_test, y_train, y_test = train_test_scaled_split(bundle.X, bundle.y)
        for max_depth in depth_values:
            start = time.perf_counter()
            forest = RandomForestClassifier(
                n_estimators=100,
                max_depth=max_depth,
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            )
            forest.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            metrics = evaluate_classifier(forest, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "sweep": "max_depth",
                    "n_estimators": 100,
                    "max_depth": max_depth,
                    "test_accuracy": metrics["accuracy"],
                    "test_f1_macro": metrics["f1_macro"],
                    "test_auc_roc": metrics["auc_roc"],
                    "oob_accuracy": forest.oob_score_,
                    "fit_seconds": elapsed,
                }
            )
    return rows


def plot_sweep(rows: list[dict], sweep_name: str, x_column: str, filename: str) -> Path:
    """Create a line plot for test accuracy and OOB accuracy."""
    ensure_output_dirs()
    output_path = FIGURES_DIR / filename
    datasets = sorted({row["dataset"] for row in rows if row["sweep"] == sweep_name})

    plt.figure(figsize=(8, 5))
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset and row["sweep"] == sweep_name]
        subset = sorted(subset, key=lambda item: int(item[x_column]))
        x_values = [int(row[x_column]) for row in subset]
        test_values = [row["test_accuracy"] for row in subset]
        oob_values = [row["oob_accuracy"] for row in subset]
        plt.plot(x_values, test_values, marker="o", label=f"{dataset} test")
        plt.plot(x_values, oob_values, marker="x", linestyle="--", label=f"{dataset} OOB")

    plt.xlabel(x_column)
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest scaling by {x_column}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a smaller smoke-test version.")
    args = parser.parse_args()

    ensure_output_dirs()
    if args.fast:
        fast_bundles = load_default_bundles()[:1]
        estimator_rows = run_n_estimators_sweep(estimator_values=[1, 3, 5], n_jobs=1, max_depth=4, bundles=fast_bundles)
        depth_rows = run_max_depth_sweep(depth_values=[1, 3, 5], n_jobs=1, bundles=fast_bundles)
    else:
        estimator_rows = run_n_estimators_sweep(n_jobs=-1)
        depth_rows = run_max_depth_sweep(n_jobs=-1)
    all_rows = estimator_rows + depth_rows
    csv_path = save_results_table(all_rows, "rf_scaling_results.csv")
    fig_1 = plot_sweep(all_rows, "n_estimators", "n_estimators", "rf_scaling_n_estimators.png")
    fig_2 = plot_sweep(all_rows, "max_depth", "max_depth", "rf_scaling_max_depth.png")
    print(f"Saved results: {csv_path}")
    print(f"Saved figures: {fig_1}, {fig_2}")


if __name__ == "__main__":
    main()
