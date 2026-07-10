"""Experiment 2 -- AdaBoost scaling on the downloaded project datasets.

This script follows the project brief directly:

* use the datasets downloaded into ``data/`` by ``download_data.sh``;
* vary ``n_estimators`` from 1 to 200;
* record training and test accuracy through ``staged_predict``;
* export tables and error-curve figures for report writing.

Run from the repository root:

    python experiments/adaboost_scaling.py
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting.adaboost import AdaBoostClassifier
from src.utils.preprocessing import DatasetBundle, load_project_datasets


@dataclass(frozen=True)
class AdaBoostScalingConfig:
    """Configuration for Experiment 2."""

    datasets: tuple[str, ...] = ("wdbc", "adult", "covertype")
    max_estimators: int = 200
    step: int = 5
    test_size: float = 0.2
    learning_rate: float = 1.0
    random_state: int = 42
    adult_max_samples: Optional[int] = 5000
    covertype_max_samples: Optional[int] = 5000
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    figure_dir: Path = field(default_factory=lambda: Path("report") / "figures")


def setup_logger(name: str) -> logging.Logger:
    """Create a simple console logger for reproducible experiment logs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy and path objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def export_json(payload: Dict[str, Any], path: Path) -> None:
    """Write a JSON file, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(payload), file, indent=2)


def export_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a CSV file from row dictionaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimator_grid(max_estimators: int, step: int) -> List[int]:
    """Return the estimator counts reported in the final table."""
    if max_estimators < 1:
        raise ValueError("max_estimators must be positive.")
    if step < 1:
        raise ValueError("step must be positive.")

    values = [1]
    values.extend(range(step, max_estimators + 1, step))
    if values[-1] != max_estimators:
        values.append(max_estimators)
    return sorted(set(values))


def slugify(value: str) -> str:
    """Create a readable file-name slug from a dataset name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def _accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the compact metrics used in the scaling curve."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def collect_staged_scores(
    model: AdaBoostClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: AdaBoostScalingConfig,
) -> List[Dict[str, Any]]:
    """Collect staged train/test scores for the requested estimator grid."""
    wanted = set(estimator_grid(config.max_estimators, config.step))
    train_predictions = model.staged_predict(X_train)
    test_predictions = model.staged_predict(X_test)

    rows: List[Dict[str, Any]] = []
    for round_index, (train_pred, test_pred) in enumerate(
        zip(train_predictions, test_predictions),
        start=1,
    ):
        if round_index not in wanted:
            continue

        train_scores = _accuracy_and_f1(y_train, train_pred)
        test_scores = _accuracy_and_f1(y_test, test_pred)
        train_accuracy = train_scores["accuracy"]
        test_accuracy = test_scores["accuracy"]
        rows.append(
            {
                "n_estimators": round_index,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_error": 1.0 - train_accuracy,
                "test_error": 1.0 - test_accuracy,
                "train_macro_f1": train_scores["macro_f1"],
                "test_macro_f1": test_scores["macro_f1"],
            }
        )

    if not rows:
        raise RuntimeError("No staged AdaBoost predictions were collected.")
    return rows


def summarize_curve(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize the scaling curve for quick report writing."""
    best_test = max(rows, key=lambda row: (row["test_accuracy"], -row["n_estimators"]))
    best_f1 = max(rows, key=lambda row: (row["test_macro_f1"], -row["n_estimators"]))
    final = rows[-1]
    min_test_error = min(rows, key=lambda row: row["test_error"])
    overfit_gap = float(final["train_accuracy"] - final["test_accuracy"])

    return {
        "best_test_accuracy": best_test,
        "best_test_macro_f1": best_f1,
        "lowest_test_error": min_test_error,
        "final_round": final,
        "final_train_test_accuracy_gap": overfit_gap,
        "overfitting_note": interpret_overfitting(rows),
    }


def interpret_overfitting(rows: List[Dict[str, Any]]) -> str:
    """Return a short, evidence-based observation for the report."""
    best = max(rows, key=lambda row: (row["test_accuracy"], -row["n_estimators"]))
    final = rows[-1]
    accuracy_drop = float(best["test_accuracy"] - final["test_accuracy"])
    train_gain = float(final["train_accuracy"] - best["train_accuracy"])

    if accuracy_drop >= 0.02 and train_gain > 0.0:
        return (
            "Test accuracy drops after the best round while training accuracy "
            "continues to improve, which is a mild overfitting signal."
        )
    if accuracy_drop >= 0.01:
        return (
            "The curve shows a small late test-performance drop, but the effect "
            "is limited on this split."
        )
    return (
        "No strong overfitting is visible on this split; test performance stays "
        "close to its best value through the final rounds."
    )


def plot_error_curve(
    rows: List[Dict[str, Any]],
    dataset_name: str,
    path: Path,
) -> Optional[Path]:
    """Save the train/test error curve when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping plot export because matplotlib is unavailable: %s", exc)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [row["n_estimators"] for row in rows]
    train_errors = [row["train_error"] for row in rows]
    test_errors = [row["test_error"] for row in rows]

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(x_values, train_errors, marker="o", linewidth=1.8, label="Training error")
    plt.plot(x_values, test_errors, marker="s", linewidth=1.8, label="Test error")
    plt.xlabel("Number of AdaBoost stumps")
    plt.ylabel("Error rate")
    plt.title(f"AdaBoost Scaling on {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def run_dataset(
    dataset: DatasetBundle,
    config: AdaBoostScalingConfig,
) -> Dict[str, Any]:
    """Run the scaling experiment for one downloaded project dataset."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.test_size,
        stratify=dataset.y,
        random_state=config.random_state,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Dataset: %s | samples=%d, features=%d | train=%d, test=%d",
        dataset.name,
        dataset.X.shape[0],
        dataset.X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
    )

    model = AdaBoostClassifier(
        n_estimators=config.max_estimators,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
    ).fit(X_train, y_train)

    rows = collect_staged_scores(model, X_train, y_train, X_test, y_test, config)
    summary = summarize_curve(rows)
    figure_path = plot_error_curve(
        rows,
        dataset.name,
        config.figure_dir / f"adaboost_scaling_{slugify(dataset.name)}.png",
    )

    logger.info(
        "Best test accuracy on %s: %.4f at %d stumps",
        dataset.name,
        summary["best_test_accuracy"]["test_accuracy"],
        summary["best_test_accuracy"]["n_estimators"],
    )
    logger.info("Observation: %s", summary["overfitting_note"])

    return {
        "dataset": {
            "name": dataset.name,
            "task": dataset.task,
            "source": dataset.source,
            "notes": dataset.notes,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
        },
        "n_estimators_fit": int(len(model.estimators_)),
        "rows": rows,
        "summary": summary,
        "figure_path": figure_path,
    }


def run_experiment(config: AdaBoostScalingConfig) -> Dict[str, Any]:
    """Run Experiment 2 across the downloaded project datasets."""
    datasets = load_project_datasets(
        names=config.datasets,
        data_dir=config.data_dir,
        adult_max_samples=config.adult_max_samples,
        covertype_max_samples=config.covertype_max_samples,
        random_state=config.random_state,
    )
    results = [run_dataset(dataset, config) for dataset in datasets]
    return {
        "config": config,
        "datasets": [result["dataset"] for result in results],
        "results": {result["dataset"]["name"]: result for result in results},
    }


def export_results(result: Dict[str, Any], config: AdaBoostScalingConfig) -> None:
    """Persist the scaling curves to results/ and plots to report/figures/."""
    rows = []
    for dataset_name, dataset_result in result["results"].items():
        for row in dataset_result["rows"]:
            rows.append(
                {
                    "dataset": dataset_name,
                    "task": dataset_result["dataset"]["task"],
                    **row,
                }
            )

    export_csv(rows, config.output_dir / "adaboost_scaling.csv")
    export_json(
        {
            "config": vars(config),
            "datasets": result["datasets"],
            "summary": {
                name: dataset_result["summary"]
                for name, dataset_result in result["results"].items()
            },
            "figure_paths": {
                name: dataset_result["figure_path"]
                for name, dataset_result in result["results"].items()
            },
            "rows": rows,
        },
        config.output_dir / "adaboost_scaling.json",
    )
    logger.info("Results exported to %s/", config.output_dir)


def main(config: Optional[AdaBoostScalingConfig] = None) -> Dict[str, Any]:
    """Run Experiment 2, export artifacts, and return the result payload."""
    config = config or AdaBoostScalingConfig()
    result = run_experiment(config)
    export_results(result, config)
    return result


if __name__ == "__main__":
    main()
