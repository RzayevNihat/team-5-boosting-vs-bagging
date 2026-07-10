"""Experiment 6 -- bias-variance decomposition for 0-1 loss.

The experiment uses the Kong and Dietterich classification decomposition:
for each test point, the main prediction is the modal prediction across
models trained on bootstrap replicates. Bias is the error of that main
prediction, and variance is the average disagreement with it.

Run from the repository root:

    python experiments/bias_variance.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting.adaboost import AdaBoostClassifier, DecisionStump
from src.utils.preprocessing import DatasetBundle, load_wdbc

try:
    from src.bagging.random_forest import RandomForestClassifier as OwnRandomForest
except ImportError:
    OwnRandomForest = None  # type: ignore[assignment]


ModelFactory = Callable[[int], Any]


@dataclass(frozen=True)
class BiasVarianceConfig:
    """Configuration for Experiment 6."""

    n_rounds: int = 100
    n_estimators: int = 100
    test_size: float = 0.5
    random_state: int = 42
    n_jobs: int = 1
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("results"))


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


def derive_seed(base_seed: int, offset: int) -> int:
    """Derive deterministic independent seeds for bootstrap rounds."""
    sequence = np.random.SeedSequence([int(base_seed), int(offset)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy and path objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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


def _is_classifier(model: Any) -> bool:
    """Check the minimal fit/predict interface used by this experiment."""
    return callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None))


def _fit_and_predict_round(
    factory: ModelFactory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train one model on one bootstrap replicate and predict the test set."""
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    sample = rng.integers(0, n_train, size=n_train)

    model = factory(seed)
    if not _is_classifier(model):
        raise TypeError(
            f"Factory produced {type(model).__name__}, which lacks fit/predict."
        )

    model.fit(X_train[sample], y_train[sample])
    predictions = np.asarray(model.predict(X_test))
    return np.searchsorted(classes, predictions).astype(np.int64)


def bias_variance_decomposition(
    factory: ModelFactory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: BiasVarianceConfig,
) -> Dict[str, Any]:
    """Estimate expected 0-1 loss, bias^2, and variance by bootstrapping."""
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Train and test sets must be non-empty.")
    if config.n_rounds < 1:
        raise ValueError("n_rounds must be positive.")

    classes = np.unique(np.concatenate([y_train, y_test]))
    y_test_encoded = np.searchsorted(classes, y_test)
    jobs = [derive_seed(config.random_state, round_idx) for round_idx in range(config.n_rounds)]

    if config.n_jobs == 1:
        prediction_rows = [
            _fit_and_predict_round(factory, X_train, y_train, X_test, classes, seed)
            for seed in jobs
        ]
    else:
        prediction_rows = Parallel(n_jobs=config.n_jobs)(
            delayed(_fit_and_predict_round)(
                factory,
                X_train,
                y_train,
                X_test,
                classes,
                seed,
            )
            for seed in jobs
        )
    predictions = np.vstack(prediction_rows)

    votes = np.stack([(predictions == index).sum(axis=0) for index in range(classes.size)])
    main_prediction = np.argmax(votes, axis=0)

    disagrees_truth = predictions != y_test_encoded[None, :]
    disagrees_main = predictions != main_prediction[None, :]
    bias_squared = float(np.mean(main_prediction != y_test_encoded))
    variance = float(disagrees_main.mean())
    expected_loss = float(disagrees_truth.mean())

    per_class: Dict[str, Dict[str, float]] = {}
    for index, label in enumerate(classes):
        mask = y_test_encoded == index
        if not np.any(mask):
            continue
        class_bias = float(np.mean(main_prediction[mask] != index))
        per_class[str(label)] = {
            "expected_loss": float(disagrees_truth[:, mask].mean()),
            "bias": class_bias,
            "bias_squared": class_bias,
            "variance": float(disagrees_main[:, mask].mean()),
        }

    return {
        "expected_loss": expected_loss,
        "bias": bias_squared,
        "bias_squared": bias_squared,
        "variance": variance,
        "per_class": per_class,
        "predictions": predictions,
        "main_prediction": classes[main_prediction],
        "classes": classes,
    }


def build_model_factories(config: BiasVarianceConfig) -> Dict[str, ModelFactory]:
    """Construct the model factories whose bias/variance is decomposed."""
    factories: Dict[str, ModelFactory] = {
        "Decision stump (ours)": lambda seed: DecisionStump(random_state=seed),
        "AdaBoost (ours)": lambda seed: AdaBoostClassifier(
            n_estimators=config.n_estimators,
            random_state=seed,
        ),
        "Random Forest (sklearn ref)": lambda seed: SklearnRandomForest(
            n_estimators=config.n_estimators,
            random_state=seed,
            n_jobs=1,
        ),
    }
    if OwnRandomForest is not None:
        factories["Random Forest (ours)"] = lambda seed: OwnRandomForest(
            n_estimators=config.n_estimators,
            random_state=seed,
        )
    return factories


def export_results(
    all_results: Dict[str, Dict[str, Any]],
    dataset: DatasetBundle,
    config: BiasVarianceConfig,
) -> None:
    """Persist scalar summaries to results/ as JSON and CSV."""
    scalar_keys = ("expected_loss", "bias_squared", "variance")
    payload = {
        "config": vars(config),
        "dataset": {
            "name": dataset.name,
            "task": dataset.task,
            "source": dataset.source,
            "notes": dataset.notes,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
        },
        "models": {
            name: {
                **{key: result[key] for key in scalar_keys},
                "per_class": result["per_class"],
            }
            for name, result in all_results.items()
        },
    }
    export_json(payload, config.output_dir / "bias_variance.json")

    rows = [
        {
            "dataset": dataset.name,
            "model": name,
            **{key: result[key] for key in scalar_keys},
        }
        for name, result in all_results.items()
    ]
    export_csv(rows, config.output_dir / "bias_variance.csv")
    logger.info("Results exported to %s/", config.output_dir)


def main(config: Optional[BiasVarianceConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Run Experiment 6 and export the decomposition table."""
    config = config or BiasVarianceConfig()
    dataset = load_wdbc(config.data_dir)
    X, y = dataset.X, dataset.y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Dataset: %s | train=%d, test=%d, rounds=%d",
        dataset.name,
        X_train.shape[0],
        X_test.shape[0],
        config.n_rounds,
    )
    logger.info("%-30s %12s %12s %12s", "model", "loss", "bias^2", "variance")

    all_results: Dict[str, Dict[str, Any]] = {}
    for name, factory in build_model_factories(config).items():
        result = bias_variance_decomposition(
            factory,
            X_train,
            y_train,
            X_test,
            y_test,
            config,
        )
        all_results[name] = result
        logger.info(
            "%-30s %12.4f %12.4f %12.4f",
            name,
            result["expected_loss"],
            result["bias_squared"],
            result["variance"],
        )
        for label, breakdown in result["per_class"].items():
            logger.info(
                "  class %-22s %12.4f %12.4f %12.4f",
                label,
                breakdown["expected_loss"],
                breakdown["bias_squared"],
                breakdown["variance"],
            )

    export_results(all_results, dataset, config)
    return all_results


if __name__ == "__main__":
    main()
