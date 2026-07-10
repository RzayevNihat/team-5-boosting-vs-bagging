"""Experiment 4 -- head-to-head comparison under fixed resources.

The comparison is run on the downloaded project datasets, not on a hidden
in-memory dataset. AdaBoost uses our from-scratch implementation. sklearn
tree/forest models are exported only as explicit reference baselines; they are
not presented as replacements for the team's required Decision Tree or Random
Forest implementations.

Run from the repository root:

    python experiments/head_to_head.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting.adaboost import AdaBoostClassifier
from src.utils.preprocessing import DatasetBundle, load_project_datasets

try:
    from src.trees.decision_tree import DecisionTree as OwnDecisionTree
except ImportError:
    OwnDecisionTree = None  # type: ignore[assignment]

try:
    from src.bagging.random_forest import RandomForestClassifier as OwnRandomForest
except ImportError:
    OwnRandomForest = None  # type: ignore[assignment]


METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "roc_auc",
)
ModelFactory = Callable[[int], Any]
FoldOutput = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata used to avoid mixing project and reference baselines."""

    name: str
    role: str
    factory: ModelFactory
    note: str


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for Experiment 4."""

    datasets: tuple[str, ...] = ("wdbc", "adult", "covertype")
    n_splits: int = 5
    n_estimators: int = 100
    learning_rate: float = 1.0
    random_state: int = 42
    n_jobs: int = 1
    confidence_level: float = 0.95
    alpha: float = 0.05
    adult_max_samples: Optional[int] = 5000
    covertype_max_samples: Optional[int] = 5000
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
    """Derive deterministic independent seeds for folds and models."""
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


def _predict_scores(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Return probability-like scores when the model exposes them."""
    if callable(getattr(model, "predict_proba", None)):
        return np.asarray(model.predict_proba(X), dtype=float)
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(X), dtype=float)
    return None


def _compute_roc_auc(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    model_classes: Optional[np.ndarray],
) -> float:
    """Compute binary or macro multi-class ROC-AUC when scores are available."""
    if y_score is None:
        return float("nan")

    y_true = np.asarray(y_true)
    present_classes = np.unique(y_true)
    if present_classes.size < 2:
        return float("nan")

    try:
        if present_classes.size == 2:
            if y_score.ndim == 1:
                positive_scores = y_score
            else:
                classes = np.asarray(model_classes) if model_classes is not None else present_classes
                positive_class = present_classes[-1]
                match = np.flatnonzero(classes == positive_class)
                column = int(match[0]) if match.size else y_score.shape[1] - 1
                positive_scores = y_score[:, column]
            return float(roc_auc_score(y_true, positive_scores))

        if y_score.ndim != 2:
            return float("nan")
        labels = np.asarray(model_classes) if model_classes is not None else present_classes
        return float(
            roc_auc_score(
                y_true,
                y_score,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    model_classes: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute all metrics required for the head-to-head experiment."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": _compute_roc_auc(y_true, y_score, model_classes),
    }


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return a t critical value, falling back to normal approximation."""
    try:
        from scipy import stats

        return float(stats.t.ppf(probability, degrees_of_freedom))
    except Exception:
        return 1.96


def _student_t_two_sided_pvalue(t_statistic: float, degrees_of_freedom: int) -> float:
    """Return a two-sided t-test p-value, with a normal fallback."""
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
    except Exception:
        return float(math.erfc(abs(t_statistic) / math.sqrt(2.0)))


def aggregate_scores(
    per_fold: Dict[str, np.ndarray],
    confidence_level: float,
) -> Dict[str, Dict[str, float]]:
    """Summarize per-fold metric arrays as mean/std/confidence interval."""
    summaries: Dict[str, Dict[str, float]] = {}
    for metric, values in per_fold.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summaries[metric] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            continue

        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        if finite.size > 1:
            tail = 0.5 + confidence_level / 2.0
            critical = _student_t_quantile(tail, finite.size - 1)
            half_width = critical * std / math.sqrt(finite.size)
        else:
            half_width = 0.0
        ci_low = max(0.0, mean - half_width)
        ci_high = min(1.0, mean + half_width)
        summaries[metric] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return summaries


def paired_t_tests_vs_reference(
    per_model: Dict[str, np.ndarray],
    reference: str,
    alpha: float,
) -> List[Dict[str, Any]]:
    """Run paired t-tests and Holm-correct p-values against one reference."""
    if reference not in per_model:
        raise KeyError(f"Reference model '{reference}' is not present in results.")

    reference_scores = np.asarray(per_model[reference], dtype=float)
    tests: List[Dict[str, Any]] = []
    for model_name, model_scores in per_model.items():
        if model_name == reference:
            continue
        model_scores = np.asarray(model_scores, dtype=float)
        mask = np.isfinite(reference_scores) & np.isfinite(model_scores)
        differences = reference_scores[mask] - model_scores[mask]
        if differences.size < 2:
            t_statistic = float("nan")
            p_value = float("nan")
        else:
            mean_difference = float(np.mean(differences))
            std_difference = float(np.std(differences, ddof=1))
            if std_difference <= 1e-15:
                t_statistic = 0.0 if abs(mean_difference) <= 1e-15 else math.inf
                p_value = 1.0 if t_statistic == 0.0 else 0.0
            else:
                t_statistic = mean_difference / (
                    std_difference / math.sqrt(differences.size)
                )
                p_value = _student_t_two_sided_pvalue(t_statistic, differences.size - 1)

        tests.append(
            {
                "model": model_name,
                "n_pairs": int(differences.size),
                "mean_difference_reference_minus_model": float(np.mean(differences))
                if differences.size
                else float("nan"),
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "p_value_holm": float("nan"),
                "significant": False,
            }
        )

    ordered = sorted(
        range(len(tests)),
        key=lambda index: (
            not math.isfinite(tests[index]["p_value"]),
            tests[index]["p_value"],
        ),
    )
    running_adjusted = 0.0
    m = len(tests)
    for rank, index in enumerate(ordered):
        p_value = tests[index]["p_value"]
        if not math.isfinite(p_value):
            adjusted = float("nan")
        else:
            adjusted = min(1.0, (m - rank) * p_value)
            running_adjusted = max(running_adjusted, adjusted)
            adjusted = running_adjusted
        tests[index]["p_value_holm"] = adjusted
        tests[index]["significant"] = bool(math.isfinite(adjusted) and adjusted <= alpha)

    return tests


def evaluate_fold(
    factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, float], FoldOutput]:
    """Train and score one model on one cross-validation fold."""
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = factory(seed)
    if not _is_classifier(model):
        raise TypeError(
            f"Factory produced {type(model).__name__}, which lacks fit/predict."
        )

    model.fit(X_train, y[train_idx])
    y_pred = np.asarray(model.predict(X_test))
    y_score = _predict_scores(model, X_test)
    model_classes = getattr(model, "classes_", None)
    scores = compute_metrics(y[test_idx], y_pred, y_score, model_classes)
    return scores, (test_idx, y[test_idx], y_pred)


def cross_validate_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    config: CrossValidationConfig,
) -> Dict[str, Any]:
    """Run stratified k-fold CV for one model."""
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    folds = list(splitter.split(X, y))
    jobs = [
        (train_idx, test_idx, derive_seed(config.random_state, fold_index))
        for fold_index, (train_idx, test_idx) in enumerate(folds)
    ]

    if config.n_jobs == 1:
        outcomes = [
            evaluate_fold(spec.factory, X, y, train_idx, test_idx, seed)
            for train_idx, test_idx, seed in jobs
        ]
    else:
        outcomes = Parallel(n_jobs=config.n_jobs)(
            delayed(evaluate_fold)(spec.factory, X, y, train_idx, test_idx, seed)
            for train_idx, test_idx, seed in jobs
        )

    for fold_index, (scores, _) in enumerate(outcomes, start=1):
        logger.info(
            "  fold %d | acc=%.4f  f1=%.4f  auc=%.4f",
            fold_index,
            scores["accuracy"],
            scores["macro_f1"],
            scores["roc_auc"],
        )

    results: Dict[str, Any] = {
        metric: np.array([scores[metric] for scores, _ in outcomes], dtype=float)
        for metric in METRIC_NAMES
    }
    results["raw"] = [raw for _, raw in outcomes]
    results["role"] = spec.role
    results["note"] = spec.note
    return results


def missing_required_custom_models() -> List[str]:
    """Return required project implementations that are not importable yet."""
    missing: List[str] = []
    if OwnDecisionTree is None:
        missing.append("src.trees.decision_tree.DecisionTree")
    if OwnRandomForest is None:
        missing.append("src.bagging.random_forest.RandomForestClassifier")
    return missing


def build_model_specs(config: CrossValidationConfig) -> List[ModelSpec]:
    """Construct project models and clearly labeled reference-only baselines."""
    specs: List[ModelSpec] = [
        ModelSpec(
            name="AdaBoost (ours)",
            role="project_implementation",
            factory=lambda seed: AdaBoostClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=seed,
            ),
            note="From-scratch SAMME AdaBoost implemented in src/boosting/adaboost.py.",
        )
    ]

    if OwnDecisionTree is not None:
        specs.append(
            ModelSpec(
                name="Single tree (ours)",
                role="project_implementation",
                factory=lambda seed: OwnDecisionTree(random_state=seed),
                note="Team DecisionTree implementation.",
            )
        )
    if OwnRandomForest is not None:
        specs.append(
            ModelSpec(
                name="Random Forest (ours)",
                role="project_implementation",
                factory=lambda seed: OwnRandomForest(
                    n_estimators=config.n_estimators,
                    random_state=seed,
                ),
                note="Team RandomForest implementation.",
            )
        )

    specs.extend(
        [
            ModelSpec(
                name="Single tree (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnDecisionTree(random_state=seed),
                note=(
                    "Reference baseline only; not a substitute for the required "
                    "from-scratch DecisionTree."
                ),
            ),
            ModelSpec(
                name="Random Forest, depth-1 (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnRandomForest(
                    n_estimators=config.n_estimators,
                    max_depth=1,
                    random_state=seed,
                    n_jobs=1,
                ),
                note=(
                    "Reference baseline only; uses depth-1 sklearn trees to match "
                    "AdaBoost weak-learner capacity."
                ),
            ),
            ModelSpec(
                name="Random Forest, full (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnRandomForest(
                    n_estimators=config.n_estimators,
                    random_state=seed,
                    n_jobs=1,
                ),
                note=(
                    "Reference baseline only; not a substitute for the required "
                    "from-scratch RandomForest."
                ),
            ),
        ]
    )
    return specs


def summarize(
    results: Dict[str, Dict[str, Any]],
    config: CrossValidationConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Log and return mean/std/CI summaries for every model and metric."""
    logger.info(
        "=== Summary (mean +/- std [%.0f%% CI] over %d folds) ===",
        100 * config.confidence_level,
        config.n_splits,
    )
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, scores in results.items():
        per_fold = {metric: scores[metric] for metric in METRIC_NAMES}
        summaries[name] = aggregate_scores(per_fold, config.confidence_level)

    for metric in METRIC_NAMES:
        logger.info("%s:", metric)
        for name, summary in summaries.items():
            entry = summary[metric]
            logger.info(
                "  %-40s %.4f +/- %.4f  [%.4f, %.4f]",
                name,
                entry["mean"],
                entry["std"],
                entry["ci_low"],
                entry["ci_high"],
            )
    return summaries


def run_significance_tests(
    results: Dict[str, Dict[str, Any]],
    reference: str,
    config: CrossValidationConfig,
    metric: str = "macro_f1",
) -> List[Dict[str, Any]]:
    """Compare every baseline against AdaBoost with paired t-tests."""
    per_model = {name: scores[metric] for name, scores in results.items()}
    tests = paired_t_tests_vs_reference(per_model, reference, config.alpha)

    logger.info(
        "=== Paired t-tests on fold %s vs '%s' (Holm-corrected, alpha=%.2f) ===",
        metric,
        reference,
        config.alpha,
    )
    for test in tests:
        verdict = "significant" if test["significant"] else "not significant"
        logger.info(
            "  vs %-34s diff=%.4f  t=%.3f  p=%.4f  p_holm=%.4f (%s)",
            test["model"],
            test["mean_difference_reference_minus_model"],
            test["t_statistic"],
            test["p_value"],
            test["p_value_holm"],
            verdict,
        )
    return tests


def dataset_payload(dataset: DatasetBundle) -> Dict[str, Any]:
    """Return report-friendly dataset metadata."""
    return {
        "name": dataset.name,
        "task": dataset.task,
        "source": dataset.source,
        "notes": dataset.notes,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
    }


def run_dataset(
    dataset: DatasetBundle,
    config: CrossValidationConfig,
) -> Dict[str, Any]:
    """Run Experiment 4 for one project dataset."""
    logger.info(
        "Dataset: %s (%d samples, %d features)",
        dataset.name,
        dataset.X.shape[0],
        dataset.X.shape[1],
    )
    logger.info(
        "Setup: %d-fold stratified CV | %d estimators | learning_rate=%.2f | seed=%d",
        config.n_splits,
        config.n_estimators,
        config.learning_rate,
        config.random_state,
    )

    results: Dict[str, Dict[str, Any]] = {}
    missing_models = missing_required_custom_models()
    if missing_models:
        logger.warning(
            "Required custom model(s) missing; sklearn rows are reference-only: %s",
            ", ".join(missing_models),
        )

    for spec in build_model_specs(config):
        logger.info("--- %s [%s] ---", spec.name, spec.role)
        results[spec.name] = cross_validate_model(spec, dataset.X, dataset.y, config)

    summaries = summarize(results, config)
    tests = run_significance_tests(results, "AdaBoost (ours)", config)
    return {
        "dataset": dataset_payload(dataset),
        "experiment_status": {
            "status": "partial" if missing_models else "complete",
            "missing_required_custom_models": missing_models,
            "reference_only_models_are_not_project_implementations": True,
        },
        "model_metadata": {
            spec.name: {"role": spec.role, "note": spec.note}
            for spec in build_model_specs(config)
        },
        "results": results,
        "summary": summaries,
        "significance_tests": tests,
    }


def export_results(
    all_results: Dict[str, Dict[str, Any]],
    config: CrossValidationConfig,
) -> None:
    """Persist the study to results/ as JSON and fold-level CSV."""
    export_json(
        {
            "config": vars(config),
            "datasets": {
                name: result["dataset"] for name, result in all_results.items()
            },
            "summary": {
                name: result["summary"] for name, result in all_results.items()
            },
            "experiment_status": {
                name: result["experiment_status"]
                for name, result in all_results.items()
            },
            "model_metadata": {
                name: result["model_metadata"]
                for name, result in all_results.items()
            },
            "significance_tests": {
                name: result["significance_tests"]
                for name, result in all_results.items()
            },
        },
        config.output_dir / "head_to_head.json",
    )

    rows = [
        {
            "dataset": dataset_name,
            "model": model_name,
            "role": scores["role"],
            "fold": fold + 1,
            **{metric: float(scores[metric][fold]) for metric in METRIC_NAMES},
        }
        for dataset_name, dataset_result in all_results.items()
        for model_name, scores in dataset_result["results"].items()
        for fold in range(config.n_splits)
    ]
    export_csv(rows, config.output_dir / "head_to_head_folds.csv")
    logger.info("Results exported to %s/", config.output_dir)


def main(config: Optional[CrossValidationConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Load downloaded project data, run Experiment 4, and export results."""
    config = config or CrossValidationConfig()
    datasets = load_project_datasets(
        names=config.datasets,
        data_dir=config.data_dir,
        adult_max_samples=config.adult_max_samples,
        covertype_max_samples=config.covertype_max_samples,
        random_state=config.random_state,
    )
    all_results = {
        dataset.name: run_dataset(dataset, config)
        for dataset in datasets
    }
    export_results(all_results, config)
    return all_results


if __name__ == "__main__":
    main()
