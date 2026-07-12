"""Experiment 5: noise robustness for Random Forest and AdaBoost.

Randomly flips 5%, 10%, and 20% of training labels, trains on corrupted labels,
and evaluates on the clean test set. Person 3 owns the Random Forest side of the
experiment. AdaBoost is included as an integration hook and will run once the
team's ``src.boosting.adaboost.AdaBoostClassifier`` is implemented.

Run from the repository root:

    python experiments/noise_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.rf_utils import (
    FIGURES_DIR,
    RANDOM_STATE,
    add_label_noise,
    ensure_output_dirs,
    evaluate_classifier,
    load_breast_cancer_bundle,
    load_digits_binary_bundle,
    save_results_table,
    train_test_scaled_split,
)
from src.bagging.random_forest import RandomForestClassifier


def _try_build_team_adaboost(n_estimators: int, random_state: int):
    try:
        from src.boosting.adaboost import AdaBoostClassifier  # type: ignore

        return AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    except Exception:
        return None


def run_noise_robustness(
    noise_levels: list[float] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = 8,
    n_jobs: int = -1,
) -> list[dict]:
    """Run label-noise robustness experiment on binary datasets."""
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20]

    rows: list[dict] = []
    bundles = [load_breast_cancer_bundle(), load_digits_binary_bundle()]

    for bundle in bundles:
        X_train, X_test, y_train, y_test = train_test_scaled_split(bundle.X, bundle.y)
        for noise_fraction in noise_levels:
            noisy_y_train = add_label_noise(
                y_train,
                noise_fraction=noise_fraction,
                random_state=RANDOM_STATE + int(noise_fraction * 1000),
            )

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            )
            rf.fit(X_train, noisy_y_train)
            rf_metrics = evaluate_classifier(rf, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "model": "team_random_forest",
                    "noise_fraction": noise_fraction,
                    "accuracy": rf_metrics["accuracy"],
                    "f1_macro": rf_metrics["f1_macro"],
                    "auc_roc": rf_metrics["auc_roc"],
                    "oob_accuracy_on_noisy_train": rf.oob_score_,
                    "status": "ok",
                }
            )

            adaboost = _try_build_team_adaboost(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
            )
            if adaboost is None:
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": "team_adaboost",
                        "noise_fraction": noise_fraction,
                        "accuracy": np.nan,
                        "f1_macro": np.nan,
                        "auc_roc": np.nan,
                        "oob_accuracy_on_noisy_train": np.nan,
                        "status": "unavailable_until_teammate_module_is_implemented",
                    }
                )
            else:
                try:
                    adaboost.fit(X_train, noisy_y_train)
                    ada_metrics = evaluate_classifier(adaboost, X_test, y_test)
                    rows.append(
                        {
                            "dataset": bundle.name,
                            "model": "team_adaboost",
                            "noise_fraction": noise_fraction,
                            "accuracy": ada_metrics["accuracy"],
                            "f1_macro": ada_metrics["f1_macro"],
                            "auc_roc": ada_metrics["auc_roc"],
                            "oob_accuracy_on_noisy_train": np.nan,
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": bundle.name,
                            "model": "team_adaboost",
                            "noise_fraction": noise_fraction,
                            "accuracy": np.nan,
                            "f1_macro": np.nan,
                            "auc_roc": np.nan,
                            "oob_accuracy_on_noisy_train": np.nan,
                            "status": f"failed: {type(exc).__name__}: {exc}",
                        }
                    )
    return rows


def plot_noise_results(rows: list[dict]) -> list[str]:
    """Create accuracy degradation plots."""
    ensure_output_dirs()
    df = pd.DataFrame(rows)
    output_paths: list[str] = []
    for dataset, subset in df[df["status"] == "ok"].groupby("dataset"):
        plt.figure(figsize=(7, 4.5))
        for model, model_rows in subset.groupby("model"):
            model_rows = model_rows.sort_values("noise_fraction")
            plt.plot(
                model_rows["noise_fraction"],
                model_rows["accuracy"],
                marker="o",
                label=model,
            )
        plt.xlabel("Training label noise fraction")
        plt.ylabel("Clean test accuracy")
        plt.title(f"Noise robustness: {dataset}")
        plt.legend()
        plt.tight_layout()
        output_path = FIGURES_DIR / f"noise_robustness_{dataset}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        output_paths.append(str(output_path))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a smaller smoke-test version.")
    args = parser.parse_args()

    ensure_output_dirs()
    rows = run_noise_robustness(
        noise_levels=[0.0, 0.10] if args.fast else None,
        n_estimators=15 if args.fast else 100,
        max_depth=5 if args.fast else 8,
        n_jobs=1 if args.fast else -1,
    )
    csv_path = save_results_table(rows, "noise_robustness_results.csv")
    figure_paths = plot_noise_results(rows)
    print(f"Saved results: {csv_path}")
    print("Saved figures:")
    for figure_path in figure_paths:
        print(f"  - {figure_path}")


if __name__ == "__main__":
    main()
