"""
Day 9 ML vs ANN comparison — notebooks/day09/day09_ml_vs_ann.ipynb.

  from src.model_comparison import build_comparison_table, save_production_model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.model_evaluation import fit_named_classifier


ModelKind = Literal["sklearn", "keras"]


def _load_day07_benchmark(project_root: Path) -> pd.DataFrame:
    """Read Day 7 comparison CSV (must exist before Day 9)."""
    path = project_root / "outputs" / "metrics" / "day07_model_benchmark.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run Day 7 first.")
    return pd.read_csv(path)


def build_comparison_table(
    project_root: Path,
    ann_test_accuracy: float | None = None,
    ann_test_f1_macro: float | None = None,
) -> pd.DataFrame:
    """
    Merge Day 7 sklearn/XGBoost rows with the Day 8 ANN test metrics.
    """
    bench = _load_day07_benchmark(project_root)
    ml_rows = bench.copy()
    ml_rows["family"] = "classic_ml"
    ml_rows["interpretability"] = ml_rows["model"].apply(
        lambda m: "high" if m == "logistic_regression" else "medium"
    )
    ml_rows["training_cost"] = "low"

    if ann_test_accuracy is None or ann_test_f1_macro is None:
        metrics_path = project_root / "outputs" / "metrics" / "day08_ann_metrics.json"
        if metrics_path.is_file():
            ann_meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            ann_test_accuracy = float(ann_meta["accuracy"])
            ann_test_f1_macro = float(ann_meta["f1_macro"])
        else:
            raise FileNotFoundError("Day 8 ANN metrics missing. Run Day 8 first.")

    ann_row = pd.DataFrame(
        [
            {
                "model": "keras_ann",
                "cv_accuracy_mean": None,
                "cv_accuracy_std": None,
                "cv_f1_macro_mean": None,
                "cv_f1_macro_std": None,
                "test_accuracy": ann_test_accuracy,
                "test_f1_macro": ann_test_f1_macro,
                "test_f1_weighted": None,
                "family": "deep_learning",
                "interpretability": "low",
                "training_cost": "medium",
            }
        ]
    )
    combined = pd.concat([ml_rows, ann_row], ignore_index=True)
    return combined.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)


def pick_production_row(comparison: pd.DataFrame) -> pd.Series:
    """Best by test macro F1 (same rule as Day 7)."""
    return comparison.iloc[0]


def save_production_model(
    comparison: pd.DataFrame,
    project_root: Path,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    career_classes: list[str] | None = None,
) -> dict[str, Path]:
    """
    Persist final model choice for the app (Day 13+).
    Sklearn winner → copy reference; keras winner → meta points to day08_ann.keras.
    """
    winner = pick_production_row(comparison)
    winner_name = str(winner["model"])
    metrics_dir = project_root / "outputs" / "metrics"
    models_dir = project_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = metrics_dir / "day09_final_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    rationale = {
        "selected_model": winner_name,
        "selection_rule": "highest test_f1_macro on held-out split (Day 3 random_state=42)",
        "test_accuracy": float(winner["test_accuracy"]),
        "test_f1_macro": float(winner["test_f1_macro"]),
        "notes": (
            "ANN may underperform tree ensembles on small tabular data — document honestly in report."
        ),
    }
    rationale_path = metrics_dir / "day09_model_selection.json"
    rationale_path.write_text(json.dumps(rationale, indent=2), encoding="utf-8")

    if winner_name == "keras_ann":
        kind: ModelKind = "keras"
        model_ref = "models/day08_ann.keras"
        bundle = {
            "kind": kind,
            "model_path": model_ref,
            "model_name": winner_name,
        }
    else:
        kind = "sklearn"
        day07_path = models_dir / "day07_best_classifier.joblib"
        day07 = joblib.load(day07_path) if day07_path.is_file() else None
        if day07 and str(day07.get("model_name")) == winner_name:
            sklearn_model = day07["model"]
            classes = day07["career_classes"]
        else:
            if X_train is None or y_train is None or career_classes is None:
                raise ValueError(
                    "Sklearn winner differs from Day 7 artifact — pass X_train, y_train, career_classes."
                )
            sklearn_model = fit_named_classifier(winner_name, X_train, y_train)
            classes = career_classes
        bundle = {
            "kind": kind,
            "model": sklearn_model,
            "career_classes": classes,
            "model_name": winner_name,
        }

    production_path = models_dir / "production_classifier.joblib"
    joblib.dump(bundle, production_path)

    return {
        "comparison_csv": comparison_path,
        "selection_json": rationale_path,
        "production_bundle": production_path,
    }
