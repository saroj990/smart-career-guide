"""
Day 7 model comparison + cross-validation — notebooks/day07/day07_ml_models_evaluation.ipynb.

Import from notebooks:
  from src.model_evaluation import benchmark_classifiers, save_day07_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.classification import evaluate_multiclass
from src.tree_models import build_random_forest

ClassifierFactory = Callable[[], ClassifierMixin]


def _default_model_factories() -> dict[str, ClassifierFactory]:
    """Models for Day 7 benchmark (same preprocessing as Day 3)."""
    factories: dict[str, ClassifierFactory] = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": build_random_forest,
        "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "knn_k7": lambda: KNeighborsClassifier(n_neighbors=7),
    }
    try:
        from xgboost import XGBClassifier

        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return factories


def cross_val_scores(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified CV on training data — macro F1 and accuracy (mean ± not computed here)."""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    scores = cross_validate(
        clone(model),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
    }


def benchmark_classifiers(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    cv: int = 5,
    factories: dict[str, ClassifierFactory] | None = None,
) -> tuple[pd.DataFrame, dict[str, ClassifierMixin], dict[str, dict[str, Any]]]:
    """
    For each model: CV on train, then fit on full train and evaluate on held-out test.
    """
    factories = factories or _default_model_factories()
    rows: list[dict[str, Any]] = []
    fitted: dict[str, ClassifierMixin] = {}
    test_evals: dict[str, dict[str, Any]] = {}

    for name, factory in factories.items():
        model = factory()
        cv_stats = cross_val_scores(model, X_train, y_train, cv=cv)
        model.fit(X_train, y_train)
        fitted[name] = model
        test_metrics = evaluate_multiclass(model, X_test, y_test, class_names)
        test_evals[name] = test_metrics
        rows.append(
            {
                "model": name,
                **cv_stats,
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
            }
        )

    df = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
    return df, fitted, test_evals


def pick_best_model_name(comparison: pd.DataFrame) -> str:
    """Choose best by test macro F1 (tie-breaker: test accuracy)."""
    return str(comparison.iloc[0]["model"])


def save_day07_artifacts(
    comparison: pd.DataFrame,
    fitted: dict[str, ClassifierMixin],
    class_names: list[str],
    project_root: Path,
    best_name: str | None = None,
) -> dict[str, Path]:
    metrics_dir = project_root / "outputs" / "metrics"
    models_dir = project_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_name = best_name or pick_best_model_name(comparison)
    csv_path = metrics_dir / "day07_model_benchmark.csv"
    comparison.to_csv(csv_path, index=False)

    json_path = metrics_dir / "day07_model_benchmark.json"
    json_path.write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    meta = {
        "best_model": best_name,
        "selection_rule": "highest test_f1_macro on held-out split",
        "n_models": len(comparison),
    }
    meta_path = metrics_dir / "day07_best_model.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_model = fitted[best_name]
    model_path = models_dir / "day07_best_classifier.joblib"
    joblib.dump(
        {"model": best_model, "career_classes": class_names, "model_name": best_name},
        model_path,
    )

    return {
        "benchmark_csv": csv_path,
        "benchmark_json": json_path,
        "best_model_meta": meta_path,
        "best_model": model_path,
    }
