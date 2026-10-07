"""
Day 6 decision tree + random forest — notebooks/day06/day06_tree_and_forest.ipynb.

Import from notebooks:
  from src.tree_models import fit_tree_models, random_forest_importance_df, save_day06_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.classification import evaluate_multiclass, plot_confusion_matrix

DEFAULT_TREE_KWARGS: dict[str, Any] = {
    "max_depth": 12,
    "random_state": 42,
}

DEFAULT_RF_KWARGS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}


def build_decision_tree(**kwargs: Any) -> DecisionTreeClassifier:
    """Single decision tree with project defaults (max_depth=12)."""
    params = {**DEFAULT_TREE_KWARGS, **kwargs}
    return DecisionTreeClassifier(**params)


def build_random_forest(**kwargs: Any) -> RandomForestClassifier:
    """Random forest with project defaults (200 trees)."""
    params = {**DEFAULT_RF_KWARGS, **kwargs}
    return RandomForestClassifier(**params)


def fit_tree_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Train a single tree and a random forest on the same train split."""
    tree = build_decision_tree()
    forest = build_random_forest()
    tree.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    return {"decision_tree": tree, "random_forest": forest}


def random_forest_importance_df(
    forest: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Global feature importances from the fitted forest (Gini-based)."""
    imp = forest.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Random Forest — top feature importances",
) -> plt.Figure:
    """Horizontal bar chart of Gini importances (global, not per student)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evaluate_named_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Run test evaluation for each named model."""
    return {
        name: evaluate_multiclass(model, X_test, y_test, class_names)
        for name, model in models.items()
    }


def comparison_rows(
    evals: dict[str, dict[str, Any]],
) -> list[dict[str, float | str]]:
    """Flatten evaluate_multiclass dicts into table rows."""
    rows: list[dict[str, float | str]] = []
    for name, metrics in evals.items():
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )
    return rows


def save_day06_artifacts(
    forest: RandomForestClassifier,
    importance_df: pd.DataFrame,
    comparison: pd.DataFrame,
    evals: dict[str, dict[str, Any]],
    class_names: list[str],
    project_root: Path,
) -> dict[str, Path]:
    """Write Day 6 CSVs, plots, and the fitted forest under models/."""
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    models_dir = project_root / "models"
    for d in (metrics_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    cmp_path = metrics_dir / "day06_model_comparison.csv"
    comparison.to_csv(cmp_path, index=False)

    summary_path = metrics_dir / "day06_model_comparison.json"
    summary_path.write_text(
        json.dumps(comparison_rows(evals), indent=2),
        encoding="utf-8",
    )

    imp_path = metrics_dir / "day06_rf_feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)

    fig = plot_feature_importance(importance_df)
    fig_path = figures_dir / "day06_rf_feature_importance.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    cm = np.asarray(evals["random_forest"]["confusion_matrix"])
    cm_fig = plot_confusion_matrix(cm, class_names, title="Random Forest — test confusion matrix")
    cm_path = figures_dir / "day06_rf_confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(cm_fig)

    model_path = models_dir / "day06_random_forest.joblib"
    joblib.dump({"model": forest, "career_classes": class_names}, model_path)

    return {
        "comparison_csv": cmp_path,
        "comparison_json": summary_path,
        "importance_csv": imp_path,
        "importance_figure": fig_path,
        "confusion_figure": cm_path,
        "model": model_path,
    }
