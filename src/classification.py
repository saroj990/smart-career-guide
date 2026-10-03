"""
Day 5 baseline classification — same logic as notebooks/day05/day05_baseline_classification.ipynb.

Import from notebooks:
  from src.classification import fit_baseline, evaluate_multiclass, save_baseline_run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DEFAULT_BASELINE_KWARGS: dict[str, Any] = {
    "max_iter": 1000,
    "random_state": 42,
    "C": 1.0,
}


def build_baseline_classifier(**kwargs: Any) -> LogisticRegression:
    """Multinomial logistic regression baseline for multi-class Career."""
    params = {**DEFAULT_BASELINE_KWARGS, **kwargs}
    return LogisticRegression(**params)


def fit_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any,
) -> LogisticRegression:
    model = build_baseline_classifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_multiclass(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Test-set metrics only — use for honest reporting."""
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix (test set)",
    figsize: tuple[float, float] = (12, 10),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted career")
    ax.set_ylabel("True career")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def save_baseline_run(
    model: LogisticRegression,
    metrics: dict[str, Any],
    career_classes: list[str],
    project_root: Path,
    run_name: str = "day05_baseline",
) -> dict[str, Path]:
    """Persist metrics JSON, confusion matrix figure, and fitted model."""
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    models_dir = project_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "LogisticRegression",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "n_classes": len(career_classes),
        "career_classes": career_classes,
    }
    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = metrics_dir / f"{run_name}_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    cm = np.asarray(metrics["confusion_matrix"])
    fig = plot_confusion_matrix(cm, career_classes)
    figure_path = figures_dir / f"{run_name}_confusion_matrix.png"
    fig.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    model_path = models_dir / f"{run_name}_logistic.joblib"
    joblib.dump(
        {"model": model, "career_classes": career_classes},
        model_path,
    )

    return {
        "metrics": metrics_path,
        "report": report_path,
        "confusion_figure": figure_path,
        "model": model_path,
    }
