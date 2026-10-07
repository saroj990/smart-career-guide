"""
Day 11 explainability — notebooks/day11/day11_explainability.ipynb.

  from src.explainability import top_feature_importances, explain_with_shap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance


def top_feature_importances(
    model: ClassifierMixin,
    feature_names: list[str],
    top_k: int = 12,
) -> pd.DataFrame:
    """Tree models expose importances; others return empty frame."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = np.asarray(model.feature_importances_)
    order = np.argsort(imp)[::-1][:top_k]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "importance": imp[order],
        }
    )


def permutation_importance_table(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1][:top_k]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "importance_mean": result.importances_mean[order],
        }
    )


def plot_importance_bar(df: pd.DataFrame, title: str = "Feature importance") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    if df.empty:
        ax.text(0.5, 0.5, "No importances for this model type", ha="center")
        return fig
    col = "importance" if "importance" in df.columns else "importance_mean"
    ax.barh(df["feature"][::-1], df[col][::-1], color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def explain_with_shap(
    model: ClassifierMixin,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    feature_names: list[str],
    max_background: int = 120,
) -> pd.DataFrame | None:
    """
    Local SHAP values for one instance. Returns None if shap is unavailable.
    """
    try:
        import shap
    except ImportError:
        return None

    bg = X_background[:max_background]
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
        if isinstance(shap_values, list):
            class_idx = int(model.predict(X_instance)[0])
            values = shap_values[class_idx][0]
        else:
            values = shap_values[0]
    else:
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_instance, nsamples=80)
        if isinstance(shap_values, list):
            class_idx = int(model.predict(X_instance)[0])
            values = shap_values[class_idx][0]
        else:
            values = shap_values[0]

    return (
        pd.DataFrame({"feature": feature_names, "shap_value": values})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(15)
        .drop(columns=["abs_shap"])
    )


def save_explainability_figure(fig: plt.Figure, project_root: Path, filename: str) -> Path:
    out = project_root / "outputs" / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
