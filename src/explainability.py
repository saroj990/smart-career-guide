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


def _predicted_class_index(model: Any, X_instance: np.ndarray) -> int:
    """Integer class index for one preprocessed row (shape (1, n_features))."""
    pred = model.predict(X_instance)
    if hasattr(pred, "ndim") and getattr(pred, "ndim", 1) > 1:
        # Keras softmax: (1, n_classes) → argmax
        return int(np.argmax(pred, axis=1)[0])
    return int(np.asarray(pred).reshape(-1)[0])


def _linear_contributions(
    model: Any,
    x: np.ndarray,
    class_idx: int,
) -> np.ndarray | None:
    """
    Contribution of each scaled feature to the predicted class logit:
    contribution_i = coef[class, i] * x_i  (Day 3 scaled values).
    """
    if not hasattr(model, "coef_"):
        return None
    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        return coef * x
    if class_idx >= coef.shape[0]:
        return None
    return coef[class_idx] * x


def local_feature_contributions(
    model: Any,
    X_instance: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Top features that pushed this profile toward the predicted career.

    Prefers logistic/linear `coef_ * x` (fast, matches production logistic).
    Falls back to SHAP when coefficients are missing. Empty table if neither works.

    Positive contribution = supports the predicted career; negative = pulls away.
    Values are on the **scaled** feature space from Day 3.
    """
    X = np.asarray(X_instance)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[0]
    class_idx = _predicted_class_index(model, X)
    career = class_names[class_idx] if class_names else str(class_idx)

    contrib = _linear_contributions(model, x, class_idx)
    method = "logistic_coef_times_value"
    if contrib is None:
        shap_df = explain_with_shap(model, X, X, feature_names)
        if shap_df is None or shap_df.empty:
            return pd.DataFrame(
                columns=["career", "feature", "contribution", "effect", "method"]
            )
        out = shap_df.rename(columns={"shap_value": "contribution"}).copy()
        out["career"] = career
        out["effect"] = np.where(
            out["contribution"] >= 0, "supports this career", "pulls away from this career"
        )
        out["method"] = "shap"
        cols = ["career", "feature", "contribution", "effect", "method"]
        return out[cols].head(top_k).reset_index(drop=True)

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    rows = []
    for i in order:
        c = float(contrib[i])
        rows.append(
            {
                "career": career,
                "feature": feature_names[i],
                "contribution": round(c, 4),
                "effect": "supports this career" if c >= 0 else "pulls away from this career",
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def top_feature_importances(
    model: ClassifierMixin,
    feature_names: list[str],
    top_k: int = 12,
) -> pd.DataFrame:
    """Global Gini importances (trees only). Empty frame for logistic / SVM / ANN."""
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
    """Global permutation importance on a held-out sample (not per-profile)."""
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
    """Bar chart for global importances (tree Gini or permutation)."""
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
    """Write a matplotlib figure under `outputs/figures/`."""
    out = project_root / "outputs" / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
