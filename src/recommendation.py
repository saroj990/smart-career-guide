"""
Day 10 career ranking — notebooks/day10/day10_career_ranking.ipynb.

  from src.recommendation import rank_careers, format_ranking_table
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    probs = model.predict(X, verbose=0)
    return np.asarray(probs)


def rank_careers(
    model: ClassifierMixin | Any,
    X: np.ndarray,
    class_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return top-N careers by predicted probability for each row in X.
    For a single profile, pass X with shape (1, n_features).
    """
    probs = _predict_proba(model, X)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    rows: list[dict[str, Any]] = []
    for row_idx, p_row in enumerate(probs):
        order = np.argsort(p_row)[::-1][:top_n]
        for rank, class_idx in enumerate(order, start=1):
            rows.append(
                {
                    "profile_index": row_idx,
                    "rank": rank,
                    "career": class_names[int(class_idx)],
                    "probability": float(p_row[class_idx]),
                    "probability_pct": round(100.0 * float(p_row[class_idx]), 2),
                }
            )
    return pd.DataFrame(rows)


def format_ranking_table(ranking: pd.DataFrame) -> str:
    """Plain-text summary for reports and Streamlit."""
    lines: list[str] = []
    for profile_index in ranking["profile_index"].unique():
        sub = ranking[ranking["profile_index"] == profile_index]
        lines.append(f"Profile #{profile_index}")
        for _, row in sub.iterrows():
            lines.append(
                f"  {int(row['rank'])}. {row['career']} — {row['probability_pct']}% "
                "(model confidence, not a guarantee)"
            )
        lines.append("")
    return "\n".join(lines).strip()
