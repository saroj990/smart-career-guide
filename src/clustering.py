"""
Day 4 K-Means helpers — same logic as notebooks/day04/day04_kmeans.ipynb.

Import from notebooks:
  from src.clustering import inertia_by_k, silhouette_by_k, cluster_profile_table
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """Fit K-Means on scaled numeric matrix X."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def inertia_by_k(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Elbow data: within-cluster sum of squares (inertia) for each K."""
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        if k < 1:
            continue
        model = fit_kmeans(X, n_clusters=k, random_state=random_state)
        rows.append({"k": k, "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def silhouette_by_k(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Average silhouette score per K (needs k >= 2)."""
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        if k < 2:
            continue
        model = fit_kmeans(X, n_clusters=k, random_state=random_state)
        score = silhouette_score(X, model.labels_)
        rows.append({"k": k, "silhouette": float(score)})
    return pd.DataFrame(rows)


def cluster_profile_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    profile_cols: list[str],
) -> pd.DataFrame:
    """Mean of selected columns per cluster (for interpretation, not prediction)."""
    work = df.copy()
    work["cluster"] = labels
    summary = work.groupby("cluster")[profile_cols].mean().round(3)
    sizes = work.groupby("cluster").size().rename("cluster_size")
    return summary.join(sizes)


def career_overlap_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    career_col: str = "Career",
) -> pd.DataFrame:
    """
    Cross-tab of cluster vs career (exploratory only — do not treat cluster as career).
    """
    work = pd.DataFrame({"cluster": labels, career_col: df[career_col].values})
    ct = pd.crosstab(work["cluster"], work[career_col], normalize="index").round(3)
    return ct
