"""
Shared inference for notebooks, tests, and Streamlit (Days 10–13).

  from src.inference import load_production_classifier, predict_profile
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ann_models import load_day08_ann
from src.cleaning import EXP_COLS, RATING_COLS
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_FEATURE_COLS,
    add_engineered_features,
    load_artifacts,
)
from src.recommendation import rank_careers
from src.skill_gap import compute_skill_gaps, load_skill_mapping, summarize_gaps


def student_profile_to_row(profile: dict[str, Any]) -> pd.Series:
    """Build one raw student row (pre-scaling) from form/API fields."""
    defaults: dict[str, Any] = {
        "Student_ID": profile.get("Student_ID", "DEMO"),
        "Age": profile.get("Age", 22),
        "CGPA": profile.get("CGPA", 7.0),
        "Interest_Area": profile.get("Interest_Area", "Technology"),
    }
    for col in RATING_COLS:
        defaults[col] = profile.get(col, 3)
    for col in EXP_COLS:
        defaults[col] = profile.get(col, 1)
    return pd.Series(defaults)


def transform_profile(profile: dict[str, Any], project_root: Path) -> np.ndarray:
    artifacts = load_artifacts(project_root)
    preprocessor = artifacts["preprocessor"]
    row = student_profile_to_row(profile)
    frame = add_engineered_features(row.to_frame().T)
    features = frame[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS]
    return preprocessor.transform(features)


def load_production_classifier(project_root: Path) -> tuple[Any, list[str], str]:
    """
    Returns (model, career_classes, model_name).
    Falls back to Day 7 best sklearn bundle if production bundle missing.
    """
    prod_path = project_root / "models" / "production_classifier.joblib"
    if prod_path.is_file():
        bundle = joblib.load(prod_path)
        if bundle.get("kind") == "keras":
            model = load_day08_ann(project_root)
            classes_path = project_root / "outputs" / "processed" / "preprocessing_meta.json"
            import json

            meta = json.loads(classes_path.read_text(encoding="utf-8"))
            return model, meta["career_classes"], str(bundle.get("model_name", "keras_ann"))
        return (
            bundle["model"],
            bundle["career_classes"],
            str(bundle.get("model_name", "sklearn")),
        )

    fallback = project_root / "models" / "day07_best_classifier.joblib"
    if fallback.is_file():
        bundle = joblib.load(fallback)
        return (
            bundle["model"],
            bundle["career_classes"],
            str(bundle.get("model_name", "day07_best")),
        )
    raise FileNotFoundError(
        "No production or Day 7 model found. Run Days 7–9 pipeline first."
    )


def predict_profile(
    profile: dict[str, Any],
    project_root: Path,
    top_n: int = 5,
) -> dict[str, Any]:
    model, class_names, model_name = load_production_classifier(project_root)
    X = transform_profile(profile, project_root)
    ranking = rank_careers(model, X, class_names, top_n=top_n)
    top_career = str(ranking.iloc[0]["career"])
    row = student_profile_to_row(profile)
    mapping = load_skill_mapping(project_root)
    gaps = compute_skill_gaps(row, top_career, mapping)
    return {
        "model_name": model_name,
        "top_career": top_career,
        "ranking": ranking,
        "skill_gaps": gaps,
        "skill_gap_summary": summarize_gaps(gaps),
    }
