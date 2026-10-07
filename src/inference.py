"""
Shared inference for notebooks, tests, and Streamlit (Days 10–13).

  from src.inference import load_production_classifier, predict_profile
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ann_models import load_day08_ann
from src.cleaning import EXP_COLS, RATING_COLS
from src.explainability import local_feature_contributions
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_FEATURE_COLS,
    add_engineered_features,
    load_artifacts,
)
from src.recommendation import rank_careers
from src.skill_gap import compute_skill_gaps, load_skill_mapping, summarize_gaps


def student_profile_to_row(profile: dict[str, Any]) -> pd.Series:
    """
    Build one raw student row (before scaling) from form or API fields.

    Missing skill ratings default to 3; experience counts default to 1 so
    the preprocessor always sees a complete column set.
    """
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
    """Apply the saved Day 3 preprocessor (fit on train only) to one profile."""
    artifacts = load_artifacts(project_root)
    preprocessor = artifacts["preprocessor"]
    row = student_profile_to_row(profile)
    # Composite scores must exist before ColumnTransformer selects columns.
    frame = add_engineered_features(row.to_frame().T)
    features = frame[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS]
    return preprocessor.transform(features)


def load_production_classifier(project_root: Path) -> tuple[Any, list[str], str]:
    """
    Load (model, career_classes, model_name) for the app.

    Prefers `models/production_classifier.joblib` (Day 9). If that file is a
    Keras pointer, load `day08_ann.keras`. Falls back to the Day 7 sklearn bundle.
    """
    prod_path = project_root / "models" / "production_classifier.joblib"
    if prod_path.is_file():
        bundle = joblib.load(prod_path)
        if bundle.get("kind") == "keras":
            model = load_day08_ann(project_root)
            classes_path = project_root / "outputs" / "processed" / "preprocessing_meta.json"
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
    """
    End-to-end guidance for one student: rank careers, local 'why', skill gaps.
    """
    model, class_names, model_name = load_production_classifier(project_root)
    artifacts = load_artifacts(project_root)
    X = transform_profile(profile, project_root)
    ranking = rank_careers(model, X, class_names, top_n=top_n)
    top_career = str(ranking.iloc[0]["career"])
    why = local_feature_contributions(
        model,
        X,
        artifacts["feature_names"],
        class_names=class_names,
        top_k=8,
    )
    row = student_profile_to_row(profile)
    mapping = load_skill_mapping(project_root)
    gaps = compute_skill_gaps(row, top_career, mapping)
    return {
        "model_name": model_name,
        "top_career": top_career,
        "ranking": ranking,
        "why_this_career": why,
        "skill_gaps": gaps,
        "skill_gap_summary": summarize_gaps(gaps),
    }
