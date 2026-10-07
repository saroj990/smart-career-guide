"""
Day 3 preprocessing — same logic as notebooks/day03/day03_preprocessing.ipynb.

Import from notebooks (add project root to sys.path):
  from src.preprocessing import prepare_train_test_bundle, save_artifacts, load_cleaned_table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.cleaning import EXP_COLS, RATING_COLS, clean_career_dataset

TARGET_COL = "Career"  # multi-class label
ID_COL = "Student_ID"  # dropped from features; kept on raw tables
CATEGORICAL_COLS = ["Interest_Area"]  # one-hot encoded after the train split

TECH_RATING_COLS = [
    "Python",
    "Java",
    "SQL",
    "Web_Development",
    "Cybersecurity",
    "Cloud_Computing",
]
ACADEMIC_RATING_COLS = ["Mathematics", "Statistics", "Data_Analysis"]
SOFT_RATING_COLS = [
    "Problem_Solving",
    "Communication",
    "Creativity",
    "Leadership",
    "Teamwork",
    "Logical_Reasoning",
    "Presentation",
]

ENGINEERED_COLS = [
    "Academic_score",
    "Technical_score",
    "Soft_skills_score",
    "Experience_score",
]

NUMERIC_FEATURE_COLS = (
    ["Age", "CGPA"] + RATING_COLS + EXP_COLS + ENGINEERED_COLS
)


def load_cleaned_table(project_root: Path) -> pd.DataFrame:
    """Prefer Day 2 CSV; fall back to clean raw data if missing."""
    cleaned = project_root / "outputs" / "AI_Career_Guidance_Dataset_cleaned.csv"
    raw = project_root / "dataset" / "AI_Career_Guidance_Dataset.csv"
    if cleaned.is_file():
        return pd.read_csv(cleaned)
    df = pd.read_csv(raw)
    clean_df, _ = clean_career_dataset(df)
    return clean_df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable composite scores (justified groupings for the report).

    Academic: math/stats/analysis skills; Technical: programming stack;
    Soft: communication and teamwork; Experience: internships, projects, certs.
    """
    out = df.copy()
    out["Academic_score"] = out[ACADEMIC_RATING_COLS + ["CGPA"]].mean(axis=1)
    out["Technical_score"] = out[TECH_RATING_COLS].mean(axis=1)
    out["Soft_skills_score"] = out[SOFT_RATING_COLS].mean(axis=1)
    out["Experience_score"] = (
        out["Internships"] + out["Projects"] * 2 + out["Certifications"]
    ) / 4.0
    return out


def build_preprocessor() -> ColumnTransformer:
    """Numeric standardization + one-hot interest (fit on train only in the notebook)."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURE_COLS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable column names after scaling + one-hot encoding."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat" and hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols).tolist())
    return names


def prepare_train_test_bundle(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Engineer features, stratified train/test split, fit preprocessor on train only.

    Returns arrays plus fitted preprocessor and label encoder for reuse (Day 4+, app).
    """
    enriched = add_engineered_features(df)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(enriched[TARGET_COL])

    feature_frame = enriched[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS]
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    feature_names = feature_names_from_preprocessor(preprocessor)

    return {
        "X_train": X_train_arr,
        "X_test": X_test_arr,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "label_encoder": y_encoder,
        "feature_names": feature_names,
        "career_classes": y_encoder.classes_.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train_arr.shape[1]),
    }


def transform_all_profiles(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Apply a train-fitted preprocessor to every row (e.g. Day 4 clustering)."""
    enriched = add_engineered_features(df)
    frame = enriched[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS]
    return preprocessor.transform(frame)


def save_artifacts(
    bundle: dict[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    """Persist pipeline, label map, and feature list under outputs/processed/."""
    out_dir = project_root / "outputs" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = out_dir / "preprocessing_pipeline.joblib"
    joblib.dump(
        {
            "preprocessor": bundle["preprocessor"],
            "label_encoder": bundle["label_encoder"],
            "feature_names": bundle["feature_names"],
            "career_classes": bundle["career_classes"],
        },
        pipeline_path,
    )

    names_path = out_dir / "feature_names.json"
    names_path.write_text(json.dumps(bundle["feature_names"], indent=2), encoding="utf-8")

    meta_path = out_dir / "preprocessing_meta.json"
    meta = {
        "n_train": bundle["n_train"],
        "n_test": bundle["n_test"],
        "n_features": bundle["n_features"],
        "career_classes": bundle["career_classes"],
        "engineered_cols": ENGINEERED_COLS,
        "numeric_feature_cols": NUMERIC_FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "pipeline": pipeline_path,
        "feature_names": names_path,
        "meta": meta_path,
    }


def load_artifacts(project_root: Path) -> dict[str, Any]:
    """Load saved preprocessing pipeline and encoders."""
    path = project_root / "outputs" / "processed" / "preprocessing_pipeline.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run Day 3 notebook or prepare_train_test_bundle + save_artifacts."
        )
    return joblib.load(path)
