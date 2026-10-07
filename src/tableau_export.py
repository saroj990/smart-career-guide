"""
Day 14 Tableau-ready exports — notebooks/day14/day14_tableau_testing.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.inference import predict_profile, student_profile_to_row
from src.preprocessing import load_cleaned_table, load_artifacts, transform_all_profiles
from src.recommendation import rank_careers
from src.inference import load_production_classifier


def export_tableau_tables(project_root: Path, sample_size: int = 200) -> dict[str, Path]:
    out_dir = project_root / "outputs" / "tableau"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cleaned_table(project_root).head(sample_size)
    model, class_names, model_name = load_production_classifier(project_root)
    artifacts = load_artifacts(project_root)
    X = transform_all_profiles(df, artifacts["preprocessor"])
    ranking = rank_careers(model, X, class_names, top_n=3)

    profiles_path = out_dir / "student_profiles_sample.csv"
    df.to_csv(profiles_path, index=False)

    preds_path = out_dir / "career_predictions_sample.csv"
    ranking.to_csv(preds_path, index=False)

    metrics_rows: list[dict[str, object]] = []
    for name in [
        "day07_model_benchmark.csv",
        "day09_final_model_comparison.csv",
    ]:
        p = project_root / "outputs" / "metrics" / name
        if p.is_file():
            part = pd.read_csv(p)
            part["source_file"] = name
            metrics_rows.append(part)
    metrics_path = out_dir / "model_metrics_long.csv"
    if metrics_rows:
        pd.concat(metrics_rows, ignore_index=True).to_csv(metrics_path, index=False)

    meta = {
        "model_name": model_name,
        "n_profiles": len(df),
        "exports": [profiles_path.name, preds_path.name],
    }
    meta_path = out_dir / "export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "profiles": profiles_path,
        "predictions": preds_path,
        "metrics": metrics_path if metrics_rows else metrics_path,
        "meta": meta_path,
    }


def demo_profile_exports(project_root: Path) -> Path:
    """Single demo profile with gaps for Tableau join examples."""
    profile = {
        "Age": 21,
        "CGPA": 8.2,
        "Interest_Area": "Technology",
        "Python": 4,
        "SQL": 3,
        "Data_Analysis": 2,
        "Projects": 1,
    }
    result = predict_profile(profile, project_root)
    out = project_root / "outputs" / "tableau" / "demo_guidance_result.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    row = student_profile_to_row(profile)
    pd.DataFrame([row]).to_csv(out.with_name("demo_profile.csv"), index=False)
    result["ranking"].to_csv(out.with_name("demo_ranking.csv"), index=False)
    result["skill_gaps"].to_csv(out.with_name("demo_skill_gaps.csv"), index=False)
    return out.parent
