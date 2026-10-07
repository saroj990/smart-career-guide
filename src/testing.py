"""
Day 14 test cases TC01–TC10 — notebooks/day14/day14_tableau_testing.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.inference import load_production_classifier, predict_profile, transform_profile
from src.preprocessing import load_artifacts, load_cleaned_table, prepare_train_test_bundle


@dataclass
class TestResult:
    case_id: str
    description: str
    passed: bool
    detail: str


def _tc01_pipeline_artifacts(project_root: Path) -> TestResult:
    path = project_root / "outputs" / "processed" / "preprocessing_pipeline.joblib"
    ok = path.is_file()
    return TestResult("TC01", "Preprocessing pipeline exists", ok, str(path))


def _tc02_production_model(project_root: Path) -> TestResult:
    try:
        load_production_classifier(project_root)
        return TestResult("TC02", "Production classifier loads", True, "ok")
    except FileNotFoundError as exc:
        return TestResult("TC02", "Production classifier loads", False, str(exc))


def _tc03_transform_shape(project_root: Path) -> TestResult:
    bundle = prepare_train_test_bundle(load_cleaned_table(project_root))
    profile = {"Age": 22, "CGPA": 7.5, "Interest_Area": "Technology"}
    X = transform_profile(profile, project_root)
    ok = X.shape[1] == bundle["n_features"]
    return TestResult(
        "TC03",
        "Profile transform feature count",
        ok,
        f"got {X.shape[1]}, expected {bundle['n_features']}",
    )


def _tc04_prediction_non_empty(project_root: Path) -> TestResult:
    result = predict_profile({"Interest_Area": "Business", "CGPA": 8.0}, project_root)
    ok = len(result["ranking"]) >= 1
    return TestResult("TC04", "predict_profile returns ranking", ok, result["top_career"])


def _tc05_ranking_probabilities_sum(project_root: Path) -> TestResult:
    result = predict_profile({"Interest_Area": "Security"}, project_root, top_n=5)
    top = result["ranking"]
    ok = (top["probability"] > 0).all() and (top["probability"] <= 1).all()
    return TestResult("TC05", "Ranking probabilities in (0,1]", ok, f"rows={len(top)}")


def _tc06_skill_gap_columns(project_root: Path) -> TestResult:
    result = predict_profile({"Python": 5, "Projects": 3}, project_root)
    gaps = result["skill_gaps"]
    ok = "gap" in gaps.columns and len(gaps) > 0
    return TestResult("TC06", "Skill gap table populated", ok, f"rows={len(gaps)}")


def _tc07_day07_metrics(project_root: Path) -> TestResult:
    path = project_root / "outputs" / "metrics" / "day07_model_benchmark.csv"
    ok = path.is_file()
    return TestResult("TC07", "Day 7 benchmark CSV exists", ok, str(path))


def _tc08_day08_ann_or_skip(project_root: Path) -> TestResult:
    path = project_root / "models" / "day08_ann.keras"
    ok = path.is_file()
    return TestResult(
        "TC08",
        "Day 8 ANN model file exists",
        ok,
        "missing — run Day 8" if not ok else str(path),
    )


def _tc09_comparison_table(project_root: Path) -> TestResult:
    path = project_root / "outputs" / "metrics" / "day09_final_model_comparison.csv"
    ok = path.is_file()
    return TestResult("TC09", "Day 9 comparison CSV exists", ok, str(path))


def _tc10_tableau_exports(project_root: Path) -> TestResult:
    path = project_root / "outputs" / "tableau" / "student_profiles_sample.csv"
    ok = path.is_file()
    return TestResult("TC10", "Tableau sample export exists", ok, str(path))


TEST_CASES: list[Callable[[Path], TestResult]] = [
    _tc01_pipeline_artifacts,
    _tc02_production_model,
    _tc03_transform_shape,
    _tc04_prediction_non_empty,
    _tc05_ranking_probabilities_sum,
    _tc06_skill_gap_columns,
    _tc07_day07_metrics,
    _tc08_day08_ann_or_skip,
    _tc09_comparison_table,
    _tc10_tableau_exports,
]


def run_all_tests(project_root: Path) -> pd.DataFrame:
    rows = [tc(project_root).__dict__ for tc in TEST_CASES]
    return pd.DataFrame(rows)


def save_test_report(project_root: Path) -> Path:
    report = run_all_tests(project_root)
    out = project_root / "outputs" / "metrics" / "day14_test_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    return out
