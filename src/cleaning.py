"""
Day 2 cleaning helpers — same logic as notebooks/day02_eda_cleaning.ipynb.

Import from notebooks:  from src.cleaning import clean_career_dataset, cleaning_audit
(Or add project root to sys.path when running scripts.)
"""

from __future__ import annotations

import pandas as pd

# Skill columns scored 1–5 in the career guidance dataset
RATING_COLS = [
    "Python",
    "Java",
    "SQL",
    "Web_Development",
    "Cybersecurity",
    "Cloud_Computing",
    "Mathematics",
    "Statistics",
    "Data_Analysis",
    "Problem_Solving",
    "Communication",
    "Creativity",
    "Leadership",
    "Teamwork",
    "Logical_Reasoning",
    "Presentation",
]
# Count fields — must be zero or positive integers
EXP_COLS = ["Internships", "Projects", "Certifications"]
# Text categories we strip and keep consistent
CAT_COLS = ["Interest_Area", "Career"]


def cleaning_audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per data-quality check (for reports and viva)."""
    checks: list[tuple[str, int, str]] = []
    checks.append(("rows", len(df), "—"))
    checks.append(("columns", len(df.columns), "—"))
    checks.append(("missing_cells", int(df.isnull().sum().sum()), "—"))
    checks.append(("duplicate_rows", int(df.duplicated().sum()), "—"))
    checks.append(("duplicate_Student_ID", int(df["Student_ID"].duplicated().sum()), "—"))
    checks.append(
        ("CGPA_out_of_0_10", int(((df["CGPA"] < 0) | (df["CGPA"] > 10)).sum()), "0–10")
    )
    bad_rating = (df[RATING_COLS] < 1).any(axis=1) | (df[RATING_COLS] > 5).any(axis=1)
    checks.append(("skill_rating_outside_1_5", int(bad_rating.sum()), "1–5"))
    bad_exp = (df[EXP_COLS] < 0).any(axis=1)
    checks.append(("negative_experience_counts", int(bad_exp.sum()), ">= 0"))
    return pd.DataFrame(checks, columns=["check", "count", "expected"])


def clean_career_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply standard cleaning steps; return cleaned DataFrame and a change log.

    Steps: drop duplicate rows/IDs, strip categories, coerce numeric types,
    drop invalid ranges and rows with NaN in critical columns.
    """
    out = df.copy()
    log: list[tuple[str, int]] = []

    n0 = len(out)
    out = out.drop_duplicates()
    log.append(("drop_duplicate_rows", n0 - len(out)))

    n0 = len(out)
    out = out.drop_duplicates(subset=["Student_ID"], keep="first")
    log.append(("drop_duplicate_Student_ID", n0 - len(out)))

    for col in CAT_COLS:
        out[col] = out[col].astype(str).str.strip()

    out["CGPA"] = pd.to_numeric(out["CGPA"], errors="coerce")
    for col in RATING_COLS + EXP_COLS + ["Age"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["CGPA"].between(0, 10)
        & out[RATING_COLS].ge(1).all(axis=1)
        & out[RATING_COLS].le(5).all(axis=1)
        & out[EXP_COLS].ge(0).all(axis=1)
        & out["Age"].between(15, 40)
    )
    n0 = len(out)
    out = out.loc[valid].reset_index(drop=True)
    log.append(("drop_invalid_ranges", n0 - len(out)))

    n0 = len(out)
    out = out.dropna(subset=["Career", "Interest_Area", "CGPA"] + RATING_COLS)
    log.append(("drop_rows_with_NaN_after_coerce", n0 - len(out)))

    change_log = pd.DataFrame(log, columns=["action", "rows_removed"])
    return out, change_log
