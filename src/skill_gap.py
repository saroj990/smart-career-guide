"""
Day 12 skill gap analysis — notebooks/day12/day12_skill_gap.ipynb.

  from src.skill_gap import load_skill_mapping, compute_skill_gaps
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning import RATING_COLS

# Labels in Career_Skill_Mapping.csv → dataset column names
SKILL_LABEL_TO_COLUMN: dict[str, str] = {
    "Python": "Python",
    "Java": "Java",
    "SQL": "SQL",
    "Web Development": "Web_Development",
    "Cybersecurity": "Cybersecurity",
    "Cloud Computing": "Cloud_Computing",
    "Mathematics": "Mathematics",
    "Statistics": "Statistics",
    "Data Analysis": "Data_Analysis",
    "Problem Solving": "Problem_Solving",
    "Communication": "Communication",
    "Creativity": "Creativity",
    "Leadership": "Leadership",
    "Teamwork": "Teamwork",
    "Logical Reasoning": "Logical_Reasoning",
    "Presentation": "Presentation",
    "Projects": "Projects",
    "Certifications": "Certifications",
    "CGPA": "CGPA",
}

DEFAULT_TARGET_RATING = 4.0
DEFAULT_TARGET_PROJECTS = 2.0
DEFAULT_TARGET_CERTIFICATIONS = 1.0
DEFAULT_TARGET_CGPA = 7.5


def load_skill_mapping(project_root: Path) -> pd.DataFrame:
    """Read Career → required skill rows from `dataset/Career_Skill_Mapping.csv`."""
    path = project_root / "dataset" / "Career_Skill_Mapping.csv"
    return pd.read_csv(path)


def required_skills_for_career(mapping: pd.DataFrame, career: str) -> list[str]:
    """Required skill labels for one career name (must match mapping CSV)."""
    return mapping.loc[mapping["Career"] == career, "Required_Skill"].tolist()


def _target_for_column(col: str) -> float:
    """Expected level used as the gap target (rule-based, not learned)."""
    if col in RATING_COLS:
        return DEFAULT_TARGET_RATING
    if col == "Projects":
        return DEFAULT_TARGET_PROJECTS
    if col == "Certifications":
        return DEFAULT_TARGET_CERTIFICATIONS
    if col == "CGPA":
        return DEFAULT_TARGET_CGPA
    return DEFAULT_TARGET_RATING


def _current_value(student_row: pd.Series, col: str) -> float:
    """Student's current value for a mapped column (rating, projects, CGPA, …)."""
    return float(student_row[col])


def compute_skill_gaps(
    student_row: pd.Series,
    career: str,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rule-based gap = max(0, target − current) for each required skill.
    """
    skills = required_skills_for_career(mapping, career)
    rows: list[dict[str, object]] = []
    for label in skills:
        col = SKILL_LABEL_TO_COLUMN.get(label)
        if col is None or col not in student_row.index:
            rows.append(
                {
                    "career": career,
                    "required_skill": label,
                    "column": None,
                    "current": None,
                    "target": None,
                    "gap": None,
                    "status": "unmapped",
                }
            )
            continue
        current = _current_value(student_row, col)
        target = _target_for_column(col)
        gap = max(0.0, target - current)
        rows.append(
            {
                "career": career,
                "required_skill": label,
                "column": col,
                "current": current,
                "target": target,
                "gap": gap,
                "status": "ok" if gap <= 0 else "needs_improvement",
            }
        )
    return pd.DataFrame(rows)


def summarize_gaps(gap_df: pd.DataFrame) -> str:
    """Plain-language list of skills below target (checklist, not ML)."""
    needs = gap_df[gap_df["status"] == "needs_improvement"]
    if needs.empty:
        return "No skill gaps above the target thresholds for this career."
    parts = ["Skills to strengthen (deterministic checklist, not ML):"]
    for _, row in needs.iterrows():
        parts.append(
            f"- {row['required_skill']}: current {row['current']}, target {row['target']} "
            f"(gap {row['gap']})"
        )
    return "\n".join(parts)
