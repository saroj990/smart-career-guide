"""
Day 13 — Streamlit career guidance app.

Run from project root:
  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import EXP_COLS, RATING_COLS
from src.explainability import permutation_importance_table, top_feature_importances
from src.inference import load_production_classifier, predict_profile
from src.preprocessing import load_artifacts, load_cleaned_table, prepare_train_test_bundle


@st.cache_data
def _background_sample(project_root: Path):
    bundle = prepare_train_test_bundle(load_cleaned_table(project_root))
    return bundle["X_test"][:80], bundle["y_test"][:80]


st.set_page_config(page_title="AI Career Guidance", layout="wide")
st.title("AI Career Guidance System")
st.caption("Educational prototype — predictions are model-based suggestions, not guarantees.")

df_raw = load_cleaned_table(ROOT)
interest_options = sorted(df_raw["Interest_Area"].dropna().unique().tolist())


def _sidebar_profile() -> dict:
    st.sidebar.header("Student profile")
    profile: dict = {
        "Age": st.sidebar.number_input("Age", 18, 35, 22),
        "CGPA": st.sidebar.slider("CGPA", 0.0, 10.0, 7.5, 0.1),
        "Interest_Area": st.sidebar.selectbox("Interest area", interest_options),
    }
    st.sidebar.subheader("Skills (1–5)")
    for col in RATING_COLS:
        profile[col] = st.sidebar.slider(col.replace("_", " "), 1, 5, 3)
    st.sidebar.subheader("Experience")
    for col in EXP_COLS:
        profile[col] = st.sidebar.number_input(col, 0, 10, 1)
    return profile


profile = _sidebar_profile()

if st.button("Get career guidance", type="primary"):
    try:
        result = predict_profile(profile, ROOT, top_n=5)
    except FileNotFoundError as exc:
        st.error(
            f"Missing trained artifacts: {exc}. "
            "Run `python scripts/run_pipeline.py` or complete Days 7–9 notebooks."
        )
        st.stop()

    st.subheader("Top career match")
    st.success(f"{result['top_career']} (model: {result['model_name']})")

    st.subheader("Ranked alternatives")
    st.dataframe(result["ranking"], use_container_width=True)

    st.subheader("Skill gap checklist")
    st.dataframe(result["skill_gaps"], use_container_width=True)
    st.text(result["skill_gap_summary"])

    try:
        model, _, _model_name = load_production_classifier(ROOT)
        artifacts = load_artifacts(ROOT)
        feature_names = artifacts["feature_names"]
        if hasattr(model, "feature_importances_"):
            imp = top_feature_importances(model, feature_names)
            st.subheader("Global feature importance")
            st.bar_chart(imp.set_index("feature"))
        else:
            X_bg, y_bg = _background_sample(ROOT)
            imp = permutation_importance_table(model, X_bg, y_bg, feature_names, top_k=10)
            st.subheader("Permutation importance (sample)")
            st.bar_chart(imp.set_index("feature"))
    except Exception as exc:  # noqa: BLE001
        st.info(f"Explanation charts skipped: {exc}")

st.divider()
st.markdown(
    "**Demo order (Day 15):** profile → ranking → skill gaps → Tableau exports in `outputs/tableau/`."
)
