# AI Career Guidance System

MCA-level **learning-by-doing** prototype: student profile → preprocessing → clustering & classification (including ANN) → ranked career guidance → explainability → skill gaps → Streamlit app and Tableau analytics.

Full curriculum: [AI_Career_Guidance_15_Day_Learning_by_Doing_Plan.md](./AI_Career_Guidance_15_Day_Learning_by_Doing_Plan.md)

## Repository layout

```text
├── dataset/                    # CSV inputs
├── notebooks/                  # One notebook per day (day01 … day15)
├── src/                        # Reusable modules (from Day 3 onward)
├── models/                     # Saved models
├── outputs/figures|metrics|predictions/
├── app.py                      # Streamlit (Day 13)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open notebooks from `notebooks/` and run cells top to bottom. **Day 1** assumes the kernel’s working directory is the `notebooks/` folder (paths use `../dataset`).

## Day-wise notebooks

| Day | Notebook | Focus |
|-----|----------|--------|
| 1 | `day01_data_understanding` | Data inspection & data dictionary |
| 2 | `day02_eda_cleaning` | Cleaning & EDA |
| 3 | `day03_preprocessing` | Features & preprocessing pipeline |
| 4 | `day04_kmeans` | K-Means clustering |
| 5 | `day05_baseline_classification` | Logistic regression baseline |
| 6 | `day06_tree_and_forest` | Decision tree & random forest |
| 7 | `day07_ml_models_evaluation` | SVM, KNN, XGBoost — **MVP milestone** |
| 8 | `day08_ann` | Neural network |
| 9 | `day09_ml_vs_ann` | Model comparison |
| 10 | `day10_career_ranking` | Top-N recommendations |
| 11 | `day11_explainability` | Feature importance & SHAP |
| 12 | `day12_skill_gap` | Skill gap analysis |
| 13 | `day13_streamlit_app` | Streamlit (`app.py`) |
| 14 | `day14_tableau_testing` | Tableau & test cases |
| 15 | `day15_documentation_demo` | Report, PPT, viva, demo |

## Datasets

- `dataset/AI_Career_Guidance_Dataset.csv` — training/experiment data  
- `dataset/Career_Skill_Mapping.csv` — career → required skills (Day 12+)

Document actual row/column counts from your runs; do not copy placeholder numbers from slides.
