# AI Career Guidance System

MCA-level **learning-by-doing** prototype: student profile → preprocessing → clustering & classification (including ANN) → ranked career guidance → explainability → skill gaps → Streamlit app and Tableau analytics.

Full curriculum: [AI_Career_Guidance_15_Day_Learning_by_Doing_Plan.md](./AI_Career_Guidance_15_Day_Learning_by_Doing_Plan.md)

## Repository layout

```text
├── dataset/                    # CSV inputs
├── notebooks/
│   ├── day01/                  # CONCEPTS.md + notebook
│   ├── day02/ … day15/
│   └── README.md               # Index of all days
├── src/                        # Reusable modules (cleaning, preprocessing, …)
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

For each day: read `notebooks/dayXX/CONCEPTS.md`, then run that folder’s `.ipynb`. Notebooks locate the project root via `find_project_root()` (kernel cwd should be `notebooks/dayXX/`).

See [notebooks/README.md](./notebooks/README.md) for the full day index.

**Run the full pipeline (Days 7+):** [EXECUTION.md](./EXECUTION.md)

## Datasets

- `dataset/AI_Career_Guidance_Dataset.csv` — training/experiment data  
- `dataset/Career_Skill_Mapping.csv` — career → required skills (Day 12+)

Document actual row/column counts from your runs; do not copy placeholder numbers from slides.
