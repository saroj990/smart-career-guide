# Execution guide (Days 1–15)

Run everything from the **project root** (`CareerGuidanceMLEngine/`).

Use **Command Prompt** or **PowerShell** on Windows; **Terminal** (bash/zsh) on macOS/Linux.

## 1. Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

If `python` is not found, try `py -m venv .venv` then `.\.venv\Scripts\activate`.

## 2. Learning path (notebooks)

For each day `XX`:

1. Read `notebooks/dayXX/CONCEPTS.md`
2. Start Jupyter with kernel cwd `notebooks/dayXX/`
3. Run `dayXX_*.ipynb`

```bash
jupyter lab
```

Days **1–6** should be run first so cleaned data and the Day 3 preprocessor exist (or rely on raw CSV fallback in `load_cleaned_table`).

## 3. Fast path — scripts (Days 7–14)

After Days 1–3 (or with raw dataset only):

```bash
python scripts/run_pipeline.py
```

Windows (same command; Python accepts `/` or `\`):

```bat
python scripts\run_pipeline.py
```

This runs:

| Step | Output |
|------|--------|
| Day 7 | `outputs/metrics/day07_*`, `models/day07_best_classifier.joblib` |
| Day 8 | `models/day08_ann.keras`, `outputs/metrics/day08_ann_metrics.json` |
| Day 9 | `outputs/metrics/day09_*`, `models/production_classifier.joblib` |
| Day 14 | `outputs/tableau/*`, `outputs/metrics/day14_test_report.csv` |

Individual scripts:

```bash
python scripts/run_day07_benchmark.py
python scripts/run_day08_ann.py
python scripts/run_day09_comparison.py
```

Windows:

```bat
python scripts\run_day07_benchmark.py
python scripts\run_day08_ann.py
python scripts\run_day09_comparison.py
```

## 4. Final application (Day 13)

Keep the venv **activated**, then from the project root:

```bash
streamlit run app.py
```

Browser: sidebar form → **Get career guidance** → top career, **why this profile**, ranking, skill gaps.

## 5. Tableau (Day 14)

Import CSVs from `outputs/tableau/` (or `outputs\tableau\` on Windows):

- `student_profiles_sample.csv`
- `career_predictions_sample.csv`
- `model_metrics_long.csv` (if present)
- `demo_*.csv` for a single guided example

Keep numbers aligned with Python metrics in `outputs/metrics/`.

## 6. Demo & viva (Day 15)

Follow `notebooks/day15/day15_documentation_demo.ipynb` — no new code. Use **your** metrics from `outputs/metrics/` only.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found (Windows) | Use `py` instead of `python` |
| Activate.ps1 cannot be loaded | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Missing preprocessing_pipeline.joblib` | Run Day 3 notebook or Day 7 script |
| XGBoost missing | `pip install xgboost` or skip (benchmark omits it) |
| TensorFlow slow on first run | Normal; Day 8 may take several minutes |
| SHAP optional in Day 11 | Install `shap`; app uses logistic coefficients when that is the production model |
