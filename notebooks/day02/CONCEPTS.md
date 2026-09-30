# Day 2 — Data cleaning + EDA

## Fundamentals (beginner)

Day 2 assumes you know **feature vs target** from Day 1. Here you learn to **trust** and **explore** the table before any algorithm runs.

### Why clean before ML?

Models assume numbers mean what they say. Dirty data causes:

- Wrong patterns (duplicate students counted twice)
- Crashes or skipped rows (missing values)
- Nonsense inputs (CGPA = 99, skill = 0)

**Cleaning** = find issues, fix or remove bad rows, document what you did.

### Missing values

| Situation | What it means |
|-----------|----------------|
| Cell is empty / `NaN` | No value recorded |
| Some algorithms | Cannot train with NaN; need drop or **imputation** (fill with mean/median/mode) |

In your audit you **count** missing cells first; then you decide a strategy. Our dataset may already be clean — the **process** still belongs in your report.

### Duplicates

- **Duplicate row:** every column identical — often delete one copy.
- **Duplicate ID:** same `Student_ID` twice — usually keep one row.

Duplicates make the model think rare patterns are common → **inflated accuracy**.

### Valid ranges (sanity checks)

For this project:

| Field | Typical rule |
|-------|----------------|
| `CGPA` | 0–10 (or your documented scale) |
| Skill ratings | 1–5 |
| Internships, projects, certs | ≥ 0 integers |

Values outside rules are often typos — drop or correct with justification.

### What is EDA?

**Exploratory Data Analysis (EDA)** = summarize and visualize data to understand:

- Shape (how many rows, which columns)
- Distributions (typical CGPA, skill levels)
- Relationships (do high Python ratings appear more with some careers?)
- Problems (imbalance, outliers)

EDA is **not** proof of cause. A chart shows **association in this dataset**, not “Python causes Data Science.”

### Common plots (intuition)

| Plot | Question it helps answer |
|------|---------------------------|
| **Bar chart** | How many per category? (careers, interests) |
| **Histogram** | How are numeric values spread? (CGPA) |
| **Boxplot** | Typical range and outliers per skill |
| **Heatmap (correlation)** | Which numeric features move together? |
| **Crosstab heatmap** | How do interest and career co-occur? |

### Correlation (preview)

**Correlation** between two numeric columns: roughly −1 to +1.

- Near **+1:** when one goes up, the other tends to go up
- Near **0:** little linear relationship
- Near **−1:** when one goes up, the other tends to go down

High correlation between two **features** can matter for some models; correlation is **not** the same as “important for predicting Career” until you model it.

### Class imbalance (important for later)

If one career has 800 students and another has 50, the dataset is **imbalanced**.

- A dumb model that always predicts the common career can get **high accuracy** but useless advice for rare careers.
- You will use **precision, recall, F1** on Day 5+ to see this clearly.

Day 2 career bar chart is your first look at imbalance.

### Causation vs association

| Statement | OK in report? |
|-----------|----------------|
| “Cloud interest and Cloud Engineer label often appear together in the data” | Yes (association) |
| “Choosing Cloud interest causes someone to become Cloud Engineer” | No (causation) without proper study |

Always use careful language in interpretations.

### Where code lives

- **Notebook:** exploration and charts
- **`src/cleaning.py`:** reusable `clean_career_dataset()` — same logic, less copy-paste

---

## Goal

**Clean** data so ML is safe, then **explore** with plots to spot imbalance, odd ranges, and associations — without claiming causation.

## Key ideas (today)

### Data cleaning

| Issue | Why it matters |
|-------|----------------|
| Duplicates | Inflates accuracy |
| Missing values | Breaks or biases models |
| Invalid ranges | Bad rows distort learning |
| Messy categories | `Web` vs ` web` → duplicate levels |

### Charts in this project (axes)

1. **Career bars** — X: count, Y: career (target balance).
2. **CGPA histogram** — X: CGPA, Y: frequency.
3. **Skill boxplots** — X: skill name, Y: rating 1–5.
4. **Interest bars** — X: interest, Y: count.
5. **Experience histograms** — X: count, Y: frequency, hue: type.
6. **Correlation heatmap** — cell = correlation between numeric features.
7. **Career vs interest** — X: career, Y: interest, color: count.

## Checkpoint

If some careers are rare, why might **accuracy** look good while **F1** for rare classes is poor?

## Before Day 3

Use `outputs/AI_Career_Guidance_Dataset_cleaned.csv` or `clean_career_dataset()` for all later steps.
