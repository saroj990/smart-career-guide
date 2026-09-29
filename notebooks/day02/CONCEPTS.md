# Day 2 — Data cleaning + EDA

## Goal
**Clean** data so ML is safe, then **explore** with plots to spot imbalance, odd ranges, and associations — without claiming causation.

## Key ideas

### Data cleaning
| Issue | Why it matters |
|-------|----------------|
| Duplicates | Inflates accuracy |
| Missing values | Breaks or biases models |
| Invalid ranges | Bad rows distort learning |
| Messy categories | `Web` vs ` web` → duplicate levels |

Reusable code lives in `src/cleaning.py`.

### EDA
Summaries and charts **before** modeling. Ask: what pattern do I see? Could it be explained another way?

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
