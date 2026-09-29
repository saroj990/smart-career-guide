# Day 1 — Understand the data and problem

## Goal
Know **what you are predicting**, which columns are **features** vs **target**, and whether the data is trustworthy before any ML.

## Key ideas

### Supervised learning
You have labels (`Career`) and want the model to learn a mapping from student profile → career. Training uses past students; testing checks on held-out rows.

### Features vs target
- **Target (y):** `Career` — the career category we predict.
- **Features (X):** CGPA, skills, interest, experience, etc. — inputs only known at recommendation time.
- **Never** put information that “comes from knowing the answer” into X ( **data leakage** ).

### Numerical vs categorical
- **Numerical:** CGPA, skill ratings 1–5, counts — math and scaling apply.
- **Categorical:** `Interest_Area`, `Career` — need encoding before many models.

## What to do in the notebook
1. Load both CSVs.
2. Inspect shape, dtypes, missing values, duplicates.
3. Build a **data dictionary** (column, meaning, type, ML role).
4. Plot `Career` distribution (preview of class balance).

## Checkpoint (viva)
**What exactly are we predicting?**  
→ The `Career` label from the student’s profile features, as guidance not a guarantee.

## Before Day 2
You should name every column and say whether it belongs in X or y.
