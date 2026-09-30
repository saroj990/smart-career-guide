# Day 5 — First classification model (baseline)

## Fundamentals (beginner)

Day 5 is your first **supervised** model: learn from `(X, y)` and predict `Career` for new students.

### What is classification?

**Classification** = assign each example to one of several **classes** (categories).

- **Binary:** 2 classes (e.g. pass/fail)
- **Multi-class:** 3+ classes — **our case** (many career titles)

Each student gets **one** primary predicted class; later (Day 10) you also rank **top-N** careers using probabilities.

### What is a baseline model?

A **baseline** is a simple model you train first to:

1. Verify the **pipeline** works (no silent bugs)
2. Set a **floor** — smarter models should beat it (or you explain why not)
3. Give a **reference** for the report and viva

**Logistic regression** is a standard baseline for tabular multi-class problems: fast, interpretable, uses probabilities.

### Logistic regression (intuition, not full math)

Despite the name “regression,” **logistic regression** is used for **classification**.

Idea:

- Compute a **score** from weighted features (like a weighted sum of CGPA, Python, …).
- Pass score through a function so outputs look like **probabilities** per class.
- Pick the class with highest probability (or use probabilities for ranking).

For **multi-class**, scikit-learn uses extensions (e.g. one-vs-rest or multinomial) — you mainly need to know it outputs **one probability per career**.

### Training vs inference

| Phase | Input | Output |
|-------|--------|--------|
| **Training** | `X_train`, `y_train` | Learned weights |
| **Inference / predict** | `X_test` or new student | Predicted class + `predict_proba` |

Never evaluate on training data alone and call it “final performance” — that **overfits** the memorization check.

### Overfitting (preview)

**Overfitting** = model memorizes training noise and does poorly on new students.

Baselines like logistic regression with regularization are often less overfit than very complex models; trees can overfit easily (Day 6).

Always report metrics on **test** (or cross-validation), not train.

### Evaluation metrics (you must know these)

Imagine one career class “Data Scientist” vs all others (one-vs-rest view):

| Metric | Question it answers | Formula intuition |
|--------|---------------------|-------------------|
| **Accuracy** | Overall, how many predictions correct? | correct / total |
| **Precision** | When we predict this career, how often right? | true positives / predicted positives |
| **Recall** | Of all true this career, how many caught? | true positives / actual positives |
| **F1** | Balance precision and recall | harmonic mean of both |

**Accuracy alone misleads** when classes are imbalanced (Day 2 chart).

### Confusion matrix

Table: **rows = true career**, **columns = predicted career**.

- Diagonal = correct predictions
- Off-diagonal = confusions (e.g. true Analyst, predicted Scientist)

Read it to see **which pairs** the model mixes up.

### Classification report

scikit-learn `classification_report` gives precision, recall, F1 **per class** and averages — paste **real numbers** from your run in the report.

### Multi-class probabilities

`model.predict_proba(X)` → one row per student, one column per career, values sum to ~1.

Used on Day 10 for **ranking** careers, not only top-1.

### What to save today

| Artifact | Path idea |
|----------|-----------|
| Metrics JSON/CSV | `outputs/metrics/` |
| Confusion matrix plot | `outputs/figures/` |
| Short note | Why logistic regression as baseline |

### How Day 5 fits the 15-day arc

```text
Day 1–2: understand + clean data
Day 3:   X_train, y_train, …
Day 4:   optional structure (clusters)
Day 5:   first end-to-end PREDICT career + METRICS  ← you are here
Day 6–7: stronger models + comparison
```

### Checkpoint answers (study)

**Why a baseline?**  
So you know the pipeline works and have a number to improve on.

**Why not only accuracy?**  
Imbalanced careers → high accuracy can hide failure on rare classes.

---

## Goal

Train **logistic regression** as a simple, interpretable baseline for multi-class `Career`.

## Key ideas (today)

- **Baseline:** sanity-check pipeline; floor for comparison.
- **Metrics:** accuracy, precision, recall, F1, confusion matrix on **test** data.
- **Multi-class:** `predict` and `predict_proba`.

## Checkpoint

Why is a baseline useful even if accuracy is modest?

## Before Day 6

Save metrics under `outputs/metrics/` with **your** experiment numbers only.
