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

### Inputs must match Day 3 exactly

| Step | What goes wrong if skipped |
|------|---------------------------|
| Same engineered columns | Model sees different features than trained preprocessor expects |
| `transform` test with **fitted** scaler/encoder | Leakage or wrong scale |
| Same `random_state` on split | Metrics not comparable across experiments |

**Rule:** Day 5 trains on `X_train`, `y_train` from `prepare_train_test_bundle()` — do not rebuild preprocessing with a different recipe.

### Regularization (why logistic regression is a sane baseline)

Logistic regression penalizes **very large weights** (L2 by default in sklearn). That tends to:

- Reduce wild swings from noisy features
- Improve generalization on **test** data vs an unregularized linear model

You do not need the full math for Day 5 — remember: **simple + regularized** = strong baseline on tabular data.

### Imbalanced classes (connect to Day 2)

If one career has 40% of rows and another has 2%:

| Scenario | What happens |
|----------|----------------|
| Model always predicts the **majority** career | Accuracy can look **high** |
| Rare careers | Low **recall** — you “miss” those students |

Use **per-class** precision/recall/F1 and the confusion matrix, not accuracy alone.

### One-vs-rest vs multinomial (sklearn detail)

For multi-class logistic regression, sklearn can:

| Mode | Idea |
|------|------|
| **multinomial** | One joint model over all careers (common default for `LogisticRegression`) |
| **ovr** (one-vs-rest) | One binary model per career vs all others |

Both output `predict` and `predict_proba`. For your report, note which you used (`multi_class` / `solver` in sklearn).

### sklearn pattern for Day 5

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

Evaluate only on **held-out** `X_test`, `y_test`.

### Common Day 5 mistakes

| Mistake | Fix |
|---------|-----|
| Training on full data, reporting train accuracy | Report **test** metrics |
| Forgetting `stratify=y` when re-splitting | Use Day 3 split or same `random_state` |
| Tuning on test set | Test is **once**; use validation/CV for tuning (Day 7 preview) |
| Comparing to Day 4 clusters as if they were predictions | Cluster ID ≠ career label |

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

### Macro vs weighted F1 (read your classification report)

| Average | What it emphasizes |
|---------|-------------------|
| **Macro** | Each career counts equally — highlights weak performance on **rare** classes |
| **Weighted** | Bigger careers pull the average more — closer to “overall” behavior |

Report **both** with accuracy so examiners see you understand imbalance.

### Train accuracy vs test accuracy

| Gap | Likely meaning |
|-----|----------------|
| Train much higher than test | Overfitting or too little data |
| Both low | Features may be weak, or classes are hard to separate |
| Both similar and modest | Honest baseline — improve with Day 6+ models |

Day 5: quote **test** metrics in your report; train scores are optional diagnostics only.

### End-to-end flow (notebook + `src/`)

```text
load_cleaned_table()
    → prepare_train_test_bundle()     # Day 3 — same random_state=42
    → fit_baseline(X_train, y_train)  # Day 5 — src/classification.py
    → evaluate_multiclass(..., test)  # metrics + confusion matrix
    → save_baseline_run()             # outputs/ + models/
```

The **preprocessor** stays separate from the **classifier**: preprocessing is fit on train in Day 3; the classifier is fit on `X_train` only in Day 5.

### Reading the confusion matrix heatmap

- **Bright cells on the diagonal** → correct top-1 predictions for that career pair.
- **Bright off-diagonal** → systematic mix-ups (e.g. two similar IT roles).
- Many careers → the plot is dense; in your write-up, call out **2–3** worst confusions with counts.

### `predict_proba` for one student

After `fit`, each test row gets a vector of probabilities. The **argmax** matches `predict()`; the full vector is what Day 10 uses to rank top-N careers.

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
