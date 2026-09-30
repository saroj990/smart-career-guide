# Day 3 — Feature engineering + preprocessing

## Fundamentals (beginner)

After clean data (Day 2), models need numbers in a **consistent shape**. Day 3 is about **preparing X and y** for training — still no fancy model required to understand the ideas.

### Raw table vs model input

| Stage | What you have |
|-------|----------------|
| Raw CSV | Mixed text and numbers, different scales |
| Model input | Numeric matrix **X**, label vector **y**, same row order |

**Preprocessing** = all steps that turn raw rows into **X** the algorithm can use.

### Feature engineering

**Feature engineering** = create **new columns** from existing ones to help the model.

Examples (only if you can justify them in words):

- **Academic score** — combine `Mathematics`, `Statistics`, `CGPA`
- **Technical score** — average of programming-related skills
- **Experience score** — weighted mix of projects, internships, certifications

Good features are **interpretable** (“we grouped related skills”) not magic formulas.

Bad features: secretly contain the target or future information (leakage).

### Encoding categorical variables

Models are math on numbers. Categories like `Interest_Area = "Cloud"` must become numbers.

| Method | Idea | When |
|--------|------|------|
| **One-hot encoding** | One column per category (0/1) | Nominal categories (no natural order) |
| **Ordinal encoding** | 1, 2, 3… if order matters | Only when order is meaningful |

`Career` is the **target** for supervised learning — you encode **y** separately (often integer labels 0…C−1), not the same as encoding interest for **X**.

### Scaling (normalization)

Features on different scales confuse distance-based methods:

- CGPA might be 5–10
- Skill ratings are 1–5

**StandardScaler** (common choice): transform each numeric feature to roughly mean 0, std 1 using **training data** statistics.

| Needs scaling often | Less sensitive to scale |
|---------------------|-------------------------|
| K-Means, KNN, SVM, neural nets | Decision trees, random forest |

You still scale for a fair comparison when mixing models later.

### Train / validation / test split

You must measure performance on data the model **did not** learn from.

| Set | Purpose | Model sees labels? |
|-----|---------|-------------------|
| **Train** | Learn parameters | Yes (for X and y) |
| **Test** | Final honest evaluation | Only X at predict time; compare to true y |
| **Validation** (optional) | Tune hyperparameters | Yes on val only during tuning |

Typical first split: **80% train / 20% test** (or 70/30). Use `stratify=y` for classification so each career appears in both sets proportionally when possible.

```
All data
   ├── Train (fit model + fit scaler/encoder)
   └── Test  (only transform with fitted scaler/encoder, then predict)
```

### Data leakage in preprocessing (critical)

**Wrong:** fit `StandardScaler` on **all** rows, then split.  
**Right:** `fit` scaler on **train only**, `transform` both train and test.

Same rule for encoders and any statistic learned from data.

### Pipelines (scikit-learn idea)

A **Pipeline** chains steps:

```text
Raw features → imputer → encoder → scaler → model
```

Benefits:

- One object to `fit` on train and `predict` on new students
- Fewer mistakes (no forgotten scaling on test)

You will implement something like this in `src/preprocessing.py`.

### What you produce today

| Output | Meaning |
|--------|---------|
| `X_train`, `X_test` | Feature matrices (numeric) |
| `y_train`, `y_test` | Career labels aligned row-by-row |
| Feature list | Column names after encoding |
| Saved pipeline or `.pkl` | Reuse in notebooks and Streamlit later |

### Vocabulary

| Term | Meaning |
|------|---------|
| **Fit** | Learn parameters from data (e.g. mean/std for scaler) |
| **Transform** | Apply learned parameters to data |
| **fit_transform** | Fit on this data, then transform it (train only) |

---

## Goal

Build a **reproducible pipeline**: raw table → encode categories → scale numbers → train/test split.

## Key ideas (today)

- **Feature engineering:** composite scores when justified.
- **Encoding:** one-hot (or similar) for `Interest_Area`.
- **Scaling:** `StandardScaler` for distance-based models later.
- **Train/test split:** fit preprocessors on **train only**.

## Deliverables

`src/preprocessing.py`, feature list, processed data or saved pipeline.

## Checkpoint

Why does scaling matter for K-Means and KNN?

## Before Day 4

You should produce `X_train`, `X_test`, `y_train`, `y_test` with one function or pipeline call.
