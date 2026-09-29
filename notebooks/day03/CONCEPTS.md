# Day 3 — Feature engineering + preprocessing

## Goal
Build a **reproducible pipeline**: raw table → encode categories → scale numbers → train/test split.

## Key ideas
- **Feature engineering:** composite scores (e.g. technical average) when justified.
- **Encoding:** one-hot or ordinal for `Interest_Area`.
- **Scaling:** `StandardScaler` — critical for K-Means, KNN, SVM, neural nets.
- **Train/test split:** fit preprocessors on **train only**, then transform test (avoid leakage).

## Deliverables
`src/preprocessing.py`, feature list, saved processed arrays or pipeline.

## Checkpoint
Why does scaling matter for K-Means and KNN?

## Before Day 4
You should be able to produce `X_train`, `X_test`, `y_train`, `y_test` with one function call.
