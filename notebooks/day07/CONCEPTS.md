# Day 7 — More models + evaluation (**MVP milestone**)

## Fundamentals (beginner)

Day 7 widens the **model zoo** and picks a defensible **best model** using the same preprocessing as Days 3–6.

### Models you will compare

| Model | Idea | Hyperparameter you should name |
|-------|------|--------------------------------|
| **Logistic regression** | Linear scores → class probabilities | `C` (regularization strength) |
| **Random Forest** | Ensemble of trees | `n_estimators`, `max_depth` |
| **SVM (RBF kernel)** | Finds separating boundaries in transformed space | `C`, `gamma` |
| **KNN** | Class = majority vote among **k** nearest training points | `n_neighbors` (k) |
| **XGBoost** (if installed) | Gradient-boosted trees | `n_estimators`, `max_depth`, `learning_rate` |

All consume the same `X_train` / `X_test` from Day 3.

### Why KNN needs scaled features

KNN uses **distance** between students in feature space → same scaling story as Day 3 / K-Means.

SVM with RBF kernel is also **distance-sensitive** — keep the Day 3 scaler.

### Train / validation / test (Day 7 workflow)

| Set | How we use it in this notebook |
|-----|--------------------------------|
| **Train** | Fit models + **cross-validation** for stable scores |
| **Test** | One honest comparison table (do not tune on test) |

**Cross-validation (CV):** split train into K folds; rotate which fold is validation. Report **mean** macro F1 across folds.

### Hyperparameters vs parameters

| | Learned from data | Chosen by you |
|---|-------------------|---------------|
| **Parameters** | Tree splits, logistic weights | — |
| **Hyperparameters** | — | k in KNN, `C` in SVM, forest size |

Day 7 uses **reasonable defaults**; full tuning can be a mini-project extension.

### How to pick the “best” model

Do **not** choose only by accuracy if classes are imbalanced.

Recommended order for this project:

1. **Test macro F1** (fair to rare careers)
2. **Test accuracy** as secondary
3. **CV macro F1** to check stability
4. **Interpretability / speed** for viva (logistic vs forest vs XGBoost)

Document **why** in `outputs/metrics/day07_best_model.json`.

### MVP milestone

By end of Day 7 you must demonstrate:

```text
student profile → preprocess (Day 3) → classifier → predicted career + metrics
```

That is the **minimum viable academic project** before neural nets (Day 8+).

### If XGBoost is not installed

The notebook and `src/model_evaluation.py` **skip** XGBoost when the package is missing — still complete the table with sklearn models.

Install optional dependency:

```bash
pip install xgboost
```

### Common mistakes

| Mistake | Fix |
|---------|-----|
| Different train/test split than Day 3 | Keep `random_state=42` |
| Tuning on test to “win” | Tune with CV on train only |
| Reporting CV score as final | Report **test** metrics for the comparison table |

---

## Goal

Benchmark **SVM, KNN, XGBoost** (if available) alongside logistic regression and random forest; produce a **comparison table** with real results.

## Key ideas (today)

- **Cross-validation** on train for stability.
- **Hyperparameters:** know what k, C, and forest size mean.
- **MVP:** end-to-end predict + evaluate.

## Before Day 8

Note which model is current best and why (macro F1 + practical reasons).
