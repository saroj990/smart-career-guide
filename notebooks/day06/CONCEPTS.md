# Day 6 — Decision tree + random forest

## Fundamentals (beginner)

Day 6 adds **tree-based** classifiers and compares them to **Day 5 logistic regression**.

### Why trees after a linear baseline?

| Model | Decision boundary idea |
|-------|------------------------|
| **Logistic regression** | Smooth scores from weighted sums of features |
| **Decision tree** | Nested **if/else rules** on features (e.g. “if Python ≥ 4 and CGPA ≥ 7 → …”) |

Trees can capture **non-linear** patterns without you hand-crafting formulas.

### How a decision tree splits

At each node the algorithm picks a **feature** and **threshold** that best separates classes (using **Gini impurity** or **entropy** in sklearn).

```text
                 [All students]
                 /            \
        CGPA < 7.5              CGPA ≥ 7.5
           /    \                  ...
    Python<3   Python≥3
```

**Leaves** assign a majority class (or class probabilities).

### Overfitting with one deep tree

A tree can grow until **every leaf is pure** → it memorizes training noise.

| Signal | Meaning |
|--------|---------|
| Train accuracy **much** higher than test | Likely overfitting |
| Shallow tree (`max_depth` small) | Simpler rules, often better test performance |

Day 6: compare a **controlled** tree (`max_depth`) vs an **ensemble**.

### Random Forest (ensemble)

**Random Forest** = many decision trees, each trained on:

- A **bootstrap** sample of rows (bagging)
- A **random subset** of features at each split

Final prediction = **vote** (classification) across trees.

| One deep tree | Random Forest |
|---------------|---------------|
| High variance, easy to overfit | Averaging reduces variance |
| Fast to train one model | More compute, often better generalization |

### Feature importance (Random Forest)

sklearn reports **importance** per input column (how much splits reduce impurity, aggregated over trees).

- Use it for **interpretation** (“which skills mattered in this model?”)
- Not a causal proof — correlation with the target in this dataset only

### Scaling and trees

Tree splits are **order-based** (is CGPA > 7.5?) — **scaling is optional** for trees.

We still use the **same Day 3 matrix** so comparisons with logistic regression and Day 7 models are fair.

### What you produce today

| Artifact | Purpose |
|----------|---------|
| Comparison table | Logistic vs tree vs forest on **test** |
| RF feature importance plot | Report narrative |
| Saved RF model | Candidate “stronger” model before Day 7 sweep |

### Checkpoint

Why can an ensemble generalize better than one very deep tree?

**Study answer:** averaging many diverse trees reduces overfitting to any single noisy partition of the data.

---

## Goal

Train **decision tree** and **random forest**, compare to **logistic regression**, analyze **feature importance**.

## Key ideas (today)

- **Tree:** rule-based splits; watch overfitting.
- **Random Forest:** bagging + feature randomness.
- **Feature importance:** global ranking from the forest.

## Before Day 7

Save `outputs/metrics/day06_model_comparison.csv` with **your** numbers only.
