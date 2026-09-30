# Day 4 — K-Means clustering

## Fundamentals (beginner)

Day 4 introduces **unsupervised learning** — no `Career` label needed for the clustering step itself.

### Supervised vs unsupervised (again)

| | Supervised (Day 5+) | Unsupervised (Day 4) |
|---|---------------------|----------------------|
| **Uses y (`Career`)?** | Yes, to learn | No — only **X** (profile features) |
| **Output** | Predicted career | **Cluster ID** (group membership) |
| **Question** | “Which career fits?” | “Which students look similar?” |

Both can live in one project: clustering **explores** data; classification **predicts** careers.

### What is clustering?

**Clustering** groups points so that:

- Points in the **same** cluster are relatively similar
- Points in **different** clusters are relatively different

There is no “correct” cluster name from the dataset — you **interpret** clusters after seeing their average CGPA, skills, etc.

### K-Means in plain language

**K-Means** algorithm (simplified):

1. Choose **K** (number of clusters).
2. Place **K centroids** (centers) in feature space.
3. Assign each student to the nearest centroid.
4. Move each centroid to the mean of its assigned students.
5. Repeat assign + move until stable.

**Requires numeric, scaled features** — otherwise a feature with large numbers (e.g. raw counts) dominates distance.

### What is K?

**K** = how many groups you ask for.

- Small K → few, broad groups
- Large K → many, narrow groups

There is no single “true” K for careers; you **choose** using methods below and domain sense.

### Elbow method

1. Run K-Means for K = 1, 2, 3, …, max_K.
2. Plot **inertia** (within-cluster sum of squared distances) vs K.
3. Look for an **elbow** — where adding another cluster helps less.

The elbow is a **heuristic**, not a theorem. Use it with silhouette and common sense.

### Silhouette score

For each point, silhouette measures how close it is to its own cluster vs other clusters. Average over all points → **silhouette score** between −1 and 1.

- **Higher** (closer to 1) → better separated clusters
- Compare scores for different K values

### Distance and scaling

K-Means uses **Euclidean distance** (straight-line distance in feature space).

If one feature is 0–10 and another is 0–1000, distance is dominated by the large-scale feature. **Always scale** (Day 3) before K-Means.

### Naming clusters

Names like “tech-heavy students” are OK only if **cluster statistics** support them (mean skills, CGPA). Do not rename clusters after career labels unless you analyze overlap carefully — that can confuse clustering with prediction.

### Why clustering in a career project?

| Reason | Explanation |
|--------|-------------|
| **EDA** | See natural student segments |
| **Feature** | Optional: add `cluster_id` as an extra input to classifiers |
| **Reporting** | “We found 4 profile types in the cohort” |

It does **not** replace predicting `Career` — that is **supervised** from Day 5.

### Common mistakes

| Mistake | Fix |
|---------|-----|
| K-Means on unscaled data | Use scaled `X` |
| Treating cluster ID as career | Cluster ≠ career label |
| Picking K only because it “sounds nice” | Use elbow + silhouette + interpretation |

### Inertia (within-cluster sum of squares)

**Inertia** measures how “tight” clusters are: for each point, square its distance to its cluster centroid, then add up over all points.

| K | Typical inertia |
|---|-----------------|
| Larger K | Usually **lower** inertia (more centroids → points closer to some center) |
| K = number of rows | Inertia → 0 (one centroid per point — useless) |

The **elbow plot** is inertia vs K. You want the point where inertia still drops but **much more slowly** — extra clusters are not buying much structure.

### Silhouette score (how to read it)

For one student:

- **a** = average distance to other students in **same** cluster (cohesion)
- **b** = average distance to students in the **nearest other** cluster (separation)
- Silhouette ≈ \((b - a) / \max(a, b)\)

| Score | Rough meaning |
|-------|----------------|
| Near **+1** | Well inside its cluster, far from neighbors |
| Near **0** | On the border between clusters |
| Negative | Possibly assigned to the wrong cluster |

Compare silhouette **across K**, not against a fixed “pass mark.”

### Euclidean distance (why scaling matters)

In 2D, distance between \((x_1, y_1)\) and \((x_2, y_2)\):

\[
\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}
\]

With many features, every dimension adds a squared difference. If `CGPA` is ~8 and a skill is ~3, raw distance is **biased** toward CGPA unless you scale (Day 3).

### K-Means limitations (honest notes)

| Limitation | Practical impact |
|------------|------------------|
| Assumes **roughly spherical** clusters | Elongated or nested groups may split oddly |
| Sensitive to **outliers** | Outliers pull centroids |
| **Random** initialization | Use `random_state` and `n_init>1` (sklearn default) for stability |
| You must choose **K** | Elbow + silhouette + domain sense |

### PCA plot vs real clustering

**PCA** (2 components) is only for **visualization** on paper/slides. K-Means in the notebook should run on the **full scaled feature matrix** from Day 3. A 2D plot can hide separation that exists in higher dimensions.

### Workflow with Day 3 artifacts

```text
Day 3: fit preprocessor on TRAIN → save pipeline
Day 4: transform ALL profiles with that same pipeline → K-Means on scaled X
```

Using the **same** scaler as supervised work keeps cluster analysis comparable to later models.

### Optional: clusters as extra features

Later you may add `cluster_id` as one more column in **X** for a classifier. Treat that as an experiment: does it help test accuracy, or just add noise?

---

## Goal

Group students by **similar profiles** (unsupervised) — not to predict career directly, but to explore structure.

## Key ideas (today)

- **K:** number of clusters; try several values.
- **Elbow method:** inertia vs K.
- **Silhouette score:** higher = better separation.
- **Scaling required** before K-Means.

## Checkpoint

Why use clustering if it does not directly output a career?

## Before Day 5

Career prediction is **supervised** from Day 5 on; you may optionally use cluster labels as extra features later.
