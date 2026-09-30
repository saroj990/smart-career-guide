# Day 1 — Understand the data and problem

## Fundamentals (beginner)

Read this section first if you are new to machine learning (ML).

### What is machine learning?

**Machine learning** means a computer learns patterns from **data** instead of you writing explicit rules like “if Python ≥ 4 then Data Scientist.”

You provide:

- **Examples** (rows in a table — here, each row is one student)
- **Inputs** (numbers and categories describing the student)
- **Outputs** (for our project: which **career** label fits that profile in the dataset)

The algorithm adjusts internal parameters so that, on new students, its predictions are as good as possible.

### AI vs ML vs this project

| Term | Simple meaning |
|------|----------------|
| **AI** | Broad field: systems that show “intelligent” behavior |
| **ML** | A way to build AI: learn from data |
| **This project** | ML for **career guidance** — suggestions, not job guarantees |

### Two main types of learning (you need both in this course)

| Type | Do you have a “correct answer” column? | Example in this project |
|------|----------------------------------------|-------------------------|
| **Supervised** | Yes — a label to predict | Predict `Career` from skills, CGPA, interest |
| **Unsupervised** | No label — find structure | Day 4: group similar students (clustering) |

Day 1 focuses on **supervised** setup: what is the label, what are the inputs?

### Dataset = table (rows and columns)

- **Row (record):** one student (`Student_ID` identifies them).
- **Column (feature or target):** one variable, e.g. `CGPA`, `Python`, `Career`.

Think of ML as: **many rows of experience** → model learns → **one new row** → model suggests a career.

### Feature vs target (most important Day 1 idea)

| Role | Symbol | In this project | When you use it |
|------|--------|-----------------|-----------------|
| **Features** | **X** | CGPA, skills, interest, internships, … | Known when a student asks for guidance |
| **Target** | **y** | `Career` | What we want to predict (known only in historical data for training) |

**Rule:** Everything in **X** must be information you would have **before** giving advice. Do not put the answer (or anything that only exists *because* they already have that career) into X.

### Data leakage (common beginner mistake)

**Data leakage** = accidentally giving the model hints about the answer during training.

Examples to avoid:

- Using `Career` (or a column derived from it) as a feature
- Fitting scaling or encoders on the **entire** dataset before splitting train/test (you fix this properly on Day 3)

If leakage happens, metrics look amazing in the notebook but the system fails in real use.

### Numerical vs categorical variables

| Type | Examples here | How models see them |
|------|---------------|---------------------|
| **Numerical** | `CGPA`, skill ratings 1–5, `Age`, project counts | Numbers; distance and averages make sense |
| **Categorical** | `Interest_Area`, `Career` | Labels; often need **encoding** (Day 3) to numbers |

### Classification vs regression (vocabulary)

- **Classification:** predict a **category** (e.g. Cloud Engineer, Data Analyst) → **our task**
- **Regression:** predict a **number** (e.g. salary, CGPA next semester) → not the main target here

We have **multi-class classification**: more than two career labels.

### What “training” and “testing” mean (preview)

- **Training:** model sees student profiles and the true `Career` and learns.
- **Testing:** model sees **new** students it was not trained on; we measure how well it predicts.

You will split data on Day 3; on Day 1 you only need to know **why** the table must be understood first.

### Files in this project (Day 1)

| File | Purpose |
|------|---------|
| `AI_Career_Guidance_Dataset.csv` | Main table: features + `Career` |
| `Career_Skill_Mapping.csv` | Career → required skills (used from Day 12; peek today is OK) |

### Mini glossary

| Term | Definition |
|------|------------|
| **Observation / sample** | One row (one student) |
| **Feature** | Input column used to predict |
| **Target / label** | Output column (`Career`) |
| **EDA** | Exploratory Data Analysis — looking at data before modeling (Day 2) |
| **Pipeline** | Repeatable steps from raw data to model input (Day 3+) |

---

## Goal

Know **what you are predicting**, which columns are **features** vs **target**, and whether the data is trustworthy before any ML.

## Key ideas (today)

### Supervised learning

You have labels (`Career`) and want the model to learn a mapping from student profile → career. Training uses past students; testing checks on held-out rows.

### Features vs target

- **Target (y):** `Career` — the career category we predict.
- **Features (X):** CGPA, skills, interest, experience, etc. — inputs only known at recommendation time.
- **Never** put information that “comes from knowing the answer” into X (**data leakage**).

### Numerical vs categorical

- **Numerical:** CGPA, skill ratings 1–5, counts — math and scaling apply.
- **Categorical:** `Interest_Area`, `Career` — need encoding before many models.

## What to do in the notebook

1. Load both CSVs.
2. Inspect shape, dtypes, missing values, duplicates.
3. Build a **data dictionary** (column, meaning, type, ML role).
4. Preview `Career` distribution (class balance).

## Checkpoint (viva)

**What exactly are we predicting?**  
→ The `Career` label from the student’s profile features, as **guidance**, not a guarantee.

## Before Day 2

You should name every column and say whether it belongs in **X** or **y**.
