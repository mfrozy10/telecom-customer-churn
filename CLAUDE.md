# Customer Churn Prediction — Project Instructions

## Project Overview

Machine Learning project for predicting customer churn, implemented as a series of Jupyter Notebooks.

## Notebook Structure

All code must be placed inside the `notebooks/` folder. Notebooks must follow sequential naming:

```
notebooks/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_feature_engineering.ipynb
├── 04_modeling.ipynb
└── 05_evaluation.ipynb
```

- Use two-digit prefix (01, 02, ...) to enforce execution order.
- Notebook names must be lowercase with underscores.
- Never place `.ipynb` files in the root or any other folder.

## Required Libraries

Use only standard ML libraries:

- **Pandas** — data loading, manipulation, and tabular operations
- **NumPy** — numerical computation
- **Matplotlib / Seaborn** — visualization
- **Scikit-Learn** — preprocessing, modeling, evaluation pipelines

Do not introduce additional ML frameworks (XGBoost, LightGBM, PyTorch, etc.) unless explicitly requested.

## Cell Authoring Rules

Every code cell must be preceded by a Markdown cell that explains what the code does and why. This is mandatory — no code cell may appear without a Markdown cell immediately above it.

Markdown cells must:
- Describe the intent and reasoning, not just restate the code
- Use storytelling style: explain the analytical decision being made
- Use headers (`##`, `###`) to separate major sections within a notebook

Example structure inside a notebook:

```
[Markdown] ## Load the Dataset
Explain why we load from this source and what format to expect.

[Code] pd.read_csv(...)

[Markdown] ### Inspect Shape and Types
Before exploring distributions, we first confirm the schema matches our expectations.

[Code] df.info(); df.describe()
```

## Coding Standards

- Prefer pipelines (`sklearn.pipeline.Pipeline`) over ad-hoc transforms.
- Keep random seeds explicit: `random_state=42`.
- Print or display outputs at the end of each code cell so the notebook renders meaningfully when run top-to-bottom.
- Avoid in-place mutations where possible; create new columns or DataFrames instead.

## Reproducibility

- The notebook must run cleanly from top to bottom with **Restart & Run All**.
- All file paths must be relative to the `notebooks/` directory (e.g., `../data/raw/churn.csv`).
