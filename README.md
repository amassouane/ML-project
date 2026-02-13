# Heart Disease ML Pipeline

This project implements a complete machine learning pipeline for heart disease detection:
- EDA and preprocessing
- Dimensionality reduction (PCA, t-SNE, NMF)
- Clustering (K-Means, Agglomerative, DBSCAN)
- Classification (7 models)
- MLflow tracking and model comparison

## Setup
1. Create a virtual environment:
   - Windows (PowerShell):
     - `py -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Data
Download the Kaggle dataset "Heart Disease UCI" and place the CSV at:
- `data/raw/heart.csv`

If your file name or location differs, pass `--data-path` to the scripts.

## Run Notebooks
- `jupyter lab`
- Open notebooks in `notebooks/` in order: 01 -> 04

## Run Scripts
- EDA:
  - `py src/run_eda.py --data-path data/raw/heart.csv`
- Dimensionality reduction:
  - `py src/run_dimensionality.py --data-path data/raw/heart.csv`
- Clustering:
  - `py src/run_clustering.py --data-path data/raw/heart.csv`
- Classification + MLflow:
  - `py src/run_classification.py --data-path data/raw/heart.csv`
- Full pipeline (EDA + DR + clustering + classification):
  - `py src/run_pipeline.py --data-path data/raw/heart.csv`

## Outputs
- Figures: `reports/figures/`
- Metrics: `reports/metrics/`
- MLflow runs: `mlruns/` (ignored by git)

## Notes
- The pipeline auto-handles missing values and categorical encoding.
- If the dataset uses `num` as the target column, it is automatically converted to a binary target.
