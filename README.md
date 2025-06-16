# 🧪 Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates an end-to-end ML pipeline using **DVC** for versioning and **MLflow** for experiment tracking. We train a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, with structured stages for preprocessing, training, and evaluation.

---

## 🚀 Key Features

### 🔁 Data Version Control (DVC)
- Tracks datasets, models, and pipeline steps.
- Ensures reproducibility by auto-re-running stages if dependencies change.
- Supports remote storage (e.g., DagsHub, S3) for collaboration.

### 📊 Experiment Tracking with MLflow
- Logs hyperparameters, metrics (accuracy), and artifacts (models).
- Supports comparison of multiple training runs and models.
- Great for model optimization and documentation.

---

## 🧱 Pipeline Stages

### 1️⃣ Preprocessing

**Script:** `src/preprocess.py`

- Reads `data/raw/data.csv`
- Performs cleaning and renaming
- Saves to `data/processed/data.csv`

**DVC Command:**

```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
