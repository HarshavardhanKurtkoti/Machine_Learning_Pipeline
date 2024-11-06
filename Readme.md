

# 📘 Machine Learning Pipeline with DVC and MLflow

This project demonstrates a robust machine learning pipeline using **DVC** (Data Version Control) for data and model versioning, and **MLflow** for experiment tracking. The model, a Random Forest Classifier, is trained on the Pima Indians Diabetes Dataset with clearly defined stages for data preprocessing, model training, and evaluation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Data](#data)
5. [Pipeline Stages](#pipeline-stages)
6. [Model Deployment and Experiment Tracking](#model-deployment-and-experiment-tracking)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [References](#references)

---

### 1. Project Overview
- **Purpose**: Create a reproducible ML pipeline for training and evaluating a Random Forest Classifier using structured stages with data and model versioning.
- **Technologies Used**: 
  - **DVC**: For version control of data, models, and pipeline stages.
  - **MLflow**: For logging experiment metrics and artifacts.
  - **Scikit-learn**: For model training.
  
### 2. Project Structure
```
├── data/                      # Data storage (raw, processed)
├── models/                    # Trained models
├── src/                       # Source code for each stage
│   ├── preprocess.py          # Data preprocessing
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
├── params.yaml                # Configurations for each stage
├── requirements.txt           # Project dependencies
├── dvc.yaml                   # DVC pipeline stages
├── README.md                  # Project README
```

### 3. Requirements
To set up the environment, install dependencies with:
```bash
pip install -r requirements.txt
```
This project uses:
- **dvc**
- **dagshub** (optional, for remote data storage)
- **scikit-learn**
- **mlflow**
- **dvc-s3** (optional, for S3 storage integration)

### 4. Data
- **Dataset**: The Pima Indians Diabetes dataset.
- **Location**: Place the raw data file at `data/raw/data.csv`.
- **Version Control**: The dataset, models, and outputs are tracked with DVC to ensure reproducibility.

### 5. Pipeline Stages

#### Preprocessing
- **Script**: `src/preprocess.py`
- **Description**: Reads the raw dataset, performs basic preprocessing (e.g., renaming columns), and saves processed data to `data/processed/data.csv`.
- **Run Command**:
  ```bash
  dvc repro preprocess
  ```

#### Training
- **Script**: `src/train.py`
- **Description**: Trains a Random Forest Classifier on the preprocessed data and saves the model to `models/model.pkl`.
- **Parameters**: Configured in `params.yaml` (e.g., `n_estimators`, `max_depth`).
- **Run Command**:
  ```bash
  dvc repro train
  ```

#### Evaluation
- **Script**: `src/evaluate.py`
- **Description**: Evaluates the trained model’s accuracy and logs the results to MLflow.
- **Run Command**:
  ```bash
  dvc repro evaluate
  ```

### 6. Model Deployment and Experiment Tracking
- **Experiment Tracking**: MLflow is used to log model parameters and metrics. This allows comparison of different training configurations for optimization.
- **Running MLflow UI**:
  ```bash
  mlflow ui
  ```
  Access the MLflow tracking server at [http://localhost:5000](http://localhost:5000).

### 7. Usage

#### Pipeline Execution
To run the entire pipeline:
```bash
dvc repro
```

#### Adding Pipeline Stages
To add new stages or update existing ones, use `dvc stage add`. Example:
```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
```

### 8. Results
- **Metrics Tracked**: Model accuracy and other relevant metrics are logged and tracked through MLflow.
- **Model Artifacts**: Saved in the `models/` directory and version-controlled with DVC.

### 9. Future Improvements
- **Model Optimization**: Experiment with other algorithms or hyperparameters.
- **Data Augmentation**: Incorporate data augmentation techniques to enhance model robustness.
- **Deployment**: Implement a model-serving API for real-time predictions.

### 10. Contributors
- **Your Name** (Primary Developer)
  
### 11. References
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
