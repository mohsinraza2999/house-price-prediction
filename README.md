# 🏠 House Price Prediction with Deep Learning

> An end-to-end machine learning system that predicts house prices from structured data using PyTorch, designed to demonstrate **production-ready ML engineering practices** such as reproducibility, testing, CI, Dockerization, and API-based inference.

---

## 🧠 Why This Project?

House price prediction is a well-known regression problem, but most implementations stop at notebooks and accuracy metrics.
This project was built to answer a more practical question:

> **How would a junior ML engineer design, train, test, and serve a model in a real production environment?**

The emphasis is on **system design, maintainability, and engineering discipline**, not just model performance.

---

## 🧠 Problem Statement

Predicting residential house prices requires modeling nonlinear relationships between structured features such as size, location indicators, and amenities.

The challenge lies not only in prediction accuracy, but in:

* Building reproducible data and training pipelines
* Separating training and inference logic
* Ensuring testability and deployment readiness

---

## 🎯 Objectives

* Build a regression model using PyTorch
* Design a modular ML pipeline:

  * Data ingestion & preprocessing
  * Model training & evaluation
  * API-based inference
* Apply production-oriented practices:

  * Configuration-driven workflows
  * Dockerization
  * CI testing
  * Logging and monitoring hooks

---

## 📊 Dataset

* **Type:** Boston House price dataset from kaggle
* **Target:** House price
* **Features:** Fourteen Numerical and encoded categorical attributes
* **Size:** 506 Observations

**Data considerations:**

* Feature scaling required
* Skewed price distribution
* Risk of overfitting without regularization

> ⚠️ This project assumes historical patterns are predictive of future prices and does not model macroeconomic shifts.

---

## 🔍 Approach & Methodology

### 1️⃣ Data Preparation

* Raw data is cleaned, validated, and transformed through a dedicated data pipeline
* Feature engineering and preprocessing logic are fully reusable and testable
* Processed data is stored separately to ensure reproducibility

### 2️⃣ Model Design

* Fully connected neural network implemented in PyTorch
* Abstract base classes used to encourage extensibility
* Config-driven architecture allows rapid experimentation

Model Architecture
```
HousePricesModel(
  (dense1): DenseLayer(
    (linear): Linear(in_features=13, out_features=13, bias=True)
    (activation): LeakyReLU(negative_slope=0.01)
  )
  (dense2): DenseLayer(
    (linear): Linear(in_features=13, out_features=1, bias=True)
  )
)
```

### 3️⃣ Training & Evaluation

* Training pipeline initializes model, optimizer, and loss function
* Metrics computed on validation data to monitor generalization
* Logging captures training progress and preprocessing behavior

---

## 🧪 Results

> *(Results may vary depending on configuration and dataset split)*

| Metric | Value   |
| ------ | ------- |
| MSE   | ~31.1380 |
| MAE   | ~4.4565  |
| R2    | ~-2.7972 |

**Observations:**

* Model converges consistently without instability
* Performance comparable to tree-based baselines
* Deep learning chosen primarily for architectural learning, not raw accuracy

---

## 📈 Visualizations

* Exploratory Data Analysis (EDA)
* Feature distributions and correlations
* Training vs validation loss curves

---

## ⚙️ Tech Stack

**Languages & Frameworks**

* Python
* PyTorch
* Pandas, NumPy

**API & Infrastructure**

* FastAPI
* Docker, Docker Compose

**Testing & CI**

* Pytest
* GitHub Actions

**Experimentation & Monitoring**

* Jupyter Notebooks
* Python logging module

---

## 🏗️ Project Structure

```text
project/
│
├── .github/workflows/ci.yml        # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/                        # training, data, inference configs
│   ├── training.yaml
│   ├── data.yaml
│   ├── paths.yaml
│   └── inference.yaml
│
├── notebooks/
│   ├── eda.ipynb
│   └── experiments.ipynb
│
├── models/
│   ├── definitions/
│   └── checkpoints/
│
├── scripts/                        # automation & CI helpers
│
├── tests/
│   ├── test_data_pipeline.py
│   └── test_api_routes.py
│
└── src/
    ├── pipeline/
    ├── layer/
    ├── trainer/
    ├── routes/
    ├── utils/
    └── cli.py                      # primary developer interface
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/mohsinraza2999/house-price-prediction.git
cd house-price-prediction
bash scripts/docker_build.sh
python src/cli.py preprocess
python src/cli.py train
python src/cli.py route
or
python src/routes/router.py
```

---

## 🔮 Making Predictions

```bash
python src/routes/predict.py
```

Example response:

```json
{
  "price_predicted": 450000
}
```

---

## 🧪 Testing

Run all unit and integration tests:

```bash
pytest tests/
```

Tests cover:

* Data preprocessing pipeline
* API routes
* Model inference behavior

---

## 🧱 Docker Build

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## 🔧 Configuration

* All hyperparameters stored in YAML files
* Data paths, training parameters, and inference behavior configurable
* Environment-agnostic (local or containerized)

---

## 🧠 Design Decisions & Trade-offs

* **Why deep learning?**
  While tree-based models perform well on tabular data, a neural network was chosen to practice model abstraction, extensibility, and deployment workflows.

* **Why config-driven pipelines?**
  To separate experimentation from code changes and improve reproducibility.

* **Why both CLI and scripts?**
  CLI serves developers; scripts support automation and CI.

---

## ⚠️ Limitations & Future Improvements

* Limited feature encoding for categorical variables
* No model explainability yet
* Future improvements:

  * SHAP-based explanations
  * Model monitoring & drift detection
  * Ensemble methods
  * Cloud deployment

---

## 🧠 Key Learnings

* ML systems should be designed as maintainable software
* Testing data pipelines prevents silent failures
* Separation of training and inference is critical
* Over-engineering can be educational when intentional

---

## 📜 CI & Automation

* GitHub Actions pipeline:

  * Runs tests on push
  * Ensures build stability
* Docker build validation included

---

## 📬 Contact

**Author:** Mohsin Raza
**Target Role:** Junior Machine Learning Engineer / AI Engineer
**GitHub:** [github/mohsinraza2999](https://github.com/mohsinraza2999)
**LinkedIn:** *[linkedin/mohsin-raza](https://www.linkedin.com/in/mohsin-raza-b7ab73328)*

---
