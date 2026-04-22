# 💎 Gemstone Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A production-ready machine learning application that predicts the price of gemstones using multiple regression algorithms, served via a REST API built with FastAPI and containerized with Docker.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Running with Docker](#running-with-docker)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

The **Gemstone Price Prediction** project is an end-to-end machine learning pipeline designed to estimate the market price of gemstones based on their physical and grading characteristics. The project follows industry-standard MLOps practices — from data ingestion and preprocessing through model selection and serving — making it suitable for both learning purposes and real-world deployment.

The trained model is exposed through a clean **FastAPI** REST endpoint, allowing seamless integration with any frontend, mobile app, or third-party service.

---

## ✨ Features

- 🔄 **End-to-end ML pipeline** — data ingestion → transformation → model training → prediction
- 🤖 **Multi-model evaluation** — automatically selects the best model via cross-validated hyperparameter tuning
- 🚀 **FastAPI REST API** — lightweight, async-ready, with auto-generated OpenAPI docs
- 🐳 **Docker containerized** — consistent, reproducible environments across all platforms
- 📓 **Exploratory notebooks** — full EDA and model experimentation included
- 📝 **Custom logging & exception handling** — structured logs and traceable errors throughout the pipeline
- ☁️ **Cloud-ready** — deployment guides included for both Azure and Render

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Web Framework** | FastAPI, Uvicorn |
| **ML Libraries** | Scikit-learn, XGBoost, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Templating** | Jinja2 |
| **Containerization** | Docker |
| **Packaging** | Setuptools |

---

## 📁 Project Structure

```
Gemstone-Price-Prediction/
│
├── app.py                          # FastAPI application entry point
├── Dockerfile                      # Docker container configuration
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
│
├── src/                            # Core source package
│   ├── __init__.py
│   ├── logger.py                   # Custom logging configuration
│   ├── exception.py                # Custom exception handler
│   ├── utils.py                    # Shared utility functions
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── data_ingestion.py       # Data loading & train/test split
│   │   ├── data_transformation.py  # Feature engineering & preprocessing
│   │   └── model_trainer.py        # Model training & evaluation
│   │
│   └── pipline/                    # Inference & training pipelines
│       ├── predict_pipline.py      # Prediction pipeline for serving
│       └── training_pipline.py     # End-to-end training pipeline
│
├── notebook/                       # Jupyter notebooks
│   ├── BASIC_EDA_PERFORME.ipynb    # Exploratory data analysis
│   ├── MODEL_TRAINING.ipynb        # Model experimentation
│   └── data/
│       └── gemstone.csv            # Raw dataset
│
├── artifacts/                      # Generated model artifacts (auto-created)
│   ├── model.pkl                   # Saved best model
│   ├── preprocessor.pkl            # Saved data preprocessor
│   ├── train.csv                   # Training split
│   └── test.csv                    # Testing split
│
├── templates/                      # HTML templates for the web UI
│   └── index.html
│
├── logs/                           # Runtime logs (auto-created)
├── AZURE_DEPLOYMENT_GUIDE.txt      # Step-by-step Azure deployment guide
└── RENDER_DEPLOYMENT_GUIDE.txt     # Step-by-step Render deployment guide
```

---

## 🤖 ML Pipeline

The project follows a modular, sequential pipeline:

```
Raw Data (gemstone.csv)
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  ── Reads CSV, splits into train/test, saves to artifacts/
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Data Transformation │  ── Encodes categoricals, scales numerics, saves preprocessor
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Model Training    │  ── Trains & tunes 5 models, selects best by R² score
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Prediction API     │  ── Loads model.pkl + preprocessor.pkl, serves predictions
└─────────────────────┘
```

### Models Evaluated

| Model | Tuning Parameters |
|---|---|
| Decision Tree | `criterion` |
| Random Forest | `n_estimators` |
| Linear Regression | — |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |

The best-performing model (by R² score on the test set) is automatically serialized and used for inference.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- pip
- Docker (optional, for containerized runs)
- Git

### Local Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NazmulHudaNabil/gemstone-price-prediction.git
   cd gemstone-price-prediction
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Train the model** *(skip if `artifacts/` already contains `model.pkl`)*

   ```bash
   python src/components/data_ingestion.py
   ```

5. **Start the API server**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Open your browser** at [http://localhost:8000](http://localhost:8000)

---

### Running with Docker

1. **Build the Docker image**

   ```bash
   docker build -t gemstone-price-prediction .
   ```

2. **Run the container**

   ```bash
   docker run -p 8000:8000 gemstone-price-prediction
   ```

3. **Access the app** at [http://localhost:8000](http://localhost:8000)

---

## 📡 API Reference

Once the server is running, interactive API docs are available at:

- **Swagger UI** → [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc** → [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

#### `GET /`
Returns the web UI landing page.

---

#### `POST /predict`
Predicts the price of a gemstone based on input features.

**Request Body** (`application/json`):

```json
{
  "carat": 0.89,
  "depth": 62.1,
  "table": 57.0,
  "x": 6.1,
  "y": 6.07,
  "z": 3.78,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS1"
}
```

**Response**:

```json
{
  "prediction": 4521.37
}
```

---

## 📊 Model Performance

Model selection is automatic. The pipeline evaluates all candidates using **GridSearchCV** and retains the model with the highest **R² score** on the held-out test set (threshold: `R² ≥ 0.6`).

Refer to [`notebook/MODEL_TRAINING.ipynb`](notebook/MODEL_TRAINING.ipynb) for a full breakdown of model comparison metrics and training results.

---

## ☁️ Deployment

Pre-written deployment guides are included in the repository root:

| Platform | Guide |
|---|---|
| **Render** | [`RENDER_DEPLOYMENT_GUIDE.txt`](RENDER_DEPLOYMENT_GUIDE.txt) |
| **Azure** | [`AZURE_DEPLOYMENT_GUIDE.txt`](AZURE_DEPLOYMENT_GUIDE.txt) |

Both guides provide step-by-step instructions leveraging the existing Docker configuration for a smooth cloud deployment.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m 'feat: add your feature'`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

Please ensure your code follows existing patterns and includes appropriate logging.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Nazmul Huda Nabil](mailto:nabil648777@gmail.com)

⭐ Star this repo if you found it helpful!

</div>
