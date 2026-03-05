# Dynamic Pricing Engine

A production-grade dynamic pricing system that predicts the **optimal nightly price** for Airbnb listings in real time. It combines ML demand forecasting, price elasticity estimation from economic theory, and mathematical revenue optimization — all built with free, open-source tools and zero paid APIs.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [ML Models — How They Work](#ml-models--how-they-work)
- [The Optimization Math](#the-optimization-math)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)
- [Testing Strategy](#testing-strategy)
- [CI/CD Pipeline](#cicd-pipeline)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What This Project Does

Given a listing's attributes (room type, beds, neighborhood, dates), the system:

1. **Forecasts demand** — predicts the probability of booking using XGBoost trained on temporal, weather, holiday, and listing features
2. **Estimates price elasticity** — models how sensitive customers are to price changes using log-log Ridge regression (grounded in microeconomic theory)
3. **Optimizes revenue** — finds the price P* that maximizes Revenue = P x Demand(P) using scipy bounded optimization
4. **Flags anomalies** — detects unusual demand patterns using Isolation Forest
5. **Serves predictions** — exposes everything via a FastAPI REST API with sub-200ms latency
6. **Visualizes results** — real-time Streamlit dashboard with price curves, demand heatmaps, and SHAP explanations

---

## System Architecture

```
 DATA SOURCES (all free, no API keys)
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │Inside Airbnb │  │  Open-Meteo   │  │  Nager.Date   │
 │ listings.csv │  │  weather API  │  │  holidays API │
 │ calendar.csv │  │  (no key)     │  │  (no key)     │
 │ reviews.csv  │  │              │  │              │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        v                 v                 v
 ┌─────────────────────────────────────────────────────┐
 │              DATA INGESTION LAYER                   │
 │         src/data/ingestion.py                       │
 │  download_airbnb_data | fetch_weather | fetch_holidays│
 └───────────────────────┬─────────────────────────────┘
                         v
 ┌─────────────────────────────────────────────────────┐
 │          PREPROCESSING & FEATURE ENGINEERING        │
 │    src/data/preprocessing.py + feature_engineering.py│
 │                                                     │
 │  Temporal (8)  | Weather (6) | Holiday (4)          │
 │  Location (1)  | Demand (3)  | Competitor (5)       │
 │  Listing (5)   = 32 total features                  │
 └───────────────────────┬─────────────────────────────┘
                         v
 ┌─────────────────────────────────────────────────────┐
 │                ML MODEL LAYER                       │
 │                                                     │
 │  ┌─────────────┐  ┌──────────────┐                 │
 │  │  XGBoost    │  │  Ridge       │                 │
 │  │  Demand     │  │  Elasticity  │                 │
 │  │  Forecaster │  │  Estimator   │                 │
 │  └──────┬──────┘  └──────┬───────┘                 │
 │         │                │         ┌──────────────┐│
 │         v                v         │  Isolation   ││
 │  ┌──────────────────────────────┐  │  Forest      ││
 │  │  scipy Revenue Optimizer     │  │  Anomaly     ││
 │  │  Revenue = P x D(P)         │  │  Detector    ││
 │  │  D(P) = D0 x (P/P0)^e      │  └──────────────┘│
 │  └──────────────┬───────────────┘                  │
 └─────────────────┼──────────────────────────────────┘
                   v
 ┌──────────────────────┐    ┌──────────────────────┐
 │  FastAPI Service     │    │  MLflow Tracking      │
 │  5 REST endpoints    │    │  params, metrics,     │
 │  Pydantic v2 schemas │    │  model artifacts      │
 │  CORS + logging      │    └──────────────────────┘
 └──────────┬───────────┘
            v
 ┌──────────────────────┐    ┌──────────────────────┐
 │  Streamlit Dashboard │    │  Evidently AI        │
 │  6 interactive panels│    │  Drift Monitoring    │
 │  real-time updates   │    │  HTML reports        │
 └──────────────────────┘    └──────────────────────┘
```

---

## Tech Stack

Every tool is 100% free and open-source. No credit card required.

| Layer | Tool | Version | Purpose |
|-------|------|---------|---------|
| **Data Ingestion** | pandas | 2.x | Data wrangling and transformation |
| | requests | Latest | HTTP fetching from APIs |
| | pyarrow | 14+ | Parquet file I/O |
| **Feature Engineering** | scikit-learn | 1.4+ | Preprocessing pipelines, KMeans clustering |
| | feature-engine | 1.6+ | Advanced feature engineering |
| **ML Models** | XGBoost | 2.x | Demand forecasting (primary model) |
| | LightGBM | Latest | Alternative gradient booster |
| | scikit-learn | 1.4+ | Ridge regression, Isolation Forest |
| **Optimization** | scipy | 1.11+ | Revenue maximization (minimize_scalar) |
| **Explainability** | SHAP | 0.44+ | Feature importance and explanations |
| **Experiment Tracking** | MLflow | 2.x | Track runs, params, metrics, models |
| **Monitoring** | Evidently AI | 0.4+ | Data drift and model drift reports |
| **API** | FastAPI | Latest | REST API serving predictions |
| | uvicorn | Latest | ASGI server |
| | Pydantic | v2 | Request/response validation |
| **Dashboard** | Streamlit | 1.30+ | Real-time pricing UI |
| | Plotly | 5.18+ | Interactive charts |
| **Testing** | pytest | 7.4+ | Unit + integration tests |
| | httpx | Latest | API testing client |
| | pytest-cov | Latest | Coverage reporting |
| **Linting** | ruff | 0.2+ | Fast Python linter and formatter |
| **Containers** | Docker | Latest | Reproducible environment |
| | Docker Compose | Latest | Multi-service orchestration |
| **CI/CD** | GitHub Actions | Free tier | Automated testing + linting on push |
| **Config** | PyYAML | 6.0+ | YAML configuration loading |

---

## Project Structure

```
dynamic-pricing-engine/
│
├── data/                           # All data files (gitignored)
│   ├── raw/                        # Downloaded CSVs (listings, calendar, reviews)
│   ├── processed/                  # Feature-engineered parquet files
│   └── external/                   # Weather & holiday data
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb# Feature development & documentation
│   ├── 03_model_experiments.ipynb  # Model training, SHAP, comparison
│   └── 04_optimization_logic.ipynb # Revenue optimization math deep-dive
│
├── src/                            # All production Python code
│   ├── data/
│   │   ├── ingestion.py            # Download & load raw data (290 lines)
│   │   ├── preprocessing.py        # Clean, validate, transform (119 lines)
│   │   └── feature_engineering.py  # 32 features across 7 categories (509 lines)
│   ├── models/
│   │   ├── demand_forecaster.py    # XGBoost demand model (278 lines)
│   │   ├── elasticity_estimator.py # Ridge log-log elasticity (248 lines)
│   │   ├── optimizer.py            # scipy revenue optimization (213 lines)
│   │   └── anomaly_detector.py     # Isolation Forest wrapper (129 lines)
│   ├── api/
│   │   ├── main.py                 # FastAPI app entry point (64 lines)
│   │   ├── routes.py               # All 5 API route definitions (235 lines)
│   │   ├── schemas.py              # Pydantic v2 request/response models (138 lines)
│   │   └── dependencies.py         # Model loading via lifespan (173 lines)
│   ├── monitoring/
│   │   └── drift_detector.py       # Evidently AI integration (152 lines)
│   └── utils/
│       ├── config.py               # YAML config loader (27 lines)
│       └── logger.py               # Structured logging (20 lines)
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard (426 lines)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (sample DataFrames)
│   ├── unit/
│   │   ├── test_preprocessing.py   # 11 tests — price cleaning, outliers
│   │   ├── test_feature_engineering.py # 14 tests — all feature categories
│   │   ├── test_demand_model.py    # 6 tests — XGBoost train/predict
│   │   ├── test_elasticity_model.py# 5 tests — elasticity < 0 enforcement
│   │   ├── test_optimizer.py       # 15 tests — bounds, edge cases, curves
│   │   ├── test_drift_detector.py  # 5 tests — anomaly detection
│   │   └── test_schemas.py         # 9 tests — Pydantic validation
│   └── integration/
│       └── test_api_predict.py     # 11 tests — full API endpoint testing
│
├── docker/
│   ├── Dockerfile.api              # API container (python:3.11-slim)
│   └── Dockerfile.dashboard        # Dashboard container
│
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions: test + lint on push
│
├── config.yaml                     # All configuration in one place
├── docker-compose.yml              # One-command deployment
├── requirements.txt                # All Python dependencies
├── pyproject.toml                  # Ruff + pytest configuration
├── Makefile                        # Shortcut commands
├── .gitignore                      # Excludes data, models, caches
└── README.md                       # This file
```

---

## Prerequisites

- **Python 3.11+** (tested on 3.11 and 3.12)
- **pip** (Python package manager)
- **Git** (version control)
- **Docker** (optional, for containerized deployment)
- **Internet connection** (for downloading data and weather/holiday APIs)

No paid API keys, no cloud accounts, no credit cards needed.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/dynamic-pricing-engine.git
cd dynamic-pricing-engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# or
make install
```

This installs all 25+ packages. Everything is free and pip-installable.

### 4. Verify installation

```bash
pytest tests/unit/test_optimizer.py -v
```

If all 15 optimizer tests pass, your environment is ready.

---

## Step-by-Step Workflow

This is the full pipeline from raw data to a live pricing dashboard.

### Step 1: Download Data

```bash
make data
# or: python -m src.data.ingestion
```

This downloads three datasets from Inside Airbnb (NYC):
- `listings.csv.gz` — 40K+ listings with price, location, amenities
- `calendar.csv.gz` — 365-day availability and pricing per listing
- `reviews.csv.gz` — historical review data

It also fetches:
- Historical weather from Open-Meteo API (free, no key)
- US public holidays from Nager.Date API (free, no key)

Files are saved to `data/raw/` and `data/external/`.

### Step 2: Run Feature Engineering

```bash
python -m src.data.feature_engineering
```

This pipeline:
1. Cleans price strings (`$1,234.56` -> `1234.56`), removes outliers
2. Builds 32 features across 7 categories (temporal, weather, holiday, location, demand, competitor, listing)
3. Clusters listings by lat/lon using KMeans (15 clusters)
4. Computes competitor pricing features (neighborhood median, percentile rank)
5. Saves to `data/processed/listings_features.parquet` and `calendar_features.parquet`

### Step 3: Explore the Data (optional but recommended)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The EDA notebook includes:
- Price distribution (raw + log-transformed)
- Room type breakdown with box plots
- Geographic scatter map of NYC listings
- Daily/weekly booking rate patterns
- Feature correlation matrix
- Review velocity as a demand proxy

### Step 4: Train Models

The recommended way is through the notebook:

```bash
jupyter notebook notebooks/03_model_experiments.ipynb
```

Or via the command line:

```bash
make train
```

This trains 3 models:

| Model | What it does | Algorithm |
|-------|-------------|-----------|
| Demand Forecaster | Predicts P(booking) for a listing/date | XGBoost Classifier |
| Elasticity Estimator | Estimates % demand change per % price change | Ridge Regression (log-log) |
| Anomaly Detector | Flags unusual demand patterns | Isolation Forest |

All runs are tracked in MLflow. View them with:

```bash
make mlflow
# Opens at http://localhost:5000
```

Models are saved to the `models/` directory.

### Step 5: Start the API

```bash
make run
# or: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API starts at `http://localhost:8000`. Auto-generated Swagger docs are at `http://localhost:8000/docs`.

### Step 6: Start the Dashboard

```bash
make dashboard
# or: streamlit run dashboard/app.py --server.port 8501
```

The dashboard opens at `http://localhost:8501`. It works with or without the API running (falls back to local computation).

### Step 7: Run Tests

```bash
make test           # Run all 81 tests
make test-cov       # Run with coverage report
make lint           # Lint with ruff
```

---

## API Reference

Base URL: `http://localhost:8000`

### POST /predict — Core Pricing Endpoint

Returns the optimal nightly price for a listing.

**Request Body:**

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| room_type | string | Yes | — | `"Entire home/apt"`, `"Private room"`, or `"Shared room"` |
| beds | int | Yes | 1-20 | Number of beds |
| bathrooms | float | Yes | 0.5-10 | Number of bathrooms |
| neighborhood | string | Yes | — | Neighborhood name from dataset |
| checkin_date | string | Yes | YYYY-MM-DD | Check-in date |
| checkout_date | string | Yes | YYYY-MM-DD | Check-out date |
| amenity_score | float | No | 0-1 | Fraction of top amenities (default: 0.5) |
| review_score | float | No | 0-5 | Average review rating (default: 4.0) |

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "Entire home/apt",
    "beds": 2,
    "bathrooms": 1.0,
    "neighborhood": "Williamsburg",
    "checkin_date": "2024-07-15",
    "checkout_date": "2024-07-18",
    "amenity_score": 0.7,
    "review_score": 4.5
  }'
```

**Example Response:**

```json
{
  "optimal_price": 187.50,
  "price_range": [168.75, 206.25],
  "expected_revenue": 562.50,
  "demand_forecast": 0.72,
  "elasticity_coeff": -1.2,
  "market_avg_price": 165.00,
  "shap_top_features": [
    {"feature": "is_weekend", "contribution": 0.15},
    {"feature": "month", "contribution": 0.12},
    {"feature": "review_score", "contribution": 0.09},
    {"feature": "beds", "contribution": 0.08},
    {"feature": "amenity_score", "contribution": 0.04}
  ],
  "is_anomaly": false,
  "model_version": "v0.1.0"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| optimal_price | float | Recommended nightly price in USD |
| price_range | [float, float] | 90% confidence interval [lower, upper] |
| expected_revenue | float | Projected total revenue over the stay |
| demand_forecast | float | Predicted occupancy rate (0-1) |
| elasticity_coeff | float | Price elasticity estimate (typically -3 to 0) |
| market_avg_price | float | Median competitor price in that neighborhood |
| shap_top_features | list | Top 5 features driving the price recommendation |
| is_anomaly | bool | True if demand pattern is unusual |
| model_version | string | Model version for traceability |

### GET /health — Liveness Check

```bash
curl http://localhost:8000/health
```

Returns `200` when models are loaded, `503` otherwise. Use this for CI/CD health checks and load balancers.

### GET /metrics — Model Performance

```bash
curl http://localhost:8000/metrics
```

Returns current model AUC, F1, elasticity coefficient, R-squared, and total predictions served.

### POST /explain — SHAP Explanations

Same request body as `/predict`. Returns full SHAP values for every feature, plus the model's base prediction value.

### GET /drift-report — Data Drift Monitoring

```bash
curl http://localhost:8000/drift-report
```

Returns drift status (`healthy` / `degraded` / `critical`), overall drift score, list of drifted features, and a link to the Evidently HTML report.

**Invalid inputs automatically return HTTP 422** with detailed Pydantic validation errors — no bad data reaches the models.

---

## Dashboard

The Streamlit dashboard has 6 interactive panels:

| Panel | What It Shows |
|-------|--------------|
| **Key Metrics** | Recommended price (big number), delta vs market, demand %, total revenue |
| **Price vs Revenue Curve** | Interactive Plotly chart showing the optimization surface with optimal and market price marked |
| **7-Day Demand Forecast** | Bar chart of predicted occupancy for the next 7 days with weekend boost |
| **SHAP Feature Drivers** | Horizontal bar chart showing which features pushed the price up or down |
| **Price Positioning** | Your recommended price vs floor / market average / ceiling |
| **Drift Monitor** | Gauge chart showing model health (green/yellow/red) with drift score |

The sidebar lets you adjust all listing parameters in real time. Auto-refresh mode updates every 30 seconds.

The dashboard works in two modes:
- **With API**: Calls `localhost:8000/predict` for full ML predictions
- **Without API**: Falls back to local computation (no API server needed)

---

## ML Models — How They Work

### 1. Demand Forecaster (XGBoost)

**Goal**: Predict P(booking) for a given listing on a given date.

- **Algorithm**: XGBoost Classifier (gradient-boosted trees)
- **Target**: `was_booked` (0 or 1, derived from calendar availability)
- **Features**: 19 features — temporal, weather, rolling occupancy
- **Validation**: TimeSeriesSplit with 5 folds (no data leakage — the future never predicts the past)
- **Metrics**: AUC, F1, Accuracy logged per fold to MLflow
- **Hyperparameter tuning**: RandomizedSearchCV over n_estimators, max_depth, learning_rate, subsample, colsample_bytree

### 2. Elasticity Estimator (Ridge Regression)

**Goal**: Estimate how sensitive demand is to price changes.

- **Algorithm**: Ridge Regression on log-transformed data
- **Specification**: `log(demand) = beta * log(price) + controls`
- **Key output**: Elasticity coefficient `beta` (e.g., -1.2 means a 1% price increase causes 1.2% demand decrease)
- **Controls**: day_of_week, month, is_weekend, season, room_type, beds, location cluster
- **Validation**: Coefficient must be negative (economic law). Bounded within [-5, 0].

### 3. Revenue Optimizer (scipy)

**Goal**: Find the price P* that maximizes expected revenue.

- **Formula**: `Revenue(P) = P * Demand(P)` where `Demand(P) = D0 * (P/P0)^elasticity`
- **Method**: `scipy.optimize.minimize_scalar` with bounded search
- **Bounds**: Price floor = market price x 0.7, ceiling = market price x 1.5
- **Output**: Optimal price, expected revenue, revenue lift %, confidence interval

### 4. Anomaly Detector (Isolation Forest)

**Goal**: Flag unusual demand patterns that may indicate data issues or special events.

- **Algorithm**: Isolation Forest (unsupervised)
- **Contamination**: 5% (flags top 5% most unusual patterns)
- **Features**: 9 demand-related features

---

## The Optimization Math

This is what separates this project from 99% of ML portfolios. Most projects predict. This project **optimizes**.

```
Revenue = Price x Demand(Price)

Demand is elastic:
  D(P) = D0 x (P / P0) ^ e

Where:
  D0 = predicted demand at current market price (from XGBoost)
  P0 = current market price (neighborhood median)
  e  = price elasticity coefficient (from Ridge regression, typically -1.5)

scipy.optimize.minimize_scalar finds P* that maximizes Revenue
  subject to: floor_price <= P* <= ceiling_price
```

**Key insight**: When |e| > 1 (elastic demand), there exists an interior optimal price where raising the price further causes revenue to *drop* because demand falls too fast. The optimizer finds this sweet spot.

---

## Feature Engineering Pipeline

32 features across 7 categories, built in `src/data/feature_engineering.py`:

| Category | Count | Features | Source |
|----------|-------|----------|--------|
| **Temporal** | 8 | day_of_week, day_of_month, week_of_year, month, quarter, is_weekend, is_month_start, is_month_end | Calendar dates |
| **Season/Checkin** | 2 | season (0-3), days_from_start | Calendar dates |
| **Weather** | 6 | temperature_mean, precipitation_sum, wind_speed_max, is_hot_day, is_cold_day, is_rainy_day | Open-Meteo API |
| **Holiday** | 4 | is_holiday, days_until_holiday, days_since_holiday, near_holiday | Nager.Date API |
| **Location** | 1 | location_cluster (KMeans, 15 clusters) | Listing lat/lon |
| **Demand Proxy** | 3 | occupancy_rate, rolling_7d_occupancy, rolling_30d_occupancy | Calendar bookings |
| **Competitor** | 5 | median_neighborhood_price, mean_neighborhood_price, std_neighborhood_price, price_vs_neighborhood, price_rank_in_neighborhood | Computed from listings |
| **Listing** | 3 | room_type_encoded, bathrooms (extracted from text), amenity_score | Listing attributes |

---

## Testing Strategy

### Test Summary

```
81 tests total
├── Unit tests (70)
│   ├── test_preprocessing.py      — 11 tests (price cleaning, outlier clipping)
│   ├── test_feature_engineering.py — 14 tests (temporal, location, competitor, listing features)
│   ├── test_demand_model.py       —  6 tests (train, predict, feature importance)
│   ├── test_elasticity_model.py   —  5 tests (elasticity < 0, bounded range)
│   ├── test_optimizer.py          — 15 tests (demand function, revenue, bounds, edge cases)
│   ├── test_drift_detector.py     —  5 tests (fit, predict, contamination rate)
│   └── test_schemas.py            —  9 tests (Pydantic rejects invalid inputs)
│   └── (additional)               —  5 tests
└── Integration tests (11)
    └── test_api_predict.py        — 11 tests (all 5 endpoints, error handling)
```

### What Each Test Module Validates

| Module | Key Assertions |
|--------|---------------|
| **test_preprocessing** | Price strings parsed correctly, outliers clipped at percentiles, missing critical fields dropped, calendar dates parsed |
| **test_feature_engineering** | Weekend detection correct, seasons mapped right, KMeans creates correct cluster count, competitor rank between 0-1 |
| **test_demand_model** | Predictions are probabilities in [0,1], feature importance non-negative, untrained model raises RuntimeError |
| **test_elasticity_model** | Elasticity coefficient is negative (economic law), bounded within [-5, 0], returns float |
| **test_optimizer** | Demand at base price = base demand, optimal price within bounds, revenue at optimal > revenue at floor and ceiling, handles zero demand, handles positive elasticity gracefully |
| **test_drift_detector** | Contamination rate matches config, anomaly scores are finite, anomalies have lower scores than normal points |
| **test_schemas** | Pydantic rejects beds=0, bathrooms=-1, review_score=6.0, amenity_score=1.5, demand_forecast > 1 |
| **test_api_predict** | POST /predict returns 200 with valid payload, 422 with missing/invalid fields, all 3 room types work, /health returns 200, /metrics returns performance stats |

### Running Tests

```bash
# All tests
pytest tests/ -v

# Just unit tests (fast, no model training)
pytest tests/unit/ -v

# Just integration tests (spins up FastAPI TestClient)
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Single test file
pytest tests/unit/test_optimizer.py -v
```

---

## CI/CD Pipeline

GitHub Actions runs automatically on every push to `main` and on pull requests.

**Pipeline steps** (`.github/workflows/ci.yml`):

1. **Setup**: Python 3.11 and 3.12 matrix
2. **Cache**: pip packages cached by requirements.txt hash
3. **Install**: `pip install -r requirements.txt`
4. **Lint**: `ruff check src/ tests/ dashboard/`
5. **Unit tests**: `pytest tests/unit/ -v`
6. **Integration tests**: `pytest tests/integration/ -v`
7. **Coverage**: `pytest --cov=src --cov-fail-under=70`

If any step fails, the push is flagged. No broken code on main.

---

## Docker Deployment

### Build and run everything with one command

```bash
docker compose up --build
```

This starts:
- **API**: `http://localhost:8000` (Swagger docs at `/docs`)
- **Dashboard**: `http://localhost:8501`

### Architecture

```
docker-compose.yml
├── api service
│   ├── Dockerfile: docker/Dockerfile.api
│   ├── Base image: python:3.11-slim
│   ├── Port: 8000
│   ├── Volumes: data/, models/, mlruns/
│   └── Health check: GET /health every 30s
│
└── dashboard service
    ├── Dockerfile: docker/Dockerfile.dashboard
    ├── Base image: python:3.11-slim
    ├── Port: 8501
    ├── Volumes: data/, models/
    └── Depends on: api service
```

### Individual containers

```bash
# Build just the API
docker build -f docker/Dockerfile.api -t pricing-api .
docker run -p 8000:8000 pricing-api

# Build just the dashboard
docker build -f docker/Dockerfile.dashboard -t pricing-dashboard .
docker run -p 8501:8501 pricing-dashboard
```

### Stop

```bash
docker compose down
```

---

## Configuration

All configuration is in `config.yaml`. Key sections:

| Section | What It Controls |
|---------|-----------------|
| `data.insideairbnb` | Download URLs for NYC Airbnb data |
| `weather` | Open-Meteo API coordinates (NYC: 40.71, -74.00) |
| `holidays` | Nager.Date country code (US) |
| `features` | Price floor/ceiling percentiles, cluster count, rolling windows |
| `model.xgboost` | n_estimators, max_depth, learning_rate, early_stopping |
| `optimization` | Price floor multiplier (0.7x), ceiling multiplier (1.5x) |
| `api` | Host, port, model path |
| `dashboard` | Host, port, API URL, refresh interval |
| `mlflow` | Tracking URI, experiment name |
| `monitoring` | Drift thresholds: degraded > 0.3, critical > 0.6 |

To switch to a different city, update the `data.insideairbnb` URLs and the `weather` coordinates in `config.yaml`.

---

## Notebooks

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `01_eda.ipynb` | Exploratory data analysis | Price distributions, geographic maps, temporal patterns, correlation matrix |
| `02_feature_engineering.ipynb` | Feature documentation | Distribution of all 32 features, correlation with price, cluster visualization |
| `03_model_experiments.ipynb` | Model training & evaluation | XGBoost AUC, elasticity coefficient, SHAP plots, model comparison table |
| `04_optimization_logic.ipynb` | Math deep-dive | Demand curves at different elasticities, revenue surfaces, analytical vs numerical solution verification |

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **TimeSeriesSplit** over random KFold | Prevents data leakage — the future cannot predict the past. This is the #1 trap in temporal ML. |
| **Log-log elasticity** model | Grounded in microeconomic theory. The constant-elasticity specification is standard in pricing research. |
| **Bounded scipy optimization** | Without bounds, the optimizer can explore nonsensical negative prices. Bounds keep recommendations within [market x 0.7, market x 1.5]. |
| **FastAPI lifespan** for model loading | Models are loaded once at startup, not per-request. This is the correct production pattern — avoids latency spikes and memory leaks. |
| **Pydantic v2 Field constraints** | `Field(ge=0, le=5)` gives free input validation. FastAPI returns 422 automatically for out-of-range inputs. |
| **MLflow tracking** for every run | Creates a permanent, reproducible audit trail. Every training run logs params, metrics, and model artifacts. |
| **Evidently AI drift detection** | Production ML models degrade silently. Drift monitoring catches distribution shifts before they affect predictions. |
| **Dashboard local fallback** | The dashboard works without the API running — it computes prices locally. This makes demos more reliable. |
| **Parquet over CSV** for processed data | 5-10x faster reads, columnar storage, type preservation. This is what real data teams use. |

---

## Makefile Commands Reference

| Command | What It Does |
|---------|-------------|
| `make install` | Install all dependencies from requirements.txt |
| `make data` | Download Inside Airbnb data + weather + holidays |
| `make train` | Train demand forecaster + elasticity estimator |
| `make run` | Start FastAPI on port 8000 with hot reload |
| `make dashboard` | Start Streamlit on port 8501 |
| `make mlflow` | Start MLflow UI on port 5000 |
| `make test` | Run all 81 tests with verbose output |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Lint with ruff (check + format) |
| `make format` | Auto-format code with ruff |
| `make docker-up` | Build and start Docker containers |
| `make docker-down` | Stop Docker containers |
| `make clean` | Remove __pycache__ and .pyc files |

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root directory, not from a subdirectory |
| Data download fails | Check internet connection. Inside Airbnb URLs change periodically — update `config.yaml` with current URLs from insideairbnb.com |
| MLflow FutureWarning about filesystem backend | This is a non-breaking warning. MLflow still works fine with the file-based backend. |
| XGBoost `use_label_encoder` warning | Harmless deprecation warning. Does not affect model performance. |
| Dashboard shows default prices | Models haven't been trained yet. Run `make train` or the model experiments notebook first. |
| Docker build fails | Ensure Docker Desktop is running. On Windows, enable WSL2 integration. |
| Tests fail with import errors | Ensure all dependencies are installed: `pip install -r requirements.txt` |
| Weather/holiday API timeout | These are free public APIs with no SLA. The pipeline gracefully skips them if unavailable. |

---

## Data Sources

All data is free, publicly available, and requires no API keys.

| Source | What It Provides | Access |
|--------|-----------------|--------|
| **Inside Airbnb** | Listings (40K+), calendar (365 days x listings), reviews | Direct CSV download from insideairbnb.com |
| **Open-Meteo** | Historical daily weather (temperature, precipitation, wind) for any city | REST API, no key needed |
| **Nager.Date** | Public holidays for 100+ countries | REST API, no key needed |

---

## License

MIT
