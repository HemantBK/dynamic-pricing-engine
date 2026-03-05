"""Dependency injection for the FastAPI service.

Loads models once at startup using FastAPI lifespan.
Models are cached in app.state — never re-loaded per request.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

from src.models.anomaly_detector import AnomalyDetector
from src.models.demand_forecaster import DemandForecaster
from src.models.elasticity_estimator import ElasticityEstimator
from src.models.optimizer import get_optimal_price
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Room type encoding map (must match training)
ROOM_TYPE_MAP = {
    "Entire home/apt": 0,
    "Hotel room": 1,
    "Private room": 2,
    "Shared room": 3,
}


@dataclass
class ModelState:
    """Container for all loaded models and runtime state."""

    demand_forecaster: DemandForecaster | None = None
    elasticity_estimator: ElasticityEstimator | None = None
    anomaly_detector: AnomalyDetector | None = None
    config: dict = field(default_factory=dict)
    model_version: str = "v0.1.0"
    is_loaded: bool = False
    total_predictions: int = 0
    # Neighborhood price lookup (precomputed from training data)
    neighborhood_prices: dict = field(default_factory=dict)


def _load_models(state: ModelState) -> None:
    """Load all models from disk into the state container."""
    config = load_config()
    state.config = config
    models_dir = PROJECT_ROOT / "models"

    # Load demand forecaster
    demand_path = models_dir / "demand_forecaster"
    if demand_path.exists():
        state.demand_forecaster = DemandForecaster(config)
        state.demand_forecaster.load(demand_path)
        logger.info("Demand forecaster loaded")
    else:
        logger.warning(f"Demand forecaster not found at {demand_path}. Training a fresh model.")
        state.demand_forecaster = DemandForecaster(config)

    # Load elasticity estimator
    elasticity_path = models_dir / "elasticity_estimator"
    if elasticity_path.exists():
        state.elasticity_estimator = ElasticityEstimator(config)
        state.elasticity_estimator.load(elasticity_path)
        logger.info("Elasticity estimator loaded")
    else:
        logger.warning(f"Elasticity estimator not found at {elasticity_path}.")
        state.elasticity_estimator = ElasticityEstimator(config)
        state.elasticity_estimator.elasticity_coeff = -1.2  # Reasonable default
        state.elasticity_estimator.metrics = {"elasticity_coeff": -1.2, "r2": 0.0, "mae": 0.0, "rmse": 0.0}

    # Load anomaly detector
    anomaly_path = models_dir / "anomaly_detector"
    if anomaly_path.exists():
        state.anomaly_detector = AnomalyDetector(config=config)
        state.anomaly_detector.load(anomaly_path)
        logger.info("Anomaly detector loaded")
    else:
        logger.warning(f"Anomaly detector not found at {anomaly_path}.")
        state.anomaly_detector = None

    # Load neighborhood prices if available
    features_path = PROJECT_ROOT / config["data"]["processed_dir"] / "listings_features.parquet"
    if features_path.exists():
        listings = pd.read_parquet(features_path)
        if "neighbourhood_cleansed" in listings.columns and "price" in listings.columns:
            state.neighborhood_prices = (
                listings.groupby("neighbourhood_cleansed")["price"]
                .median()
                .to_dict()
            )
            logger.info(f"Loaded prices for {len(state.neighborhood_prices)} neighborhoods")
    else:
        logger.warning("No listings features found. Neighborhood prices unavailable.")

    state.is_loaded = True
    logger.info("All models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load models at startup, cleanup at shutdown."""
    state = ModelState()
    try:
        _load_models(state)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        state.is_loaded = False
    app.state.models = state
    logger.info("API startup complete")
    yield
    logger.info("API shutting down")


def get_model_state(request: Request) -> ModelState:
    """FastAPI dependency: get the model state from app.state."""
    return request.app.state.models


def build_features_from_request(
    room_type: str,
    beds: int,
    bathrooms: float,
    neighborhood: str,
    checkin_date: str,
    amenity_score: float,
    review_score: float,
) -> dict:
    """Convert API request fields into model feature dict."""
    dt = pd.Timestamp(checkin_date)

    features = {
        "day_of_week": dt.dayofweek,
        "day_of_month": dt.day,
        "week_of_year": dt.isocalendar()[1],
        "month": dt.month,
        "quarter": dt.quarter,
        "is_weekend": int(dt.dayofweek >= 5),
        "season": (
            0 if dt.month in [12, 1, 2] else
            1 if dt.month in [3, 4, 5] else
            2 if dt.month in [6, 7, 8] else 3
        ),
        "days_from_start": 0,
        "room_type_encoded": ROOM_TYPE_MAP.get(room_type, 0),
        "beds": beds,
        "bathrooms": bathrooms,
        "amenity_score": amenity_score,
        "review_score": review_score,
        # Defaults for features that need historical data
        "rolling_7d_occupancy": 0.5,
        "rolling_30d_occupancy": 0.5,
        "temperature_mean": 20.0,
        "precipitation_sum": 0.0,
        "wind_speed_max": 10.0,
        "is_hot_day": 0,
        "is_cold_day": 0,
        "is_rainy_day": 0,
        "is_holiday": 0,
        "days_until_holiday": 30,
        "near_holiday": 0,
        "location_cluster": 0,
        "occupancy_rate": 0.5,
        "price_rank_in_neighborhood": 0.5,
        "price_vs_neighborhood": 1.0,
    }
    return features
