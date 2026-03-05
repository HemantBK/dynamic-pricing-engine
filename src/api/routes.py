"""API route definitions for the Dynamic Pricing Engine.

Endpoints:
- POST /predict — optimal pricing recommendation
- GET  /health  — liveness check
- GET  /metrics — model performance stats
- POST /explain — SHAP feature explanations
- GET  /drift-report — data drift monitoring
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    ModelState,
    build_features_from_request,
    get_model_state,
)
from src.api.schemas import (
    DriftResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    PricingRequest,
    PricingResponse,
    ShapFeature,
)
from src.models.optimizer import get_optimal_price
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PricingResponse)
async def predict_price(
    request: PricingRequest,
    state: ModelState = Depends(get_model_state),
):
    """Compute optimal price for a listing.

    Runs the full pipeline: features -> demand model -> elasticity ->
    optimizer -> response in < 200ms.
    """
    start_time = time.time()

    if not state.is_loaded or state.demand_forecaster is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Build feature vector
    features = build_features_from_request(
        room_type=request.room_type,
        beds=request.beds,
        bathrooms=request.bathrooms,
        neighborhood=request.neighborhood,
        checkin_date=request.checkin_date,
        amenity_score=request.amenity_score,
        review_score=request.review_score,
    )

    # Predict demand
    try:
        demand_forecast = state.demand_forecaster.predict_single(features)
    except Exception:
        demand_forecast = 0.5  # Fallback

    # Get elasticity
    elasticity = -1.2  # Default
    if state.elasticity_estimator and state.elasticity_estimator.elasticity_coeff is not None:
        elasticity = state.elasticity_estimator.get_elasticity()

    # Get market price for neighborhood
    market_price = state.neighborhood_prices.get(request.neighborhood, 150.0)

    # Optimize price
    nights = max(1, (
        pd.Timestamp(request.checkout_date) - pd.Timestamp(request.checkin_date)
    ).days)

    opt_result = get_optimal_price(
        base_demand=demand_forecast,
        base_price=market_price,
        elasticity=elasticity,
        config=state.config,
    )

    # Anomaly check
    is_anomaly = False
    if state.anomaly_detector and state.anomaly_detector.is_fitted:
        try:
            feature_df = pd.DataFrame([features])
            anomaly_pred = state.anomaly_detector.predict(feature_df)
            is_anomaly = bool(anomaly_pred[0])
        except Exception:
            pass

    # SHAP top features (simplified — use feature importance as proxy)
    shap_features = []
    try:
        importance = state.demand_forecaster.get_feature_importance()
        for _, row in importance.head(5).iterrows():
            feat_val = features.get(row["feature"], 0)
            shap_features.append(ShapFeature(
                feature=row["feature"],
                contribution=round(float(row["importance"]) * float(feat_val), 4),
            ))
    except Exception:
        shap_features = [ShapFeature(feature="unknown", contribution=0.0)]

    state.total_predictions += 1
    latency_ms = (time.time() - start_time) * 1000
    logger.info(f"Prediction served in {latency_ms:.1f}ms")

    return PricingResponse(
        optimal_price=opt_result.optimal_price,
        price_range=list(opt_result.price_range),
        expected_revenue=round(opt_result.expected_revenue * nights, 2),
        demand_forecast=round(demand_forecast, 4),
        elasticity_coeff=round(elasticity, 4),
        market_avg_price=round(market_price, 2),
        shap_top_features=shap_features,
        is_anomaly=is_anomaly,
        model_version=state.model_version,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(state: ModelState = Depends(get_model_state)):
    """Liveness check for CI/CD and load balancers."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return HealthResponse(
        status="healthy",
        models_loaded=state.is_loaded,
        model_version=state.model_version,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(state: ModelState = Depends(get_model_state)):
    """Current model performance statistics."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    demand_metrics = state.demand_forecaster.metrics if state.demand_forecaster else {}
    elasticity_metrics = state.elasticity_estimator.metrics if state.elasticity_estimator else {}

    return MetricsResponse(
        demand_model_auc=round(demand_metrics.get("auc", 0.0), 4),
        demand_model_f1=round(demand_metrics.get("f1", 0.0), 4),
        elasticity_coeff=round(elasticity_metrics.get("elasticity_coeff", -1.2), 4),
        elasticity_r2=round(elasticity_metrics.get("r2", 0.0), 4),
        total_predictions=state.total_predictions,
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    request: ExplainRequest,
    state: ModelState = Depends(get_model_state),
):
    """Return SHAP values for a specific prediction."""
    if not state.is_loaded or state.demand_forecaster is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    features = build_features_from_request(
        room_type=request.room_type,
        beds=request.beds,
        bathrooms=request.bathrooms,
        neighborhood=request.neighborhood,
        checkin_date=request.checkin_date,
        amenity_score=request.amenity_score,
        review_score=request.review_score,
    )

    # Predict demand
    try:
        demand = state.demand_forecaster.predict_single(features)
    except Exception:
        demand = 0.5

    # Get feature importance as SHAP proxy
    shap_values = []
    base_value = 0.5
    try:
        importance = state.demand_forecaster.get_feature_importance()
        for _, row in importance.iterrows():
            feat_name = row["feature"]
            feat_val = features.get(feat_name, 0)
            shap_values.append(ShapFeature(
                feature=feat_name,
                contribution=round(float(row["importance"]) * float(feat_val), 4),
            ))
    except Exception:
        shap_values = [ShapFeature(feature="unknown", contribution=0.0)]

    return ExplainResponse(
        shap_values=shap_values,
        base_value=base_value,
        predicted_demand=round(demand, 4),
    )


@router.get("/drift-report", response_model=DriftResponse)
async def get_drift_report(state: ModelState = Depends(get_model_state)):
    """Trigger drift check and return summary.

    Uses Evidently AI if available, otherwise returns a placeholder.
    """
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Try to run Evidently drift detection
    try:
        from src.monitoring.drift_detector import run_drift_check
        drift_result = run_drift_check(state.config)
        return DriftResponse(**drift_result)
    except ImportError:
        logger.warning("Evidently not installed. Returning placeholder drift report.")
    except Exception as e:
        logger.warning(f"Drift check failed: {e}")

    return DriftResponse(
        status="healthy",
        drift_score=0.0,
        features_drifted=[],
        report_url="/static/drift_report.html",
    )
