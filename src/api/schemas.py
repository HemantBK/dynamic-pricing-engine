"""Pydantic v2 request/response models for the Pricing API."""

from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    """Request body for POST /predict."""

    room_type: str = Field(
        ...,
        description="Room type: 'Entire home/apt', 'Private room', or 'Shared room'",
        examples=["Entire home/apt"],
    )
    beds: int = Field(..., ge=1, le=20, description="Number of beds")
    bathrooms: float = Field(..., ge=0.5, le=10, description="Number of bathrooms")
    neighborhood: str = Field(
        ...,
        description="Neighborhood name from the dataset taxonomy",
        examples=["Williamsburg"],
    )
    checkin_date: str = Field(
        ...,
        description="Check-in date in YYYY-MM-DD format",
        examples=["2024-07-15"],
    )
    checkout_date: str = Field(
        ...,
        description="Check-out date in YYYY-MM-DD format",
        examples=["2024-07-18"],
    )
    amenity_score: float = Field(
        0.5, ge=0, le=1,
        description="Fraction of top amenities present (0-1)",
    )
    review_score: float = Field(
        4.0, ge=0, le=5,
        description="Average review rating (0-5)",
    )


class ShapFeature(BaseModel):
    """A single SHAP feature contribution."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution to price")


class PricingResponse(BaseModel):
    """Response body for POST /predict."""

    optimal_price: float = Field(..., description="Recommended nightly price in USD")
    price_range: list[float] = Field(
        ...,
        min_length=2, max_length=2,
        description="[lower_bound, upper_bound] at 90% confidence",
    )
    expected_revenue: float = Field(..., description="Projected revenue over the date range")
    demand_forecast: float = Field(
        ..., ge=0, le=1,
        description="Predicted occupancy rate (0-1)",
    )
    elasticity_coeff: float = Field(
        ...,
        description="Price elasticity estimate (typically -3 to 0)",
    )
    market_avg_price: float = Field(
        ...,
        description="Median competitor price in the neighborhood",
    )
    shap_top_features: list[ShapFeature] = Field(
        ...,
        description="Top 5 features and their SHAP contribution to price",
    )
    is_anomaly: bool = Field(
        ...,
        description="True if demand pattern is unusual",
    )
    model_version: str = Field(
        ...,
        description="MLflow model version tag",
    )


class ExplainRequest(BaseModel):
    """Request body for POST /explain."""

    room_type: str = Field(..., description="Room type")
    beds: int = Field(..., ge=1, le=20, description="Number of beds")
    bathrooms: float = Field(..., ge=0.5, le=10, description="Number of bathrooms")
    neighborhood: str = Field(..., description="Neighborhood name")
    checkin_date: str = Field(..., description="Check-in date YYYY-MM-DD")
    amenity_score: float = Field(0.5, ge=0, le=1, description="Amenity score")
    review_score: float = Field(4.0, ge=0, le=5, description="Review score")


class ExplainResponse(BaseModel):
    """Response body for POST /explain."""

    shap_values: list[ShapFeature] = Field(
        ...,
        description="All SHAP feature contributions",
    )
    base_value: float = Field(..., description="Model base prediction value")
    predicted_demand: float = Field(..., description="Predicted booking probability")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    model_version: str = Field("", description="Current model version")


class MetricsResponse(BaseModel):
    """Response body for GET /metrics."""

    demand_model_auc: float = Field(..., description="Demand forecaster AUC")
    demand_model_f1: float = Field(..., description="Demand forecaster F1")
    elasticity_coeff: float = Field(..., description="Estimated price elasticity")
    elasticity_r2: float = Field(..., description="Elasticity model R-squared")
    total_predictions: int = Field(..., description="Total predictions served")


class DriftResponse(BaseModel):
    """Response body for GET /drift-report."""

    status: str = Field(
        ...,
        description="'healthy', 'degraded', or 'critical'",
    )
    drift_score: float = Field(..., description="Overall drift score (0-1)")
    features_drifted: list[str] = Field(
        ...,
        description="List of features that have drifted",
    )
    report_url: str = Field(..., description="URL to the Evidently HTML report")
