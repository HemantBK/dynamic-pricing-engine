"""Unit tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ExplainRequest,
    HealthResponse,
    PricingRequest,
    PricingResponse,
    ShapFeature,
)


class TestPricingRequest:
    def test_valid_request(self):
        req = PricingRequest(
            room_type="Entire home/apt",
            beds=2,
            bathrooms=1.0,
            neighborhood="Williamsburg",
            checkin_date="2024-07-15",
            checkout_date="2024-07-18",
        )
        assert req.beds == 2
        assert req.amenity_score == 0.5  # default

    def test_rejects_zero_beds(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=0,
                bathrooms=1.0,
                neighborhood="SoHo",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
            )

    def test_rejects_negative_bathrooms(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Shared room",
                beds=1,
                bathrooms=-1,
                neighborhood="Harlem",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
            )

    def test_rejects_review_score_out_of_range(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                review_score=6.0,
            )

    def test_rejects_amenity_score_out_of_range(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                amenity_score=1.5,
            )

    def test_serializes_correctly(self):
        req = PricingRequest(
            room_type="Private room",
            beds=1,
            bathrooms=1.0,
            neighborhood="Astoria",
            checkin_date="2024-08-01",
            checkout_date="2024-08-03",
            amenity_score=0.8,
            review_score=4.5,
        )
        data = req.model_dump()
        assert data["beds"] == 1
        assert data["amenity_score"] == 0.8


class TestPricingResponse:
    def test_valid_response(self):
        resp = PricingResponse(
            optimal_price=185.50,
            price_range=[160.0, 210.0],
            expected_revenue=556.50,
            demand_forecast=0.72,
            elasticity_coeff=-1.3,
            market_avg_price=165.0,
            shap_top_features=[
                ShapFeature(feature="is_weekend", contribution=0.15),
            ],
            is_anomaly=False,
            model_version="v0.1.0",
        )
        assert resp.optimal_price == 185.50

    def test_rejects_invalid_demand(self):
        with pytest.raises(ValidationError):
            PricingResponse(
                optimal_price=100,
                price_range=[90, 110],
                expected_revenue=300,
                demand_forecast=1.5,  # > 1
                elasticity_coeff=-1.0,
                market_avg_price=100,
                shap_top_features=[],
                is_anomaly=False,
                model_version="v1",
            )


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(status="healthy", models_loaded=True, model_version="v1")
        assert resp.status == "healthy"
