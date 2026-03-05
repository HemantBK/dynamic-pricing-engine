"""Integration tests for the FastAPI pricing API."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "room_type": "Entire home/apt",
    "beds": 2,
    "bathrooms": 1.0,
    "neighborhood": "Williamsburg",
    "checkin_date": "2024-07-15",
    "checkout_date": "2024-07-18",
    "amenity_score": 0.7,
    "review_score": 4.5,
}


class TestPredictEndpoint:
    def test_valid_prediction(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "optimal_price" in data
        assert "price_range" in data
        assert "demand_forecast" in data
        assert data["optimal_price"] > 0
        assert 0 <= data["demand_forecast"] <= 1

    def test_missing_fields_returns_422(self, client):
        response = client.post("/predict", json={"room_type": "Private room"})
        assert response.status_code == 422

    def test_invalid_beds_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "beds": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_review_score_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "review_score": 10.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_response_contains_shap_features(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "shap_top_features" in data
        assert isinstance(data["shap_top_features"], list)

    def test_response_has_model_version(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "model_version" in data
        assert len(data["model_version"]) > 0

    def test_different_room_types(self, client):
        for room_type in ["Entire home/apt", "Private room", "Shared room"]:
            payload = {**VALID_PAYLOAD, "room_type": room_type}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "demand_model_auc" in data
        assert "elasticity_coeff" in data
        assert "total_predictions" in data


class TestExplainEndpoint:
    def test_explain_returns_shap_values(self, client):
        payload = {
            "room_type": "Private room",
            "beds": 1,
            "bathrooms": 1.0,
            "neighborhood": "Harlem",
            "checkin_date": "2024-08-01",
            "amenity_score": 0.5,
            "review_score": 4.0,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "shap_values" in data
        assert "predicted_demand" in data


class TestDriftEndpoint:
    def test_drift_report_returns_200(self, client):
        response = client.get("/drift-report")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "drift_score" in data
