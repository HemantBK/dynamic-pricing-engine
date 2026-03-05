"""Unit tests for the demand forecaster."""

import numpy as np
import pandas as pd
import pytest

from src.models.demand_forecaster import DemandForecaster


@pytest.fixture
def trained_forecaster():
    """Create a demand forecaster trained on synthetic data."""
    np.random.seed(42)
    n = 1000
    calendar_df = pd.DataFrame({
        "listing_id": np.random.randint(1, 10, n),
        "date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "was_booked": np.random.randint(0, 2, n),
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
        "is_weekend": np.random.randint(0, 2, n),
        "season": np.random.randint(0, 4, n),
        "rolling_7d_occupancy": np.random.uniform(0, 1, n),
        "rolling_30d_occupancy": np.random.uniform(0, 1, n),
        "day_of_month": np.random.randint(1, 29, n),
        "week_of_year": np.random.randint(1, 53, n),
        "quarter": np.random.randint(1, 5, n),
        "days_from_start": np.arange(n),
    })

    forecaster = DemandForecaster()
    forecaster.train(calendar_df)
    return forecaster


class TestDemandForecaster:
    def test_loads_without_error(self):
        forecaster = DemandForecaster()
        assert forecaster.model is None

    def test_predict_returns_probabilities(self, trained_forecaster):
        X = pd.DataFrame({
            "day_of_week": [5],
            "month": [7],
            "is_weekend": [1],
            "season": [2],
            "rolling_7d_occupancy": [0.6],
            "rolling_30d_occupancy": [0.5],
            "day_of_month": [15],
            "week_of_year": [28],
            "quarter": [3],
            "days_from_start": [100],
        })
        pred = trained_forecaster.predict(X)
        assert len(pred) == 1
        assert 0 <= pred[0] <= 1

    def test_predict_single(self, trained_forecaster):
        features = {
            "day_of_week": 3,
            "month": 6,
            "is_weekend": 0,
            "season": 2,
            "rolling_7d_occupancy": 0.5,
            "rolling_30d_occupancy": 0.45,
            "day_of_month": 10,
            "week_of_year": 24,
            "quarter": 2,
            "days_from_start": 50,
        }
        pred = trained_forecaster.predict_single(features)
        assert isinstance(pred, float)
        assert 0 <= pred <= 1

    def test_feature_importance(self, trained_forecaster):
        importance = trained_forecaster.get_feature_importance()
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) > 0
        assert (importance["importance"] >= 0).all()

    def test_handles_extra_features(self, trained_forecaster):
        # Provide all expected features plus extras — should still work
        X = pd.DataFrame({
            "day_of_week": [3],
            "month": [6],
            "is_weekend": [0],
            "season": [2],
            "rolling_7d_occupancy": [0.5],
            "rolling_30d_occupancy": [0.45],
            "day_of_month": [10],
            "week_of_year": [24],
            "quarter": [2],
            "days_from_start": [50],
            "extra_col": [999],
        })
        pred = trained_forecaster.predict(X)
        assert len(pred) == 1

    def test_raises_without_training(self):
        forecaster = DemandForecaster()
        with pytest.raises(RuntimeError, match="not trained"):
            forecaster.predict(pd.DataFrame({"day_of_week": [1]}))
