"""Unit tests for the elasticity estimator."""

import numpy as np
import pandas as pd
import pytest

from src.models.elasticity_estimator import ElasticityEstimator


@pytest.fixture
def trained_estimator():
    """Create an elasticity estimator trained on synthetic data."""
    np.random.seed(42)
    n = 2000

    # Create synthetic data with known negative elasticity
    prices = np.random.uniform(50, 300, n)
    # Demand decreases with price (negative elasticity)
    noise = np.random.normal(0, 0.1, n)
    demand = 0.9 * (prices / 150) ** (-1.2) + noise
    demand = np.clip(demand, 0.01, 1.0)

    calendar_df = pd.DataFrame({
        "listing_id": np.random.randint(1, 20, n),
        "date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "price": prices,
        "was_booked": (np.random.random(n) < demand).astype(int),
        "rolling_7d_occupancy": demand,
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
        "is_weekend": np.random.randint(0, 2, n),
        "season": np.random.randint(0, 4, n),
        "is_holiday": np.random.randint(0, 2, n),
        "near_holiday": np.random.randint(0, 2, n),
    })

    estimator = ElasticityEstimator()
    estimator.train(calendar_df)
    return estimator


class TestElasticityEstimator:
    def test_elasticity_is_negative(self, trained_estimator):
        """Economic law: higher price -> lower demand."""
        assert trained_estimator.elasticity_coeff < 0

    def test_elasticity_bounded(self, trained_estimator):
        """Elasticity should be within reasonable range."""
        assert -5 < trained_estimator.elasticity_coeff < 0

    def test_get_elasticity_returns_float(self, trained_estimator):
        e = trained_estimator.get_elasticity()
        assert isinstance(e, float)

    def test_coefficients_dataframe(self, trained_estimator):
        coefs = trained_estimator.get_coefficients()
        assert "feature" in coefs.columns
        assert "coefficient" in coefs.columns
        assert "log_price" in coefs["feature"].values

    def test_raises_without_training(self):
        estimator = ElasticityEstimator()
        with pytest.raises(RuntimeError, match="not trained"):
            estimator.get_elasticity()
