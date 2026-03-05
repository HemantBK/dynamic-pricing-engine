"""Unit tests for the anomaly detector."""

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import AnomalyDetector


@pytest.fixture
def sample_demand_df():
    """Create sample demand data for anomaly detection."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
        "is_weekend": np.random.randint(0, 2, n),
        "rolling_7d_occupancy": np.random.uniform(0.3, 0.8, n),
        "rolling_30d_occupancy": np.random.uniform(0.3, 0.8, n),
    })


class TestAnomalyDetector:
    def test_fit_and_predict(self, sample_demand_df):
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(sample_demand_df)
        anomalies = detector.predict(sample_demand_df)
        assert len(anomalies) == len(sample_demand_df)
        assert anomalies.dtype == bool

    def test_contamination_rate(self, sample_demand_df):
        contamination = 0.1
        detector = AnomalyDetector(contamination=contamination)
        detector.fit(sample_demand_df)
        anomalies = detector.predict(sample_demand_df)
        anomaly_rate = anomalies.mean()
        # Should be approximately equal to contamination rate
        assert abs(anomaly_rate - contamination) < 0.05

    def test_score_samples(self, sample_demand_df):
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(sample_demand_df)
        scores = detector.score_samples(sample_demand_df)
        assert len(scores) == len(sample_demand_df)
        # Anomaly scores should be finite
        assert np.isfinite(scores).all()

    def test_anomalies_have_lower_scores(self, sample_demand_df):
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(sample_demand_df)
        anomalies = detector.predict(sample_demand_df)
        scores = detector.score_samples(sample_demand_df)

        mean_anomaly_score = scores[anomalies].mean()
        mean_normal_score = scores[~anomalies].mean()
        assert mean_anomaly_score < mean_normal_score

    def test_raises_without_fit(self):
        detector = AnomalyDetector()
        df = pd.DataFrame({"day_of_week": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(df)
