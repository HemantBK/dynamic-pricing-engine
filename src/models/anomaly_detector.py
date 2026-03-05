"""Anomaly Detection for demand patterns.

Uses Isolation Forest from scikit-learn to flag unusual demand
spikes or dips that may indicate data quality issues, special
events, or model drift.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ANOMALY_FEATURES = [
    "day_of_week", "month", "is_weekend",
    "rolling_7d_occupancy", "rolling_30d_occupancy",
    "temperature_mean", "precipitation_sum",
    "is_holiday", "near_holiday",
]


class AnomalyDetector:
    """Isolation Forest wrapper for demand anomaly detection."""

    def __init__(self, contamination: float = 0.05, config: dict | None = None):
        """Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.01 to 0.5).
            config: Configuration dict.
        """
        if config is None:
            config = load_config()
        self.config = config
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=config["model"]["random_state"],
            n_estimators=200,
            n_jobs=-1,
        )
        self.feature_names = None
        self.is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select available anomaly features from DataFrame."""
        available = [f for f in ANOMALY_FEATURES if f in df.columns]
        if not available:
            raise ValueError(f"No anomaly features found. Expected some of: {ANOMALY_FEATURES}")
        self.feature_names = available
        return df[available].fillna(0)

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the anomaly detector on training data.

        Args:
            df: DataFrame with demand features.

        Returns:
            Self for chaining.
        """
        X = self._get_features(df)
        self.model.fit(X)
        self.is_fitted = True

        # Count anomalies in training data
        labels = self.model.predict(X)
        n_anomalies = (labels == -1).sum()
        logger.info(
            f"Anomaly detector fitted on {len(X):,} samples. "
            f"Found {n_anomalies:,} anomalies ({n_anomalies/len(X)*100:.1f}%)"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict whether observations are anomalous.

        Args:
            df: DataFrame with demand features.

        Returns:
            Boolean array where True = anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        features = [f for f in self.feature_names if f in df.columns]
        X = df[features].fillna(0)
        labels = self.model.predict(X)
        return labels == -1  # True for anomalies

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous).

        Args:
            df: DataFrame with demand features.

        Returns:
            Array of anomaly scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        features = [f for f in self.feature_names if f in df.columns]
        X = df[features].fillna(0)
        return self.model.score_samples(X)

    def save(self, path: Path | None = None):
        """Save detector to disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "anomaly_detector"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        logger.info(f"Anomaly detector saved to {path}")

    def load(self, path: Path | None = None):
        """Load detector from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "anomaly_detector"
        self.model = joblib.load(path / "model.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.is_fitted = True
        logger.info(f"Anomaly detector loaded from {path}")
