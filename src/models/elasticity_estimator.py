"""Price Elasticity Estimator.

Estimates price elasticity of demand using log-log regression:
    log(demand) = beta * log(price) + controls

The elasticity coefficient beta represents the % change in demand
for a 1% change in price. Economic theory expects beta < 0
(higher price -> lower demand).

Uses Ridge Regression with regularization for stable estimates.
"""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Control variables for the elasticity model
CONTROL_FEATURES = [
    "day_of_week", "month", "is_weekend", "season",
    "is_holiday", "near_holiday",
    "room_type_encoded", "beds", "bathrooms",
    "location_cluster", "amenity_score",
    "temperature_mean", "is_rainy_day",
]


class ElasticityEstimator:
    """Log-log price elasticity model with Ridge Regression."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.elasticity_coeff = None
        self.metrics = {}

    def _prepare_data(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare log-transformed features and target.

        X includes log(price) and control variables.
        y is log(demand_proxy) where demand_proxy = was_booked smoothed.
        """
        df = calendar_df.copy()

        # Merge listing features if available
        if listings_df is not None and "listing_id" in df.columns:
            listing_cols = ["id", "room_type_encoded", "beds", "bathrooms",
                           "amenity_score", "location_cluster"]
            merge_cols = [c for c in listing_cols if c in listings_df.columns]
            if merge_cols:
                df = df.merge(
                    listings_df[merge_cols],
                    left_on="listing_id",
                    right_on="id",
                    how="left",
                )

        # Need price and demand columns
        if "price" not in df.columns:
            raise ValueError("'price' column required for elasticity estimation")
        if "was_booked" not in df.columns:
            raise ValueError("'was_booked' column required for elasticity estimation")

        # Filter valid prices
        df = df[df["price"] > 0].copy()

        # Create demand proxy: rolling average booking rate per listing
        # (smoother than raw binary was_booked)
        if "rolling_7d_occupancy" in df.columns:
            demand_col = "rolling_7d_occupancy"
        else:
            df["demand_smooth"] = (
                df.groupby("listing_id")["was_booked"]
                .transform(lambda x: x.rolling(7, min_periods=1).mean())
            )
            demand_col = "demand_smooth"

        # Remove zero demand (can't take log)
        df = df[df[demand_col] > 0].copy()

        # Log transforms
        df["log_price"] = np.log(df["price"])
        df["log_demand"] = np.log(df[demand_col])

        # Build feature matrix: log_price + controls
        available_controls = [c for c in CONTROL_FEATURES if c in df.columns]
        self.feature_names = ["log_price"] + available_controls
        X = df[self.feature_names].fillna(0)
        y = df["log_demand"]

        logger.info(
            f"Elasticity data: {len(X):,} samples, "
            f"{len(self.feature_names)} features (1 price + {len(available_controls)} controls)"
        )
        return X, y

    def train(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
        alpha: float = 1.0,
    ) -> dict:
        """Train the elasticity model.

        Args:
            calendar_df: Calendar data with price and was_booked columns.
            listings_df: Optional listings data for control variables.
            alpha: Ridge regularization strength.

        Returns:
            Dict of evaluation metrics including elasticity coefficient.
        """
        X, y = self._prepare_data(calendar_df, listings_df)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names,
            index=X.index,
        )

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=self.config["model"]["cv_splits"])

        # MLflow tracking
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="elasticity_estimator"):
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("n_samples", len(X))

            self.model = Ridge(alpha=alpha)

            # Evaluate across folds
            fold_metrics = []
            for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train = X_scaled.iloc[train_idx]
                X_val = X_scaled.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)

                fold_metrics.append({
                    "r2": r2_score(y_val, y_pred),
                    "mae": mean_absolute_error(y_val, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                })

            # Final fit on all data
            self.model.fit(X_scaled, y)

            # Extract elasticity coefficient (coefficient of log_price)
            log_price_idx = self.feature_names.index("log_price")
            # Unscale the coefficient: beta_original = beta_scaled / std(log_price)
            self.elasticity_coeff = self.model.coef_[log_price_idx] / self.scaler.scale_[log_price_idx]

            self.metrics = {
                "elasticity_coeff": float(self.elasticity_coeff),
                "r2": np.mean([m["r2"] for m in fold_metrics]),
                "mae": np.mean([m["mae"] for m in fold_metrics]),
                "rmse": np.mean([m["rmse"] for m in fold_metrics]),
            }

            logger.info(f"Elasticity coefficient: {self.elasticity_coeff:.4f}")
            logger.info(f"Mean R2: {self.metrics['r2']:.4f}")

            mlflow.log_metrics(self.metrics)
            mlflow.sklearn.log_model(self.model, "elasticity_model")

        return self.metrics

    def get_elasticity(
        self,
        neighborhood: str | None = None,
        room_type: str | None = None,
    ) -> float:
        """Return the estimated elasticity coefficient.

        In a more advanced version, this could return segment-specific
        elasticity. Currently returns the global estimate.

        Returns:
            Elasticity coefficient (typically between -3 and 0).
        """
        if self.elasticity_coeff is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.elasticity_coeff

    def get_coefficients(self) -> pd.DataFrame:
        """Return all model coefficients as a DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        # Unscale coefficients
        coefs_unscaled = self.model.coef_ / self.scaler.scale_

        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs_unscaled,
            "abs_coefficient": np.abs(coefs_unscaled),
        }).sort_values("abs_coefficient", ascending=False)

    def save(self, path: Path | None = None):
        """Save model, scaler, and metadata to disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "elasticity_estimator"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        joblib.dump(self.elasticity_coeff, path / "elasticity_coeff.joblib")
        joblib.dump(self.metrics, path / "metrics.joblib")
        logger.info(f"Elasticity estimator saved to {path}")

    def load(self, path: Path | None = None):
        """Load model, scaler, and metadata from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "elasticity_estimator"
        self.model = joblib.load(path / "model.joblib")
        self.scaler = joblib.load(path / "scaler.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.elasticity_coeff = joblib.load(path / "elasticity_coeff.joblib")
        self.metrics = joblib.load(path / "metrics.joblib")
        logger.info(f"Elasticity estimator loaded from {path}")
