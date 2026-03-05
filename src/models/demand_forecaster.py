"""Demand Forecaster using XGBoost.

Predicts occupancy rate (probability of booking) for a listing
on a given date, using temporal, weather, holiday, and listing features.

Uses TimeSeriesSplit for cross-validation to prevent data leakage.
All runs are tracked with MLflow.
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Features used for demand prediction
DEMAND_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "season", "days_from_start",
    "temperature_mean", "precipitation_sum", "wind_speed_max",
    "is_hot_day", "is_cold_day", "is_rainy_day",
    "is_holiday", "days_until_holiday", "near_holiday",
    "rolling_7d_occupancy", "rolling_30d_occupancy",
]

# Hyperparameter search space for RandomizedSearchCV
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}


class DemandForecaster:
    """XGBoost-based demand forecaster with MLflow tracking."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.model_cfg = config["model"]
        self.model = None
        self.feature_names = None
        self.metrics = {}

    def _get_available_features(self, df: pd.DataFrame) -> list[str]:
        """Return the intersection of expected features and available columns."""
        available = [f for f in DEMAND_FEATURES if f in df.columns]
        if not available:
            raise ValueError(
                f"No demand features found in DataFrame. "
                f"Expected some of: {DEMAND_FEATURES}. "
                f"Got columns: {df.columns.tolist()}"
            )
        return available

    def _prepare_data(
        self, calendar_df: pd.DataFrame, listings_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target y from calendar data."""
        df = calendar_df.copy()

        # Merge listing-level features if provided
        if listings_df is not None and "listing_id" in df.columns:
            listing_cols = ["id", "room_type_encoded", "beds", "bathrooms",
                           "amenity_score", "review_score", "location_cluster",
                           "occupancy_rate", "price_rank_in_neighborhood",
                           "price_vs_neighborhood"]
            merge_cols = [c for c in listing_cols if c in listings_df.columns]
            if merge_cols:
                df = df.merge(
                    listings_df[merge_cols],
                    left_on="listing_id",
                    right_on="id",
                    how="left",
                )

        # Target
        if "was_booked" not in df.columns:
            raise ValueError("Target column 'was_booked' not found in calendar data")
        y = df["was_booked"].astype(int)

        # Features
        self.feature_names = self._get_available_features(df)
        X = df[self.feature_names].copy()

        # Fill NaN with 0 for features
        X = X.fillna(0)

        logger.info(f"Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
        return X, y

    def train(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
        tune_hyperparams: bool = False,
        n_iter: int = 20,
    ) -> dict:
        """Train the demand forecasting model.

        Args:
            calendar_df: Feature-engineered calendar DataFrame with 'was_booked' target.
            listings_df: Optional listings DataFrame for listing-level features.
            tune_hyperparams: Whether to run RandomizedSearchCV.
            n_iter: Number of random search iterations.

        Returns:
            Dict of evaluation metrics.
        """
        X, y = self._prepare_data(calendar_df, listings_df)

        # TimeSeriesSplit — prevents future data leakage
        tscv = TimeSeriesSplit(n_splits=self.model_cfg["cv_splits"])

        # Set up MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="demand_forecaster"):
            if tune_hyperparams:
                logger.info(f"Running RandomizedSearchCV with {n_iter} iterations...")
                base_model = XGBClassifier(
                    random_state=self.model_cfg["random_state"],
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
                search = RandomizedSearchCV(
                    base_model,
                    PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    cv=tscv,
                    scoring="roc_auc",
                    random_state=self.model_cfg["random_state"],
                    n_jobs=-1,
                    verbose=1,
                )
                search.fit(X, y)
                self.model = search.best_estimator_
                best_params = search.best_params_
                logger.info(f"Best params: {best_params}")
                mlflow.log_params(best_params)
            else:
                xgb_cfg = self.model_cfg["xgboost"]
                params = {
                    "n_estimators": xgb_cfg["n_estimators"],
                    "max_depth": xgb_cfg["max_depth"],
                    "learning_rate": xgb_cfg["learning_rate"],
                    "subsample": xgb_cfg["subsample"],
                    "colsample_bytree": xgb_cfg["colsample_bytree"],
                    "random_state": self.model_cfg["random_state"],
                    "eval_metric": "logloss",
                    "use_label_encoder": False,
                    "early_stopping_rounds": xgb_cfg["early_stopping_rounds"],
                }
                mlflow.log_params(params)
                self.model = XGBClassifier(**params)

                # Train with early stopping using last fold as eval set
                splits = list(tscv.split(X))
                train_idx, val_idx = splits[-1]
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            # Evaluate across all CV folds
            fold_metrics = []
            for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                y_pred_proba = self.model.predict_proba(X_val_fold)[:, 1]
                y_pred = self.model.predict(X_val_fold)

                fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
                fold_acc = accuracy_score(y_val_fold, y_pred)
                fold_f1 = f1_score(y_val_fold, y_pred)
                fold_metrics.append({
                    "fold": fold_i,
                    "auc": fold_auc,
                    "accuracy": fold_acc,
                    "f1": fold_f1,
                })
                logger.info(f"Fold {fold_i}: AUC={fold_auc:.4f}, Acc={fold_acc:.4f}, F1={fold_f1:.4f}")

            # Average metrics
            self.metrics = {
                "auc": np.mean([m["auc"] for m in fold_metrics]),
                "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
                "f1": np.mean([m["f1"] for m in fold_metrics]),
                "auc_std": np.std([m["auc"] for m in fold_metrics]),
            }
            logger.info(f"Mean AUC: {self.metrics['auc']:.4f} +/- {self.metrics['auc_std']:.4f}")

            mlflow.log_metrics(self.metrics)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("features", json.dumps(self.feature_names))

            # Log model
            mlflow.sklearn.log_model(self.model, "demand_model")

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict booking probability for given features.

        Args:
            X: DataFrame with demand features.

        Returns:
            Array of booking probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        features = [f for f in self.feature_names if f in X.columns]
        X_input = X[features].fillna(0)
        return self.model.predict_proba(X_input)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict booking probability for a single observation."""
        df = pd.DataFrame([features])
        return float(self.predict(df)[0])

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance

    def save(self, path: Path | None = None):
        """Save model and metadata to disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        joblib.dump(self.metrics, path / "metrics.joblib")
        logger.info(f"Demand forecaster saved to {path}")

    def load(self, path: Path | None = None):
        """Load model and metadata from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"
        self.model = joblib.load(path / "model.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.metrics = joblib.load(path / "metrics.joblib")
        logger.info(f"Demand forecaster loaded from {path}")
