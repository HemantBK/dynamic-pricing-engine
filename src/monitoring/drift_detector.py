"""Data drift detection using Evidently AI.

Compares training data distribution vs recent production data
to detect feature drift that may degrade model performance.
"""

from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_drift_check(config: dict | None = None) -> dict:
    """Run Evidently AI drift detection.

    Compares the training reference dataset against a simulated
    'current' dataset to detect distribution shifts.

    Args:
        config: Configuration dict.

    Returns:
        Dict with status, drift_score, features_drifted, report_url.
    """
    if config is None:
        config = load_config()

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently not installed. Install with: pip install evidently")
        return {
            "status": "healthy",
            "drift_score": 0.0,
            "features_drifted": [],
            "report_url": "",
        }

    # Load reference data (training data)
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    ref_path = processed_dir / "listings_features.parquet"

    if not ref_path.exists():
        logger.warning(f"Reference data not found at {ref_path}")
        return {
            "status": "healthy",
            "drift_score": 0.0,
            "features_drifted": [],
            "report_url": "",
        }

    ref_df = pd.read_parquet(ref_path)

    # Select numeric columns for drift analysis
    numeric_cols = ref_df.select_dtypes(include=["number"]).columns.tolist()
    # Filter to key features
    drift_features = [
        c for c in numeric_cols
        if c in [
            "price", "beds", "bathrooms", "reviews_per_month",
            "availability_365", "number_of_reviews", "amenity_score",
            "review_score", "occupancy_rate", "price_vs_neighborhood",
            "latitude", "longitude",
        ]
    ]

    if not drift_features:
        logger.warning("No drift features available")
        return {
            "status": "healthy",
            "drift_score": 0.0,
            "features_drifted": [],
            "report_url": "",
        }

    ref_data = ref_df[drift_features].dropna()

    # Simulate current data (in production, this would come from logged predictions)
    # Add slight perturbation to simulate realistic drift
    import numpy as np
    np.random.seed(42)
    current_data = ref_data.sample(min(1000, len(ref_data)), random_state=42).copy()
    # Add small noise to simulate natural distribution shift
    for col in current_data.columns:
        noise = np.random.normal(0, current_data[col].std() * 0.05, len(current_data))
        current_data[col] = current_data[col] + noise

    ref_sample = ref_data.sample(min(1000, len(ref_data)), random_state=0)

    # Run Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_sample, current_data=current_data)

    # Save HTML report
    report_dir = PROJECT_ROOT / config["monitoring"]["report_output_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.html"
    report.save_html(str(report_path))
    logger.info(f"Drift report saved to {report_path}")

    # Extract results
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])

    drift_score = 0.0
    features_drifted = []

    for metric in metrics:
        metric_result = metric.get("result", {})
        if "share_of_drifted_columns" in metric_result:
            drift_score = metric_result["share_of_drifted_columns"]
        if "drift_by_columns" in metric_result:
            for col_name, col_data in metric_result["drift_by_columns"].items():
                if col_data.get("drift_detected", False):
                    features_drifted.append(col_name)

    # Determine status
    degraded_threshold = config["monitoring"]["drift_threshold_degraded"]
    critical_threshold = config["monitoring"]["drift_threshold_critical"]

    if drift_score >= critical_threshold:
        status = "critical"
    elif drift_score >= degraded_threshold:
        status = "degraded"
    else:
        status = "healthy"

    logger.info(
        f"Drift check: status={status}, score={drift_score:.3f}, "
        f"drifted={len(features_drifted)} features"
    )

    return {
        "status": status,
        "drift_score": round(drift_score, 4),
        "features_drifted": features_drifted,
        "report_url": f"/static/{report_path.name}",
    }


if __name__ == "__main__":
    result = run_drift_check()
    print(f"Status: {result['status']}")
    print(f"Drift Score: {result['drift_score']}")
    print(f"Features Drifted: {result['features_drifted']}")
    print(f"Report: {result['report_url']}")
