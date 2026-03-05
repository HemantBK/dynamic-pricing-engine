"""Unit tests for the drift detector / monitoring module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring.drift_detector import run_drift_check


class TestRunDriftCheck:
    def test_returns_healthy_when_evidently_not_installed(self):
        """If Evidently is missing, should return a healthy fallback."""
        config = {
            "data": {"processed_dir": "data/processed"},
            "monitoring": {
                "report_output_dir": "monitoring/reports",
                "drift_threshold_degraded": 0.3,
                "drift_threshold_critical": 0.6,
            },
        }
        with patch.dict("sys.modules", {"evidently": None, "evidently.metric_preset": None, "evidently.report": None}):
            # Force the import to fail by patching builtins __import__
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if "evidently" in name:
                    raise ImportError("No module named 'evidently'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Re-import to trigger the fallback
                import importlib
                import src.monitoring.drift_detector as dd
                importlib.reload(dd)
                result = dd.run_drift_check(config=config)

            # Reload back to normal
            importlib.reload(dd)

        assert result["status"] == "healthy"
        assert result["drift_score"] == 0.0
        assert result["features_drifted"] == []

    def test_returns_healthy_when_reference_data_missing(self, tmp_path):
        """If processed data doesn't exist, should return healthy fallback."""
        config = {
            "data": {"processed_dir": str(tmp_path / "nonexistent")},
            "monitoring": {
                "report_output_dir": str(tmp_path / "reports"),
                "drift_threshold_degraded": 0.3,
                "drift_threshold_critical": 0.6,
            },
        }
        with patch("src.monitoring.drift_detector.PROJECT_ROOT", tmp_path):
            result = run_drift_check(config=config)

        assert result["status"] == "healthy"
        assert result["drift_score"] == 0.0

    def test_returns_healthy_when_no_drift_features(self, tmp_path):
        """If reference data has no numeric drift features, return healthy."""
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        ref_path = processed_dir / "listings_features.parquet"

        # Create parquet with only non-drift columns
        df = pd.DataFrame({
            "name": ["listing1", "listing2"],
            "description": ["desc1", "desc2"],
        })
        df.to_parquet(ref_path, index=False)

        config = {
            "data": {"processed_dir": "data/processed"},
            "monitoring": {
                "report_output_dir": "monitoring/reports",
                "drift_threshold_degraded": 0.3,
                "drift_threshold_critical": 0.6,
            },
        }
        with patch("src.monitoring.drift_detector.PROJECT_ROOT", tmp_path):
            result = run_drift_check(config=config)

        assert result["status"] == "healthy"
        assert result["drift_score"] == 0.0
        assert result["features_drifted"] == []

    def test_default_config_loaded_when_none(self):
        """When config is None, it should load from config.yaml."""
        with patch("src.monitoring.drift_detector.load_config") as mock_load:
            mock_load.return_value = {
                "data": {"processed_dir": "fake/nonexistent"},
                "monitoring": {
                    "report_output_dir": "monitoring/reports",
                    "drift_threshold_degraded": 0.3,
                    "drift_threshold_critical": 0.6,
                },
            }
            with patch("src.monitoring.drift_detector.PROJECT_ROOT", __import__("pathlib").Path("/tmp/fake")):
                result = run_drift_check(config=None)
            mock_load.assert_called_once()
        assert result["status"] == "healthy"
