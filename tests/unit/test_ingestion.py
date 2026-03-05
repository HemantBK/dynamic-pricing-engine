"""Unit tests for the data ingestion module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ingestion import (
    EXPECTED_CALENDAR_COLUMNS,
    EXPECTED_LISTINGS_COLUMNS,
    EXPECTED_REVIEWS_COLUMNS,
    download_file,
    load_csv,
    validate_columns,
)


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def test_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert validate_columns(df, ["a", "b"], "test") is True

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        assert validate_columns(df, ["a", "b", "c"], "test") is False

    def test_empty_expected(self):
        df = pd.DataFrame({"a": [1]})
        assert validate_columns(df, [], "test") is True

    def test_exact_match(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        assert validate_columns(df, ["x", "y"], "test") is True

    def test_listings_columns_constant(self):
        """Verify expected listings columns list is non-empty."""
        assert len(EXPECTED_LISTINGS_COLUMNS) > 5

    def test_calendar_columns_constant(self):
        assert "listing_id" in EXPECTED_CALENDAR_COLUMNS
        assert "date" in EXPECTED_CALENDAR_COLUMNS

    def test_reviews_columns_constant(self):
        assert "listing_id" in EXPECTED_REVIEWS_COLUMNS


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_loads_uncompressed_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(csv_path, index=False)
        result = load_csv(csv_path, compressed=False)
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_csv(missing, compressed=False)

    def test_loads_gzip_csv(self, tmp_path):
        csv_gz_path = tmp_path / "test.csv.gz"
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        df.to_csv(csv_gz_path, index=False, compression="gzip")
        result = load_csv(csv_gz_path, compressed=True)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# download_file (mocked network)
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_skips_existing_file(self, tmp_path):
        existing = tmp_path / "existing.csv"
        existing.write_text("a,b\n1,2\n")
        result = download_file("http://example.com/file.csv", existing, force=False)
        assert result == existing

    @patch("src.data.ingestion.requests.get")
    def test_downloads_new_file(self, mock_get, tmp_path):
        output = tmp_path / "subdir" / "new_file.csv"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"col1,col2\n1,2\n"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_file("http://example.com/data.csv", output)
        assert result == output
        assert output.exists()

    @patch("src.data.ingestion.requests.get")
    def test_force_redownloads(self, mock_get, tmp_path):
        existing = tmp_path / "file.csv"
        existing.write_text("old data")

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "50"}
        mock_response.iter_content.return_value = [b"new data"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        download_file("http://example.com/file.csv", existing, force=True)
        assert existing.read_text() == "new data"


# ---------------------------------------------------------------------------
# fetch_weather_data (mocked network)
# ---------------------------------------------------------------------------

class TestFetchWeatherData:
    @patch("src.data.ingestion.requests.get")
    def test_fetches_and_parses_weather(self, mock_get, tmp_path):
        from src.data.ingestion import fetch_weather_data

        api_response = {
            "daily": {
                "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "temperature_2m_mean": [5.0, 6.0, 7.0],
                "precipitation_sum": [0.0, 2.5, 0.0],
                "wind_speed_10m_max": [10.0, 15.0, 8.0],
            }
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {
            "weather": {
                "api_url": "https://api.open-meteo.com/v1/forecast",
                "latitude": 40.7,
                "longitude": -74.0,
            },
            "data": {"external_dir": str(tmp_path)},
        }

        with patch("src.data.ingestion.PROJECT_ROOT", tmp_path):
            df = fetch_weather_data(config=config)

        assert len(df) == 3
        assert "temperature_mean" in df.columns
        assert "precipitation_sum" in df.columns
        assert "wind_speed_max" in df.columns


# ---------------------------------------------------------------------------
# fetch_holiday_data (mocked network)
# ---------------------------------------------------------------------------

class TestFetchHolidayData:
    @patch("src.data.ingestion.requests.get")
    def test_fetches_and_parses_holidays(self, mock_get, tmp_path):
        from src.data.ingestion import fetch_holiday_data

        api_response = [
            {"date": "2024-01-01", "localName": "New Year's Day", "name": "New Year's Day"},
            {"date": "2024-07-04", "localName": "Independence Day", "name": "Independence Day"},
            {"date": "2024-12-25", "localName": "Christmas Day", "name": "Christmas Day"},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        config = {
            "holidays": {
                "api_url": "https://date.nager.at/api/v3/PublicHolidays",
                "country_code": "US",
            },
            "data": {"external_dir": str(tmp_path)},
        }

        with patch("src.data.ingestion.PROJECT_ROOT", tmp_path):
            df = fetch_holiday_data(year=2024, config=config)

        assert len(df) == 3
        assert "holiday_local_name" in df.columns
        assert "holiday_name" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------------------
# download_airbnb_data (mocked)
# ---------------------------------------------------------------------------

class TestDownloadAirbnbData:
    @patch("src.data.ingestion.download_file")
    def test_downloads_three_datasets(self, mock_download, tmp_path):
        from src.data.ingestion import download_airbnb_data

        mock_download.side_effect = lambda url, path, force=False: path

        config = {
            "data": {
                "raw_dir": str(tmp_path),
                "insideairbnb": {
                    "listings_url": "http://example.com/listings.csv.gz",
                    "calendar_url": "http://example.com/calendar.csv.gz",
                    "reviews_url": "http://example.com/reviews.csv.gz",
                },
            }
        }

        with patch("src.data.ingestion.PROJECT_ROOT", tmp_path):
            files = download_airbnb_data(config=config)

        assert "listings" in files
        assert "calendar" in files
        assert "reviews" in files
        assert mock_download.call_count == 3
