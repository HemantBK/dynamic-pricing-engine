"""Unit tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_price_column,
    clip_outliers,
    preprocess_calendar,
    preprocess_listings,
)


class TestCleanPriceColumn:
    def test_string_prices(self):
        s = pd.Series(["$1,234.56", "$99.00", "$10.50"])
        result = clean_price_column(s)
        assert result.iloc[0] == pytest.approx(1234.56)
        assert result.iloc[1] == pytest.approx(99.0)
        assert result.iloc[2] == pytest.approx(10.50)

    def test_numeric_prices(self):
        s = pd.Series([100.0, 200.0, 300.0])
        result = clean_price_column(s)
        assert result.iloc[0] == pytest.approx(100.0)

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        result = clean_price_column(s)
        assert len(result) == 0


class TestClipOutliers:
    def test_clips_extreme_values(self):
        df = pd.DataFrame({"price": list(range(1, 101))})
        result = clip_outliers(df, "price", 10, 90)
        assert result["price"].min() >= 10
        assert result["price"].max() <= 90

    def test_preserves_middle_values(self):
        df = pd.DataFrame({"price": list(range(1, 101))})
        result = clip_outliers(df, "price", 5, 95)
        # All values between 5th and 95th percentile should be preserved
        assert len(result) >= 85


class TestPreprocessListings:
    def test_removes_zero_price(self, sample_listings_df):
        df = sample_listings_df.copy()
        df.loc[0, "price"] = 0
        result = preprocess_listings(df)
        assert (result["price"] > 0).all()

    def test_fills_missing_reviews(self, sample_listings_df):
        df = sample_listings_df.copy()
        df.loc[0, "reviews_per_month"] = np.nan
        result = preprocess_listings(df)
        assert result["reviews_per_month"].isna().sum() == 0

    def test_drops_missing_critical_fields(self, sample_listings_df):
        df = sample_listings_df.copy()
        df.loc[0, "latitude"] = np.nan
        df.loc[1, "room_type"] = np.nan
        result = preprocess_listings(df)
        assert result["latitude"].isna().sum() == 0


class TestPreprocessCalendar:
    def test_parses_dates(self, sample_calendar_df):
        result = preprocess_calendar(sample_calendar_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_creates_was_booked(self, sample_calendar_df):
        result = preprocess_calendar(sample_calendar_df)
        assert "was_booked" in result.columns
        assert set(result["was_booked"].unique()).issubset({0, 1})

    def test_cleans_price(self, sample_calendar_df):
        result = preprocess_calendar(sample_calendar_df)
        assert pd.api.types.is_float_dtype(result["price"])
