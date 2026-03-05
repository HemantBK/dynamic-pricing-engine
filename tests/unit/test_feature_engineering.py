"""Unit tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_checkin_features,
    add_competitor_features,
    add_demand_features,
    add_holiday_features,
    add_listing_features,
    add_location_features,
    add_temporal_features,
    add_weather_features,
)


class TestTemporalFeatures:
    def test_adds_all_temporal_columns(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30)})
        result = add_temporal_features(df)
        expected = [
            "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "is_weekend",
            "is_month_start", "is_month_end",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_weekend_detection(self):
        # 2024-01-06 is Saturday, 2024-01-07 is Sunday
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"])
        })
        result = add_temporal_features(df)
        assert result.iloc[0]["is_weekend"] == 0  # Friday
        assert result.iloc[1]["is_weekend"] == 1  # Saturday
        assert result.iloc[2]["is_weekend"] == 1  # Sunday
        assert result.iloc[3]["is_weekend"] == 0  # Monday

    def test_handles_string_dates(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-06-15", "2024-12-31"]})
        result = add_temporal_features(df)
        assert result.iloc[0]["month"] == 1
        assert result.iloc[1]["month"] == 6
        assert result.iloc[2]["month"] == 12


class TestCheckinFeatures:
    def test_adds_season(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"])
        })
        result = add_checkin_features(df)
        assert "season" in result.columns
        assert result.iloc[0]["season"] == 0  # Winter
        assert result.iloc[1]["season"] == 1  # Spring
        assert result.iloc[2]["season"] == 2  # Summer
        assert result.iloc[3]["season"] == 3  # Fall

    def test_adds_days_from_start(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
        result = add_checkin_features(df)
        assert result.iloc[0]["days_from_start"] == 0
        assert result.iloc[9]["days_from_start"] == 9


class TestLocationFeatures:
    def test_creates_clusters(self, sample_listings_df):
        result = add_location_features(sample_listings_df, n_clusters=5)
        assert "location_cluster" in result.columns
        assert result["location_cluster"].nunique() <= 5

    def test_handles_few_points(self):
        df = pd.DataFrame({
            "latitude": [40.7, 40.8],
            "longitude": [-74.0, -73.9],
        })
        result = add_location_features(df, n_clusters=10)
        assert "location_cluster" in result.columns


class TestCompetitorFeatures:
    def test_adds_neighborhood_stats(self, sample_listings_df):
        result = add_competitor_features(sample_listings_df)
        assert "median_neighborhood_price" in result.columns
        assert "price_vs_neighborhood" in result.columns
        assert "price_rank_in_neighborhood" in result.columns

    def test_price_vs_neighborhood_positive(self, sample_listings_df):
        result = add_competitor_features(sample_listings_df)
        assert (result["price_vs_neighborhood"] > 0).all()

    def test_rank_between_0_and_1(self, sample_listings_df):
        result = add_competitor_features(sample_listings_df)
        assert result["price_rank_in_neighborhood"].min() >= 0
        assert result["price_rank_in_neighborhood"].max() <= 1


class TestListingFeatures:
    def test_encodes_room_type(self, sample_listings_df):
        result = add_listing_features(sample_listings_df)
        assert "room_type_encoded" in result.columns
        assert result["room_type_encoded"].dtype in [np.int32, np.int64, np.intp]

    def test_extracts_bathrooms(self, sample_listings_df):
        result = add_listing_features(sample_listings_df)
        assert "bathrooms" in result.columns
        assert result["bathrooms"].notna().all()

    def test_adds_amenity_score(self, sample_listings_df):
        result = add_listing_features(sample_listings_df)
        assert "amenity_score" in result.columns

    def test_adds_review_score(self, sample_listings_df):
        result = add_listing_features(sample_listings_df)
        assert "review_score" in result.columns

    def test_amenity_score_from_text(self):
        df = pd.DataFrame({
            "room_type": ["Private room"],
            "amenities": ['["wifi", "kitchen", "heating", "tv"]'],
            "beds": [2],
        })
        result = add_listing_features(df)
        assert result.iloc[0]["amenity_score"] > 0

    def test_default_amenity_score_without_column(self):
        df = pd.DataFrame({
            "room_type": ["Shared room"],
            "beds": [1],
        })
        result = add_listing_features(df)
        assert result.iloc[0]["amenity_score"] == 0.5

    def test_missing_room_type_column(self):
        df = pd.DataFrame({"beds": [1, 2], "price": [100, 200]})
        result = add_listing_features(df)
        assert "room_type_encoded" not in result.columns

    def test_review_scores_rating_used(self):
        df = pd.DataFrame({
            "room_type": ["Private room"],
            "review_scores_rating": [4.5],
        })
        result = add_listing_features(df)
        assert result.iloc[0]["review_score"] == pytest.approx(4.5)

    def test_review_scores_value_fallback(self):
        df = pd.DataFrame({
            "room_type": ["Private room"],
            "review_scores_value": [3.8],
        })
        result = add_listing_features(df)
        assert result.iloc[0]["review_score"] == pytest.approx(3.8)


# ---------------------------------------------------------------------------
# Weather Features
# ---------------------------------------------------------------------------

class TestWeatherFeatures:
    def test_merges_weather_data(self):
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({"date": dates, "value": range(10)})
        weather_df = pd.DataFrame({
            "date": dates,
            "temperature_mean": np.random.uniform(0, 30, 10),
            "precipitation_sum": np.random.uniform(0, 20, 10),
            "wind_speed_max": np.random.uniform(0, 50, 10),
        })
        result = add_weather_features(df, weather_df=weather_df)
        assert "temperature_mean" in result.columns
        assert "precipitation_sum" in result.columns
        assert "wind_speed_max" in result.columns
        assert len(result) == 10

    def test_adds_derived_weather_flags(self):
        dates = pd.date_range("2024-07-01", periods=5)
        df = pd.DataFrame({"date": dates, "value": range(5)})
        weather_df = pd.DataFrame({
            "date": dates,
            "temperature_mean": [35.0, 2.0, 20.0, 31.0, 4.0],
            "precipitation_sum": [0.0, 10.0, 0.0, 6.0, 0.0],
            "wind_speed_max": [5.0, 20.0, 10.0, 15.0, 8.0],
        })
        result = add_weather_features(df, weather_df=weather_df)
        assert "is_hot_day" in result.columns
        assert "is_cold_day" in result.columns
        assert "is_rainy_day" in result.columns
        # First row: temp 35 > 30, should be hot
        assert result.iloc[0]["is_hot_day"] == 1
        # Second row: temp 2 < 5, should be cold
        assert result.iloc[1]["is_cold_day"] == 1
        # Second row: precip 10 > 5, should be rainy
        assert result.iloc[1]["is_rainy_day"] == 1

    def test_handles_partial_weather_merge(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        # Only 5 days of weather
        weather_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "temperature_mean": [10.0] * 5,
            "precipitation_sum": [0.0] * 5,
            "wind_speed_max": [5.0] * 5,
        })
        result = add_weather_features(df, weather_df=weather_df)
        # Missing values should be filled with median
        assert result["temperature_mean"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Holiday Features
# ---------------------------------------------------------------------------

class TestHolidayFeatures:
    def test_adds_holiday_columns(self):
        dates = pd.date_range("2024-01-01", periods=30)
        df = pd.DataFrame({"date": dates, "value": range(30)})
        holiday_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "holiday_local_name": ["New Year", "MLK Day"],
            "holiday_name": ["New Year's Day", "Martin Luther King Jr. Day"],
        })
        result = add_holiday_features(df, holiday_df=holiday_df)
        assert "is_holiday" in result.columns
        assert "days_until_holiday" in result.columns
        assert "days_since_holiday" in result.columns
        assert "near_holiday" in result.columns

    def test_is_holiday_flag_correct(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"date": dates})
        holiday_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "holiday_local_name": ["New Year"],
            "holiday_name": ["New Year's Day"],
        })
        result = add_holiday_features(df, holiday_df=holiday_df)
        assert result.iloc[0]["is_holiday"] == 1
        assert result.iloc[1]["is_holiday"] == 0
        assert result.iloc[2]["is_holiday"] == 0

    def test_near_holiday_flag(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"])
        df = pd.DataFrame({"date": dates})
        holiday_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "holiday_local_name": ["New Year"],
            "holiday_name": ["New Year's Day"],
        })
        result = add_holiday_features(df, holiday_df=holiday_df)
        # Jan 1 is holiday and near_holiday
        assert result.iloc[0]["near_holiday"] == 1
        # Jan 2 is 1 day since holiday, so near_holiday
        assert result.iloc[1]["near_holiday"] == 1
        # Jan 10 is far from any holiday
        assert result.iloc[2]["near_holiday"] == 0

    def test_days_until_holiday(self):
        dates = pd.to_datetime(["2024-01-05"])
        df = pd.DataFrame({"date": dates})
        holiday_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "holiday_local_name": ["H1", "H2"],
            "holiday_name": ["Holiday 1", "Holiday 2"],
        })
        result = add_holiday_features(df, holiday_df=holiday_df)
        # 5 days until Jan 10
        assert result.iloc[0]["days_until_holiday"] == 5


# ---------------------------------------------------------------------------
# Demand Features
# ---------------------------------------------------------------------------

class TestDemandFeatures:
    def test_computes_occupancy_rate(self):
        listings_df = pd.DataFrame({
            "id": [1, 2],
            "price": [100, 200],
        })
        calendar_df = pd.DataFrame({
            "listing_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "date": pd.date_range("2024-01-01", periods=4).tolist() * 2,
            "was_booked": [1, 1, 0, 0, 1, 0, 0, 0],
        })
        config = {"features": {"rolling_window_days": 3}}

        listings_out, cal_out = add_demand_features(listings_df, calendar_df, config=config)
        assert "occupancy_rate" in listings_out.columns
        # Listing 1: 2/4 = 0.5 occupancy
        assert listings_out[listings_out["id"] == 1]["occupancy_rate"].iloc[0] == pytest.approx(0.5)

    def test_rolling_occupancy_columns(self):
        listings_df = pd.DataFrame({"id": [1], "price": [100]})
        calendar_df = pd.DataFrame({
            "listing_id": [1] * 30,
            "date": pd.date_range("2024-01-01", periods=30),
            "was_booked": [1, 0] * 15,
        })
        config = {"features": {"rolling_window_days": 7}}

        _, cal_out = add_demand_features(listings_df, calendar_df, config=config)
        assert "rolling_7d_occupancy" in cal_out.columns
        assert "rolling_30d_occupancy" in cal_out.columns

    def test_creates_was_booked_from_available(self):
        listings_df = pd.DataFrame({"id": [1], "price": [100]})
        calendar_df = pd.DataFrame({
            "listing_id": [1, 1, 1],
            "date": pd.date_range("2024-01-01", periods=3),
            "available": ["f", "t", "f"],
        })
        config = {"features": {"rolling_window_days": 7}}

        _, cal_out = add_demand_features(listings_df, calendar_df, config=config)
        assert "was_booked" in cal_out.columns


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_location_no_lat_lon(self):
        df = pd.DataFrame({"price": [100, 200]})
        result = add_location_features(df, n_clusters=3)
        assert "location_cluster" not in result.columns

    def test_competitor_missing_columns(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = add_competitor_features(df)
        assert "median_neighborhood_price" not in result.columns
