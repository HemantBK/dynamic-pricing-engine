"""Unit tests for the API dependencies module."""

import pytest

from src.api.dependencies import ROOM_TYPE_MAP, ModelState, build_features_from_request


class TestBuildFeaturesFromRequest:
    def test_returns_dict_with_all_features(self):
        features = build_features_from_request(
            room_type="Entire home/apt",
            beds=2,
            bathrooms=1.0,
            neighborhood="Harlem",
            checkin_date="2024-07-15",
            amenity_score=0.7,
            review_score=4.5,
        )
        assert isinstance(features, dict)
        # Check key features exist
        assert "day_of_week" in features
        assert "month" in features
        assert "is_weekend" in features
        assert "season" in features
        assert "room_type_encoded" in features
        assert "beds" in features
        assert "bathrooms" in features
        assert "rolling_7d_occupancy" in features

    def test_correct_room_type_encoding(self):
        features = build_features_from_request(
            room_type="Private room",
            beds=1,
            bathrooms=1.0,
            neighborhood="Chelsea",
            checkin_date="2024-03-10",
            amenity_score=0.5,
            review_score=4.0,
        )
        assert features["room_type_encoded"] == ROOM_TYPE_MAP["Private room"]

    def test_weekend_detection(self):
        # 2024-07-13 is Saturday
        features = build_features_from_request(
            room_type="Entire home/apt",
            beds=2,
            bathrooms=1.5,
            neighborhood="SoHo",
            checkin_date="2024-07-13",
            amenity_score=0.8,
            review_score=4.8,
        )
        assert features["is_weekend"] == 1

    def test_weekday_detection(self):
        # 2024-07-10 is Wednesday
        features = build_features_from_request(
            room_type="Entire home/apt",
            beds=2,
            bathrooms=1.5,
            neighborhood="SoHo",
            checkin_date="2024-07-10",
            amenity_score=0.8,
            review_score=4.8,
        )
        assert features["is_weekend"] == 0

    def test_summer_season(self):
        features = build_features_from_request(
            room_type="Entire home/apt",
            beds=1,
            bathrooms=1.0,
            neighborhood="Williamsburg",
            checkin_date="2024-07-15",
            amenity_score=0.5,
            review_score=4.0,
        )
        assert features["season"] == 2  # Summer

    def test_winter_season(self):
        features = build_features_from_request(
            room_type="Shared room",
            beds=1,
            bathrooms=1.0,
            neighborhood="Astoria",
            checkin_date="2024-01-15",
            amenity_score=0.3,
            review_score=3.5,
        )
        assert features["season"] == 0  # Winter

    def test_unknown_room_type_defaults_to_zero(self):
        features = build_features_from_request(
            room_type="Unknown type",
            beds=1,
            bathrooms=1.0,
            neighborhood="Harlem",
            checkin_date="2024-05-01",
            amenity_score=0.5,
            review_score=4.0,
        )
        assert features["room_type_encoded"] == 0

    def test_preserves_numeric_inputs(self):
        features = build_features_from_request(
            room_type="Entire home/apt",
            beds=3,
            bathrooms=2.5,
            neighborhood="Chelsea",
            checkin_date="2024-06-01",
            amenity_score=0.9,
            review_score=4.7,
        )
        assert features["beds"] == 3
        assert features["bathrooms"] == 2.5
        assert features["amenity_score"] == 0.9
        assert features["review_score"] == 4.7


class TestModelState:
    def test_default_state(self):
        state = ModelState()
        assert state.demand_forecaster is None
        assert state.elasticity_estimator is None
        assert state.anomaly_detector is None
        assert state.is_loaded is False
        assert state.total_predictions == 0
        assert state.model_version == "v0.1.0"

    def test_room_type_map_complete(self):
        assert "Entire home/apt" in ROOM_TYPE_MAP
        assert "Private room" in ROOM_TYPE_MAP
        assert "Shared room" in ROOM_TYPE_MAP
        assert "Hotel room" in ROOM_TYPE_MAP
