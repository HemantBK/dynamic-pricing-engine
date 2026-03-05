"""Shared test fixtures for the Dynamic Pricing Engine."""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config()


@pytest.fixture
def sample_listings_df():
    """Create a sample listings DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"Listing {i}" for i in range(1, n + 1)],
        "host_id": np.random.randint(1000, 9999, n),
        "neighbourhood_group_cleansed": np.random.choice(
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n
        ),
        "neighbourhood_cleansed": np.random.choice(
            ["Harlem", "Williamsburg", "Astoria", "SoHo", "Chelsea"], n
        ),
        "latitude": np.random.uniform(40.5, 40.9, n),
        "longitude": np.random.uniform(-74.1, -73.7, n),
        "room_type": np.random.choice(
            ["Entire home/apt", "Private room", "Shared room"], n, p=[0.5, 0.4, 0.1]
        ),
        "price": np.random.uniform(50, 500, n),
        "minimum_nights": np.random.randint(1, 30, n),
        "number_of_reviews": np.random.randint(0, 200, n),
        "reviews_per_month": np.random.uniform(0, 10, n),
        "availability_365": np.random.randint(0, 365, n),
        "beds": np.random.randint(1, 6, n),
        "bathrooms_text": np.random.choice(["1 bath", "2 baths", "1 shared bath"], n),
    })


@pytest.fixture
def sample_calendar_df():
    """Create a sample calendar DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    listing_ids = [1, 2, 3]
    rows = []
    for lid in listing_ids:
        for d in dates:
            rows.append({
                "listing_id": lid,
                "date": d.strftime("%Y-%m-%d"),
                "available": np.random.choice(["t", "f"], p=[0.6, 0.4]),
                "price": f"${np.random.uniform(80, 300):.2f}",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_weather_df():
    """Create a sample weather DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    return pd.DataFrame({
        "date": dates,
        "temperature_mean": np.random.uniform(-5, 35, len(dates)),
        "precipitation_sum": np.random.uniform(0, 50, len(dates)),
        "wind_speed_max": np.random.uniform(0, 80, len(dates)),
    })
