"""Feature engineering pipeline for the Dynamic Pricing Engine.

Builds all features from raw data:
- Temporal features (day_of_week, month, is_weekend, etc.)
- Weather features (temperature, precipitation, wind)
- Holiday features (is_holiday, days_until_holiday)
- Location features (KMeans clusters)
- Demand proxy features (review velocity, rolling occupancy)
- Competitor pricing features (neighborhood median, percentile rank)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from src.data.ingestion import fetch_holiday_data, fetch_weather_data
from src.data.preprocessing import preprocess_calendar, preprocess_listings
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Temporal Features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Extract temporal features from a date column.

    Features: hour (if datetime), day_of_week, day_of_month, week_of_year,
    month, quarter, is_weekend, is_month_start, is_month_end.
    """
    df = df.copy()
    dt = df[date_column]

    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt)
        df[date_column] = dt

    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    logger.info(f"Added 8 temporal features from '{date_column}'")
    return df


def add_checkin_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add days-until-checkin style features relative to a reference date."""
    df = df.copy()
    dt = pd.to_datetime(df[date_column])

    # Days from today (useful for future bookings)
    reference = dt.min()
    df["days_from_start"] = (dt - reference).dt.days

    # Season encoding
    month = dt.dt.month
    df["season"] = np.where(
        month.isin([12, 1, 2]), 0,  # Winter
        np.where(
            month.isin([3, 4, 5]), 1,  # Spring
            np.where(
                month.isin([6, 7, 8]), 2,  # Summer
                3  # Fall
            )
        )
    )

    logger.info("Added checkin and season features")
    return df


# ---------------------------------------------------------------------------
# Weather Features
# ---------------------------------------------------------------------------

def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    date_column: str = "date",
    config: dict | None = None,
) -> pd.DataFrame:
    """Merge weather data into the main DataFrame.

    If weather_df is not provided, attempts to load from disk or fetch from API.
    """
    df = df.copy()
    if config is None:
        config = load_config()

    if weather_df is None:
        weather_path = PROJECT_ROOT / config["data"]["external_dir"] / "weather.parquet"
        if weather_path.exists():
            weather_df = pd.read_parquet(weather_path)
            logger.info(f"Loaded weather data from {weather_path}")
        else:
            date_col = pd.to_datetime(df[date_column])
            start = date_col.min().strftime("%Y-%m-%d")
            end = date_col.max().strftime("%Y-%m-%d")
            weather_df = fetch_weather_data(start_date=start, end_date=end, config=config)

    # Ensure date types match
    df[date_column] = pd.to_datetime(df[date_column])
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    # Merge on date
    df = df.merge(
        weather_df[["date", "temperature_mean", "precipitation_sum", "wind_speed_max"]],
        left_on=date_column,
        right_on="date",
        how="left",
        suffixes=("", "_weather"),
    )

    # Drop duplicate date column if created
    if "date_weather" in df.columns:
        df = df.drop(columns=["date_weather"])

    # Fill missing weather with median
    for col in ["temperature_mean", "precipitation_sum", "wind_speed_max"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Derived weather features
    if "temperature_mean" in df.columns:
        df["is_hot_day"] = (df["temperature_mean"] > 30).astype(int)
        df["is_cold_day"] = (df["temperature_mean"] < 5).astype(int)

    if "precipitation_sum" in df.columns:
        df["is_rainy_day"] = (df["precipitation_sum"] > 5).astype(int)

    logger.info("Added weather features (temperature, precipitation, wind + derived)")
    return df


# ---------------------------------------------------------------------------
# Holiday Features
# ---------------------------------------------------------------------------

def add_holiday_features(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame | None = None,
    date_column: str = "date",
    config: dict | None = None,
) -> pd.DataFrame:
    """Add holiday-related features.

    Features: is_holiday, days_until_next_holiday, days_since_last_holiday.
    """
    df = df.copy()
    if config is None:
        config = load_config()

    if holiday_df is None:
        holiday_path = PROJECT_ROOT / config["data"]["external_dir"] / "holidays.parquet"
        if holiday_path.exists():
            holiday_df = pd.read_parquet(holiday_path)
            logger.info(f"Loaded holiday data from {holiday_path}")
        else:
            years = pd.to_datetime(df[date_column]).dt.year.unique()
            frames = []
            for year in years:
                frames.append(fetch_holiday_data(year=int(year), config=config))
            holiday_df = pd.concat(frames, ignore_index=True)

    df[date_column] = pd.to_datetime(df[date_column])
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])

    holiday_dates = set(holiday_df["date"].dt.date)
    df["is_holiday"] = df[date_column].dt.date.isin(holiday_dates).astype(int)

    # Days until next holiday
    sorted_holidays = sorted(holiday_df["date"].unique())

    def days_to_next_holiday(d):
        for h in sorted_holidays:
            h_date = pd.Timestamp(h)
            if h_date >= d:
                return (h_date - d).days
        return 365  # No upcoming holiday found

    def days_since_last_holiday(d):
        for h in reversed(sorted_holidays):
            h_date = pd.Timestamp(h)
            if h_date <= d:
                return (d - h_date).days
        return 365

    # For performance, compute on unique dates then map back
    unique_dates = df[date_column].unique()
    next_holiday_map = {d: days_to_next_holiday(d) for d in unique_dates}
    last_holiday_map = {d: days_since_last_holiday(d) for d in unique_dates}

    df["days_until_holiday"] = df[date_column].map(next_holiday_map)
    df["days_since_holiday"] = df[date_column].map(last_holiday_map)

    # Near-holiday flag (within 3 days of a holiday)
    df["near_holiday"] = ((df["days_until_holiday"] <= 3) | (df["days_since_holiday"] <= 1)).astype(int)

    logger.info("Added holiday features (is_holiday, days_until/since, near_holiday)")
    return df


# ---------------------------------------------------------------------------
# Location Features
# ---------------------------------------------------------------------------

def add_location_features(
    df: pd.DataFrame,
    n_clusters: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Cluster listings by latitude/longitude using KMeans.

    Returns the original DataFrame with a 'location_cluster' column.
    Also returns the fitted KMeans model for inference.
    """
    df = df.copy()
    if config is None:
        config = load_config()
    if n_clusters is None:
        n_clusters = config["features"]["location_clusters"]

    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("Skipping location features: lat/lon not found")
        return df

    coords = df[["latitude", "longitude"]].dropna()
    if len(coords) < n_clusters:
        logger.warning(f"Not enough points ({len(coords)}) for {n_clusters} clusters")
        df["location_cluster"] = 0
        return df

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[coords.index, "location_cluster"] = kmeans.fit_predict(coords)
    df["location_cluster"] = df["location_cluster"].fillna(0).astype(int)

    logger.info(f"Added location_cluster feature with {n_clusters} clusters")
    return df


# ---------------------------------------------------------------------------
# Demand Proxy Features
# ---------------------------------------------------------------------------

def add_demand_features(
    listings_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Compute demand proxy features from calendar and listing data.

    Features:
    - review_velocity (reviews_per_month from listings — already exists)
    - rolling_7d_occupancy: 7-day rolling average booking rate per listing
    - rolling_30d_occupancy: 30-day rolling average
    - occupancy_rate: overall occupancy rate per listing
    """
    if config is None:
        config = load_config()

    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    if "was_booked" not in cal.columns:
        cal["was_booked"] = (cal["available"] == "f").astype(int)

    # Overall occupancy per listing
    occupancy = cal.groupby("listing_id")["was_booked"].mean().reset_index()
    occupancy.columns = ["listing_id", "occupancy_rate"]

    # Rolling occupancy (per listing, sorted by date)
    cal = cal.sort_values(["listing_id", "date"])
    rolling_7d = config["features"]["rolling_window_days"]

    cal["rolling_7d_occupancy"] = (
        cal.groupby("listing_id")["was_booked"]
        .transform(lambda x: x.rolling(rolling_7d, min_periods=1).mean())
    )
    cal["rolling_30d_occupancy"] = (
        cal.groupby("listing_id")["was_booked"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )

    # Merge occupancy rate into listings
    listings_out = listings_df.copy()
    if "id" in listings_out.columns:
        listings_out = listings_out.merge(
            occupancy, left_on="id", right_on="listing_id", how="left"
        )
        if "listing_id" in listings_out.columns and "id" in listings_out.columns:
            listings_out = listings_out.drop(columns=["listing_id"])
    listings_out["occupancy_rate"] = listings_out.get("occupancy_rate", pd.Series(dtype=float)).fillna(0)

    logger.info("Added demand features (occupancy_rate, rolling occupancy)")
    return listings_out, cal


# ---------------------------------------------------------------------------
# Competitor Pricing Features
# ---------------------------------------------------------------------------

def add_competitor_features(df: pd.DataFrame, neighborhood_col: str = "neighbourhood_cleansed") -> pd.DataFrame:
    """Compute competitor pricing features within each neighborhood.

    Features:
    - median_neighborhood_price: median price in the same neighborhood
    - mean_neighborhood_price: mean price in the same neighborhood
    - price_rank_in_neighborhood: percentile rank of listing price
    - price_vs_neighborhood: ratio of listing price to neighborhood median
    """
    df = df.copy()

    if neighborhood_col not in df.columns or "price" not in df.columns:
        logger.warning(f"Skipping competitor features: '{neighborhood_col}' or 'price' not found")
        return df

    neighborhood_stats = df.groupby(neighborhood_col)["price"].agg(
        median_neighborhood_price="median",
        mean_neighborhood_price="mean",
        std_neighborhood_price="std",
        count_in_neighborhood="count",
    ).reset_index()

    df = df.merge(neighborhood_stats, on=neighborhood_col, how="left")

    # Price relative to neighborhood
    df["price_vs_neighborhood"] = df["price"] / df["median_neighborhood_price"].replace(0, np.nan)
    df["price_vs_neighborhood"] = df["price_vs_neighborhood"].fillna(1.0)

    # Percentile rank within neighborhood
    df["price_rank_in_neighborhood"] = df.groupby(neighborhood_col)["price"].rank(pct=True)

    logger.info("Added competitor pricing features (neighborhood median, rank, ratio)")
    return df


# ---------------------------------------------------------------------------
# Listing-Level Features
# ---------------------------------------------------------------------------

def add_listing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode listing-specific features.

    Features:
    - room_type_encoded: label-encoded room type
    - amenity_score: placeholder (computed from amenities text if available)
    - beds, bathrooms: extracted from text columns if needed
    """
    df = df.copy()

    # Encode room_type
    if "room_type" in df.columns:
        le = LabelEncoder()
        df["room_type_encoded"] = le.fit_transform(df["room_type"].fillna("Unknown"))

    # Extract bathrooms from text
    if "bathrooms_text" in df.columns and "bathrooms" not in df.columns:
        df["bathrooms"] = (
            df["bathrooms_text"]
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)
        )
        df["bathrooms"] = df["bathrooms"].fillna(1.0)

    # Fill beds
    if "beds" in df.columns:
        df["beds"] = df["beds"].fillna(1)

    # Amenity score (fraction of popular amenities present)
    if "amenities" in df.columns:
        top_amenities = [
            "wifi", "kitchen", "heating", "air conditioning",
            "washer", "dryer", "tv", "elevator", "parking",
            "pool", "gym", "hot tub",
        ]
        df["amenity_score"] = df["amenities"].apply(
            lambda x: sum(1 for a in top_amenities if a.lower() in str(x).lower()) / len(top_amenities)
            if pd.notna(x) else 0.0
        )
    else:
        df["amenity_score"] = 0.5  # Default when amenities not available

    # Review score
    if "review_scores_rating" in df.columns:
        df["review_score"] = df["review_scores_rating"].fillna(df["review_scores_rating"].median())
    elif "review_scores_value" in df.columns:
        df["review_score"] = df["review_scores_value"].fillna(df["review_scores_value"].median())
    else:
        df["review_score"] = 4.0  # Default

    logger.info("Added listing features (room_type_encoded, bathrooms, amenity_score, review_score)")
    return df


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    listings_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    holiday_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full feature engineering pipeline.

    Args:
        listings_df: Raw listings DataFrame.
        calendar_df: Raw calendar DataFrame.
        weather_df: Optional weather DataFrame.
        holiday_df: Optional holiday DataFrame.
        config: Configuration dict.

    Returns:
        Tuple of (enriched_listings, enriched_calendar) DataFrames.
    """
    if config is None:
        config = load_config()

    logger.info("=== Starting Feature Engineering Pipeline ===")

    # Step 1: Preprocess
    logger.info("Step 1/7: Preprocessing...")
    listings = preprocess_listings(listings_df, config)
    calendar = preprocess_calendar(calendar_df)

    # Step 2: Listing features
    logger.info("Step 2/7: Listing features...")
    listings = add_listing_features(listings)

    # Step 3: Location features
    logger.info("Step 3/7: Location clustering...")
    listings = add_location_features(listings, config=config)

    # Step 4: Competitor features
    logger.info("Step 4/7: Competitor pricing features...")
    listings = add_competitor_features(listings)

    # Step 5: Demand features
    logger.info("Step 5/7: Demand features...")
    listings, calendar = add_demand_features(listings, calendar, config)

    # Step 6: Temporal features on calendar
    logger.info("Step 6/7: Temporal features...")
    calendar = add_temporal_features(calendar, date_column="date")
    calendar = add_checkin_features(calendar, date_column="date")

    # Step 7: Weather & holiday features on calendar
    logger.info("Step 7/7: Weather & holiday features...")
    try:
        calendar = add_weather_features(calendar, weather_df=weather_df, config=config)
    except Exception as e:
        logger.warning(f"Weather features skipped: {e}")

    try:
        calendar = add_holiday_features(calendar, holiday_df=holiday_df, config=config)
    except Exception as e:
        logger.warning(f"Holiday features skipped: {e}")

    # Save to parquet
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    listings_path = processed_dir / "listings_features.parquet"
    calendar_path = processed_dir / "calendar_features.parquet"

    listings.to_parquet(listings_path, index=False)
    calendar.to_parquet(calendar_path, index=False)

    logger.info(f"Saved listings features: {listings.shape} -> {listings_path}")
    logger.info(f"Saved calendar features: {calendar.shape} -> {calendar_path}")
    logger.info("=== Feature Engineering Pipeline Complete ===")

    return listings, calendar


if __name__ == "__main__":
    from src.data.ingestion import load_airbnb_data

    config = load_config()
    print("Loading raw data...")
    datasets = load_airbnb_data(config)

    print("Running feature engineering pipeline...")
    listings, calendar = build_feature_matrix(
        datasets["listings"],
        datasets["calendar"],
        config=config,
    )

    print(f"\nFinal listings: {listings.shape}")
    print(f"Final calendar: {calendar.shape}")
    print(f"\nListings columns ({len(listings.columns)}):")
    print(listings.columns.tolist())
    print(f"\nCalendar columns ({len(calendar.columns)}):")
    print(calendar.columns.tolist())
