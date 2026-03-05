"""Data ingestion module for the Dynamic Pricing Engine.

Downloads and loads Inside Airbnb data, weather data from Open-Meteo,
and holiday data from Nager.Date API.
"""

import gzip
import io
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_LISTINGS_COLUMNS = [
    "id", "name", "host_id", "neighbourhood_group_cleansed",
    "neighbourhood_cleansed", "latitude", "longitude", "room_type",
    "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "availability_365",
]

EXPECTED_CALENDAR_COLUMNS = [
    "listing_id", "date", "available", "price",
]

EXPECTED_REVIEWS_COLUMNS = [
    "listing_id", "date",
]


def download_file(url: str, output_path: Path, force: bool = False) -> Path:
    """Download a file from a URL if it doesn't already exist.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded file.
    """
    if output_path.exists() and not force:
        logger.info(f"File already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} ...")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def load_csv(filepath: Path, compressed: bool = True) -> pd.DataFrame:
    """Load a CSV file, optionally gzip-compressed.

    Args:
        filepath: Path to the CSV or CSV.GZ file.
        compressed: Whether the file is gzip-compressed.

    Returns:
        Loaded DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    compression = "gzip" if compressed or filepath.suffix == ".gz" else None
    df = pd.read_csv(filepath, compression=compression, low_memory=False)
    logger.info(f"Loaded {filepath.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> bool:
    """Validate that a DataFrame contains the expected columns.

    Args:
        df: DataFrame to validate.
        expected: List of expected column names.
        name: Name of the dataset (for logging).

    Returns:
        True if all expected columns are present.
    """
    missing = set(expected) - set(df.columns)
    if missing:
        logger.warning(f"{name}: Missing columns: {missing}")
        return False
    logger.info(f"{name}: All {len(expected)} expected columns present")
    return True


def download_airbnb_data(config: dict | None = None, force: bool = False) -> dict[str, Path]:
    """Download all Inside Airbnb datasets for the configured city.

    Args:
        config: Configuration dict. Loaded from config.yaml if None.
        force: If True, re-download even if files exist.

    Returns:
        Dict mapping dataset names to file paths.
    """
    if config is None:
        config = load_config()

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    urls = config["data"]["insideairbnb"]

    files = {}
    for name in ["listings", "calendar", "reviews"]:
        url = urls[f"{name}_url"]
        filename = f"{name}.csv.gz"
        output_path = raw_dir / filename
        files[name] = download_file(url, output_path, force=force)

    return files


def load_airbnb_data(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Load all Inside Airbnb datasets into DataFrames.

    Downloads the data first if it hasn't been downloaded yet.

    Args:
        config: Configuration dict. Loaded from config.yaml if None.

    Returns:
        Dict mapping dataset names to DataFrames.
    """
    if config is None:
        config = load_config()

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    datasets = {}

    for name in ["listings", "calendar", "reviews"]:
        filepath = raw_dir / f"{name}.csv.gz"
        if not filepath.exists():
            logger.info(f"{name}.csv.gz not found. Downloading...")
            download_airbnb_data(config)

        df = load_csv(filepath)
        datasets[name] = df

    # Validate columns
    validate_columns(datasets["listings"], EXPECTED_LISTINGS_COLUMNS, "Listings")
    validate_columns(datasets["calendar"], EXPECTED_CALENDAR_COLUMNS, "Calendar")
    validate_columns(datasets["reviews"], EXPECTED_REVIEWS_COLUMNS, "Reviews")

    return datasets


def fetch_weather_data(
    latitude: float | None = None,
    longitude: float | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    config: dict | None = None,
) -> pd.DataFrame:
    """Fetch historical weather data from Open-Meteo API (free, no key).

    Args:
        latitude: Location latitude. Uses config default if None.
        longitude: Location longitude. Uses config default if None.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        config: Configuration dict.

    Returns:
        DataFrame with daily weather data.
    """
    if config is None:
        config = load_config()

    weather_cfg = config["weather"]
    lat = latitude or weather_cfg["latitude"]
    lon = longitude or weather_cfg["longitude"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "America/New_York",
    }

    logger.info(f"Fetching weather data for ({lat}, {lon}) from {start_date} to {end_date}")
    response = requests.get(weather_cfg["api_url"].replace("/forecast", "/archive"), params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    daily = data.get("daily", {})

    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temperature_mean": daily.get("temperature_2m_mean", []),
        "precipitation_sum": daily.get("precipitation_sum", []),
        "wind_speed_max": daily.get("wind_speed_10m_max", []),
    })

    output_path = PROJECT_ROOT / config["data"]["external_dir"] / "weather.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Weather data saved: {df.shape[0]} days -> {output_path}")

    return df


def fetch_holiday_data(
    year: int = 2024,
    country_code: str | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Fetch public holiday data from Nager.Date API (free, no key).

    Args:
        year: Year to fetch holidays for.
        country_code: ISO country code. Uses config default if None.
        config: Configuration dict.

    Returns:
        DataFrame with holiday dates and names.
    """
    if config is None:
        config = load_config()

    holiday_cfg = config["holidays"]
    cc = country_code or holiday_cfg["country_code"]

    url = f"{holiday_cfg['api_url']}/{year}/{cc}"
    logger.info(f"Fetching holidays for {cc} in {year}")

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    holidays = response.json()
    df = pd.DataFrame(holidays)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "localName", "name"]].rename(columns={
        "localName": "holiday_local_name",
        "name": "holiday_name",
    })

    output_path = PROJECT_ROOT / config["data"]["external_dir"] / "holidays.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Holiday data saved: {df.shape[0]} holidays -> {output_path}")

    return df


if __name__ == "__main__":
    config = load_config()
    print("=== Dynamic Pricing Engine - Data Ingestion ===\n")

    # Download and load Airbnb data
    print("1. Downloading Inside Airbnb data for NYC...")
    files = download_airbnb_data(config)
    print(f"   Downloaded: {list(files.keys())}\n")

    print("2. Loading datasets into memory...")
    datasets = load_airbnb_data(config)
    for name, df in datasets.items():
        print(f"   {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n3. Fetching weather data from Open-Meteo...")
    weather_df = fetch_weather_data(config=config)
    print(f"   Weather: {weather_df.shape[0]} days")

    print("\n4. Fetching holiday data from Nager.Date...")
    holiday_df = fetch_holiday_data(config=config)
    print(f"   Holidays: {holiday_df.shape[0]} entries")

    print("\n=== Data ingestion complete! ===")
