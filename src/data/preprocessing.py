"""Data preprocessing module for the Dynamic Pricing Engine.

Handles cleaning, validation, and basic transformations.
Full feature engineering is in Phase 2.
"""

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_price_column(series: pd.Series) -> pd.Series:
    """Convert price strings like '$1,234.56' to float."""
    if series.dtype == object:
        return (
            series.str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return series.astype(float)


def clip_outliers(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 5,
    upper_percentile: float = 95,
) -> pd.DataFrame:
    """Clip values outside the given percentile range."""
    lower = np.percentile(df[column].dropna(), lower_percentile)
    upper = np.percentile(df[column].dropna(), upper_percentile)
    before_count = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    clipped = before_count - len(df)
    logger.info(
        f"Clipped {clipped:,} rows from '{column}' "
        f"(kept [{lower:.2f}, {upper:.2f}])"
    )
    return df


def preprocess_listings(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Clean and preprocess the listings dataset.

    Args:
        df: Raw listings DataFrame.
        config: Configuration dict.

    Returns:
        Cleaned DataFrame.
    """
    if config is None:
        config = load_config()

    df = df.copy()
    logger.info(f"Preprocessing listings: {df.shape[0]:,} rows")

    # Clean price column
    if "price" in df.columns:
        df["price"] = clean_price_column(df["price"])
        df = df[df["price"] > 0]

    # Clip price outliers
    floor_pct = config["features"]["price_floor_percentile"]
    ceil_pct = config["features"]["price_ceiling_percentile"]
    df = clip_outliers(df, "price", floor_pct, ceil_pct)

    # Convert numeric columns
    numeric_cols = ["latitude", "longitude", "number_of_reviews",
                    "reviews_per_month", "availability_365"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing reviews_per_month with 0 (no reviews = 0 velocity)
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Drop rows with missing critical fields
    critical = ["price", "latitude", "longitude", "room_type"]
    critical_present = [c for c in critical if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_present)
    logger.info(f"Dropped {before - len(df):,} rows with missing critical fields")

    logger.info(f"Preprocessed listings: {df.shape[0]:,} rows remaining")
    return df


def preprocess_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the calendar dataset.

    Args:
        df: Raw calendar DataFrame.

    Returns:
        Cleaned DataFrame with parsed dates and prices.
    """
    df = df.copy()
    logger.info(f"Preprocessing calendar: {df.shape[0]:,} rows")

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Clean price
    if "price" in df.columns:
        df["price"] = clean_price_column(df["price"])

    # Convert available to boolean
    if "available" in df.columns:
        df["was_booked"] = (df["available"] == "f").astype(int)

    logger.info(f"Preprocessed calendar: {df.shape[0]:,} rows")
    return df
