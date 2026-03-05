"""Configuration loader for the Dynamic Pricing Engine."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_data_paths(config: dict | None = None) -> dict[str, Path]:
    """Return resolved data directory paths."""
    if config is None:
        config = load_config()
    data_cfg = config["data"]
    return {
        "raw": PROJECT_ROOT / data_cfg["raw_dir"],
        "processed": PROJECT_ROOT / data_cfg["processed_dir"],
        "external": PROJECT_ROOT / data_cfg["external_dir"],
    }
