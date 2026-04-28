from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration.

    Inspired by the clean configuration style encouraged by the project instructions.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Create output folders if they do not exist.
    """
    Path(config["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["figures_dir"]).mkdir(parents=True, exist_ok=True)