from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    # check if the config file exists
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with path.open("r", encoding="utf-8") as file:
        # read the yaml and return its contents as a dictionary
        return yaml.safe_load(file)

def ensure_output_dirs(config: Dict[str, Any]) -> None:
    Path(config["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["figures_dir"]).mkdir(parents=True, exist_ok=True)