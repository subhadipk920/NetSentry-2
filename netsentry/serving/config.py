"""
Serving Configuration Schema (`netsentry.serving.config`).
---------------------------------------------------------
Loads server settings and MLflow connection parameters.
Threshold is intentionally NOT loaded here—it belongs to the Champion model version.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


import os

@dataclass(frozen=True)
class ServingConfig:
    """Configuration for NetSentry FastAPI serving microservice."""
    model_name: str = "NetSentry"
    champion_alias: str = "champion"
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    api_title: str = "NetSentry API"
    api_version: str = "2.0.0"


def load_serving_config(config_source: Optional[Union[str, Path, Dict[str, Any]]] = None) -> ServingConfig:
    """Loads and validates ServingConfig from YAML file or defaults."""
    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if config_source is None:
        default_path = Path("configs/serving.yaml")
        if default_path.exists():
            config_source = default_path
        else:
            return ServingConfig()

    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            return ServingConfig()
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        return ServingConfig()

    srv_raw = raw_cfg.get("serving", {}) if isinstance(raw_cfg, dict) else {}
    tracking_uri = env_tracking_uri or str(srv_raw.get("tracking_uri", "sqlite:///mlruns.db"))
    return ServingConfig(
        model_name=str(srv_raw.get("model_name", "NetSentry")),
        champion_alias=str(srv_raw.get("champion_alias", "champion")),
        tracking_uri=tracking_uri,
        api_title=str(srv_raw.get("api_title", "NetSentry API")),
        api_version=str(srv_raw.get("api_version", "2.0.0")),
    )
