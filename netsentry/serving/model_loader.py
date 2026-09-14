"""
Champion Model Loader (`netsentry.serving.model_loader`).
---------------------------------------------------------
Loads the active @champion model pipeline and its bound decision threshold
directly from the MLflow Model Registry. Guarantees model artifact and threshold
always match the exact same version.
"""

from dataclasses import dataclass
from typing import Any, Optional
import mlflow
from netsentry.registry.client import RegistryClient
from netsentry.serving.config import ServingConfig, load_serving_config


@dataclass(frozen=True)
class ChampionModel:
    """Encapsulates the loaded model pipeline, model version, and frozen threshold."""
    model: Any
    version: str
    threshold: float
    model_name: str


class ModelLoader:
    """Loads and caches the current champion model and threshold from MLflow."""

    def __init__(self, config: Optional[ServingConfig] = None, client: Optional[RegistryClient] = None):
        self.config = config or load_serving_config()
        self.client = client or RegistryClient(tracking_uri=self.config.tracking_uri)
        self._champion_model: Optional[ChampionModel] = None

    def load(self) -> ChampionModel:
        """
        1. Queries @champion alias from MLflow Model Registry.
        2. Resolves actual immutable model version.
        3. Loads Python model pipeline artifact.
        4. Extracts matching decision_threshold tag.
        5. Returns atomic ChampionModel object.
        """
        # Configure tracking URI
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        mv = self.client.get_model_version_by_alias(
            name=self.config.model_name,
            alias=self.config.champion_alias,
        )
        if mv is None:
            raise RuntimeError(
                f"No model version found for alias '@{self.config.champion_alias}' "
                f"on registered model '{self.config.model_name}'."
            )

        version_str = str(mv.version)
        threshold = self.client.get_model_threshold(
            model_name=self.config.model_name,
            version=version_str,
        )

        model_pipeline = self.client.load_model_by_alias(
            name=self.config.model_name,
            alias=self.config.champion_alias,
        )

        self._champion_model = ChampionModel(
            model=model_pipeline,
            version=version_str,
            threshold=threshold,
            model_name=self.config.model_name,
        )
        return self._champion_model

    @property
    def champion_model(self) -> ChampionModel:
        if self._champion_model is None:
            raise RuntimeError("Champion model has not been loaded. Call load() first.")
        return self._champion_model
