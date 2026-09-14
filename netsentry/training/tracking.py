"""
MLflow Tracking Abstraction (`netsentry.training.tracking`).
-----------------------------------------------------------
Manages experiment initialization, run lifecycle, metric/parameter logging,
and model artifact recording without coupling training logic to MLflow.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import mlflow
import mlflow.sklearn


class MLflowTracker:
    """Encapsulates MLflow experiment tracking operations."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = "file:./mlruns", enabled: bool = True):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled

        import os
        # Map Cloudflare R2 credentials to standard AWS/Boto3 environment variables
        if os.getenv("R2_ACCESS_KEY_ID") and not os.getenv("AWS_ACCESS_KEY_ID"):
            os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("R2_ACCESS_KEY_ID")
        if os.getenv("R2_SECRET_ACCESS_KEY") and not os.getenv("AWS_SECRET_ACCESS_KEY"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("R2_SECRET_ACCESS_KEY")
        if os.getenv("R2_ENDPOINT_URL") and not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("R2_ENDPOINT_URL")
        if os.getenv("R2_ENDPOINT_URL") and not os.getenv("AWS_ENDPOINT_URL"):
            os.environ["AWS_ENDPOINT_URL"] = os.getenv("R2_ENDPOINT_URL")

        if self.enabled and self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Use R2 as default artifact root if R2 credentials exist
            bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
            artifact_location = f"s3://{bucket}/mlflow-artifacts" if os.getenv("MLFLOW_S3_ENDPOINT_URL") else None

            from mlflow.tracking import MlflowClient
            client = MlflowClient(self.tracking_uri)
            try:
                exp = client.get_experiment_by_name(self.experiment_name)
                if exp is None:
                    client.create_experiment(name=self.experiment_name, artifact_location=artifact_location)
                elif exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                try:
                    mlflow.set_experiment(self.experiment_name)
                except Exception:
                    raise e

    @contextmanager
    def start_run(self, run_name: Optional[str] = None) -> Generator[Optional[str], None, None]:
        """Context manager to scope an MLflow run safely."""
        if not self.enabled:
            yield None
            return

        # Support nested run if an existing run is already active
        nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            yield run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs configuration parameters and dataset metadata."""
        if not self.enabled:
            return

        # Flatten nested structures or complex objects into string/primitive representations
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                flat_params[k] = str(v)
            else:
                flat_params[k] = v

        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs evaluation metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Logs the complete model object (preprocessor + estimator)."""
        if not self.enabled:
            return
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a local file or directory as an MLflow artifact."""
        if not self.enabled:
            return
        mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Logs a Python dictionary as a JSON artifact directly to MLflow."""
        if not self.enabled:
            return
        mlflow.log_dict(dictionary, artifact_file)
