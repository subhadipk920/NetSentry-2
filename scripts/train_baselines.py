"""
NetSentry Baseline Training Script (`scripts/train_baselines.py`).
-----------------------------------------------------------------
CLI entry point to discover baseline model configs, prepare data splits,
train and evaluate all candidate models, rank by validation Recall, and persist the winner.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --config-dir configs/models --metric recall
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for standalone script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from netsentry.data.pipeline import run_data_pipeline
from netsentry.features.engineer import engineer_network_features
from netsentry.features.split import get_stratified_splits
from netsentry.training.baseline import BaselineOrchestrator
from netsentry.training.tracking import MLflowTracker

import warnings
warnings.filterwarnings("ignore", message=r".*artifact_path.*is deprecated.*")
warnings.filterwarnings("ignore", message=r".*Saving scikit-learn models in the pickle or cloudpickle format.*")


def parse_args():
    parser = argparse.ArgumentParser(description="NetSentry Baseline Model Training & Selection Tournament.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/models",
        help="Directory containing baseline model YAML configurations.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        help="Primary selection metric for baseline ranking (default: recall).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/clean_network_flows.parquet",
        help="Path to preprocessed clean parquet dataset.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory with raw CIC-IDS2017 network traffic CSVs.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional sampling fraction for quick benchmark iteration (e.g. 0.05).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("NETSENTRY v2: BASELINE MODEL TRAINING TOURNAMENT")
    print("=" * 60)

    import polars as pl
    import os

    parquet_file = Path(args.data_path)
    if parquet_file.exists():
        print(f"Loading cleaned dataset directly from {parquet_file}...")
        clean_df = pl.read_parquet(parquet_file)
    else:
        # 1. Discover raw capture files
        raw_dir = Path(args.raw_data_dir)
        raw_csvs = list(raw_dir.glob("*.csv"))
        if not raw_csvs:
            print(f"Error: Neither {parquet_file} nor raw CSV files found in {raw_dir}")
            sys.exit(1)

        print(f"Found {len(raw_csvs)} raw flow capture files.")
        pipeline_res = run_data_pipeline(raw_csvs)
        clean_df = pl.read_parquet(pipeline_res["output_path"])

    if args.sample_frac is not None and 0.0 < args.sample_frac < 1.0:
        clean_df = clean_df.sample(fraction=args.sample_frac, seed=42)
        print(f"Sampled {len(clean_df)} rows ({args.sample_frac * 100:.1f}%) for fast tournament.")

    # 3. Domain Feature Engineering
    print("Applying stateless network feature engineering...")
    feature_df = engineer_network_features(clean_df)

    # 4. Stratified 70/15/15 Splitting
    print("Partitioning stratified 70/15/15 splits (preserving test set untouched)...")
    splits = get_stratified_splits(feature_df)

    # 5. Execute Tournament via BaselineOrchestrator
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    tracker = MLflowTracker(
        experiment_name="netsentry-baseline-tournament",
        tracking_uri=tracking_uri,
        enabled=True,
    )

    orchestrator = BaselineOrchestrator(
        config_dir=args.config_dir,
        primary_metric=args.metric,
        direction="maximize",
        tracker=tracker,
    )

    result = orchestrator.run(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
    )

    print(f"🏆 Baseline Tournament Winner: {result.winner.upper()}")
    print(f"Artifact stored at: {result.output_artifact_path}")
    print("Next step: Pass winner configuration to Phase 6 Optuna Tuning.")


if __name__ == "__main__":
    main()
