"""
NetSentry v2 — End-to-End Production Pipeline Entrypoint
=========================================================
Automates the full industry lifecycle:
1. Baseline Winner Resolution (from artifacts/baselines/latest.json or fallback tournament).
2. Hyperparameter Optimization via Optuna (using configs/tuning/<model>.yaml).
3. Final Retraining on Train + Val with tuned hyperparameters.
4. Optimal Decision Threshold (tau*) Search on Validation Set.
5. Strict Untouched Test Evaluation & Quality Gate Auditing.
6. MLflow Model Registration with Threshold Tag Binding.
7. Conditional Promotion to @champion Alias.

Usage:
    python scripts/run_pipeline.py [--model-name xgboost] [--skip-tuning]
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*artifact_path.*")
warnings.filterwarnings("ignore", message=r".*pickle.*")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from netsentry.data.pipeline import run_data_pipeline
from netsentry.evaluation.drift import DataDriftDetector
from netsentry.evaluation.evaluator import Evaluator
from netsentry.evaluation.metrics import compute_binary_metrics
from netsentry.evaluation.threshold import apply_threshold, find_optimal_threshold
from netsentry.features.engineer import engineer_network_features
from netsentry.features.split import get_stratified_splits
from netsentry.models.config import load_model_config
from netsentry.models.factory import create_model
from netsentry.registry.lifecycle import ModelLifecycleManager
from netsentry.training.baseline import BaselineOrchestrator
from netsentry.training.tracking import MLflowTracker
from netsentry.tuning.config import load_tuning_config
from netsentry.tuning.tuner import Tuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("netsentry.pipeline")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)


def resolve_baseline_models(baseline_summary_path: Path, top_k: int = 3) -> list[str]:
    """Reads baseline artifact and returns the top_k best performing model names."""
    if not baseline_summary_path.exists():
        logger.error(
            "\n" + "=" * 65 + "\n"
            "❌ ERROR: No baseline tournament results found!\n"
            f"Expected artifact at: {baseline_summary_path}\n\n"
            "You must execute the baseline comparison tournament first:\n"
            "    python scripts/train_baselines.py\n\n"
            "Or specify an explicit model to train:\n"
            "    python scripts/run_pipeline.py --model-name <model_name>\n"
            + "=" * 65 + "\n"
        )
        sys.exit(1)

    try:
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        models_list = data.get("models", [])
        primary_metric = data.get("metric", "recall")
        direction = data.get("direction", "maximize")

        successful = [m for m in models_list if m.get("status") == "SUCCESS"]
        if not successful:
            winner = data.get("winner")
            return [winner] if winner else []

        def ranking_key(item: dict):
            m = item.get("metrics", {})
            return (
                m.get(primary_metric, 0.0),
                m.get("f1", 0.0),
                m.get("precision", 0.0),
                m.get("roc_auc", 0.0),
            )

        ranked = sorted(successful, key=ranking_key, reverse=(direction == "maximize"))
        top_models = [m["name"] for m in ranked[:top_k]]
        logger.info(f"Resolved top {len(top_models)} baseline models from {baseline_summary_path}: {top_models}")
        return top_models
    except Exception as e:
        logger.error(f"Failed to read baseline summary from {baseline_summary_path}: {e}")
        logger.error("Please rerun the baseline tournament: python scripts/train_baselines.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="NetSentry v2 End-to-End Pipeline Entrypoint")
    parser.add_argument("--model-name", type=str, default=None, help="Explicit model to train/tune (e.g. xgboost)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top baseline models to tune and compare (default: 3)")
    parser.add_argument("--tuning-trials", type=int, default=None, help="Override number of Optuna HPO trials")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna HPO and train baseline directly")
    parser.add_argument("--force-tuning", action="store_true", help="Force Optuna HPO even if no drift is detected")
    parser.add_argument("--drift-threshold", type=float, default=0.20, help="Fraction of drifted features triggering emergency HPO (default: 0.20)")
    import os
    default_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    parser.add_argument("--tracking-uri", type=str, default=default_tracking_uri, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="netsentry-production-pipeline", help="MLflow experiment")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Setup execution log capture file
    import datetime
    log_dir = project_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_run_{timestamp_str}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # 1. Initialize Tracker
    tracker = MLflowTracker(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        enabled=True,
    )

    # 2. Data Preparation & Partitioning
    logger.info("Step 1/6: Running Data Pipeline & Feature Engineering...")
    import polars as pl
    pipeline_res = run_data_pipeline(
        raw_source=project_root / "data" / "raw",
        output_path=project_root / "data" / "processed" / "clean_network_flows.parquet",
        force_recompute=False,
    )
    clean_parquet_path = pipeline_res["output_path"]
    logger.info(f"Loading cleaned flows from: {clean_parquet_path}")
    df_clean = pl.read_parquet(clean_parquet_path)

    df_features = engineer_network_features(df_clean)
    splits = get_stratified_splits(df_features, seed=42)

    X_train = splits.X_train
    y_train = splits.y_train
    X_val = splits.X_val
    y_val = splits.y_val
    X_test = splits.X_test
    y_test = splits.y_test
    labels_test = splits.labels_test

    logger.info(
        f"Data partitioned: Train={len(y_train):,} rows, Val={len(y_val):,} rows, Test={len(y_test):,} rows"
    )

    # 3. Model Architecture Selection (Top-K from Baseline Tournament)
    baseline_artifact = project_root / "artifacts" / "baselines" / "latest.json"
    if args.model_name:
        candidate_models = [args.model_name]
        logger.info(f"Target model explicitly configured by user: '{args.model_name}'")
    else:
        candidate_models = resolve_baseline_models(
            baseline_summary_path=baseline_artifact,
            top_k=args.top_k,
        )
        logger.info(f"Selected Top-{len(candidate_models)} baseline models for tuning & comparison: {candidate_models}")

    # 4. Data Drift Evaluation (Adaptive Trigger: Fast Retrain vs. Emergency HPO)
    logger.info("Step 2/6: Evaluating Feature Distribution Drift against Reference Baseline...")
    drift_detector = DataDriftDetector(dataset_drift_threshold=args.drift_threshold)
    drift_report = drift_detector.calculate_drift(
        reference_data=X_train,
        current_data=X_val,
        feature_names=splits.feature_names if hasattr(splits, "feature_names") else None,
    )
    drift_artifact_path = project_root / "artifacts" / "drift" / "latest.json"
    drift_detector.save_report(drift_report, drift_artifact_path)

    trigger_hpo = drift_report.drift_detected or args.force_tuning
    if trigger_hpo:
        reason = "Drift Detected" if drift_report.drift_detected else "Forced via CLI"
        logger.warning(f"🚨 TRIGGERING EMERGENCY OPTUNA HPO ({reason})! Significant feature shift observed.")
    else:
        logger.info("✅ NO SIGNIFICANT DRIFT DETECTED. Proceeding with Fast Production Retraining on existing architectures.")

    # 5. Model Retraining or Emergency HPO
    tuned_models = {}

    for idx, model_name in enumerate(candidate_models, 1):
        logger.info("=" * 60)
        logger.info(f"[{idx}/{len(candidate_models)}] Processing Candidate Architecture: '{model_name}'")
        logger.info("=" * 60)

        model_instance = None
        current_opt_name = model_name

        # Case A: Emergency HPO Triggered (or tuning explicitly requested without skip)
        if trigger_hpo and not args.skip_tuning:
            tuning_config_file = project_root / "configs" / "tuning" / f"{model_name}.yaml"
            if tuning_config_file.exists():
                logger.info(f"Running Optuna HPO for '{model_name}' ({tuning_config_file.name})...")
                tuning_cfg = load_tuning_config(str(tuning_config_file))
                if args.tuning_trials:
                    from dataclasses import replace
                    updated_settings = replace(tuning_cfg.tuning, n_trials=args.tuning_trials)
                    tuning_cfg = replace(tuning_cfg, tuning=updated_settings)

                tuner = Tuner(
                    tuning_config=tuning_cfg,
                    tracker=tracker,
                )
                tuning_res = tuner.tune(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    output_dir=str(project_root / "configs" / "models"),
                    retrain_on_train_val=True,
                )
                model_instance = tuning_res.final_model
                current_opt_name = f"{model_name}_tuned"
                logger.info(f"Tuning finished for '{model_name}'. Best score ({tuning_cfg.tuning.metric}): {tuning_res.best_score:.4f}")
                logger.info(f"Saved tuned configuration to: {tuning_res.tuned_model_config_path}")
            else:
                logger.warning(
                    f"No HPO config found at '{tuning_config_file}'. Falling back to fast retrain."
                )

        # Case B: Fast Retrain (No drift) - Retrain winning architecture on Train + Val
        if model_instance is None:
            existing_tuned_yaml = project_root / "configs" / "models" / f"{model_name}_tuned.yaml"
            base_yaml = project_root / "configs" / "models" / f"{model_name}.yaml"
            target_cfg_file = existing_tuned_yaml if existing_tuned_yaml.exists() else base_yaml

            logger.info(f"⚡ Fast Retraining model '{model_name}' using config '{target_cfg_file.name}' on Train + Val...")
            model_cfg = load_model_config(str(target_cfg_file))
            model_instance = create_model(model_cfg)
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            model_instance.fit(X_combined, y_combined)
            current_opt_name = target_cfg_file.stem

        tuned_models[model_name] = {
            "model": model_instance,
            "optimal_name": current_opt_name,
        }

    # If multiple candidates, evaluate on validation set to pick the overall best candidate
    if len(tuned_models) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARING TUNED CANDIDATES ON VALIDATION SET")
        logger.info("=" * 60)
        best_candidate_name = None
        best_val_score = -1.0
        primary_metric = "recall"

        eval_config_file = project_root / "configs" / "evaluation.yaml"
        temp_evaluator = Evaluator(
            config=str(eval_config_file) if eval_config_file.exists() else None,
            tracker=tracker,
        )

        for name, entry in tuned_models.items():
            model = entry["model"]
            val_probs = model.predict_proba(X_val)
            th_search = find_optimal_threshold(
                y_val_true=y_val,
                y_val_prob=val_probs,
                config=temp_evaluator.config.threshold,
            )
            val_preds = apply_threshold(val_probs, threshold=th_search.selected_threshold)
            val_metrics = compute_binary_metrics(y_true=y_val, y_pred=val_preds, y_prob=val_probs)
            score = val_metrics.get(primary_metric, 0.0)
            logger.info(
                f"Candidate '{entry['optimal_name']}': Val {primary_metric.capitalize()}={score:.4f}, "
                f"Tau*={th_search.selected_threshold:.4f}, F1={val_metrics.get('f1', 0.0):.4f}, "
                f"ROC-AUC={val_metrics.get('roc_auc', 0.0):.4f}"
            )
            if score > best_val_score:
                best_val_score = score
                best_candidate_name = name

        logger.info("=" * 60)
        logger.info(f"🏆 Best Architecture Selected: '{best_candidate_name}' (Score: {best_val_score:.4f})")
        logger.info("=" * 60 + "\n")
        final_model = tuned_models[best_candidate_name]["model"]
        optimal_model_name = tuned_models[best_candidate_name]["optimal_name"]
    else:
        only_name = list(tuned_models.keys())[0]
        final_model = tuned_models[only_name]["model"]
        optimal_model_name = tuned_models[only_name]["optimal_name"]

    # 5. Strict Evaluation & Quality Gates
    logger.info("Step 3/6: Threshold Optimization & Untouched Test Set Evaluation...")
    eval_config_file = project_root / "configs" / "evaluation.yaml"
    evaluator = Evaluator(
        config=str(eval_config_file) if eval_config_file.exists() else None,
        tracker=tracker,
    )

    with tracker.start_run(run_name=f"pipeline_eval_{optimal_model_name}") as run_id:
        eval_result = evaluator.evaluate(
            model=final_model,
            model_name=optimal_model_name,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels_test=labels_test,
        )

        logger.info("=" * 60)
        logger.info("EVALUATION & QUALITY GATE REPORT")
        logger.info("=" * 60)
        logger.info("Default Decision Threshold:        0.5000")
        logger.info(f"Optimal Decision Threshold (tau*): {eval_result.optimal_threshold:.4f}")
        logger.info("-" * 60)

        def_m = eval_result.default_metrics
        opt_m = eval_result.selected_threshold_metrics

        logger.info(f"{'Metric':<20} | {'Default (tau=0.50)':<18} | {'Optimal (tau*)':<18}")
        logger.info("-" * 60)
        logger.info(f"{'Recall':<20} | {def_m.get('recall', 0.0):<18.4f} | {opt_m.get('recall', 0.0):<18.4f}")
        logger.info(f"{'F1 Score':<20} | {def_m.get('f1', 0.0):<18.4f} | {opt_m.get('f1', 0.0):<18.4f}")
        logger.info(f"{'Precision':<20} | {def_m.get('precision', 0.0):<18.4f} | {opt_m.get('precision', 0.0):<18.4f}")
        logger.info(f"{'False Negative Rate':<20} | {def_m.get('false_negative_rate', 0.0):<18.4f} | {opt_m.get('false_negative_rate', 0.0):<18.4f}")
        logger.info(f"{'False Positive Rate':<20} | {def_m.get('false_positive_rate', 0.0):<18.4f} | {opt_m.get('false_positive_rate', 0.0):<18.4f}")
        logger.info("-" * 60)

        # Formatted Confusion Matrix Comparison
        logger.info("CONFUSION MATRIX COMPARISON ON UNTOUCHED TEST SET:")
        logger.info(f"  [Default tau=0.50]                 [Optimal tau*={eval_result.optimal_threshold:.4f}]")
        logger.info(f"  TN: {def_m.get('true_negatives', 0):<8,}  FP: {def_m.get('false_positives', 0):<8,}     TN: {opt_m.get('true_negatives', 0):<8,}  FP: {opt_m.get('false_positives', 0):<8,}")
        logger.info(f"  FN: {def_m.get('false_negatives', 0):<8,}  TP: {def_m.get('true_positives', 0):<8,}     FN: {opt_m.get('false_negatives', 0):<8,}  TP: {opt_m.get('true_positives', 0):<8,}")
        logger.info("=" * 60)
        logger.info(f"Latency (p95):   {eval_result.latency.p95_ms:.2f} ms")
        logger.info(f"Quality Gates Passed: {eval_result.quality_gate.passed}")
        if not eval_result.quality_gate.passed:
            logger.warning(f"Quality Gate Failures: {eval_result.quality_gate.failure_reasons}")

        # Save machine-readable confusion matrix artifact
        cm_artifact_path = project_root / "artifacts" / "evaluation"
        cm_artifact_path.mkdir(parents=True, exist_ok=True)
        cm_report_file = cm_artifact_path / f"{optimal_model_name}_confusion_matrix.json"
        with open(cm_report_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": optimal_model_name,
                "default_threshold": 0.50,
                "optimal_threshold": eval_result.optimal_threshold,
                "default_confusion_matrix": {
                    "tn": def_m.get("true_negatives", 0),
                    "fp": def_m.get("false_positives", 0),
                    "fn": def_m.get("false_negatives", 0),
                    "tp": def_m.get("true_positives", 0),
                    "fnr": def_m.get("false_negative_rate", 0.0),
                    "fpr": def_m.get("false_positive_rate", 0.0),
                },
                "optimal_confusion_matrix": {
                    "tn": opt_m.get("true_negatives", 0),
                    "fp": opt_m.get("false_positives", 0),
                    "fn": opt_m.get("false_negatives", 0),
                    "tp": opt_m.get("true_positives", 0),
                    "fnr": opt_m.get("false_negative_rate", 0.0),
                    "fpr": opt_m.get("false_positive_rate", 0.0),
                },
            }, f, indent=2)
        logger.info(f"Confusion matrix report saved to: {cm_report_file}")

        # Log confusion matrix metrics and JSON artifact to MLflow
        if tracker.enabled:
            tracker.log_metrics({
                "default_tn": float(def_m.get("true_negatives", 0)),
                "default_fp": float(def_m.get("false_positives", 0)),
                "default_fn": float(def_m.get("false_negatives", 0)),
                "default_tp": float(def_m.get("true_positives", 0)),
                "default_fnr": float(def_m.get("false_negative_rate", 0.0)),
                "default_fpr": float(def_m.get("false_positive_rate", 0.0)),
                "optimal_tn": float(opt_m.get("true_negatives", 0)),
                "optimal_fp": float(opt_m.get("false_positives", 0)),
                "optimal_fn": float(opt_m.get("false_negatives", 0)),
                "optimal_tp": float(opt_m.get("true_positives", 0)),
                "optimal_fnr": float(opt_m.get("false_negative_rate", 0.0)),
                "optimal_fpr": float(opt_m.get("false_positive_rate", 0.0)),
            })
            tracker.log_artifact(local_path=str(cm_report_file), artifact_path="evaluation")

        # Log complete model artifact
        tracker.log_model(model=final_model, artifact_path="model")

        # 6. Lifecycle Management & Promotion
        logger.info("Step 4/6: Registry & Champion Promotion...")
        registry_config_file = project_root / "configs" / "registry" / "promotion.yaml"
        from netsentry.registry.client import RegistryClient
        lifecycle = ModelLifecycleManager(
            config=str(registry_config_file) if registry_config_file.exists() else None,
            registry_client=RegistryClient(tracking_uri=tracker.tracking_uri),
            tracker=tracker,
        )

        lifecycle_result = lifecycle.promote_candidate(
            candidate_model=final_model,
            run_id=run_id,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels_test=labels_test,
            metadata={
                "model_name": optimal_model_name,
                "tuned": not args.skip_tuning,
                "optimal_threshold": eval_result.optimal_threshold,
                "quality_gate_passed": eval_result.quality_gate.passed,
            },
            optimal_threshold=eval_result.optimal_threshold,
        )

        logger.info("=" * 60)
        logger.info("REGISTRY PROMOTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Registered Version:     v{lifecycle_result.registered_version}")
        logger.info(f"Promotion Passed:       {lifecycle_result.decision.passed}")
        logger.info(f"Promotion Reason:       {lifecycle_result.decision.reason}")
        logger.info(f"Active Champion Version: v{lifecycle_result.champion_version_after}")
        logger.info(f"Is New Champion:        {lifecycle_result.is_new_champion}")
        logger.info("=" * 60)
        logger.info("Pipeline execution completed successfully.")

        # Flush and upload execution log to Cloudflare R2 and MLflow
        if log_file.exists():
            file_handler.flush()
            tracker.log_artifact(local_path=str(log_file), artifact_path="logs")
            try:
                from netsentry.data.storage import upload_file_to_r2
                r2_log_path = f"logs/{log_file.name}"
                upload_file_to_r2(local_path=str(log_file), r2_path=r2_log_path)
                logger.info(f"Execution log uploaded to Cloudflare R2: {r2_log_path}")
            except Exception as e:
                logger.warning(f"Could not upload log to R2: {e}")


if __name__ == "__main__":
    main()
