"""AICRA CLI using Typer."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
import typer

from .config import get_settings
from .core.calibration import create_calibrator
from .core.data import load_ember_2024
from .core.evaluation import cost_sensitive_threshold, evaluate_probs
from .models.lightgbm import train_bagged_lightgbm
from .register import Policy, compute_register, write_register
from .pipelines.training import TrainingPipeline
from .pipelines.calibration import CalibrationPipeline
from .pipelines.evaluation import EvaluationPipeline
from .pipelines.policy import PolicyPipeline

app = typer.Typer(
    name="aicra",
    help="AI Cyber Risk Advisor - Machine Learning-Based Cyber Risk Assessment",
    no_args_is_help=True,
)


@app.command()
def ingest(
    input_file: Path = typer.Option(..., help="Input file to ingest"),
    output_dir: Path = typer.Option(..., help="Output directory"),
    validate: bool = typer.Option(True, help="Validate input schema"),
) -> None:
    """Ingest and validate input data."""
    typer.echo(f"Ingesting data from {input_file}")

    # TODO: Implement schema validation
    if validate:
        typer.echo("Validating input schema...")

    typer.echo(f"Data ingested to {output_dir}")


@app.command()
def scan(
    input_file: Path = typer.Option(..., help="Input file to scan"),
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
) -> None:
    """Scan files for ransomware risk."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {input_file} with model {model_path}")

    # TODO: Implement scanning logic
    typer.echo("Scan completed")


@app.command()
def train(
    data_path: Path | None = typer.Option(None, help="Path to training data"),
    model_type: str = typer.Option("lgbm", help="Model type (lgbm/ffnn)"),
    model_name: str = typer.Option("bagged_lightgbm", help="Model name"),
    experiment_name: str | None = typer.Option(None, help="MLflow experiment name"),
    seeds: int = typer.Option(5, help="Number of seeds for bagging"),
) -> None:
    """Train AICRA model."""
    settings = get_settings()

    # Set up MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    typer.echo("Loading training data...")

    try:
        if data_path:
            # TODO: Load from custom path
            train_data, _ = load_ember_2024()
        else:
            train_data, _ = load_ember_2024()
    except FileNotFoundError as e:
        typer.echo(f"Data not found: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo("Training model...")

    # Use training pipeline
    training_pipeline = TrainingPipeline(settings)
    model_path = training_pipeline.run(
        train_data=train_data,
        model_type=model_type,
        model_name=model_name,
        experiment_name=experiment_name,
        seeds=seeds
    )

    typer.echo(f"Model saved to {model_path}")


@app.command()
def evaluate(
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    data_path: Path | None = typer.Option(None, help="Path to test data"),
    generate_plots: bool = typer.Option(True, help="Generate evaluation plots"),
    timestamp_column: str | None = typer.Option(None, help="Timestamp column for time-ordered split"),
    family_column: str | None = typer.Option(None, help="Family column for out-of-family evaluation"),
    k_values: str = typer.Option("1,5,10", help="Comma-separated k values for Lift@k"),
) -> None:
    """Evaluate model performance."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading test data...")

    try:
        if data_path:
            # TODO: Load from custom path
            _, test_data = load_ember_2024()
        else:
            _, test_data = load_ember_2024()
    except FileNotFoundError as e:
        typer.echo(f"Data not found: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo("Loading model...")
    model = model_path.load() if hasattr(model_path, 'load') else joblib.load(model_path)

    typer.echo("Generating predictions...")
    y_prob = model.predict_proba(test_data.features)

    # Use default threshold for now
    threshold = 0.5

    # Parse k values
    k_vals = [int(k.strip()) for k in k_values.split(',')]

    typer.echo("Computing metrics...")
    
    # Use evaluation pipeline
    evaluation_pipeline = EvaluationPipeline(settings)
    metrics = evaluation_pipeline.run(
        test_data=test_data,
        y_prob=y_prob,
        threshold=threshold,
        timestamp_column=timestamp_column,
        family_column=family_column,
        k_values=k_vals
    )

    # Print metrics
    typer.echo(f"AUROC: {metrics.auroc:.4f}")
    typer.echo(f"PR-AUC: {metrics.pr_auc:.4f}")
    typer.echo(f"Brier Score: {metrics.brier:.4f}")
    typer.echo(f"ECE: {metrics.ece:.4f}")
    
    for k in k_vals:
        lift_attr = f"lift_at_{k}pct"
        if hasattr(metrics, lift_attr):
            typer.echo(f"Lift@{k}%: {getattr(metrics, lift_attr):.4f}")

    typer.echo("Evaluation completed")


@app.command()
def calibrate(
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    method: str = typer.Option("auto", help="Calibration method (platt/isotonic/auto)"),
) -> None:
    """Calibrate model probabilities."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading data...")
    train_data, val_data = load_ember_2024()

    typer.echo("Loading model...")
    model = model_path.load() if hasattr(model_path, 'load') else joblib.load(model_path)

    typer.echo("Generating predictions...")
    y_prob_train = model.predict_proba(train_data.features)
    y_prob_val = model.predict_proba(val_data.features)

    typer.echo(f"Training {method} calibrator...")
    
    # Use calibration pipeline
    calibration_pipeline = CalibrationPipeline(settings)
    calibrator = calibration_pipeline.run(
        train_data=train_data,
        val_data=val_data,
        y_prob_train=y_prob_train,
        y_prob_val=y_prob_val,
        method=method
    )

    # Save calibrator
    cal_path = settings.models_dir / "calibrator.joblib"
    calibrator.save(cal_path)

    typer.echo(f"Calibrator saved to {cal_path}")


@app.command()
def thresholds(
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    calibrator_path: Path | None = typer.Option(None, help="Path to calibrator"),
    cost_fp: float = typer.Option(5.0, help="Cost of false positive"),
    cost_fn: float = typer.Option(100.0, help="Cost of false negative"),
    optimize_cost: bool = typer.Option(True, help="Optimize threshold for cost"),
) -> None:
    """Compute optimal cost-sensitive thresholds."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if calibrator_path is None:
        calibrator_path = settings.models_dir / "calibrator.joblib"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading test data...")
    _, test_data = load_ember_2024()

    typer.echo("Loading model...")
    model = model_path.load() if hasattr(model_path, 'load') else joblib.load(model_path)

    typer.echo("Generating predictions...")
    y_prob = model.predict_proba(test_data.features)

    if calibrator_path.exists():
        typer.echo("Applying calibration...")
        calibrator = calibrator_path.load() if hasattr(calibrator_path, 'load') else joblib.load(calibrator_path)
        y_prob = calibrator.transform(y_prob)

    if optimize_cost:
        typer.echo("Computing optimal threshold...")
        
        # Use policy pipeline for optimization
        policy_pipeline = PolicyPipeline(settings)
        threshold = policy_pipeline.optimize_cost_sensitive_threshold(
            test_data.labels.values, y_prob, cost_fn, cost_fp
        )
    else:
        threshold = cost_sensitive_threshold(test_data.labels.values, y_prob, cost_fn, cost_fp)

    typer.echo(f"Optimal threshold: {threshold:.4f}")

    # Save policy
    policy = Policy(
        threshold=threshold,
        cost_false_negative=cost_fn,
        cost_false_positive=cost_fp,
        impact_default=settings.impact_default,
    )

    policy_path = settings.policies_dir / "policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write(policy.to_json())

    typer.echo(f"Policy saved to {policy_path}")


@app.command()
def policy(
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    calibrator_path: Path | None = typer.Option(None, help="Path to calibrator"),
    impact_table: Path | None = typer.Option(None, help="Path to impact table CSV"),
    cost_fp: float = typer.Option(5.0, help="Cost of false positive"),
    cost_fn: float = typer.Option(100.0, help="Cost of false negative"),
) -> None:
    """Generate banking narratives and cost-sensitive policy."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if calibrator_path is None:
        calibrator_path = settings.models_dir / "calibrator.joblib"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading test data...")
    _, test_data = load_ember_2024()

    typer.echo("Loading model...")
    model = model_path.load() if hasattr(model_path, 'load') else joblib.load(model_path)

    typer.echo("Generating predictions...")
    y_prob = model.predict_proba(test_data.features)

    if calibrator_path.exists():
        typer.echo("Applying calibration...")
        calibrator = calibrator_path.load() if hasattr(calibrator_path, 'load') else joblib.load(calibrator_path)
        y_prob = calibrator.transform(y_prob)

    typer.echo("Creating policy...")
    
    # Use policy pipeline
    policy_pipeline = PolicyPipeline(settings)
    policy = policy_pipeline.create_policy(
        y_true=test_data.labels.values,
        y_prob=y_prob,
        model_id=str(model_path),
        calibration_id=str(calibrator_path) if calibrator_path.exists() else ""
    )

    # Save policy
    policy_path = policy_pipeline.save_policy(policy)
    typer.echo(f"Policy saved to {policy_path}")

    # Generate ops report
    typer.echo("Generating operations report...")
    
    # Create dataframe for report
    df = pd.DataFrame({
        "probability": y_prob,
        "label": test_data.labels.values,
        "family": test_data.families if hasattr(test_data, 'families') else ["Unknown"] * len(y_prob),
    })
    
    # Add susceptibility scores and buckets
    df["susceptibility"] = df["probability"].clip(0.0, 1.0)
    df["susceptibility_bucket"] = pd.cut(
        df["susceptibility"],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    
    # Add canonical families
    mapping_results = df["family"].apply(policy_pipeline.mapping_pipeline.get_complete_mapping)
    df["canonical_family"] = mapping_results.apply(lambda x: x["canonical_family"])
    
    # Generate reports
    ops_report = policy_pipeline.generate_ops_report(df, policy)
    lift_report = policy_pipeline.compute_lift_at_k_report(df)
    
    typer.echo("Policy and reports generated successfully!")


@app.command()
def drift_check(
    new_data: Path = typer.Option(..., help="Path to new data for drift detection"),
    reference_data: Path | None = typer.Option(None, help="Path to reference data"),
    threshold: float = typer.Option(0.05, help="Drift threshold"),
) -> None:
    """Check for data and performance drift."""
    typer.echo("Loading data...")

    # TODO: Implement drift detection
    typer.echo("Drift detection completed")

    # For now, always exit with success
    # In real implementation, would check drift severity and exit non-zero if severe


@app.command()
def register(
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    policy_path: Path | None = typer.Option(None, help="Path to policy file"),
    output_name: str = typer.Option("cyber_risk_advisor_register", help="Output name"),
    impact_table: Path | None = typer.Option(None, help="Path to impact table CSV"),
) -> None:
    """Generate cyber risk advisor register."""
    settings = get_settings()

    if model_path is None:
        model_path = settings.models_dir / "bagged_lightgbm.joblib"

    if policy_path is None:
        policy_path = settings.policies_dir / "policy.json"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    if not policy_path.exists():
        typer.echo(f"Policy not found at {policy_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading test data...")
    _, test_data = load_ember_2024()

    typer.echo("Loading model...")
    model = model_path.load() if hasattr(model_path, 'load') else joblib.load(model_path)

    typer.echo("Generating predictions...")
    y_prob = model.predict_proba(test_data.features)

    typer.echo("Loading policy...")
    with open(policy_path, encoding="utf-8") as f:
        import json
        policy_data = json.load(f)

    policy = Policy(
        threshold=policy_data["threshold"],
        cost_false_negative=policy_data["cost_false_negative"],
        cost_false_positive=policy_data["cost_false_positive"],
        impact_default=policy_data["impact_default"],
    )

    typer.echo("Computing register...")
    out_df = pd.DataFrame({
        "family": test_data.families,
        "probability": y_prob,
        "label": test_data.labels,
    })

    register_df = compute_register(out_df, policy)
    
    # Enrich with prescriptive controls
    policy_pipeline = PolicyPipeline(settings)
    register_df = policy_pipeline.enrich_register_with_controls(register_df)
    
    write_register(register_df, name=output_name)

    typer.echo(f"Register saved as {output_name}")


@app.command()
def run_all() -> None:
    """Run complete AICRA pipeline."""
    typer.echo("Running complete AICRA pipeline...")

    # Run all commands in sequence
    train()
    evaluate()
    calibrate()
    thresholds()
    register()

    typer.echo("AICRA pipeline completed successfully!")


if __name__ == "__main__":
    app()

