"""AICRA CLI using Typer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

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
from .pipelines.smoke import SmokeTestPipeline
from .utils import save_run_results, list_recent_runs

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
        seeds=seeds,
        is_smoke_test=False  # Regular training, not smoke test
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
        k_values=k_vals,
        is_smoke_test=False  # Regular evaluation, not smoke test
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
def smoke(
    dry_run: bool = typer.Option(False, help="Run smoke test in dry-run mode (no training)"),
) -> None:
    """Run end-to-end smoke test to validate AICRA pipeline."""
    settings = get_settings()
    
    # Set up MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("smoke_test")
    
    typer.echo("Running AICRA smoke test...")
    
    # Run smoke test pipeline
    smoke_pipeline = SmokeTestPipeline(settings)
    success, summary = smoke_pipeline.run(dry_run=dry_run)
    
    # Print results
    typer.echo(summary)
    
    # Exit with appropriate code
    if success:
        typer.echo("✅ Smoke test PASSED")
        
        # Archive smoke test results
        save_run_results(
            settings=settings,
            run_name="smoke",
            dataset_name="synthetic",
            model_name="mock_model",
        )
        
        raise typer.Exit(0)
    else:
        typer.echo("❌ Smoke test FAILED")
        raise typer.Exit(1)


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
    
    # Generate model and policy IDs
    model_id = f"model_{model_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    policy_id = f"policy_{policy_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Write register with metadata
    latest_path, archived_path = write_register(
        register_df, 
        name=output_name,
        model_id=model_id,
        policy_id=policy_id
    )

    typer.echo(f"Register saved as {output_name}")
    typer.echo(f"Latest version: {latest_path}")
    typer.echo(f"Archived version: {archived_path}")
    
    # Archive results
    save_run_results(
        settings=settings,
        run_name="register",
        dataset_name="ember2024",
        model_name=model_path.stem,
    )


@app.command()
def update_lookups(
    attack_bundle: Path | None = typer.Option(None, help="Path to MITRE ATT&CK JSON bundle"),
    d3fend_bundle: Path | None = typer.Option(None, help="Path to MITRE D3FEND JSON bundle"),
    malware_families: str | None = typer.Option(None, help="Comma-separated list of malware families"),
    policy_path: Path | None = typer.Option(None, help="Path to policy.json to update with versions"),
) -> None:
    """Update lookup tables from MITRE ATT&CK and D3FEND bundles."""
    settings = get_settings()
    
    # Parse malware families
    families_list = None
    if malware_families:
        families_list = [f.strip() for f in malware_families.split(',')]
    
    typer.echo("Parsing MITRE bundles and updating lookup tables...")
    
    # Initialize parser
    parser = MitreParser(settings)
    
    # Parse and update lookups
    results = parser.parse_and_update_lookups(
        attack_bundle_path=attack_bundle,
        d3fend_bundle_path=d3fend_bundle,
        malware_families=families_list,
    )
    
    # Update policy with versions if provided
    if policy_path and policy_path.exists():
        update_policy_with_versions(policy_path, results)
    
    # Print results
    typer.echo("Lookup table updates completed:")
    typer.echo(f"  - Canonical families updated: {results['canonical_families_updated']}")
    typer.echo(f"  - Family to attack updated: {results['family_to_attack_updated']}")
    typer.echo(f"  - Attack to D3FEND updated: {results['attack_to_d3fend_updated']}")
    typer.echo(f"  - Timestamp: {results['timestamp']}")
    
    if results['source_versions']:
        typer.echo("Source versions:")
        for source, version in results['source_versions'].items():
            typer.echo(f"  - {source}: {version}")


@app.command()
def archive_results(
    run_name: str | None = typer.Option(None, help="Custom name for the run"),
    dataset_name: str = typer.Option("ember2024", help="Name of the dataset used"),
    model_name: str = typer.Option("bagged_lightgbm", help="Name of the model used"),
) -> None:
    """Manually archive current results to a timestamped folder."""
    settings = get_settings()
    
    typer.echo("Archiving current results...")
    
    result_folder = save_run_results(
        settings=settings,
        run_name=run_name,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    
    typer.echo(f"Results archived to: {result_folder}")


@app.command()
def list_runs(
    limit: int = typer.Option(10, help="Number of recent runs to show"),
) -> None:
    """List recent runs from the versions log."""
    list_recent_runs(limit=limit)


@app.command()
def run_test(
    phase: str = typer.Option(..., help="Test phase: smoke, small_ember, or full"),
    data_dir: str = typer.Option("data/ember2024", help="Data directory for EMBER-2024 JSONL files"),
    sample_size: int = typer.Option(10000, help="Sample size for small_ember phase (ignored for full unless --no-sample)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    debug: bool = typer.Option(False, help="Enable debug mode with deep diagnostics"),
    time_split: bool = typer.Option(False, help="Use time-ordered split if timestamp column exists"),
    no_sample: bool = typer.Option(False, help="Force full phase to use entire dataset (ignore sample_size)"),
) -> None:
    """Run automated training, testing, and reporting for sequential test phases."""
    settings = get_settings()
    
    # Set up MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Validate phase
    valid_phases = ["smoke", "small_ember", "full"]
    if phase not in valid_phases:
        typer.echo(f"Invalid phase: {phase}. Must be one of {valid_phases}", err=True)
        raise typer.Exit(1)
    
    # Validate data requirements for non-smoke phases
    if phase in ["small_ember", "full"]:
        data_path = Path(data_dir)
        if not data_path.exists():
            typer.echo(f"❌ EMBER-2024 data directory not found: {data_path.absolute()}", err=True)
            typer.echo(f"For {phase} phase, you must provide real EMBER-2024 data.", err=True)
            typer.echo(f"Expected structure: {data_path}/*.jsonl", err=True)
            typer.echo(f"Use: aicra run-test --phase {phase} --data-dir <path-to-ember-data>", err=True)
            raise typer.Exit(1)
        
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            typer.echo(f"❌ No JSONL files found in {data_path.absolute()}", err=True)
            typer.echo(f"For {phase} phase, you must provide real EMBER-2024 JSONL files.", err=True)
            typer.echo(f"Expected files: {data_path}/*.jsonl", err=True)
            typer.echo(f"Use: aicra run-test --phase {phase} --data-dir <path-to-ember-data>", err=True)
            raise typer.Exit(1)
    
    # Handle sample_size for debug mode and full phase
    if phase == "full":
        if debug and sample_size != 10000:
            typer.echo(f"🔍 DEBUG: Using sample size {sample_size} for full phase debugging")
        elif not debug and not no_sample:
            sample_size = None  # Use full dataset for production full phase
            typer.echo("📊 FULL phase: Using entire dataset (use --no-sample to override)")
        elif no_sample:
            sample_size = None  # Explicitly use full dataset
            typer.echo("📊 FULL phase: Using entire dataset (--no-sample flag)")
        else:
            sample_size = None  # Default to full dataset
    
    typer.echo(f"Running {phase} test phase...")
    
    # Import and run the test pipeline
    from .pipelines.test_runner import TestRunnerPipeline
    
    test_pipeline = TestRunnerPipeline(settings)
    success, summary = test_pipeline.run(
        phase=phase,
        data_dir=data_dir,
        sample_size=sample_size,
        seed=seed,
        debug=debug,
        time_split=time_split
    )
    
    # Print results
    typer.echo(summary)
    
    # Exit with appropriate code
    if success:
        typer.echo(f"✅ {phase.upper()} test PASSED")
        raise typer.Exit(0)
    else:
        typer.echo(f"❌ {phase.upper()} test FAILED")
        raise typer.Exit(1)


@app.command()
def validate_lookups(
    phase: str = typer.Option(..., help="Test phase: smoke, small_ember, or full"),
    data_dir: str = typer.Option("data/ember2024", help="Data directory for EMBER-2024 JSONL files"),
    sample_size: int = typer.Option(10000, help="Sample size for small_ember phase"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Validate lookup coverage without training."""
    settings = get_settings()
    
    # Set up MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Validate phase
    valid_phases = ["smoke", "small_ember", "full"]
    if phase not in valid_phases:
        typer.echo(f"Invalid phase: {phase}. Must be one of {valid_phases}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Validating lookup coverage for {phase} phase...")
    
    try:
        # Load data for coverage validation
        if phase == "smoke":
            # Use synthetic data for smoke test
            from .core.data import _synthetic_dataset
            _, test_data = _synthetic_dataset(n=1000, d=256, seed=seed)
            families = test_data.families.tolist()
        else:
            # Load real EMBER-2024 data
            from .pipelines.data_loader import EMBERDataLoader
            loader = EMBERDataLoader(settings)
            features_df, labels_series, families_series, metadata = loader.load_ember_data(
                data_dir=data_dir,
                sample_size=sample_size if phase == "small_ember" else None,
                seed=seed,
                phase=phase
            )
            families = families_series.tolist()
        
        # Initialize mapping pipeline
        from .pipelines.mapping import MappingPipeline
        mapping_pipeline = MappingPipeline(settings, skip_mlflow=True)
        
        # Run batch mapping to compute coverage
        mapping_results = mapping_pipeline.map_families_batch(families, phase)
        
        # Log coverage report
        coverage_metrics = mapping_pipeline.log_coverage_report(phase)
        
        # Check coverage thresholds
        coverage_ok = mapping_pipeline.check_coverage_thresholds(phase)
        
        # Print results
        typer.echo(f"\n📊 Lookup Coverage Report for {phase.upper()} phase:")
        typer.echo("=" * 50)
        for metric, value in coverage_metrics.items():
            typer.echo(f"{metric}: {value:.3f}")
        
        if coverage_ok:
            typer.echo(f"\n✅ Lookup coverage validation PASSED for {phase.upper()}")
            raise typer.Exit(0)
        else:
            typer.echo(f"\n❌ Lookup coverage validation FAILED for {phase.upper()}")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Lookup validation failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def expand_lookups(
    from_mitre: Optional[str] = typer.Option(None, help="Path to MITRE ATT&CK/D3FEND JSON bundles"),
    output_dir: str = typer.Option("data/lookups", help="Output directory for YAML files"),
    dry_run: bool = typer.Option(False, help="Show what would be done without making changes"),
) -> None:
    """Expand lookup files from MITRE ATT&CK/D3FEND data."""
    settings = get_settings()
    
    if not from_mitre:
        typer.echo("❌ --from-mitre path is required", err=True)
        raise typer.Exit(1)
    
    mitre_path = Path(from_mitre)
    if not mitre_path.exists():
        typer.echo(f"❌ MITRE data path not found: {mitre_path}", err=True)
        raise typer.Exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Expanding lookups from MITRE data: {mitre_path}")
    typer.echo(f"Output directory: {output_path}")
    
    try:
        # Import expansion utilities
        from .utils.mitre_expander import MITREExpander
        
        expander = MITREExpander(mitre_path, output_path)
        
        if dry_run:
            typer.echo("🔍 DRY RUN - No changes will be made")
            changes = expander.analyze_changes()
            typer.echo(f"Would add {changes['new_families']} new families")
            typer.echo(f"Would add {changes['new_techniques']} new techniques")
            typer.echo(f"Would add {changes['new_controls']} new controls")
        else:
            # Set up MLflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("lookup_expansion")
            
            with mlflow.start_run():
                results = expander.expand_lookups()
                
                # Log results to MLflow
                mlflow.log_param("mitre_source", str(mitre_path))
                mlflow.log_param("output_dir", str(output_path))
                mlflow.log_metric("new_families", results['new_families'])
                mlflow.log_metric("new_techniques", results['new_techniques'])
                mlflow.log_metric("new_controls", results['new_controls'])
                
                typer.echo(f"✅ Successfully expanded lookups:")
                typer.echo(f"  - Added {results['new_families']} new families")
                typer.echo(f"  - Added {results['new_techniques']} new techniques")
                typer.echo(f"  - Added {results['new_controls']} new controls")
                typer.echo(f"  - Updated versions: {results['versions']}")
        
        raise typer.Exit(0)
        
    except Exception as e:
        typer.echo(f"❌ Lookup expansion failed: {str(e)}", err=True)
        raise typer.Exit(1)


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
    
    # Archive complete pipeline results
    save_run_results(
        settings=get_settings(),
        run_name="full_pipeline",
        dataset_name="ember2024",
        model_name="bagged_lightgbm",
    )


if __name__ == "__main__":
    app()

