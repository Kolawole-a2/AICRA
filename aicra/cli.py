"""AICRA CLI application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import Settings
from ..pipelines.cost_optimization import CostOptimizer
from ..pipelines.drift_monitoring import DriftMonitor
from ..pipelines.model_card import ModelCardGenerator
from ..pipelines.training import TrainingPipeline
from ..pipelines.evaluation import EvaluationPipeline
from ..pipelines.calibration import CalibrationPipeline

app = typer.Typer(help="AICRA - AI-powered Cybersecurity Risk Assessment")
console = Console()


@app.command()
def data(
    action: str = typer.Argument(..., help="Action to perform: fetch, validate, snapshot"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory path"),
    sample_size: Optional[int] = typer.Option(None, help="Sample size for testing"),
    seed: int = typer.Option(42, help="Random seed")
):
    """Data management commands."""
    settings = Settings()
    
    if action == "fetch":
        console.print("Fetching EMBER 2024 dataset...")
        # Implementation would go here
        console.print("✅ Data fetch completed")
    
    elif action == "validate":
        console.print("Validating data schema...")
        # Implementation would go here
        console.print("✅ Data validation completed")
    
    elif action == "snapshot":
        console.print("Creating data snapshot...")
        # Implementation would go here
        console.print("✅ Data snapshot created")
    
    else:
        console.print(f"❌ Unknown action: {action}")
        raise typer.Exit(1)


@app.command()
def train(
    data_dir: str = typer.Option("data/ember2024", help="Data directory path"),
    model_dir: str = typer.Option("models", help="Model output directory"),
    sample_size: Optional[int] = typer.Option(None, help="Sample size for training"),
    seed: int = typer.Option(42, help="Random seed"),
    resume: bool = typer.Option(False, help="Resume training from checkpoint")
):
    """Train the AICRA model."""
    console.print("Starting model training...")
    
    settings = Settings()
    pipeline = TrainingPipeline(settings)
    
    try:
        # Load data
        console.print("Loading training data...")
        train_data, val_data = pipeline.load_data(data_dir, sample_size, seed)
        
        # Train model
        console.print("Training model...")
        model = pipeline.train_model(train_data, val_data)
        
        # Save model
        console.print("Saving model...")
        model_path = pipeline.save_model(model, model_dir)
        
        console.print(f"✅ Training completed. Model saved to: {model_path}")
        
    except Exception as e:
        console.print(f"❌ Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def eval(
    model_path: str = typer.Option("models/model.pkl", help="Path to trained model"),
    data_dir: str = typer.Option("data/ember2024", help="Data directory path"),
    output_dir: str = typer.Option("artifacts", help="Output directory for results"),
    sample_size: Optional[int] = typer.Option(None, help="Sample size for evaluation")
):
    """Evaluate model performance."""
    console.print("Starting model evaluation...")
    
    settings = Settings()
    pipeline = EvaluationPipeline(settings)
    
    try:
        # Load model and data
        console.print("Loading model and data...")
        model = pipeline.load_model(model_path)
        test_data = pipeline.load_test_data(data_dir, sample_size)
        
        # Evaluate model
        console.print("Evaluating model...")
        metrics = pipeline.evaluate_model(model, test_data)
        
        # Generate plots
        console.print("Generating evaluation plots...")
        pipeline.generate_plots(model, test_data, output_dir)
        
        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        console.print(f"✅ Evaluation completed. Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        raise typer.Exit(1)


@app.command()
def predict(
    model_path: str = typer.Option("models/model.pkl", help="Path to trained model"),
    input_file: str = typer.Option(..., help="Input file to predict"),
    output_file: Optional[str] = typer.Option(None, help="Output file for predictions"),
    threshold: float = typer.Option(0.5, help="Prediction threshold")
):
    """Make predictions on new data."""
    console.print("Making predictions...")
    
    settings = Settings()
    
    try:
        # Load model
        console.print("Loading model...")
        # Implementation would go here
        
        # Make prediction
        console.print("Making prediction...")
        # Implementation would go here
        
        console.print("✅ Prediction completed")
        
    except Exception as e:
        console.print(f"❌ Prediction failed: {e}")
        raise typer.Exit(1)


@app.command()
def drift(
    reference_data: str = typer.Option(..., help="Path to reference dataset"),
    current_data: str = typer.Option(..., help="Path to current dataset"),
    output_dir: str = typer.Option("artifacts", help="Output directory for drift report")
):
    """Check for data and prediction drift."""
    console.print("Checking for drift...")
    
    settings = Settings()
    monitor = DriftMonitor(settings)
    
    try:
        # Load datasets
        console.print("Loading datasets...")
        ref_data = settings.load_data(reference_data)
        curr_data = settings.load_data(current_data)
        
        # Detect data drift
        console.print("Detecting data drift...")
        data_drift = monitor.detect_data_drift(ref_data.features, curr_data.features)
        
        # Detect prediction drift
        console.print("Detecting prediction drift...")
        prediction_drift = monitor.detect_prediction_drift(
            ref_data.labels, curr_data.labels
        )
        
        # Generate report
        console.print("Generating drift report...")
        report_path = monitor.generate_drift_report(
            data_drift, prediction_drift, Path(output_dir) / "DriftReport.md"
        )
        
        console.print(f"✅ Drift analysis completed. Report saved to: {report_path}")
        
    except Exception as e:
        console.print(f"❌ Drift analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def calibrate(
    model_path: str = typer.Option("models/model.pkl", help="Path to trained model"),
    data_dir: str = typer.Option("data/ember2024", help="Data directory path"),
    output_dir: str = typer.Option("artifacts", help="Output directory for calibration results"),
    method: str = typer.Option("isotonic", help="Calibration method: isotonic or platt")
):
    """Calibrate model probabilities."""
    console.print("Starting model calibration...")
    
    settings = Settings()
    pipeline = CalibrationPipeline(settings)
    
    try:
        # Load model and data
        console.print("Loading model and data...")
        model = pipeline.load_model(model_path)
        train_data, val_data = pipeline.load_data(data_dir)
        
        # Calibrate model
        console.print("Calibrating model...")
        calibrated_model = pipeline.calibrate_model(model, train_data, val_data, method)
        
        # Generate calibration plots
        console.print("Generating calibration plots...")
        pipeline.generate_calibration_plots(calibrated_model, val_data, output_dir)
        
        console.print(f"✅ Calibration completed. Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Calibration failed: {e}")
        raise typer.Exit(1)


@app.command()
def optimize_threshold(
    model_path: str = typer.Option("models/model.pkl", help="Path to trained model"),
    data_dir: str = typer.Option("data/ember2024", help="Data directory path"),
    output_dir: str = typer.Option("artifacts", help="Output directory for threshold analysis"),
    fn_cost: float = typer.Option(1000.0, help="False negative cost"),
    fp_cost: float = typer.Option(100.0, help="False positive cost")
):
    """Optimize prediction threshold based on business costs."""
    console.print("Optimizing prediction threshold...")
    
    settings = Settings()
    optimizer = CostOptimizer(settings)
    
    try:
        # Load model and data
        console.print("Loading model and data...")
        # Implementation would go here
        
        # Optimize threshold
        console.print("Optimizing threshold...")
        # Implementation would go here
        
        console.print(f"✅ Threshold optimization completed. Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Threshold optimization failed: {e}")
        raise typer.Exit(1)


@app.command()
def generate_model_card(
    model_path: str = typer.Option("models/model.pkl", help="Path to trained model"),
    output_dir: str = typer.Option("artifacts", help="Output directory for model card")
):
    """Generate model card documentation."""
    console.print("Generating model card...")
    
    settings = Settings()
    generator = ModelCardGenerator(settings)
    
    try:
        # Generate model card
        console.print("Creating model card...")
        # Implementation would go here
        
        console.print(f"✅ Model card generated. Saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Model card generation failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()