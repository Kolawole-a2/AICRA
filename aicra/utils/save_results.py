"""Automatic versioned result archiving for AICRA runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Settings


def save_run_results(
    settings: Settings,
    run_name: Optional[str] = None,
    dataset_name: str = "ember2024",
    model_name: str = "bagged_lightgbm",
    additional_files: Optional[List[Path]] = None,
) -> Path:
    """
    Save run results to a timestamped folder and update versions log.
    
    Args:
        settings: AICRA settings object
        run_name: Optional custom name for the run (defaults to timestamp)
        dataset_name: Name of the dataset used
        model_name: Name of the model used
        additional_files: Additional files to copy beyond the standard set
        
    Returns:
        Path to the created results folder
    """
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamped folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    if run_name:
        folder_name = f"run_{timestamp}_{run_name}"
    else:
        folder_name = f"run_{timestamp}"
    
    run_folder = results_dir / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Files to copy from artifacts/
    artifact_files = [
        "metrics.json",
        "policy.json", 
        "risk_register.csv",
        "roc.png",
        "pr.png",
        "reliability.png",
        "confusion_at_ops.png",
        "lift_curve.png",  # Optional, may not exist
    ]
    
    # Files to copy from models/
    model_files = [
        "bagged_lightgbm.joblib",
        "calibrator.joblib",
    ]
    
    # Additional context files
    context_files = [
        "data/impact.csv",
    ]
    
    copied_files = []
    
    # Copy artifact files
    for filename in artifact_files:
        src_path = settings.artifacts_dir / filename
        if src_path.exists():
            dst_path = run_folder / filename
            shutil.copy2(src_path, dst_path)
            copied_files.append(str(dst_path))
    
    # Copy model files
    for filename in model_files:
        src_path = settings.models_dir / filename
        if src_path.exists():
            dst_path = run_folder / filename
            shutil.copy2(src_path, dst_path)
            copied_files.append(str(dst_path))
    
    # Copy context files
    for filepath in context_files:
        src_path = Path(filepath)
        if src_path.exists():
            dst_path = run_folder / src_path.name
            shutil.copy2(src_path, dst_path)
            copied_files.append(str(dst_path))
    
    # Copy additional files if provided
    if additional_files:
        for src_path in additional_files:
            if src_path.exists():
                dst_path = run_folder / src_path.name
                shutil.copy2(src_path, dst_path)
                copied_files.append(str(dst_path))
    
    # Update versions log
    _update_versions_log(run_folder, dataset_name, model_name, settings)
    
    # Print summary
    print(f"\n✅ Results archived at {run_folder}/")
    print(f"   Files copied: {len(copied_files)}")
    for file_path in copied_files:
        print(f"   - {Path(file_path).name}")
    
    return run_folder


def _update_versions_log(
    run_folder: Path, 
    dataset_name: str, 
    model_name: str, 
    settings: Settings
) -> None:
    """Update the versions log CSV with run information."""
    versions_log_path = Path("results") / "versions_log.csv"
    
    # Try to load metrics.json to get performance metrics
    metrics_path = run_folder / "metrics.json"
    metrics_data = {}
    
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Extract metrics with defaults
    auroc = metrics_data.get("auroc", 0.0)
    pr_auc = metrics_data.get("pr_auc", 0.0)
    ece = metrics_data.get("ece", 0.0)
    lift_at_5 = metrics_data.get("lift_at_5pct", 0.0)
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_folder.name,
        "dataset": dataset_name,
        "model": model_name,
        "AUROC": round(auroc, 4),
        "PR-AUC": round(pr_auc, 4),
        "ECE": round(ece, 4),
        "Lift@5": round(lift_at_5, 4),
        "folder_path": str(run_folder),
    }
    
    # Append to CSV
    df = pd.DataFrame([log_entry])
    
    if versions_log_path.exists():
        # Append to existing file
        df.to_csv(versions_log_path, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(versions_log_path, index=False)
    
    print(f"   Updated versions log: {versions_log_path}")


def list_recent_runs(limit: int = 10) -> pd.DataFrame:
    """List recent runs from the versions log."""
    versions_log_path = Path("results") / "versions_log.csv"
    
    if not versions_log_path.exists():
        print("No runs found in versions log.")
        return pd.DataFrame()
    
    df = pd.read_csv(versions_log_path)
    recent_runs = df.tail(limit)
    
    print(f"\n📊 Recent {len(recent_runs)} runs:")
    print(recent_runs.to_string(index=False))
    
    return recent_runs


def get_run_summary(run_folder: Path) -> Dict[str, Any]:
    """Get a summary of a specific run from its folder."""
    summary = {
        "folder": str(run_folder),
        "timestamp": run_folder.name,
        "files": [],
        "metrics": {},
    }
    
    # List files in the folder
    if run_folder.exists():
        summary["files"] = [f.name for f in run_folder.iterdir() if f.is_file()]
    
    # Load metrics if available
    metrics_path = run_folder / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                summary["metrics"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    return summary
