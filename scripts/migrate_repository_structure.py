"""
Repository Restructure Migration Script

⚠️ WARNING: This script performs a MAJOR restructuring of the codebase.
It will move files and update imports. Use with caution.

Run with --dry-run first to see what changes would be made.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Mapping of old paths to new paths
FILE_MOVEMENTS: Dict[str, str] = {
    # Data preparation
    "aicra/pipelines/features_pe.py": "aicra/data_prep/pe_features.py",
    "aicra/utils/data_loader.py": "aicra/data_prep/data_loader.py",
    
    # Models
    "aicra/pipelines/training.py": "aicra/models/training.py",
    
    # Calibration
    "aicra/pipelines/calibration.py": "aicra/calibration/pipeline.py",
    
    # Mapping
    "aicra/pipelines/mapping.py": "aicra/mapping/pipeline.py",
    "aicra/mappings/learned_ml_mapping.py": "aicra/mapping/learned.py",
    
    # Evaluation
    "aicra/core/evaluation.py": "aicra/evaluation/metrics.py",
    "aicra/core/benchmarks.py": "aicra/evaluation/benchmarks.py",
    
    # Experiments
    "aicra/experiments/h1_classification.py": "experiments/h1_main/run.py",
    "aicra/experiments/h2_calibration_thresholds.py": "experiments/h2_calibration_transfer/run.py",
    "aicra/experiments/h3_evaluation.py": "experiments/h3_mapping_comparison/run.py",
}

# Import path mappings (old -> new)
IMPORT_MAPPINGS: Dict[str, str] = {
    "aicra.pipelines.features_pe": "aicra.data_prep.pe_features",
    "aicra.utils.data_loader": "aicra.data_prep.data_loader",
    "aicra.pipelines.training": "aicra.models.training",
    "aicra.pipelines.calibration": "aicra.calibration.pipeline",
    "aicra.pipelines.mapping": "aicra.mapping.pipeline",
    "aicra.mappings.learned_ml_mapping": "aicra.mapping.learned",
    "aicra.core.evaluation": "aicra.evaluation.metrics",
    "aicra.core.benchmarks": "aicra.evaluation.benchmarks",
    "aicra.experiments.h1_classification": "experiments.h1_main.run",
    "aicra.experiments.h2_calibration_thresholds": "experiments.h2_calibration_transfer.run",
    "aicra.experiments.h3_evaluation": "experiments.h3_mapping_comparison.run",
}


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in the repository."""
    python_files = []
    for path in root.rglob("*.py"):
        # Skip virtual environments and build directories
        if any(skip in str(path) for skip in [".venv", "venv", "__pycache__", ".git", "build", "dist"]):
            continue
        python_files.append(path)
    return python_files


def update_imports_in_file(file_path: Path, dry_run: bool = False) -> List[str]:
    """
    Update imports in a Python file.
    
    Returns list of changes made.
    """
    changes = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        
        # Update each import mapping
        for old_import, new_import in IMPORT_MAPPINGS.items():
            # Pattern for "from old_import import ..."
            pattern_from = rf"from\s+{re.escape(old_import)}\s+import"
            replacement_from = f"from {new_import} import"
            
            if re.search(pattern_from, content):
                content = re.sub(pattern_from, replacement_from, content)
                changes.append(f"Updated 'from {old_import} import' to 'from {new_import} import'")
            
            # Pattern for "import old_import"
            pattern_import = rf"import\s+{re.escape(old_import)}\b"
            replacement_import = f"import {new_import}"
            
            if re.search(pattern_import, content):
                content = re.sub(pattern_import, replacement_import, content)
                changes.append(f"Updated 'import {old_import}' to 'import {new_import}'")
        
        # Write updated content if changes were made
        if content != original_content and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            changes.append(f"File updated: {file_path}")
        elif content != original_content:
            changes.append(f"[DRY RUN] Would update: {file_path}")
    
    except Exception as e:
        changes.append(f"Error processing {file_path}: {e}")
    
    return changes


def migrate_files(dry_run: bool = False) -> Dict[str, List[str]]:
    """
    Migrate files to new structure.
    
    Returns dictionary of file_path -> list of changes.
    """
    repo_root = Path.cwd()
    all_changes: Dict[str, List[str]] = {}
    
    # Create new directories
    new_dirs = [
        "aicra/data_prep",
        "aicra/models",
        "aicra/calibration",
        "aicra/mapping",
        "aicra/evaluation",
        "experiments/h1_main",
        "experiments/h2_calibration_transfer",
        "experiments/h3_mapping_comparison",
        "artifacts/metrics/h1",
        "artifacts/metrics/h2",
        "artifacts/metrics/h3",
        "artifacts/benchmarks",
        "artifacts/improvement_reports",
        "artifacts/risk_registers",
        "artifacts/policies",
        "artifacts/models",
        "docs",
    ]
    
    for dir_path in new_dirs:
        full_path = repo_root / dir_path
        if not dry_run:
            full_path.mkdir(parents=True, exist_ok=True)
            all_changes[str(full_path)] = [f"Created directory: {dir_path}"]
        else:
            all_changes[str(full_path)] = [f"[DRY RUN] Would create directory: {dir_path}"]
    
    # Move files
    for old_path_str, new_path_str in FILE_MOVEMENTS.items():
        old_path = repo_root / old_path_str
        new_path = repo_root / new_path_str
        
        if old_path.exists():
            if not dry_run:
                # Create parent directory
                new_path.parent.mkdir(parents=True, exist_ok=True)
                # Move file
                shutil.move(str(old_path), str(new_path))
                all_changes[str(new_path)] = [f"Moved from {old_path_str} to {new_path_str}"]
            else:
                all_changes[str(new_path)] = [f"[DRY RUN] Would move {old_path_str} to {new_path_str}"]
        else:
            all_changes[str(new_path)] = [f"Source file not found: {old_path_str}"]
    
    # Update imports in all Python files
    python_files = find_python_files(repo_root)
    for py_file in python_files:
        changes = update_imports_in_file(py_file, dry_run=dry_run)
        if changes:
            all_changes[str(py_file)] = changes
    
    return all_changes


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate repository to new structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm migration (required for actual migration)"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.confirm:
        print("⚠️  WARNING: This will restructure the repository!")
        print("Use --dry-run to see what would change, or --confirm to proceed.")
        return
    
    print("=" * 80)
    print("Repository Restructure Migration")
    print("=" * 80)
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No changes will be made\n")
    else:
        print("\n[MIGRATION MODE] - Files will be moved and imports updated\n")
    
    changes = migrate_files(dry_run=args.dry_run)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Migration Summary")
    print("=" * 80)
    
    total_files = len([c for c in changes.values() if c])
    print(f"\nTotal files affected: {total_files}")
    
    if args.dry_run:
        print("\nRun without --dry-run and with --confirm to apply changes.")
    else:
        print("\nMigration complete!")
        print("\n⚠️  IMPORTANT: Review all changes and test imports before committing.")


if __name__ == "__main__":
    main()

