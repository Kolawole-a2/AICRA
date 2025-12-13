#!/usr/bin/env python3
"""
CI Guardrails: Prevents common repo-breaking mistakes.

Scans git-tracked files for:
- Large files (> 50 MB)
- Forbidden paths (ember2024_real/ anywhere, or data/ember2024_real/)
- Forbidden extensions (.jsonl)
- Forbidden code patterns (allow_pickle=True)

Exits with non-zero status if violations found.
"""

import subprocess
import sys
from pathlib import Path

# Constants
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Allowed paths (exceptions to forbidden paths)
ALLOWED_PATHS = [
    "data/lookups/",
    "data/mappings/",
    "data/ember/README.md",
]

# Allowed file patterns (exceptions to forbidden extensions)
ALLOWED_FILE_PATTERNS = [
    "data/",  # Allow data/*.csv files
]

# Forbidden paths - must match exactly or be subdirectories
FORBIDDEN_PATHS = [
    "ember2024_real/",  # Anywhere in path
    "data/ember2024_real/",  # Specific path
]

FORBIDDEN_EXTENSIONS = [
    ".jsonl",
]

FORBIDDEN_PATTERNS = [
    ("allow_pickle=True", "Unsafe pickle loading enabled"),
    ("allow_pickle = True", "Unsafe pickle loading enabled"),
    ("allow_pickle= True", "Unsafe pickle loading enabled"),
    ("allow_pickle =True", "Unsafe pickle loading enabled"),
]

REQUIRED_GITIGNORE_PATTERNS = [
    "data/",
    "*.jsonl",
]


def get_tracked_files() -> list[str]:
    """Get list of all git-tracked files."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to get tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "ERROR: git not found. This script must be run in a git repository.",
            file=sys.stderr,
        )
        sys.exit(1)


def check_file_size(filepath: Path) -> tuple[bool, str]:
    """Check if file exceeds size limit."""
    try:
        size = filepath.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            size_mb = size / (1024 * 1024)
            return False, f"File exceeds {MAX_FILE_SIZE_MB} MB limit: {size_mb:.2f} MB"
    except (OSError, FileNotFoundError):
        return True, ""  # File doesn't exist or can't be read (might be deleted)
    return True, ""


def check_forbidden_paths(filepath: str) -> tuple[bool, str]:
    """Check if file path matches forbidden patterns, excluding allowed paths."""
    normalized = filepath.replace("\\", "/")  # Normalize Windows paths

    # Check if path is in allowed list first (exact prefix match or contains)
    for allowed in ALLOWED_PATHS:
        if normalized.startswith(allowed):
            return True, ""  # Allowed path, skip check

    # Check if it's a CSV file in data/ (allowed)
    if normalized.startswith("data/") and normalized.endswith(".csv"):
        return True, ""

    # Check if it's in data/lookups/ or data/mappings/ (allowed subdirectories)
    if normalized.startswith("data/lookups/") or normalized.startswith(
        "data/mappings/"
    ):
        return True, ""

    # Now check forbidden paths - ONLY fail for ember2024_real/ paths
    for forbidden in FORBIDDEN_PATHS:
        if forbidden in normalized:
            return False, f"Forbidden path detected: {forbidden}"
    return True, ""


def check_forbidden_extensions(filepath: str) -> tuple[bool, str]:
    """Check if file has forbidden extension."""
    for ext in FORBIDDEN_EXTENSIONS:
        if filepath.endswith(ext):
            return False, f"Forbidden extension: {ext}"
    return True, ""


def check_code_patterns(filepath: Path) -> list[tuple[int, str, str]]:
    """Check for forbidden code patterns in Python files."""
    violations = []
    if not filepath.suffix == ".py":
        return violations

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip docstrings and comments
                stripped = line.strip()
                if (
                    stripped.startswith('"""')
                    or stripped.startswith("'''")
                    or stripped.startswith("#")
                ):
                    continue
                # Skip if pattern is in a string literal (docstring content)
                if '"""' in line or "'''" in line:
                    continue
                for pattern, message in FORBIDDEN_PATTERNS:
                    if pattern in line:
                        violations.append((line_num, pattern, message))
    except (OSError, UnicodeDecodeError):
        pass  # Skip files that can't be read
    return violations


def check_gitignore() -> tuple[bool, list[str]]:
    """Check that .gitignore includes required patterns."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        return False, [".gitignore file not found"]

    try:
        content = gitignore_path.read_text(encoding="utf-8")
        missing = []
        for pattern in REQUIRED_GITIGNORE_PATTERNS:
            if pattern not in content:
                missing.append(f"Missing .gitignore rule: {pattern}")
        return len(missing) == 0, missing
    except OSError:
        return False, ["Could not read .gitignore"]


def main() -> int:
    """Run all guardrail checks."""
    print("=" * 80)
    print("CI GUARDRAILS: Checking for repo-breaking violations")
    print("=" * 80)
    print()

    errors = []

    # Check .gitignore first
    print("[1/5] Checking .gitignore rules...")
    gitignore_ok, gitignore_errors = check_gitignore()
    if not gitignore_ok:
        errors.extend(gitignore_errors)
        for err in gitignore_errors:
            print(f"  [FAIL] {err}")
    else:
        print("  [PASS] .gitignore contains required patterns")
    print()

    # Get tracked files
    print("[2/5] Scanning git-tracked files...")
    tracked_files = get_tracked_files()
    print(f"  Found {len(tracked_files)} tracked files")
    print()

    # Check file sizes
    print(f"[3/5] Checking file sizes (max {MAX_FILE_SIZE_MB} MB)...")
    large_files = []
    for filepath_str in tracked_files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            continue  # File might be deleted but still tracked
        ok, msg = check_file_size(filepath)
        if not ok:
            large_files.append((filepath_str, msg))
            errors.append(f"{filepath_str}: {msg}")

    if large_files:
        print(
            f"  [FAIL] Found {len(large_files)} file(s) exceeding {MAX_FILE_SIZE_MB} MB:"
        )
        for filepath, msg in large_files:
            print(f"    - {filepath}: {msg}")
    else:
        print(f"  [PASS] No files exceed {MAX_FILE_SIZE_MB} MB")
    print()

    # Check forbidden paths
    print("[4/5] Checking for forbidden paths...")
    forbidden_path_files = []
    for filepath_str in tracked_files:
        ok, msg = check_forbidden_paths(filepath_str)
        if not ok:
            forbidden_path_files.append((filepath_str, msg))
            errors.append(f"{filepath_str}: {msg}")

    if forbidden_path_files:
        print(
            f"  [FAIL] Found {len(forbidden_path_files)} file(s) with forbidden paths:"
        )
        for filepath, msg in forbidden_path_files:
            print(f"    - {filepath}: {msg}")
    else:
        print("  [PASS] No forbidden paths detected")
    print()

    # Check forbidden extensions
    print("[5/5] Checking for forbidden file extensions...")
    forbidden_ext_files = []
    for filepath_str in tracked_files:
        ok, msg = check_forbidden_extensions(filepath_str)
        if not ok:
            forbidden_ext_files.append((filepath_str, msg))
            errors.append(f"{filepath_str}: {msg}")

    if forbidden_ext_files:
        print(
            f"  [FAIL] Found {len(forbidden_ext_files)} file(s) with forbidden extensions:"
        )
        for filepath, msg in forbidden_ext_files:
            print(f"    - {filepath}: {msg}")
    else:
        print("  [PASS] No forbidden extensions detected")
    print()

    # Check code patterns
    print("[6/6] Checking for unsafe code patterns...")
    pattern_violations = []
    for filepath_str in tracked_files:
        filepath = Path(filepath_str)
        # Skip the guardrails script itself (it documents what it checks)
        if filepath_str == "scripts/ci_guardrails.py":
            continue
        violations = check_code_patterns(filepath)
        if violations:
            for line_num, pattern, message in violations:
                pattern_violations.append((filepath_str, line_num, pattern, message))
                errors.append(
                    f"{filepath_str}:{line_num}: {message} (found '{pattern}')"
                )

    if pattern_violations:
        for filepath, line_num, pattern, message in pattern_violations:
            print(f"  [WARN] {filepath}:{line_num}: {message}")
            print(f"      Found: '{pattern}'")
    else:
        print("  [PASS] No unsafe code patterns detected")
    print()

    # Summary
    print("=" * 80)
    if errors:
        print(f"[FAILED] Found {len(errors)} violation(s)")
        print()
        print("REMEDIATION:")
        print("1. Remove large files from Git: git rm --cached <file>")
        print("2. Remove forbidden paths/extensions: git rm --cached <file>")
        print("3. Update .gitignore to exclude data/ and *.jsonl")
        print(
            "4. Fix unsafe code patterns (use allow_pickle=False or safe alternatives)"
        )
        print("5. Commit fixes and push again")
        print()
        print("Violations:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1
    else:
        print("[PASSED] All guardrails checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
