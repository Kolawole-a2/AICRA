#!/usr/bin/env python3
import sys
from pathlib import Path

# Write to a test file
test_file = Path("test_output.txt")
with open(test_file, "w") as f:
    f.write("Python is working\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current directory: {Path.cwd()}\n")
    
    # Try importing
    try:
        from aicra.experiments.h3_evaluation import run_h3_evaluation
        f.write("Import successful\n")
    except Exception as e:
        f.write(f"Import failed: {e}\n")
        import traceback
        f.write(traceback.format_exc())

print("Test file created: test_output.txt")
