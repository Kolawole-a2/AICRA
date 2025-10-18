"""Schema validation utilities for AICRA."""

import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError


def validate_input_schema(data: dict[str, Any]) -> None:
    """Validate input data against AICRA schema."""
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "input_schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, encoding='utf-8') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(data, schema)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e.message}") from e


def validate_json_file(file_path: Path) -> None:
    """Validate JSON file against AICRA schema."""
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    validate_input_schema(data)
