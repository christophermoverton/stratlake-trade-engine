from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class ContractValidationError(ValueError):
    """Raised when a JSON payload violates a static artifact contract."""


def validate_json(data: dict[str, Any], schema_path: Path) -> None:
    schema_file = Path(schema_path)
    schema_name = schema_file.name
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: (
            _format_instance_path(error.absolute_path),
            _format_schema_path(error.absolute_schema_path),
            error.message,
        ),
    )
    if not errors:
        return

    error = errors[0]
    instance_path = _format_instance_path(error.absolute_path)
    raise ContractValidationError(
        f"JSON contract validation failed for '{schema_name}' at '{instance_path}': {error.message}"
    )


def _format_instance_path(path: Any) -> str:
    parts = list(path)
    if not parts:
        return "$"

    formatted = "$"
    for part in parts:
        if isinstance(part, int):
            formatted += f"[{part}]"
        else:
            formatted += f".{part}"
    return formatted


def _format_schema_path(path: Any) -> str:
    return "/".join(str(part) for part in path)
