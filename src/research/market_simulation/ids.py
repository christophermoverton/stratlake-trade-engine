from __future__ import annotations

import hashlib
from typing import Any, Mapping

from src.research.registry import canonicalize_value, serialize_canonical_json


def generate_scenario_id(
    *,
    simulation_name: str,
    scenario_name: str,
    simulation_type: str,
    seed: int,
    path_count: int,
    source_window_start: str | None = None,
    source_window_end: str | None = None,
    method_config: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "simulation_name": simulation_name,
        "scenario_name": scenario_name,
        "simulation_type": simulation_type,
        "seed": seed,
        "path_count": path_count,
        "source_window_start": source_window_start,
        "source_window_end": source_window_end,
        "method_config": canonicalize_value(dict(method_config or {})),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_slugify(scenario_name)}_{digest}"


def generate_path_id(
    *,
    scenario_id: str,
    path_index: int,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "path_index": path_index,
        "seed": seed,
        "metadata": canonicalize_value(dict(metadata or {})),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{scenario_id}_path_{path_index:06d}_{digest}"


def _slugify(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "scenario"
