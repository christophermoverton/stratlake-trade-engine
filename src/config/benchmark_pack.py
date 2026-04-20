from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class BenchmarkPackConfigError(ValueError):
    """Raised when a benchmark-pack configuration is malformed."""


_ROOT_KEYS = frozenset({"pack_id", "description", "dataset", "campaign", "batching", "outputs"})
_DATASET_KEYS = frozenset(
    {"generator", "features_root", "dataset_name", "start_date", "periods", "frequency", "symbols"}
)
_CAMPAIGN_KEYS = frozenset({"config_path", "overrides"})
_BATCHING_KEYS = frozenset({"batch_size"})
_OUTPUT_KEYS = frozenset({"root", "inventory_excludes"})
_SUPPORTED_DATASET_GENERATORS = frozenset({"synthetic_features_daily"})


@dataclass(frozen=True)
class BenchmarkDatasetConfig:
    generator: str
    features_root: str
    dataset_name: str
    start_date: str
    periods: int
    frequency: str
    symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "features_root": self.features_root,
            "dataset_name": self.dataset_name,
            "start_date": self.start_date,
            "periods": self.periods,
            "frequency": self.frequency,
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True)
class BenchmarkCampaignConfig:
    config_path: str
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "overrides": _canonicalize_value(dict(self.overrides)),
        }


@dataclass(frozen=True)
class BenchmarkBatchingConfig:
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}


@dataclass(frozen=True)
class BenchmarkOutputsConfig:
    root: str | None
    inventory_excludes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "inventory_excludes": list(self.inventory_excludes),
        }


@dataclass(frozen=True)
class BenchmarkPackConfig:
    pack_id: str
    description: str | None
    dataset: BenchmarkDatasetConfig
    campaign: BenchmarkCampaignConfig
    batching: BenchmarkBatchingConfig
    outputs: BenchmarkOutputsConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "description": self.description,
            "dataset": self.dataset.to_dict(),
            "campaign": self.campaign.to_dict(),
            "batching": self.batching.to_dict(),
            "outputs": self.outputs.to_dict(),
        }


def load_benchmark_pack_config(path: Path) -> BenchmarkPackConfig:
    if not path.exists():
        raise BenchmarkPackConfigError(f"Benchmark-pack config does not exist: {path.as_posix()}")

    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise BenchmarkPackConfigError(
                f"Unsupported benchmark-pack config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack config must contain a top-level mapping.")
    return BenchmarkPackConfig.from_mapping(payload, config_dir=path.parent)


@classmethod
def _from_mapping(
    cls,
    payload: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
) -> BenchmarkPackConfig:
    unknown_keys = sorted(set(payload) - _ROOT_KEYS)
    if unknown_keys:
        raise BenchmarkPackConfigError(
            f"Benchmark-pack config contains unsupported keys: {unknown_keys}."
        )

    pack_id = _normalize_required_string(payload.get("pack_id"), field_name="pack_id")
    description = _normalize_optional_string(payload.get("description"), field_name="description")
    dataset = _resolve_dataset(payload.get("dataset"), config_dir=config_dir)
    campaign = _resolve_campaign(payload.get("campaign"), config_dir=config_dir)
    batching = _resolve_batching(payload.get("batching"))
    outputs = _resolve_outputs(payload.get("outputs"), config_dir=config_dir)
    return BenchmarkPackConfig(
        pack_id=pack_id,
        description=description,
        dataset=dataset,
        campaign=campaign,
        batching=batching,
        outputs=outputs,
    )


BenchmarkPackConfig.from_mapping = _from_mapping  # type: ignore[attr-defined]


def _resolve_dataset(payload: Any, *, config_dir: Path | None) -> BenchmarkDatasetConfig:
    if not isinstance(payload, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack field 'dataset' must be a mapping.")
    unknown_keys = sorted(set(payload) - _DATASET_KEYS)
    if unknown_keys:
        raise BenchmarkPackConfigError(
            f"Benchmark-pack field 'dataset' contains unsupported keys: {unknown_keys}."
        )

    generator = _normalize_required_string(payload.get("generator"), field_name="dataset.generator")
    if generator not in _SUPPORTED_DATASET_GENERATORS:
        expected = ", ".join(sorted(_SUPPORTED_DATASET_GENERATORS))
        raise BenchmarkPackConfigError(
            f"Benchmark-pack field 'dataset.generator' must be one of: {expected}."
        )

    features_root = _normalize_relative_path_string(
        payload.get("features_root", "workspace"),
        field_name="dataset.features_root",
    )
    dataset_name = _normalize_required_string(
        payload.get("dataset_name", "features_daily"),
        field_name="dataset.dataset_name",
    )
    start_date = _normalize_required_string(
        payload.get("start_date"),
        field_name="dataset.start_date",
    )
    periods = _resolve_positive_int(payload.get("periods"), field_name="dataset.periods")
    frequency = _normalize_required_string(
        payload.get("frequency", "D"),
        field_name="dataset.frequency",
    )
    symbols = _normalize_string_sequence(payload.get("symbols"), field_name="dataset.symbols")
    if not symbols:
        raise BenchmarkPackConfigError("Benchmark-pack field 'dataset.symbols' must not be empty.")
    return BenchmarkDatasetConfig(
        generator=generator,
        features_root=features_root,
        dataset_name=dataset_name,
        start_date=start_date,
        periods=periods,
        frequency=frequency,
        symbols=symbols,
    )


def _resolve_campaign(payload: Any, *, config_dir: Path | None) -> BenchmarkCampaignConfig:
    if not isinstance(payload, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack field 'campaign' must be a mapping.")
    unknown_keys = sorted(set(payload) - _CAMPAIGN_KEYS)
    if unknown_keys:
        raise BenchmarkPackConfigError(
            f"Benchmark-pack field 'campaign' contains unsupported keys: {unknown_keys}."
        )
    config_path = _normalize_path_string(
        payload.get("config_path"),
        field_name="campaign.config_path",
        config_dir=config_dir,
    )
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack field 'campaign.overrides' must be a mapping.")
    return BenchmarkCampaignConfig(
        config_path=config_path,
        overrides=_canonicalize_value(dict(overrides)),
    )


def _resolve_batching(payload: Any) -> BenchmarkBatchingConfig:
    if not isinstance(payload, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack field 'batching' must be a mapping.")
    unknown_keys = sorted(set(payload) - _BATCHING_KEYS)
    if unknown_keys:
        raise BenchmarkPackConfigError(
            f"Benchmark-pack field 'batching' contains unsupported keys: {unknown_keys}."
        )
    return BenchmarkBatchingConfig(
        batch_size=_resolve_positive_int(payload.get("batch_size"), field_name="batching.batch_size")
    )


def _resolve_outputs(payload: Any, *, config_dir: Path | None) -> BenchmarkOutputsConfig:
    if payload is None:
        return BenchmarkOutputsConfig(
            root=None,
            inventory_excludes=(
                "checkpoint.json",
                "manifest.json",
                "summary.json",
                "comparisons/*",
                "inventory.json",
                "snapshots/*",
            ),
        )
    if not isinstance(payload, Mapping):
        raise BenchmarkPackConfigError("Benchmark-pack field 'outputs' must be a mapping.")
    unknown_keys = sorted(set(payload) - _OUTPUT_KEYS)
    if unknown_keys:
        raise BenchmarkPackConfigError(
            f"Benchmark-pack field 'outputs' contains unsupported keys: {unknown_keys}."
        )

    root_value = payload.get("root")
    root = None
    if root_value is not None:
        root = _normalize_path_string(root_value, field_name="outputs.root", config_dir=config_dir)

    excludes = payload.get(
        "inventory_excludes",
        (
            "checkpoint.json",
            "manifest.json",
            "summary.json",
            "comparisons/*",
            "inventory.json",
            "snapshots/*",
        ),
    )
    if not isinstance(excludes, (list, tuple)):
        raise BenchmarkPackConfigError(
            "Benchmark-pack field 'outputs.inventory_excludes' must be a sequence of glob patterns."
        )
    normalized_excludes = tuple(
        _normalize_required_string(item, field_name="outputs.inventory_excludes[]")
        for item in excludes
    )
    return BenchmarkOutputsConfig(root=root, inventory_excludes=normalized_excludes)


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkPackConfigError(f"Benchmark-pack field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise BenchmarkPackConfigError(f"Benchmark-pack field '{field_name}' must be a non-empty string.")
    return normalized


def _normalize_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_required_string(value, field_name=field_name)


def _normalize_path_string(value: Any, *, field_name: str, config_dir: Path | None) -> str:
    normalized = _normalize_required_string(value, field_name=field_name)
    path = Path(normalized)
    if not path.is_absolute() and config_dir is not None:
        path = (config_dir / path).resolve()
    return path.as_posix()


def _normalize_relative_path_string(value: Any, *, field_name: str) -> str:
    normalized = _normalize_required_string(value, field_name=field_name)
    return Path(normalized).as_posix()


def _resolve_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkPackConfigError(f"Benchmark-pack field '{field_name}' must be a positive integer.")
    return value


def _normalize_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise BenchmarkPackConfigError(f"Benchmark-pack field '{field_name}' must be a sequence.")
    return tuple(
        item
        for item in (
            _normalize_required_string(candidate, field_name=field_name)
            for candidate in value
        )
    )


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_value(item) for item in value]
    return value


__all__ = [
    "BenchmarkBatchingConfig",
    "BenchmarkCampaignConfig",
    "BenchmarkDatasetConfig",
    "BenchmarkOutputsConfig",
    "BenchmarkPackConfig",
    "BenchmarkPackConfigError",
    "load_benchmark_pack_config",
]
