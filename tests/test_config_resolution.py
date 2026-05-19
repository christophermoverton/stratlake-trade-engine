from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config.resolution import ConfigResolutionError, resolve_runtime_profile_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_runtime_profile_config_default_only_has_default_provenance() -> None:
    result = resolve_runtime_profile_config()

    assert result.profile_name is None
    assert result.config.settings["artifacts_root"] == "artifacts"
    assert result.config.settings["features_root"] == "data"
    assert result.config.settings["marketlake_root"] is None
    assert result.config.workflow_configs["execution_config"] == "configs/execution.yml"
    assert result.config.boundaries["direct_scan"] is True
    assert result.config.boundaries["derived_outputs_authoritative"] is False
    assert result.provenance["settings.artifacts_root"].source == "default"
    assert result.provenance["runtime.execution.execution_delay"].source == "default"


def test_profile_overrides_defaults_with_profile_provenance() -> None:
    result = resolve_runtime_profile_config("ci")

    assert result.profile_name == "ci"
    assert result.profile_path == "configs/profiles/ci.yml"
    assert result.config.settings["artifacts_root"] == "artifacts/ci"
    assert result.config.review["output"]["emit_plots"] is False
    assert result.config.runtime["strict_mode"] == {"enabled": True, "source": "config"}
    assert result.provenance["settings.artifacts_root"].source == "profile"
    assert result.provenance["settings.artifacts_root"].source_detail == "configs/profiles/ci.yml"
    assert result.provenance["runtime.strict_mode.enabled"].source == "profile"
    assert result.provenance["runtime.strict_mode.source"].source == "profile"
    assert "runtime resolver" in result.provenance["runtime.strict_mode.source"].source_detail


def test_environment_overrides_profile_settings_with_environment_provenance() -> None:
    result = resolve_runtime_profile_config(
        "local",
        environment={
            "ARTIFACTS_ROOT": "artifacts/env",
            "FEATURES_ROOT": "data/env_features",
            "LOG_LEVEL": "DEBUG",
        },
    )

    assert result.config.settings["artifacts_root"] == "artifacts/env"
    assert result.config.settings["features_root"] == "data/env_features"
    assert result.config.settings["log_level"] == "DEBUG"
    assert result.provenance["settings.artifacts_root"].source == "environment"
    assert result.provenance["settings.artifacts_root"].source_detail == "ARTIFACTS_ROOT"
    assert result.provenance["settings.duckdb_path"].source == "profile"


def test_cli_overrides_environment_and_profile_with_cli_provenance() -> None:
    result = resolve_runtime_profile_config(
        "local",
        environment={
            "ARTIFACTS_ROOT": "artifacts/env",
            "FEATURES_ROOT": "data/env_features",
        },
        cli_overrides={
            "settings": {
                "artifacts_root": "artifacts/cli",
            },
            "workflow_configs": {
                "evaluation_config": "configs/evaluation.yml",
            },
            "runtime": {
                "execution": {
                    "transaction_cost_bps": 7.5,
                },
                "risk": {
                    "target_volatility": 0.12,
                },
            },
            "review": {
                "output": {
                    "path": "artifacts/review/leaderboard.csv",
                    "emit_plots": False,
                },
            },
        },
    )

    assert result.config.settings["artifacts_root"] == "artifacts/cli"
    assert result.config.settings["features_root"] == "data/env_features"
    assert result.config.workflow_configs["evaluation_config"] == "configs/evaluation.yml"
    assert result.config.runtime["execution"]["transaction_cost_bps"] == pytest.approx(7.5)
    assert result.config.runtime["execution"]["enabled"] is True
    assert result.config.runtime["risk"]["target_volatility"] == pytest.approx(0.12)
    assert result.config.review["output"]["path"] == "artifacts/review/leaderboard.csv"
    assert result.provenance["settings.artifacts_root"].source == "cli_override"
    assert result.provenance["settings.features_root"].source == "environment"
    assert result.provenance["runtime.execution.transaction_cost_bps"].source == "cli_override"
    assert result.provenance["runtime.execution.enabled"].source == "cli_override"
    assert "runtime resolver" in result.provenance["runtime.execution.enabled"].source_detail
    assert result.provenance["review.output.path"].source == "cli_override"


def test_unknown_keys_fail_clearly() -> None:
    with pytest.raises(ConfigResolutionError, match="unsupported keys"):
        resolve_runtime_profile_config(cli_overrides={"mystery": {"value": True}})

    with pytest.raises(ConfigResolutionError, match="settings"):
        resolve_runtime_profile_config(cli_overrides={"settings": {"api_key": "not-allowed"}})

    with pytest.raises(ConfigResolutionError, match="unsupported keys"):
        resolve_runtime_profile_config(
            cli_overrides={"runtime": {"execution": {"not_a_runtime_key": True}}}
        )


def test_invalid_paths_remain_rejected() -> None:
    with pytest.raises(ConfigResolutionError, match="repository-relative"):
        resolve_runtime_profile_config(environment={"ARTIFACTS_ROOT": "C:/Users/example/artifacts"})

    with pytest.raises(ConfigResolutionError, match="review.output.path|repository-relative"):
        resolve_runtime_profile_config(
            cli_overrides={
                "review": {
                    "output": {
                        "path": "/home/example/review.csv",
                    }
                }
            }
        )


def test_repeated_resolution_and_serialization_are_deterministic() -> None:
    kwargs = {
        "profile": "pipeline",
        "environment": {"ARTIFACTS_ROOT": "artifacts/env_pipeline"},
        "cli_overrides": {
            "runtime": {"execution": {"execution_delay": 2}},
            "review": {"output": {"emit_plots": False}},
        },
    }

    first = resolve_runtime_profile_config(**kwargs)
    second = resolve_runtime_profile_config(**kwargs)

    assert first.to_json_dict() == second.to_json_dict()
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_json_dict()
    assert list(first.to_json_dict()["provenance"]) == sorted(first.to_json_dict()["provenance"])


def test_resolution_from_explicit_profile_path_does_not_create_or_modify_outputs(tmp_path: Path) -> None:
    profile_path = tmp_path / "custom_profile.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {
                    "artifacts_root": "artifacts/custom",
                    "features_root": "data/custom",
                },
                "boundaries": {
                    "direct_scan": True,
                    "derived_outputs_authoritative": False,
                    "mutates_canonical_artifacts": False,
                    "requires_network": False,
                    "requires_credentials": False,
                    "requires_live_market_data": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    before = _snapshot_files(tmp_path)

    result = resolve_runtime_profile_config(profile_path=profile_path)

    after = _snapshot_files(tmp_path)
    assert after == before
    assert result.config.settings["artifacts_root"] == "artifacts/custom"
    assert result.config.boundaries["direct_scan"] is True
    assert result.config.boundaries["derived_outputs_authoritative"] is False


def test_resolution_does_not_mutate_starter_profiles() -> None:
    before = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "configs" / "profiles").glob("*.yml"))
    }

    resolve_runtime_profile_config(
        "ci",
        environment={"ARTIFACTS_ROOT": "artifacts/env"},
        cli_overrides={"settings": {"artifacts_root": "artifacts/cli"}},
    )

    after = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "configs" / "profiles").glob("*.yml"))
    }
    assert after == before


def _snapshot_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
