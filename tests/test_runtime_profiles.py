from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.profiles import (
    RuntimeProfileError,
    SUPPORTED_RUNTIME_PROFILES,
    load_runtime_profile,
    validate_runtime_profile,
)
from src.validation.docs_path_lint import lint_guarded_surfaces


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def test_starter_runtime_profiles_are_parseable_and_portable() -> None:
    profile_paths = sorted(PROFILE_DIR.glob("*.yml"))

    assert {path.stem for path in profile_paths} == SUPPORTED_RUNTIME_PROFILES

    for path in profile_paths:
        profile = load_runtime_profile(str(path))

        assert profile.profile == path.stem
        assert profile.schema_version == 1
        assert profile.boundaries["direct_scan"] is True
        assert profile.boundaries["derived_outputs_authoritative"] is False
        assert profile.boundaries["mutates_canonical_artifacts"] is False
        assert profile.boundaries["requires_network"] is False
        assert profile.boundaries["requires_credentials"] is False
        assert profile.boundaries["requires_live_market_data"] is False


def test_runtime_profile_rejects_unknown_keys() -> None:
    with pytest.raises(RuntimeProfileError, match="unsupported keys"):
        validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {"artifacts_root": "artifacts"},
                "secret_token": "not-allowed",
            }
        )

    with pytest.raises(RuntimeProfileError, match="settings"):
        validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {"api_key": "not-allowed"},
            }
        )


def test_runtime_profile_rejects_invalid_path_values() -> None:
    base_payload = {
        "schema_version": 1,
        "profile": "local",
    }

    for bad_path in [
        "C:/Users/example/data",
        "/home/example/data",
        "~/data",
        "file://data/curated",
        "../data/curated",
        "data\\curated",
    ]:
        with pytest.raises(RuntimeProfileError, match="repository-relative|URI|home|POSIX|normalized"):
            validate_runtime_profile(
                {
                    **base_payload,
                    "settings": {"marketlake_root": bad_path},
                }
            )

    with pytest.raises(RuntimeProfileError, match="review.output.path"):
        validate_runtime_profile(
            {
                **base_payload,
                "review": {"output": {"path": "C:/Users/example/review.csv"}},
            }
        )


def test_runtime_profile_rejects_invalid_boundaries() -> None:
    with pytest.raises(RuntimeProfileError, match="direct_scan"):
        validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "ci",
                "boundaries": {"direct_scan": False},
            }
        )

    with pytest.raises(RuntimeProfileError, match="requires_credentials"):
        validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "ci",
                "boundaries": {"requires_credentials": True},
            }
        )


def test_runtime_profile_examples_do_not_include_machine_local_absolute_paths() -> None:
    report = lint_guarded_surfaces(
        REPO_ROOT,
        guarded_surfaces=("configs/profiles/**/*.yml", "docs/runtime_profiles.md"),
    )

    assert report["status"] == "passed", report["findings"]
    assert report["finding_count"] == 0


def test_runtime_profile_examples_do_not_request_external_services() -> None:
    for path in sorted(PROFILE_DIR.glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        boundaries = payload["boundaries"]

        assert boundaries["requires_network"] is False
        assert boundaries["requires_credentials"] is False
        assert boundaries["requires_live_market_data"] is False
