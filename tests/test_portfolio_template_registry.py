from __future__ import annotations

from pathlib import Path

import pytest

from src.portfolio.registry import (
    load_portfolio_template_registry,
    register_portfolio_template,
    resolve_portfolio_template_definition,
)
from src.research.registry import RegistryError


def test_register_and_load_portfolio_template_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "portfolios.jsonl"

    register_portfolio_template(
        name="core_equal_weight",
        version="1.0.0",
        definition={
            "allocator": "equal_weight",
            "initial_capital": 1.0,
            "alignment_policy": "intersection",
        },
        registry_path=registry_path,
    )

    entries = load_portfolio_template_registry(registry_path)
    assert len(entries) == 1
    assert entries[0]["portfolio_name"] == "core_equal_weight"
    assert entries[0]["version"] == "1.0.0"
    assert entries[0]["definition"]["allocator"] == "equal_weight"


def test_register_portfolio_template_is_idempotent(tmp_path: Path) -> None:
    registry_path = tmp_path / "portfolios.jsonl"
    payload = {
        "allocator": "equal_weight",
        "initial_capital": 1.0,
        "alignment_policy": "intersection",
    }

    first = register_portfolio_template(
        name="core_equal_weight",
        version="1.0.0",
        definition=payload,
        registry_path=registry_path,
    )
    second = register_portfolio_template(
        name="core_equal_weight",
        version="1.0.0",
        definition=payload,
        registry_path=registry_path,
    )

    assert first == second
    assert len(load_portfolio_template_registry(registry_path)) == 1


def test_register_portfolio_template_rejects_conflicts(tmp_path: Path) -> None:
    registry_path = tmp_path / "portfolios.jsonl"

    register_portfolio_template(
        name="core_equal_weight",
        version="1.0.0",
        definition={"allocator": "equal_weight"},
        registry_path=registry_path,
    )

    with pytest.raises(RegistryError, match="conflicting"):
        register_portfolio_template(
            name="core_equal_weight",
            version="1.0.0",
            definition={"allocator": "max_sharpe"},
            registry_path=registry_path,
        )


def test_resolve_portfolio_template_definition_prefers_latest_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "portfolios.jsonl"

    register_portfolio_template(
        name="core_equal_weight",
        version="1.0.0",
        definition={"allocator": "equal_weight", "initial_capital": 1.0},
        registry_path=registry_path,
    )
    register_portfolio_template(
        name="core_equal_weight",
        version="2.0.0",
        definition={"allocator": "equal_weight", "initial_capital": 10.0},
        registry_path=registry_path,
    )

    entries = load_portfolio_template_registry(registry_path)
    latest = resolve_portfolio_template_definition(entries, name="core_equal_weight", version=None)
    explicit = resolve_portfolio_template_definition(entries, name="core_equal_weight", version="1.0.0")

    assert latest is not None
    assert latest["initial_capital"] == 10.0
    assert explicit is not None
    assert explicit["initial_capital"] == 1.0
