from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

from src.session_archive.manifest import (
    SessionArchiveError,
    SessionArchiveLogicalGroup,
    validate_session_archive_manifest,
)
from src.session_archive.writer import (
    SessionArchiveIncludePolicy,
    SessionArchiveWriteRequest,
    build_session_archive_plan,
    write_session_archive_pack,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _repo(root: Path, *, reverse: bool = False) -> Path:
    files = [
        ("data/curated/features_daily/AAPL.parquet", "feature-a\n"),
        ("data/curated/features_daily/MSFT.parquet", "feature-m\n"),
        ("artifacts/strategies/run_a/manifest.json", '{"run_id":"run_a"}\n'),
        ("artifacts/strategies/run_a/metrics.json", '{"sharpe":1.0}\n'),
        ("configs/profiles/notebook.yml", "schema_version: 1\nprofile: notebook\n"),
        ("configs/strategies.yml", "strategies: []\n"),
    ]
    for relative, text in reversed(files) if reverse else files:
        _write(root / relative, text)
    return root


def _request(
    root: Path,
    *,
    groups: tuple[SessionArchiveLogicalGroup, ...] = (
        SessionArchiveLogicalGroup.FEATURES,
        SessionArchiveLogicalGroup.ARTIFACTS,
        SessionArchiveLogicalGroup.CONFIGS,
    ),
    max_entries_per_shard: int = 1000,
) -> SessionArchiveWriteRequest:
    return SessionArchiveWriteRequest(
        archive_id="session-a",
        repository_root=root,
        output_root="artifacts/_derived/session_archives",
        include_policy=SessionArchiveIncludePolicy(
            include_groups=groups,
            include_paths={
                SessionArchiveLogicalGroup.FEATURES: ("data/curated/features_daily",),
                SessionArchiveLogicalGroup.ARTIFACTS: ("artifacts/strategies",),
                SessionArchiveLogicalGroup.CONFIGS: ("configs",),
            },
        ),
        max_entries_per_shard=max_entries_per_shard,
        max_shard_size_bytes=1024,
        session_id="notebook-session-a",
        source_runtime_profile="notebook",
        source_profile_path="configs/profiles/notebook.yml",
    )


def _plan_signature(request: SessionArchiveWriteRequest) -> dict[str, object]:
    plan = build_session_archive_plan(request)
    return {
        "archive_index": plan.archive_index,
        "checksums": plan.checksums,
        "manifest": plan.manifest.to_dict(),
        "restore_plan": plan.restore_plan,
    }


def test_deterministic_shard_planning_for_identical_trees(tmp_path: Path) -> None:
    first = _repo(tmp_path / "first")
    second = _repo(tmp_path / "second")

    assert _plan_signature(_request(first)) == _plan_signature(_request(second))


def test_stable_traversal_ordering_independent_of_file_creation_order(tmp_path: Path) -> None:
    first = _repo(tmp_path / "first")
    second = _repo(tmp_path / "second", reverse=True)

    assert _plan_signature(_request(first)) == _plan_signature(_request(second))


def test_archive_creation_writes_expected_layout_and_valid_manifest(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")

    result = write_session_archive_pack(_request(root, max_entries_per_shard=1))

    assert result.archive_root == root / "artifacts/_derived/session_archives/session-a"
    assert result.manifest_path.is_file()
    assert result.archive_index_path.is_file()
    assert result.checksums_path.is_file()
    assert result.restore_plan_path.is_file()
    assert result.shard_paths
    assert all(path.parent == result.archive_root / "shards" for path in result.shard_paths)

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    validate_session_archive_manifest(manifest_payload)
    assert manifest_payload["boundaries"]["derived"] is True
    assert manifest_payload["boundaries"]["canonical_storage"] is False
    sidecar_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            result.manifest_path,
            result.archive_index_path,
            result.checksums_path,
            result.restore_plan_path,
        )
    )
    assert str(root) not in sidecar_text
    assert root.as_posix() not in sidecar_text


def test_checksum_stability_for_identical_inputs(tmp_path: Path) -> None:
    first = write_session_archive_pack(_request(_repo(tmp_path / "first")))
    second = write_session_archive_pack(_request(_repo(tmp_path / "second")))

    assert first.plan.checksums == second.plan.checksums
    assert [shard.checksum for shard in first.plan.shards] == [
        shard.checksum for shard in second.plan.shards
    ]


@pytest.mark.parametrize(
    ("group", "expected_prefix"),
    [
        (SessionArchiveLogicalGroup.FEATURES, "data/curated/features_daily"),
        (SessionArchiveLogicalGroup.ARTIFACTS, "artifacts/strategies"),
        (SessionArchiveLogicalGroup.CONFIGS, "configs"),
    ],
)
def test_include_groups_independently(
    tmp_path: Path,
    group: SessionArchiveLogicalGroup,
    expected_prefix: str,
) -> None:
    root = _repo(tmp_path / "repo")

    result = write_session_archive_pack(_request(root, groups=(group,)))

    assert {entry.logical_group for entry in result.plan.entries} == {group}
    assert all(entry.source_path.startswith(expected_prefix) for entry in result.plan.entries)
    validate_session_archive_manifest(result.plan.manifest)


def test_combined_groups_create_grouped_shards(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")

    result = write_session_archive_pack(_request(root))

    assert {shard.logical_group for shard in result.plan.shards} == {
        SessionArchiveLogicalGroup.FEATURES,
        SessionArchiveLogicalGroup.ARTIFACTS,
        SessionArchiveLogicalGroup.CONFIGS,
    }


def test_default_exclude_policy_skips_noisy_and_archive_output_paths(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    _write(root / ".git/config", "git\n")
    _write(root / ".venv/pyvenv.cfg", "venv\n")
    _write(root / "configs/__pycache__/ignored.pyc", "cache\n")
    _write(root / "configs/notes.tmp", "tmp\n")
    _write(root / "artifacts/_derived/session_archives/session-a/shards/old.tar", "old\n")

    plan = build_session_archive_plan(_request(root))
    included = {entry.source_path for entry in plan.entries}

    assert ".git/config" not in included
    assert ".venv/pyvenv.cfg" not in included
    assert "configs/__pycache__/ignored.pyc" not in included
    assert "configs/notes.tmp" not in included
    assert "artifacts/_derived/session_archives/session-a/shards/old.tar" not in included


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside",
        "/absolute/path",
        "C:relative/path",
        "file://configs",
        "https://example.test/configs",
        "~/configs",
    ],
)
def test_unsafe_include_paths_are_rejected(tmp_path: Path, bad_path: str) -> None:
    root = _repo(tmp_path / "repo")
    request = SessionArchiveWriteRequest(
        archive_id="session-a",
        repository_root=root,
        include_policy=SessionArchiveIncludePolicy(
            include_groups=(SessionArchiveLogicalGroup.CONFIGS,),
            include_paths={SessionArchiveLogicalGroup.CONFIGS: (bad_path,)},
        ),
    )

    with pytest.raises(SessionArchiveError):
        build_session_archive_plan(request)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    outside = _write(tmp_path / "outside.txt", "outside\n")
    link = root / "configs" / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(SessionArchiveError, match="configured root"):
        build_session_archive_plan(_request(root, groups=(SessionArchiveLogicalGroup.CONFIGS,)))


def test_dry_run_planning_writes_no_files(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")

    plan = build_session_archive_plan(_request(root))

    assert plan.shards
    assert not (root / "artifacts/_derived/session_archives/session-a").exists()


def test_duckdb_memory_snapshot_metadata_is_optional(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    request = SessionArchiveWriteRequest(
        archive_id="session-a",
        repository_root=root,
        include_policy=SessionArchiveIncludePolicy(
            include_groups=(SessionArchiveLogicalGroup.CONFIGS,),
            include_paths={SessionArchiveLogicalGroup.CONFIGS: ("configs",)},
        ),
        duckdb_snapshot_source_path=":memory:",
        duckdb_snapshot_description="metadata only",
    )

    plan = build_session_archive_plan(request)

    assert plan.manifest.duckdb_snapshot is not None
    assert plan.manifest.duckdb_snapshot.source_path == ":memory:"
    assert {entry.logical_group for entry in plan.entries} == {SessionArchiveLogicalGroup.CONFIGS}


def test_file_backed_duckdb_snapshot_can_be_sharded_when_selected(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    _write(root / "artifacts/session.duckdb", "duckdb bytes\n")
    request = SessionArchiveWriteRequest(
        archive_id="session-a",
        repository_root=root,
        include_policy=SessionArchiveIncludePolicy(
            include_groups=(
                SessionArchiveLogicalGroup.CONFIGS,
                SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT,
            ),
            include_paths={SessionArchiveLogicalGroup.CONFIGS: ("configs",)},
        ),
        duckdb_snapshot_source_path="artifacts/session.duckdb",
    )

    result = write_session_archive_pack(request)

    assert result.plan.manifest.duckdb_snapshot is not None
    assert result.plan.manifest.duckdb_snapshot.included is True
    assert SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT in {
        entry.logical_group for entry in result.plan.entries
    }
    validate_session_archive_manifest(result.plan.manifest)


def test_tar_members_are_repository_relative_sorted_and_metadata_normalized(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")

    result = write_session_archive_pack(
        _request(root, groups=(SessionArchiveLogicalGroup.CONFIGS,))
    )

    with tarfile.open(result.shard_paths[0], mode="r") as archive:
        members = archive.getmembers()

    names = [member.name for member in members]
    assert names == sorted(names)
    assert all(not name.startswith("/") and "\\" not in name for name in names)
    assert {member.mtime for member in members} == {0}
    assert {member.uid for member in members} == {0}
    assert {member.gid for member in members} == {0}
    assert {member.uname for member in members} == {""}
    assert {member.gname for member in members} == {""}
