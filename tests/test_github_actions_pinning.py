from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_NOTES = REPO_ROOT / "docs" / "m36_release_notes.md"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
USES_RE = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)

EXPECTED_ACTION_PINS = {
    "actions/checkout": "34e114876b0b11c390a56381ad16ebd13914f8d5",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "softprops/action-gh-release": "3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
}


def test_external_workflow_actions_are_full_sha_pinned() -> None:
    external_uses = _external_uses()

    assert external_uses
    for action, ref in external_uses:
        assert SHA_RE.fullmatch(ref), f"{action}@{ref} must use a full commit SHA"


def test_workflow_action_inventory_matches_reviewed_pin_set() -> None:
    action_to_refs: dict[str, set[str]] = {}
    for action, ref in _external_uses():
        action_to_refs.setdefault(action, set()).add(ref)

    assert set(action_to_refs) == set(EXPECTED_ACTION_PINS)
    for action, expected_ref in EXPECTED_ACTION_PINS.items():
        assert action_to_refs[action] == {expected_ref}


def test_release_notes_explicitly_document_no_unpinned_exceptions() -> None:
    text = " ".join(RELEASE_NOTES.read_text(encoding="utf-8").split())

    assert "no local reusable actions" in text
    assert "no intentionally unpinned external action" in text


def _external_uses() -> list[tuple[str, str]]:
    uses: list[tuple[str, str]] = []
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for target in USES_RE.findall(text):
            if target.startswith("./"):
                continue
            action, ref = target.rsplit("@", maxsplit=1)
            uses.append((action, ref))
    return uses
