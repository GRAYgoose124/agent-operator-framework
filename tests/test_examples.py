"""Validate example pipeline TOML files."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import assert_valid_pipeline_toml


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_tomls():
    """List of example pipeline TOML paths."""
    return list(EXAMPLES_DIR.glob("*.toml"))


def test_all_examples_load(example_tomls):
    """Each example TOML parses and has valid structure."""
    assert len(example_tomls) >= 5
    for path in example_tomls:
        assert_valid_pipeline_toml(path)


def test_example_steps_have_required_fields(example_tomls):
    """Each step has system_prompt or equivalent."""
    import tomllib

    for path in example_tomls:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for i, step in enumerate(data.get("steps", [])):
            assert "system_prompt" in step or "name" in step, (
                f"{path}: step {i} missing system_prompt or name"
            )


def test_example_roles_referenced(example_tomls):
    """Steps reference roles (defaults to general if omitted)."""
    import tomllib

    for path in example_tomls:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for step in data.get("steps", []):
            role = step.get("role", "general")
            assert isinstance(role, str) and len(role) > 0
