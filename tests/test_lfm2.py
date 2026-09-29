"""Tests for LFM2.5 / Liquid Nanos pipeline support."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.config import load_config
from aof.inference.parsing import detect_model_family


def test_detect_model_family_lfm2():
    """detect_model_family returns lfm2 for LFM2 paths."""
    assert detect_model_family("LFM2-1.2B-Tool-GGUF/LFM2-1.2B-Tool-Q4_K_M.gguf") == "lfm2"
    assert detect_model_family("LFM2.5-1.2B-Instruct-GGUF/model.gguf") == "lfm2"
    assert detect_model_family("/path/to/lfm2_rag/model.gguf") == "lfm2"
    assert detect_model_family("liquid-nanos/LFM2-1.2B.gguf") == "lfm2"


def test_detect_model_family_qwen3():
    """detect_model_family returns qwen3 for Qwen paths."""
    assert detect_model_family("Qwen3-0.6B-GGUF/model.gguf") == "qwen3"
    assert detect_model_family("/path/to/qwen3/model.gguf") == "qwen3"


def test_resolve_role_path_lfm2_roles():
    """_resolve_role_path returns non-empty path for lfm2 roles."""
    from aof.cli import _resolve_role_path

    config = load_config(None)
    lfm2_roles = (
        "lfm2_tool", "lfm2_rag", "lfm2_extract", "lfm2_extract_350m",
        "lfm2_math", "lfm2_transcript", "lfm2_vl", "lfm2_jp",
        "orchestrator", "thinker",
    )
    for role in lfm2_roles:
        path = _resolve_role_path(config, role)
        assert path, f"Role {role} must resolve"
        assert "LFM2" in path or "lfm2" in path.lower()


def test_get_pipeline_step_roles_extracts_lfm2():
    """_get_pipeline_step_roles extracts lfm2 roles from lfm2_nanos.toml."""
    from aof.cli import _get_pipeline_step_roles

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    toml_path = examples_dir / "lfm2_nanos.toml"
    if not toml_path.exists():
        pytest.skip("lfm2_nanos.toml not found")
    roles = _get_pipeline_step_roles(str(toml_path))
    assert "lfm2_tool" in roles
    assert "lfm2_extract" in roles
    assert "lfm2_rag" in roles


def test_lfm2_full_loads_all_six_models():
    """lfm2_full.toml uses all 6 LFM2 models; each role resolves to a valid path."""
    from aof.cli import _get_pipeline_step_roles, _resolve_role_path

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    toml_path = examples_dir / "lfm2_full.toml"
    if not toml_path.exists():
        pytest.skip("lfm2_full.toml not found")

    roles = _get_pipeline_step_roles(str(toml_path))
    expected_roles = {"lfm2_tool", "lfm2_extract", "reasoning", "lfm2_math", "lfm2_rag", "fast"}
    assert roles == expected_roles, f"Expected {expected_roles}, got {roles}"

    config = load_config(None)
    for role in expected_roles:
        path = _resolve_role_path(config, role)
        assert path, f"Role {role} must resolve to a non-empty path"
        assert len(path) > 10, f"Role {role} path '{path}' looks invalid"
