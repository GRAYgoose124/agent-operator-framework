"""Tests for configuration loading."""

import tempfile
from pathlib import Path

from aof.config import AppConfig, config_with_workspace, load_config


def test_load_defaults():
    config = load_config(None)
    assert isinstance(config, AppConfig)
    assert config.pool.max_agents == 8
    assert config.model.n_ctx == 2048
    assert config.evaluator.health_threshold == 0.5


def test_load_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[model]\nn_ctx = 4096\n\n[pool]\nmax_agents = 16\n')
        f.flush()
        config = load_config(Path(f.name))

    assert config.model.n_ctx == 4096
    assert config.pool.max_agents == 16
    # Other values should be defaults
    assert config.model.temperature == 0.7


def test_load_nonexistent_file():
    config = load_config("nonexistent.toml")
    assert isinstance(config, AppConfig)
    assert config.pool.max_agents == 8


def test_frozen_config():
    config = load_config(None)
    try:
        config.model.n_ctx = 999  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass  # Expected: frozen dataclass


def test_config_with_workspace():
    config = load_config(None)
    ws_root = Path("data/workspaces/test_ws").resolve()
    overridden = config_with_workspace(config, ws_root)
    assert overridden.memory.db_path == str(ws_root / "memory.db")
    assert overridden.memory.notes_dir == str(ws_root / "memory")
    assert overridden.research.queue_path == str(ws_root / "queue.json")
    assert overridden.research.interrupt_flag_path == str(ws_root / "interrupt.flag")


def test_model_paths_expand_user(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[model]\npath = "~/m.gguf"\n[models]\ndirectory = "~/models"\n')
    config = load_config(cfg_file)
    assert "~" not in config.model.path
    assert "~" not in config.models.directory


def test_resolve_role_path_falls_back_to_liquidai_sibling(tmp_path):
    """Liquid Nanos often live under models/LiquidAI while models.directory is lmstudio-community."""
    from aof.config import resolve_role_path

    community = tmp_path / "lmstudio-community"
    community.mkdir()
    liquid = tmp_path / "LiquidAI" / "LFM2-1.2B-Tool-GGUF"
    liquid.mkdir(parents=True)
    model = liquid / "LFM2-1.2B-Tool-Q4_K_M.gguf"
    model.write_bytes(b"gguf")

    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        f'[models]\ndirectory = "{community.as_posix()}"\n'
        '[roles]\nlfm2_tool = "LFM2-1.2B-Tool-GGUF/LFM2-1.2B-Tool-Q4_K_M.gguf"\n'
    )
    config = load_config(cfg_file)
    resolved = Path(resolve_role_path(config, "lfm2_tool"))
    assert resolved.resolve() == model.resolve()


def test_resolve_role_path_prefers_models_directory(tmp_path):
    """When the file exists under models.directory, do not prefer LiquidAI."""
    from aof.config import resolve_role_path

    community = tmp_path / "lmstudio-community" / "LFM2-1.2B-Tool-GGUF"
    community.mkdir(parents=True)
    primary = community / "LFM2-1.2B-Tool-Q4_K_M.gguf"
    primary.write_bytes(b"primary")
    liquid = tmp_path / "LiquidAI" / "LFM2-1.2B-Tool-GGUF"
    liquid.mkdir(parents=True)
    (liquid / "LFM2-1.2B-Tool-Q4_K_M.gguf").write_bytes(b"liquid")

    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        f'[models]\ndirectory = "{(tmp_path / "lmstudio-community").as_posix()}"\n'
        '[roles]\nlfm2_tool = "LFM2-1.2B-Tool-GGUF/LFM2-1.2B-Tool-Q4_K_M.gguf"\n'
    )
    config = load_config(cfg_file)
    resolved = Path(resolve_role_path(config, "lfm2_tool"))
    assert resolved.resolve() == primary.resolve()
