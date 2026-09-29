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
