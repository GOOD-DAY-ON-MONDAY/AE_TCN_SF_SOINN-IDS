"""Tests for the three-layer config merge (Runner ticket 01).

Contract under test (from CONTEXT.md Q7 / spec.md):
- Layers: base_config.yaml -> <model_dir>/model.yaml -> --config override.
- Later layers override matching keys; unspecified keys inherit from the
  layer below.
- The merge is in memory only: base_config.yaml on disk is NEVER modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    deep_merge,
    load_config,
    load_merged_config,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# deep_merge unit behavior
# ---------------------------------------------------------------------------


def test_deep_merge_nested_dicts_merge_key_by_key() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    override = {"a": {"x": 9}}
    deep_merge(base, override)
    assert base == {"a": {"x": 1}}


def test_deep_merge_replaces_non_mapping_values_wholesale() -> None:
    base = {"list": [1, 2, 3], "scalar": 1}
    override = {"list": [9], "scalar": {"now": "a mapping"}}
    merged = deep_merge(base, override)
    assert merged == {"list": [9], "scalar": {"now": "a mapping"}}


# ---------------------------------------------------------------------------
# load_merged_config behavior
# ---------------------------------------------------------------------------


def test_base_only_merge_equals_load_config() -> None:
    via_merge = load_merged_config()
    via_legacy = load_config()
    assert via_merge.__dict__ == via_legacy.__dict__


def test_model_layer_overrides_and_inherits(tmp_path: Path) -> None:
    model_yaml = _write_yaml(
        tmp_path / "model.yaml",
        "training:\n  learning_rate: 0.5\n",
    )
    cfg = load_merged_config(model_config_path=model_yaml)
    # Override applied...
    assert cfg.training.learning_rate == 0.5
    # ...unspecified keys inherited from the layer below.
    assert cfg.training.batch_size == load_config().training.batch_size
    assert cfg.training.random_seed == load_config().training.random_seed


def test_override_layer_wins_over_model_layer(tmp_path: Path) -> None:
    model_yaml = _write_yaml(
        tmp_path / "model.yaml",
        "training:\n  learning_rate: 0.5\n",
    )
    override_yaml = _write_yaml(
        tmp_path / "override.yaml",
        "training:\n  learning_rate: 9.9\n  batch_size: 7\n",
    )
    cfg = load_merged_config(model_config_path=model_yaml, override_path=override_yaml)
    # All three layers defined learning_rate: the later (--config) wins.
    assert cfg.training.learning_rate == 9.9
    # Override wins over base for batch_size; model layer didn't define it.
    assert cfg.training.batch_size == 7


def test_missing_optional_layers_fall_through() -> None:
    cfg = load_merged_config()
    assert cfg.__dict__ == load_config().__dict__


def test_base_file_on_disk_is_never_modified(tmp_path: Path) -> None:
    before = _digest(DEFAULT_CONFIG_PATH)
    model_yaml = _write_yaml(
        tmp_path / "model.yaml",
        "training:\n  learning_rate: 0.5\n",
    )
    override_yaml = _write_yaml(
        tmp_path / "override.yaml",
        "training:\n  learning_rate: 9.9\n",
    )
    load_merged_config(model_config_path=model_yaml, override_path=override_yaml)
    assert _digest(DEFAULT_CONFIG_PATH) == before, (
        "base_config.yaml must never be modified by the merge"
    )


def test_missing_layer_file_raises_specific_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_merged_config(model_config_path=tmp_path / "does_not_exist.yaml")
    with pytest.raises(ConfigError, match="Config file not found"):
        load_merged_config(override_path=tmp_path / "does_not_exist.yaml")


def test_non_mapping_layer_file_raises(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "bad.yaml", "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="does not contain a YAML mapping"):
        load_merged_config(model_config_path=bad)


def test_empty_layer_file_contributes_nothing(tmp_path: Path) -> None:
    empty = _write_yaml(tmp_path / "empty.yaml", "")
    cfg = load_merged_config(model_config_path=empty)
    assert cfg.__dict__ == load_config().__dict__


def test_merged_result_still_validates_against_required_keys(tmp_path: Path) -> None:
    # A layer must not be able to remove required keys — removal isn't part
    # of the merge semantics, so a full merge always validates; this asserts
    # the merged namespace carries the required sections end-to-end.
    model_yaml = _write_yaml(
        tmp_path / "model.yaml",
        "training:\n  learning_rate: 0.5\n",
    )
    cfg = load_merged_config(model_config_path=model_yaml)
    for section in ("data", "splitting", "model", "training", "device"):
        assert hasattr(cfg, section)
