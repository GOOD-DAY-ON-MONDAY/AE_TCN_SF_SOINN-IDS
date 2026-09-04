"""Configuration loading and validation for the netml-cl project.

Reads ``configs/base_config.yaml`` into a nested, attribute-accessible
object and validates that every required key is present before the rest
of the pipeline uses it. All errors are specific: the raised message
names the exact missing (or invalid) key, so failures are never silent
and never generic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Repo root = three levels up from this file's directory (src/utils/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base_config.yaml"

# Both datasets use the same raw flow schema: 121 features per record.
EXPECTED_FEATURE_DIM = 121

_DATASETS = ("netml2020", "cicids2017")

_DATA_KEYS = (
    "raw_dir",
    "training_set",
    "training_annotations",
    "test_std_dir",
    "test_challenge_dir",
    "labels_available",
    "feature_dim",
    "num_classes",
    "class_map",
)

_MODEL_SECTIONS = {
    "denoising_gate": ("latent_dim",),
    "temporal_engine": ("channels", "kernel_size", "dilations", "dropout"),
    "zeroday_hunter": ("similarity_threshold",),
    "routing": ("confidence_threshold",),
}

_TRAINING_KEYS = ("batch_size", "learning_rate", "random_seed")
_DEVICE_KEYS = ("preference",)


class ConfigError(Exception):
    """Raised when configs/base_config.yaml is missing keys or has bad values."""


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    """Return mapping[key] or raise ConfigError naming the exact missing key."""
    if key not in mapping:
        raise ConfigError(f"Missing required config key: '{path}'")
    return mapping[key]


def _validate(cfg: Mapping[str, Any]) -> None:
    for section in ("data", "splitting", "model", "training", "device"):
        _require(cfg, section, section)

    # ---- data --------------------------------------------------------
    data = cfg["data"]
    _require(data, "processed_dir", "data.processed_dir")
    for name in _DATASETS:
        ds = _require(data, name, f"data.{name}")
        for key in _DATA_KEYS:
            _require(ds, key, f"data.{name}.{key}")
        if ds["feature_dim"] != EXPECTED_FEATURE_DIM:
            raise ConfigError(
                f"Invalid config value: 'data.{name}.feature_dim' must be "
                f"{EXPECTED_FEATURE_DIM}, got {ds['feature_dim']!r}"
            )
        if not isinstance(ds["class_map"], Mapping) or not ds["class_map"]:
            raise ConfigError(
                f"Invalid config value: 'data.{name}.class_map' must be a "
                "non-empty mapping of class name -> index"
            )
        if len(ds["class_map"]) != ds["num_classes"]:
            raise ConfigError(
                f"Invalid config value: 'data.{name}.class_map' has "
                f"{len(ds['class_map'])} entries but "
                f"'data.{name}.num_classes' is {ds['num_classes']}"
            )
        if not isinstance(ds["labels_available"], bool):
            raise ConfigError(
                f"Invalid config value: 'data.{name}.labels_available' "
                "must be true or false"
            )

    # ---- splitting ---------------------------------------------------
    splitting = cfg["splitting"]
    _require(splitting, "val_split", "splitting.val_split")
    _require(splitting, "random_seed", "splitting.random_seed")
    zero_day = _require(splitting, "zero_day_classes", "splitting.zero_day_classes")
    for name in _DATASETS:
        classes = _require(zero_day, name, f"splitting.zero_day_classes.{name}")
        if not isinstance(classes, Sequence) or isinstance(classes, str):
            raise ConfigError(
                f"Invalid config value: 'splitting.zero_day_classes.{name}' "
                "must be a YAML list of class names (it may be empty while "
                "the TODO is pending, but the key itself must exist)"
            )

    # ---- model -------------------------------------------------------
    model = cfg["model"]
    for section, keys in _MODEL_SECTIONS.items():
        sub = _require(model, section, f"model.{section}")
        for key in keys:
            _require(sub, key, f"model.{section}.{key}")

    # ---- training ----------------------------------------------------
    training = cfg["training"]
    for key in _TRAINING_KEYS:
        _require(training, key, f"training.{key}")

    # ---- device ------------------------------------------------------
    device = cfg["device"]
    for key in _DEVICE_KEYS:
        _require(device, key, f"device.{key}")


def _to_namespace(value: Any) -> Any:
    """Recursively convert mappings to SimpleNamespace for attribute access."""
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """Load and validate the project YAML config.

    Returns a nested namespace, e.g. ``cfg.data.netml2020.feature_dim``.
    Raises :class:`ConfigError` naming the specific missing key if any
    required key is absent, or a specific message if a known invariant
    (feature_dim == 121, class_map size == num_classes, etc.) is violated.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: '{path}'")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file '{path}' does not contain a YAML mapping")
    _validate(cfg)
    return _to_namespace(cfg)


if __name__ == "__main__":
    loaded = load_config()
    print(f"Config OK: {DEFAULT_CONFIG_PATH}")
    print(f"  datasets: {', '.join(_DATASETS)}")
    print(f"  val_split: {loaded.splitting.val_split}")
    print(f"  device preference: {loaded.device.preference}")
