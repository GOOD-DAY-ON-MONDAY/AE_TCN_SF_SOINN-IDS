"""Tests for the ``show-config`` subcommand (Runner ticket 10)."""

from __future__ import annotations

import pytest

from src import train


def _make_model_dir(tmp_path, with_model_yaml: bool = True):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "main.py").write_text("def create_model(cfg):\n    return None\n")
    if with_model_yaml:
        (model_dir / "model.yaml").write_text("training:\n  epochs: 99\n")
    return model_dir


def test_show_config_runs_without_executing(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    model_dir = _make_model_dir(tmp_path)
    rc = train.main(
        [
            "show-config",
            "--model",
            str(model_dir),
            "--dataset",
            "netml2020",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Known overridden key from model.yaml appears, marked with its layer.
    assert "epochs: 99" in out
    assert "model.yaml" in out
    # An unspecified key inherits its base value.
    assert "val_split: 0.15" in out


def test_show_config_marks_override_layer(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    model_dir = _make_model_dir(tmp_path)
    override = tmp_path / "override.yaml"
    override.write_text("splitting:\n  val_split: 0.2\n")
    rc = train.main(
        [
            "show-config",
            "--model",
            str(model_dir),
            "--dataset",
            "cicids2017",
            "--config",
            str(override),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "val_split: 0.2" in out
    assert "splitting.val_split <- --config" in out
    assert "training.epochs <- model.yaml" in out


def test_show_config_base_only_when_no_optional_layers(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    model_dir = _make_model_dir(tmp_path, with_model_yaml=False)
    rc = train.main(
        [
            "show-config",
            "--model",
            str(model_dir),
            "--dataset",
            "netml2020",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "No keys overridden" in out
    assert "val_split: 0.15" in out


def test_show_config_rejects_bad_model(tmp_path, capsys: pytest.CaptureFixture) -> None:
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    rc = train.main(["show-config", "--model", str(bogus), "--dataset", "netml2020"])
    assert rc == 2
    assert "Unknown model directory" in capsys.readouterr().err
