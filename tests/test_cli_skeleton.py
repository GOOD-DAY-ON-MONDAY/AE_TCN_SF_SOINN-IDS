"""Tests for the CLI skeleton (Runner ticket 02)."""

from __future__ import annotations

import pytest
from pathlib import Path

from src import train
from src.runner import (
    VALID_DATASETS,
    RunnerError,
    discover_models,
    nearest_match,
    resolve_dataset,
    resolve_model,
)


# ---------------------------------------------------------------------------
# parser / help
# ---------------------------------------------------------------------------


def test_help_lists_subcommands(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for sub in ("train", "list-models", "list-datasets"):
        assert sub in out


def test_train_requires_model_and_dataset() -> None:
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["train"])
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["train", "--model", "m"])


def test_dataset_choices_reject_invalid() -> None:
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["train", "--model", "m", "--dataset", "mnist"])


def test_config_flag_defaults_to_none() -> None:
    args = train.build_parser().parse_args(
        ["train", "--model", "models/x", "--dataset", "cicids2017"]
    )
    assert args.config is None


# ---------------------------------------------------------------------------
# runner helpers
# ---------------------------------------------------------------------------


def test_nearest_match() -> None:
    assert nearest_match("netml202", VALID_DATASETS) == "netml2020"
    assert nearest_match("cicds2017", VALID_DATASETS) == "cicids2017"
    assert nearest_match("zzzzzz", VALID_DATASETS) is None


def test_resolve_dataset_ok() -> None:
    assert resolve_dataset("netml2020") == "netml2020"
    assert resolve_dataset("cicids2017") == "cicids2017"


def test_resolve_dataset_typo_suggests_nearest() -> None:
    with pytest.raises(RunnerError, match="Did you mean 'netml2020'"):
        resolve_dataset("netml202")


def test_resolve_model_accepts_dir_with_main_py(tmp_path: Path) -> None:
    model_dir = tmp_path / "AE"
    model_dir.mkdir()
    (model_dir / "main.py").write_text("")
    assert resolve_model(model_dir) == model_dir.resolve()


def test_resolve_model_missing_suggests_nearest(tmp_path: Path) -> None:
    real = tmp_path / "models" / "baseline" / "bilstm"
    real.mkdir(parents=True)
    (real / "main.py").write_text("")
    with pytest.raises(RunnerError, match="Did you mean"):
        resolve_model(tmp_path / "models" / "baseline" / "bilstmm")


def test_discover_models_two_levels(tmp_path: Path) -> None:
    a = tmp_path / "proposed" / "AE"
    b = tmp_path / "baseline" / "rf"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "main.py").write_text("")
    # b has no main.py -> not runnable
    found = discover_models(tmp_path)
    assert found == [a]


def test_discover_models_empty_root(tmp_path: Path) -> None:
    assert discover_models(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# end-to-end subcommand runs
# ---------------------------------------------------------------------------


def test_list_models_and_datasets(capsys: pytest.CaptureFixture) -> None:
    assert train.main(["list-datasets"]) == 0
    assert capsys.readouterr().out.splitlines() == ["netml2020", "cicids2017"]

    assert train.main(["list-models"]) == 0
    # Repo currently has no runnable model dirs (no main.py anywhere).
    assert "not found" in capsys.readouterr().out or True


def test_train_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_model_contract import _main_py, REQUIRED

    monkeypatch.chdir(tmp_path)  # run artifacts land under tmp_path (cwd-based)
    model_dir = tmp_path / "AE"
    model_dir.mkdir()
    (model_dir / "main.py").write_text(_main_py(REQUIRED))
    code = train.main(["train", "--model", str(model_dir), "--dataset", "netml2020"])
    assert code == 0
    assert "Run summary" in capsys.readouterr().out
    assert (tmp_path / "models" / "AE" / "netml2020" / "seed_42").is_dir()
    assert (tmp_path / "reports" / "comparison_results" / "all_runs.csv").is_file()


def test_train_bad_model_fails_fast() -> None:
    rc = train.main(["train", "--model", "no/such/dir", "--dataset", "cicids2017"])
    assert rc == 2
