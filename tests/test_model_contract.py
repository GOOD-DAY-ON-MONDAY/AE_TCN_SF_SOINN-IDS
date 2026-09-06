"""Tests for model contract validation (ticket 04)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.runner import (
    ContractViolation,
    RunnerError,
    check_predict_contract,
    load_model,
    validate_model,
)

CLASS_HEADER = """\
class StubModel:
    supports_incremental = {incremental}
"""

METHODS = {
    "fit": """\
    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_ = np.unique(y)
        self.n_fits_ = getattr(self, "n_fits_", 0) + 1
""",
    "predict": """\
    def predict(self, X):
        # Deterministic pseudo-classes: pure read (seeded rng).
        rng = np.random.default_rng(0)
        n = max(len(self.classes_), 1)
        return rng.integers(0, n, size=len(X))
""",
    "save": """\
    def save(self, path):
        Path(path).write_text("stub")
""",
    "partial_fit": """\
    def partial_fit(self, X, y):
        pass
""",
    "predict_and_adapt": """\
    def predict_and_adapt(self, X):
        return self.predict(X)
""",
}

FACTORY = """\

def create_model(cfg):
    return StubModel()
"""


def _main_py(include: tuple[str, ...], incremental: bool = False) -> str:
    parts = ["from pathlib import Path\nimport numpy as np\n\n"]
    parts.append(CLASS_HEADER.format(incremental=incremental))
    for name in include:
        parts.append(METHODS[name])
    parts.append(FACTORY)
    return "\n".join(parts)


def _write_model(tmp_path: Path, source: str, name: str = "stub_model") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir(exist_ok=True)
    (model_dir / "main.py").write_text(source, encoding="utf-8")
    return model_dir


def _make_cfg() -> object:
    from types import SimpleNamespace

    cm8 = {f"c{i}": i for i in range(8)}
    cm21 = {f"c{i}": i for i in range(21)}
    return SimpleNamespace(
        training=SimpleNamespace(random_seed=42),
        data=SimpleNamespace(
            netml2020=SimpleNamespace(
                class_map=cm21, num_classes=21, feature_dim=121
            ),
            cicids2017=SimpleNamespace(
                class_map=cm8, num_classes=8, feature_dim=121
            ),
        ),
        splitting=SimpleNamespace(val_split=0.15, random_seed=42),
    )


class _Cfg:
    """Minimal merged-config stand-in for tests."""

    def __new__(cls) -> object:  # pragma: no cover - thin alias
        return _make_cfg()


REQUIRED = ("fit", "predict", "save")


def test_stub_model_satisfying_contract_loads_and_validates(tmp_path: Path) -> None:
    model_dir = _write_model(tmp_path, _main_py(REQUIRED))
    model = load_model(model_dir, _Cfg())
    assert model is not None
    X = np.zeros((4, 3))
    model.fit(X, np.array([0, 1, 0, 1]))
    check_predict_contract(model, X)
    model.save(tmp_path / "artifact")
    assert (tmp_path / "artifact").is_file()


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_required_method_fails_fast(tmp_path: Path, missing: str) -> None:
    included = tuple(m for m in REQUIRED if m != missing)
    model_dir = _write_model(tmp_path, _main_py(included))
    with pytest.raises(ContractViolation, match=f"missing required method '{missing}'"):
        load_model(model_dir, _Cfg())


def test_missing_entry_file_or_create_model_fails(tmp_path: Path) -> None:
    empty = tmp_path / "no_entry"
    empty.mkdir()
    with pytest.raises(RunnerError, match="entry file"):
        load_model(empty, _Cfg())

    no_factory = _write_model(tmp_path, _main_py(REQUIRED).replace(FACTORY, "\n"))
    with pytest.raises(ContractViolation, match="create_model"):
        load_model(no_factory, _Cfg())


def test_incremental_missing_partial_fit_fails(tmp_path: Path) -> None:
    model_dir = _write_model(tmp_path, _main_py(REQUIRED, incremental=True))
    with pytest.raises(ContractViolation, match="missing 'partial_fit'"):
        load_model(model_dir, _Cfg())


def test_incremental_missing_predict_and_adapt_fails(tmp_path: Path) -> None:
    model_dir = _write_model(
        tmp_path,
        _main_py(REQUIRED + ("partial_fit",), incremental=True),
    )
    with pytest.raises(ContractViolation, match="missing 'predict_and_adapt'"):
        load_model(model_dir, _Cfg())


def test_incremental_with_both_extra_methods_passes(tmp_path: Path) -> None:
    model_dir = _write_model(
        tmp_path,
        _main_py(REQUIRED + ("partial_fit", "predict_and_adapt"), incremental=True),
    )
    assert load_model(model_dir, _Cfg()) is not None


def test_non_incremental_model_without_extra_methods_passes(tmp_path: Path) -> None:
    model = validate_model(load_model(_write_model(tmp_path, _main_py(REQUIRED)), _Cfg()))
    assert not getattr(model, "supports_incremental", False)


def test_predict_is_pure_read() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    check_predict_contract(_Fitted(), X)  # repeated calls -> identical output


def test_predict_impure_read_fails() -> None:
    class _Impure(_Fitted):
        calls = 0

        def predict(self, X):
            _Impure.calls += 1
            return np.full(len(X), _Impure.calls, dtype=int)

    with pytest.raises(ContractViolation, match="not a pure read"):
        check_predict_contract(_Impure(), np.zeros((3, 2)))


def test_predict_must_return_hard_integer_indices() -> None:
    class _Proba(_Fitted):
        def predict(self, X):
            return np.full(len(X), 0.5)

    with pytest.raises(ContractViolation, match="hard integer class indices"):
        check_predict_contract(_Proba(), np.zeros((3, 2)))


def test_fit_recallable_without_reset(tmp_path: Path) -> None:
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    model = load_model(_write_model(tmp_path, _main_py(REQUIRED)), _Cfg())
    model.fit(X, y)
    model.fit(X, y, X_val=X, y_val=y)  # re-callable, with optional val data
    assert model.n_fits_ == 2  # state accumulated, not reset


class _Fitted:
    """In-memory stub used for runtime contract checks."""

    supports_incremental = False

    def predict(self, X):
        return np.zeros(len(X), dtype=int)
