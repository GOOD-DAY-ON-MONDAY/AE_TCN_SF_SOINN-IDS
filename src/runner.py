"""Runner-side logic: model discovery, dataset registry, model loading/contract
validation, and nearest-match error messages. Kept separate from the CLI layer
(``src/train.py``) per CONTEXT.md.

Model contract: ``main.py`` exposes ``create_model(cfg)``; models implement
``fit(X, y, X_val=None, y_val=None)`` (re-callable), ``predict(X)`` returning
hard integer class indices (pure read), and ``save(path)``. Models declaring
``supports_incremental = True`` must also implement ``partial_fit(X, y)`` and
``predict_and_adapt(X)`` (contract-only in v1).
"""

from __future__ import annotations

import difflib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import REPO_ROOT

VALID_DATASETS = ("netml2020", "cicids2017")

MODELS_ROOT = REPO_ROOT / "models"

# A directory counts as a runnable model if it contains the fixed entry file.
ENTRY_FILE = "main.py"


class RunnerError(Exception):
    """Fast-fail Runner error with a suggest-the-fix message."""


def nearest_match(candidate: str, valid: list[str] | tuple[str, ...]) -> str | None:
    """Return the closest valid name to ``candidate``, or None if nothing is close."""
    matches = difflib.get_close_matches(candidate, list(valid), n=1, cutoff=0.4)
    return matches[0] if matches else None


def discover_models(models_root: str | Path | None = None) -> list[Path]:
    """Return runnable model directories (those containing ``main.py``).

    Searches two levels deep under ``models/``: ``models/<family>/<model>/``.
    """
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    if not root.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Family dir: look one level down for model dirs.
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / ENTRY_FILE).is_file():
                found.append(sub)
        # Also allow models placed directly under models/ (family-less).
        if (entry / ENTRY_FILE).is_file():
            found.append(entry)
    return found


def resolve_model(model_path: str | Path) -> Path:
    """Validate ``model_path`` is a directory with an entry file.

    The directory path is the model's identity/display name. A typo'd or
    nonexistent path fails fast, suggesting the nearest matching model.
    """
    path = Path(model_path)
    if path.is_dir() and (path / ENTRY_FILE).is_file():
        return path.resolve()

    # Search both the repo-wide models tree and the neighborhood of the
    # given path (so typos like `.../bilstmm` can be matched to `.../bilstm`).
    known = [str(p) for p in discover_models()]
    search_root = path.parent.parent if path.parent.parent.is_dir() else None
    if search_root is not None:
        known += [str(p) for p in discover_models(search_root)]
    seen: set[str] = set()
    known = [k for k in known if not (k in seen or seen.add(k))]
    hint = nearest_match(str(path), known)
    msg = f"Unknown model directory: '{path}'"
    if hint:
        msg += f"\nDid you mean '{hint}'?"
    if known:
        msg += "\nAvailable models:\n  " + "\n  ".join(known)
    raise RunnerError(msg)


def resolve_dataset(dataset: str) -> str:
    """Validate the dataset name, suggesting the nearest valid name on typo."""
    if dataset in VALID_DATASETS:
        return dataset
    hint = nearest_match(dataset, VALID_DATASETS)
    msg = f"Unknown dataset: '{dataset}' (valid: {', '.join(VALID_DATASETS)})"
    if hint:
        msg += f"\nDid you mean '{hint}'?"
    raise RunnerError(msg)


# ---------------------------------------------------------------------------
# Model loading + contract validation (ticket 04)
# ---------------------------------------------------------------------------


_REQUIRED_METHODS = ("fit", "predict", "save")
_INCREMENTAL_METHODS = ("partial_fit", "predict_and_adapt")


class ContractViolation(RunnerError):
    """Model does not satisfy the Runner contract; message names the fix."""


def load_model(model_dir: str | Path, cfg: Any) -> Any:
    """Load ``create_model(cfg)`` from ``<model_dir>/main.py`` and validate it."""
    model_dir = Path(model_dir)
    entry = model_dir / ENTRY_FILE
    if not entry.is_file():
        raise RunnerError(
            f"Model directory '{model_dir}' has no '{ENTRY_FILE}' entry file.\n"
            f"Fix: add '{ENTRY_FILE}' exposing create_model(cfg)."
        )
    spec = importlib.util.spec_from_file_location(
        f"runner_model_{abs(hash(str(model_dir)))}", entry
    )
    if spec is None or spec.loader is None:
        raise RunnerError(f"Could not import model entry file: '{entry}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_model"):
        raise ContractViolation(
            f"'{entry}' does not expose create_model(cfg).\n"
            "Fix: define create_model(cfg) returning the model object."
        )
    model = module.create_model(cfg)
    return validate_model(model, model_dir)


def validate_model(model: Any, origin: str = "<model>") -> Any:
    """Fail fast if ``model`` violates the contract; return it unchanged."""
    for name in _REQUIRED_METHODS:
        if not callable(getattr(model, name, None)):
            raise ContractViolation(
                f"Model '{origin}' is missing required method '{name}'.\n"
                f"Fix: implement {name}() per the Runner contract "
                f"(required: {', '.join(_REQUIRED_METHODS)})."
            )
    if getattr(model, "supports_incremental", False):
        for name in _INCREMENTAL_METHODS:
            if not callable(getattr(model, name, None)):
                raise ContractViolation(
                    f"Incremental model '{origin}' declares "
                    f"supports_incremental=True but is missing '{name}'.\n"
                    f"Fix: implement {name}(), or remove the "
                    "supports_incremental declaration."
                )
    return model


def check_predict_contract(model: Any, X: Any) -> None:
    """Post-fit runtime checks: hard integer indices + pure read.

    Called on the val/test arrays before metrics are computed.
    """
    out1 = model.predict(X)
    out2 = model.predict(X)
    arr1 = np.asarray(out1)
    if arr1.ndim != 1:
        raise ContractViolation(
            f"predict() returned shape {arr1.shape}; expected 1-D hard "
            "class indices (no probabilities, no 2-D outputs)."
        )
    if arr1.size > 0 and not np.issubdtype(arr1.dtype, np.integer):
        raise ContractViolation(
            f"predict() returned dtype {arr1.dtype}; the contract requires "
            "hard integer class indices. Convert inside the model "
            "(e.g. argmax over probabilities)."
        )
    if not np.array_equal(arr1, np.asarray(out2)):
        raise ContractViolation(
            "predict() is not a pure read: repeated calls with the same "
            "input returned different outputs. predict() must have no "
            "side effects (adaptation belongs in partial_fit / "
            "predict_and_adapt)."
        )


# ---------------------------------------------------------------------------
# Single-seed tracer run (ticket 05)
# ---------------------------------------------------------------------------

REPORTS_DIR = REPO_ROOT / "reports" / "comparison_results"
ALL_RUNS_CSV = REPORTS_DIR / "all_runs.csv"

# Single source of truth for the CSV/metrics column list (ticket 05).
RUN_COLUMNS = (
    "model",
    "dataset",
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "latency_ms",
    "peak_mem_gb",
    "train_time_s",
    "run_dir",
    "timestamp",
)

from src.utils.ui import chunked_predict_with_progress, console, phase
from src.utils.ui import yellow as _yellow


def _peak_mem_gb() -> float:
    """Return the process peak RSS in GB (measured by the Runner, never model code)."""
    import resource
    import sys

    # ru_maxrss is bytes on macOS, KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / 1024**3
    return peak / (1024**2)


def _latency_ms_per_flow(model: Any, X: Any) -> float:
    """Return mean wall-clock predict latency per flow, in milliseconds."""
    import time

    n = max(len(X), 1)
    start = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - start) * 1000.0 / n


def _confusion_plot(cm: Any, run_dir: Path, display_names: list[str]) -> None:
    """Render a labeled confusion-matrix heatmap to ``<run_dir>/confusion_matrix.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n = len(display_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=90, fontsize=6)
    ax.set_yticklabels(display_names, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Zero-day loop with v1 gating (ticket 08)
# ---------------------------------------------------------------------------


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a SimpleNamespace or a dict-like config node."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_zero_day_loop(
    model: Any,
    cfg: Any,
    dataset: str,
    X_withheld: Any,
    y_withheld: Any,
    X_known: Any,
    y_known: Any,
    run_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Zero-day protocol: predict (unknown flag) -> partial_fit -> re-predict.

    Gated on ``supports_incremental = True`` and a non-empty
    ``splitting.zero_day_classes.<dataset>``; otherwise prints a skip notice and
    returns None. ``predict_and_adapt`` is never called (contract-only in v1).
    Returns the results dict, also written to ``<run_dir>/zero_day.json`` when
    ``run_dir`` is given.
    """
    import json

    import numpy as np

    zd_cfg = _cfg_get(_cfg_get(cfg, "splitting"), "zero_day_classes")
    zero_day_classes = list(_cfg_get(zd_cfg, dataset, []) or [])

    if not zero_day_classes:
        console.print(
            f"[yellow]zero-day loop skipped (no zero-day classes configured for {dataset})[/yellow]"
        )
        return None
    if not getattr(model, "supports_incremental", False):
        console.print("[yellow]zero-day loop skipped (not an incremental model)[/yellow]")
        return None

    ds_cfg = _cfg_get(_cfg_get(cfg, "data"), dataset)
    raw_map = _cfg_get(ds_cfg, "class_map")
    class_map = dict(vars(raw_map)) if hasattr(raw_map, "__dict__") else dict(raw_map)
    zd_indices = {class_map[name] for name in zero_day_classes if name in class_map}
    known_labels = sorted(set(np.asarray(y_known).tolist()) - zd_indices)

    if len(X_withheld) == 0 or len(X_known) == 0:
        console.print(
            "[yellow]zero-day loop skipped (no withheld/known samples in "
            "the evaluation split)[/yellow]"
        )
        return None

    # Step 1: predict on the withheld class — is it flagged unknown?
    pre_withheld_pred = np.asarray(model.predict(X_withheld))
    known_label_set = set(known_labels)
    flagged_unknown = int(
        np.mean([int(p) not in known_label_set for p in pre_withheld_pred])
    )

    # Retention snapshot on known classes BEFORE teaching.
    pre_known_pred = np.asarray(model.predict(X_known))

    # Step 2: teach the withheld class (reproducible, seed-driven data).
    model.partial_fit(X_withheld, y_withheld)

    # Step 3: re-predict known classes to measure retention.
    post_known_pred = np.asarray(model.predict(X_known))
    retention = float(
        np.mean(pre_known_pred == post_known_pred)
    )  # prediction stability across the teach step
    post_accuracy = float(np.mean(post_known_pred == np.asarray(y_known)))

    results = {
        "zero_day_classes": zero_day_classes,
        "n_withheld": len(X_withheld),
        "n_known": len(X_known),
        "flagged_unknown_rate": flagged_unknown,
        "retention_prediction_stability": retention,
        "post_teach_known_accuracy": post_accuracy,
        "used_predict_and_adapt": False,
    }
    print(
        f"zero-day loop: flagged_unknown={flagged_unknown:.2f} "
        f"retention_stability={retention:.4f} "
        f"post_teach_accuracy={post_accuracy:.4f}"
    )
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "zero_day.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def run_single_seed(
    model_dir: str | Path,
    dataset: str,
    seed: int,
    cfg: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any = None,
    y_test: Any = None,
    report_root: str | Path | None = None,
    csv_path: str | Path | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """One full train -> predict -> evaluate cycle.

    Evaluates on the test split when the seam carries one, otherwise on the val
    split (test data blocked — see src/data/seam.py). Returns the row dict
    appended to all_runs.csv.
    """
    import csv
    import inspect
    import json
    import time
    from datetime import datetime, timezone

    import numpy as np
    from sklearn import metrics as skmetrics

    model_dir = Path(model_dir)
    model = load_model(model_dir, cfg)
    validate_model(model, model_dir)

    # ---- fit (per-phase progress: indeterminate spinner or step callback)
    t0 = time.perf_counter()
    fit_kwargs = {"X_val": X_val, "y_val": y_val}
    sig = inspect.signature(model.fit)
    if progress is not None and ("progress_callback" in sig.parameters or "callback" in sig.parameters):
        # Epoch/iteration-based model: fit() reports real steps through the
        # callback (advance=1 per epoch/batch, total=epochs). The bar moves
        # only when the model actually reports a step — never synthetically.
        # NOTE: implemented but unverified — no epoch-based model exists yet.
        train_task = progress.add_task(
            f"[bold]training[/bold] [dim]({model_dir.name}: epoch-based)[/dim]",
            total=None,
        )

        def _cb(advance: int = 1, total: int | None = None, description: str | None = None) -> None:
            """Forward one model progress step to the Rich train task.

            Args:
                advance (int): steps completed since last call.
                total (int | None): reset the task total when known.
                description (str | None): replace the task label when given.
            """
            kw: dict[str, Any] = {"advance": advance}
            if total is not None:
                kw["total"] = total
            if description is not None:
                kw["description"] = description
            progress.update(train_task, **kw)

        cb_name = "progress_callback" if "progress_callback" in sig.parameters else "callback"
        fit_kwargs[cb_name] = _cb
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        finally:
            train_time_s = time.perf_counter() - t0
            progress.update(
                train_task,
                completed=1,
                total=1,
                description=f"[green]training completed[/green] [dim]({train_time_s:.2f}s)[/dim]",
            )
    else:
        # One-shot fit-call model (e.g. sklearn SVM/RF): .fit() is a single
        # blocking call with no intermediate progress hooks. Per the CLI
        # design rules, show an indeterminate spinner labeled with what is
        # running — never a fabricated percentage.
        phase(
            f"training [dim]({model_dir.name}: one-shot fit, "
            f"no intermediate progress)[/dim]",
            lambda: model.fit(X_train, y_train, **fit_kwargs),
            progress=progress,
        )
        train_time_s = time.perf_counter() - t0

    # ---- evaluate (test if wired, else val — blocker documented) --------
    if X_test is not None:
        X_eval, y_eval = X_test, y_test
    else:
        X_eval, y_eval = X_val, y_val
        console.print(
            "[yellow]note: test data unavailable (labels_available: false); "
            "evaluating on the validation split[/yellow]"
        )

    # ---- testing phase: chunked predict with a determinate bar ----------
    y_pred = chunked_predict_with_progress(
        model, X_eval, label="evaluating", progress=progress
    )
    check_predict_contract(model, X_eval)
    y_eval = np.asarray(y_eval).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ds_cfg = getattr(cfg.data, dataset)
    raw_map = ds_cfg.class_map
    # The merged namespace converts mappings to SimpleNamespace; accept both.
    class_map = dict(vars(raw_map)) if hasattr(raw_map, "__dict__") else dict(raw_map)
    display_names = sorted(class_map, key=class_map.__getitem__)
    labels = list(range(len(display_names)))

    # ---- metrics (measured by the Runner) ------------------------------
    row: dict[str, Any] = {
        "model": str(model_dir),
        "dataset": dataset,
        "seed": seed,
        "accuracy": skmetrics.accuracy_score(y_eval, y_pred),
        "precision": skmetrics.precision_score(
            y_eval, y_pred, average="weighted", zero_division=0
        ),
        "recall": skmetrics.recall_score(
            y_eval, y_pred, average="weighted", zero_division=0
        ),
        "f1": skmetrics.f1_score(y_eval, y_pred, average="weighted", zero_division=0),
        "macro_precision": skmetrics.precision_score(
            y_eval, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": skmetrics.recall_score(
            y_eval, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": skmetrics.f1_score(
            y_eval, y_pred, average="macro", zero_division=0
        ),
        "latency_ms": _latency_ms_per_flow(model, X_eval),
        "peak_mem_gb": _peak_mem_gb(),
        "train_time_s": round(train_time_s, 6),
        "run_dir": "",  # filled below
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    cm = skmetrics.confusion_matrix(y_eval, y_pred, labels=labels).tolist()

    # ---- artifacts ------------------------------------------------------
    root = Path(report_root) if report_root is not None else Path.cwd() / "models"
    resolved = model_dir.resolve()
    if resolved.is_relative_to(MODELS_ROOT.resolve()):
        rel = resolved.relative_to(MODELS_ROOT.resolve())
    else:
        rel = Path(model_dir.name)
    run_dir = root / rel / dataset / f"seed_{seed}"
    if run_dir.exists():
        console.print(f"[yellow]overwriting existing run directory: {run_dir}[/yellow]")
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps({**row, "run_dir": str(run_dir), "confusion_matrix": cm}, indent=2),
        encoding="utf-8",
    )
    _confusion_plot(np.asarray(cm), run_dir, display_names)
    model.save(run_dir / "model_artifact")
    row["run_dir"] = str(run_dir)

    # ---- zero-day loop (gated, ticket 08) --------------------------------
    # Samples in the evaluation split whose label belongs to the
    # configured zero-day classes are "withheld"; the rest are known.
    # The gate itself (incremental model + non-empty config) lives in
    # run_zero_day_loop; if the split carries no zero-day-class samples,
    # the loop self-skips there too.
    y_eval_arr = np.asarray(y_eval)
    ds_cfg_zd = getattr(cfg.data, dataset)
    raw_map_zd = ds_cfg_zd.class_map
    cmap_zd = (
        dict(vars(raw_map_zd)) if hasattr(raw_map_zd, "__dict__") else dict(raw_map_zd)
    )
    zd_cfg = getattr(cfg.splitting, "zero_day_classes", None)
    zd_names = list(getattr(zd_cfg, dataset, []) or []) if zd_cfg is not None else []
    zd_idx = {cmap_zd[n] for n in zd_names if n in cmap_zd}
    if zd_idx:
        zd_mask = np.isin(y_eval_arr, list(zd_idx))
        run_zero_day_loop(
            model=model,
            cfg=cfg,
            dataset=dataset,
            X_withheld=X_eval[zd_mask],
            y_withheld=y_eval_arr[zd_mask],
            X_known=X_eval[~zd_mask],
            y_known=y_eval_arr[~zd_mask],
            run_dir=run_dir,
        )
    else:
        run_zero_day_loop(
            model=model,
            cfg=cfg,
            dataset=dataset,
            X_withheld=X_eval,
            y_withheld=y_eval_arr,
            X_known=X_eval,
            y_known=y_eval_arr,
            run_dir=run_dir,
        )

    # ---- all_runs.csv ----------------------------------------------------
    csv_file = (
        Path(csv_path)
        if csv_path is not None
        else Path.cwd() / "reports" / "comparison_results" / "all_runs.csv"
    )
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_file.exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RUN_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    # ---- end-of-run summary table ----------------------------------------
    _print_summary(row, len(y_eval))
    return row


def _print_summary(row: dict[str, Any], n_eval: int) -> None:
    """Print the end-of-run summary as two side-by-side Rich tables in a Panel."""
    from rich import box
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table

    def _make_table(title: str, metrics: dict[str, float]) -> Table:
        """Build a two-column metric table with 4-decimal values.

        Args:
            title (str): table title.
            metrics (dict[str, float]): metric name to value.

        Returns:
            Table: Rich table for the run summary panel.
        """
        t = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False, title_style="bold")
        t.add_column("metric", style="dim", no_wrap=True)
        t.add_column("value", justify="right", style="bold green")
        for k, v in metrics.items():
            t.add_row(k, f"{v:.4f}")
        return t

    classification = _make_table(
        "Classification",
        {
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "macro_precision": row["macro_precision"],
            "macro_recall": row["macro_recall"],
            "macro_f1": row["macro_f1"],
        },
    )
    performance = _make_table(
        "Performance",
        {
            "latency_ms": row["latency_ms"],
            "peak_mem_gb": row["peak_mem_gb"],
            "train_time_s": row["train_time_s"],
        },
    )

    # Use a relative run_dir in the subtitle to keep lines short.
    try:
        rel_run_dir = str(Path(row["run_dir"]).relative_to(REPO_ROOT))
    except ValueError:
        rel_run_dir = row["run_dir"]

    model_name = Path(row["model"]).name
    console.print(
        Panel.fit(
            Columns([classification, performance], equal=False, expand=False),
            title=f"[bold]Run summary[/bold] — {model_name} / {row['dataset']} / seed {row['seed']}",
            subtitle=f"eval flows: {n_eval}  ·  run_dir: {rel_run_dir}",
            border_style="blue",
        )
    )


# ---------------------------------------------------------------------------
# Multi-seed loop + mean/std aggregation (ticket 06)
# ---------------------------------------------------------------------------

# Columns aggregated as mean ± std (all numeric run metrics).
AGGREGATED_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "latency_ms",
    "peak_mem_gb",
    "train_time_s",
)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Aggregate run rows into per-metric (mean, std) pairs.

    Uses population std (ddof=0); a single row yields std 0.0. Non-numeric
    columns are not aggregated.
    """
    if not rows:
        raise RunnerError("aggregate_rows() called with no run rows.")
    agg: dict[str, tuple[float, float]] = {}
    for key in AGGREGATED_COLUMNS:
        values = np.asarray([float(r[key]) for r in rows], dtype=float)
        agg[key] = (float(values.mean()), float(values.std()))
    return agg


def print_aggregate(agg: dict[str, tuple[float, float]], n_seeds: int) -> None:
    """Print the mean ± std aggregation as a Rich table."""
    from rich import box
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table

    def _agg_table(title: str, keys: tuple[str, ...]) -> Table:
        """Build a mean ± std table for ``keys`` from the closure ``agg``.

        Args:
            title (str): table title.
            keys (tuple[str, ...]): metric names to pull from ``agg``.

        Returns:
            Table: Rich table with mean and std columns.
        """
        t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
        t.add_column("metric", style="dim", no_wrap=True)
        t.add_column("mean", justify="right", style="bold green")
        t.add_column("± std", justify="right", style="dim")
        for k in keys:
            mean, std = agg[k]
            t.add_row(k, f"{mean:.4f}", f"{std:.4f}")
        return t

    classification_keys = (
        "accuracy", "precision", "recall", "f1",
        "macro_precision", "macro_recall", "macro_f1",
    )
    performance_keys = ("latency_ms", "peak_mem_gb", "train_time_s")

    console.print(
        Panel.fit(
            Columns(
                [
                    _agg_table("Classification", classification_keys),
                    _agg_table("Performance", performance_keys),
                ],
                equal=False,
                expand=False,
            ),
            title=f"[bold]Aggregated over {n_seeds} seed(s)[/bold]",
            border_style="blue",
        )
    )


def run_seeds(
    model_dir: str | Path,
    dataset: str,
    seeds: list[int],
    cfg: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any = None,
    y_test: Any = None,
    report_root: str | Path | None = None,
    csv_path: str | Path | None = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """One Run per seed, aggregated into mean ± std.

    Returns the per-seed row list; aggregation is printed. Rerunning a seed
    overwrites its ``seed_<n>/`` directory.
    """
    if not seeds:
        raise RunnerError("--seeds received no values; give at least one seed.")
    rows = [
        run_single_seed(
            model_dir=model_dir,
            dataset=dataset,
            seed=seed,
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            report_root=report_root,
            csv_path=csv_path,
            progress=progress,
        )
        for seed in seeds
    ]
    print_aggregate(aggregate_rows(rows), len(rows))
    return rows
