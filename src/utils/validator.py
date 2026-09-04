"""Data validation for the AE_TCN_SFSONIN_IDS project.

Validates the raw datasets described in configs/base_config.yaml and
produces the per-class count reports needed to pick zero_day_classes:

  * Counts flows per class in each dataset's training annotations.
  * Scans test_std_dir / test_challenge_dir for label/annotation files
    and prints whether ground-truth labels appear to be available —
    this is the pending check referenced by ``labels_available: false``
    in configs/base_config.yaml and docs/CONFIG.md.
  * Writes reports/data/<dataset>_data_report.png (bar chart with a
    log-scale y-axis so rare attack classes stay visible next to the
    heavy benign traffic) and reports/data/class_counts.csv (exact
    counts, rarest first, to pick the 3-5 zero_day_classes).

NOTE ON ANNOTATION FORMATS: the raw annotation payload has not been
inspected yet, so the loader below is intentionally defensive. It
handles JSON / gzipped JSON / JSON-lines payloads shaped as a list of
label strings, a list of dicts with a label-ish key, or a mapping of
id -> label. If the real files use a different shape, adjust
``_LABEL_KEYS`` / ``_extract_labels`` — everything else stays the same.

Usage (from the repo root):
    python -m src.utils.validate_data

Exit code 0 = both datasets validated; 1 = at least one dataset could
not be validated (e.g. raw data not downloaded yet).
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a window
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from src.utils.config import REPO_ROOT, ConfigError, load_config

REPORTS_DIR = REPO_ROOT / "reports" / "data"

# Documented training-set sizes (see the DATASET CONTEXT comment in
# configs/base_config.yaml). A mismatch is a warning, not an error.
EXPECTED_TRAINING_ROWS = {"netml2020": 387_268, "cicids2017": 441_116}

# Keys checked (in order) when a record is a dict and we need its label.
_LABEL_KEYS = ("label", "Label", "class", "Class", "category", "attack_type", "type")

# Substrings that mark a file inside a test dir as a candidate label file.
# NOTE: feature files (e.g. 1_test-std_set.json.gz) also end in .json.gz,
# so name matching — not extension matching — is what separates
# annotations from features here.
_LABEL_FILE_HINTS = ("annotation", "label")


def _read_json_payload(path: Path) -> Any:
    """Read a .json or .json.gz file, falling back to JSON-lines."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some challenge payloads are JSON-lines (one record per line).
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"'{path}' is neither valid JSON nor JSON-lines ({exc})"
            ) from exc


def _find_annotation_file(annotations_path: Path, dataset_name: str) -> Path:
    """Locate the label file inside (or at) data.<ds>.training_annotations."""
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"data.{dataset_name}.training_annotations does not exist: "
            f"'{annotations_path}' (raw data may not be downloaded yet — "
            f"see src/utils/dwnld_data.py)"
        )
    if annotations_path.is_file():
        return annotations_path
    files = sorted(p for p in annotations_path.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"data.{dataset_name}.training_annotations directory is empty: "
            f"'{annotations_path}'"
        )
    preferred = [
        p for p in files if any(hint in p.name.lower() for hint in _LABEL_FILE_HINTS)
    ]
    candidates = preferred or files
    json_like = [p for p in candidates if p.name.endswith((".json", ".json.gz"))]
    if len(json_like) == 1:
        return json_like[0]
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in files)
    raise RuntimeError(
        f"data.{dataset_name}.training_annotations is ambiguous — multiple "
        f"candidate label files found in '{annotations_path}': {names}"
    )


def _extract_labels(payload: Any, source: Path) -> list[str]:
    """Flatten a parsed annotation payload into a plain list of label strings."""
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return list(payload)
        labels = []
        for i, record in enumerate(payload):
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"{source}: record {i} is {type(record).__name__}, expected "
                    f"a dict with one of the label keys {_LABEL_KEYS}"
                )
            for key in _LABEL_KEYS:
                if key in record:
                    labels.append(str(record[key]))
                    break
            else:
                keys = ", ".join(sorted(record))
                raise RuntimeError(
                    f"{source}: record {i} has no label key (tried "
                    f"{_LABEL_KEYS}); available keys: {keys}"
                )
        return labels
    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            return list(payload.values())
        return _extract_labels(list(payload.values()), source)
    raise RuntimeError(
        f"{source}: unsupported annotation payload — top-level "
        f"{type(payload).__name__}, expected a list or dict"
    )


def count_training_classes(dataset_name: str, ds_cfg: Any) -> tuple[Counter[str], int]:
    """Return {class_name: flow_count} and the total row count for a dataset."""
    training_set = Path(ds_cfg.training_set)
    if not training_set.exists():
        print(
            f"[WARN] {dataset_name}: feature file missing: '{training_set}' — "
            f"counts below are label-only; download features before training."
        )

    annotation_file = _find_annotation_file(
        Path(ds_cfg.training_annotations), dataset_name
    )
    labels = _extract_labels(_read_json_payload(annotation_file), annotation_file)
    if not labels:
        raise RuntimeError(f"{annotation_file}: no labels were parsed")

    expected = EXPECTED_TRAINING_ROWS.get(dataset_name)
    if expected is not None and len(labels) != expected:
        print(
            f"[WARN] {dataset_name}: parsed {len(labels):,} training labels, "
            f"but the challenge docs say {expected:,} — verify the annotation file."
        )

    class_map = dict(ds_cfg.class_map)
    unknown = sorted(set(labels) - set(class_map))
    if unknown:
        print(
            f"[WARN] {dataset_name}: {len(unknown)} label(s) not present in "
            f"data.{dataset_name}.class_map: {', '.join(unknown)}"
        )
    missing = sorted(set(class_map) - set(labels))
    if missing:
        print(
            f"[WARN] {dataset_name}: {len(missing)} class_map class(es) have "
            f"0 training flows: {', '.join(missing)}"
        )
    return Counter(labels), len(labels)


def _first_record(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict):
        return next(iter(payload.values()), None)
    return payload


def _record_has_label(record: Any) -> bool:
    if isinstance(record, str):
        return True  # a bare string entry is itself a label
    if isinstance(record, dict):
        return any(key in record for key in _LABEL_KEYS)
    return False


def _record_keys(record: Any) -> str:
    if isinstance(record, dict):
        return ", ".join(sorted(record))
    return type(record).__name__ if record is not None else "None"


def check_test_label_availability(dataset_name: str, ds_cfg: Any) -> None:
    """Print whether test_std/test_challenge contain label/annotation files.

    This is the record-level check that configs/base_config.yaml deferred:
    if label files (or label-bearing records) turn up here, flip that
    dataset's ``labels_available`` to true in configs/base_config.yaml.
    """
    for attr in ("test_std_dir", "test_challenge_dir"):
        dir_path = Path(getattr(ds_cfg, attr))
        header = f"[{dataset_name}] {attr} ({dir_path})"
        if not dir_path.exists():
            print(f"{header}: directory missing — raw data not downloaded yet.")
            continue
        if not dir_path.is_dir():
            print(
                f"{header}: not a directory — check the path in "
                f"configs/base_config.yaml."
            )
            continue
        files = sorted(p for p in dir_path.rglob("*") if p.is_file())
        if not files:
            print(f"{header}: directory exists but is empty.")
            continue
        candidates = [
            p for p in files if any(h in p.name.lower() for h in _LABEL_FILE_HINTS)
        ]
        if not candidates:
            names = ", ".join(p.name for p in files[:5])
            more = f" (+{len(files) - 5} more)" if len(files) > 5 else ""
            print(
                f"{header}: {len(files)} file(s), none named like annotations "
                f"[{names}{more}] → no ground-truth labels found; treat as "
                f"UNLABELED (unsupervised inference only)."
            )
            continue
        for candidate in candidates:
            try:
                first = _first_record(_read_json_payload(candidate))
            except (OSError, ValueError, RuntimeError) as exc:
                print(
                    f"{header}: candidate label file '{candidate.name}' could "
                    f"not be read ({exc})."
                )
                continue
            if _record_has_label(first):
                print(
                    f"{header}: candidate label file '{candidate.name}' found AND "
                    f"its first record carries a label field → labels appear "
                    f"AVAILABLE. Verify the full payload, then set "
                    f"data.{dataset_name}.labels_available: true."
                )
            else:
                print(
                    f"{header}: file '{candidate.name}' found, but its first "
                    f"record shows no label field ({_record_keys(first)}) → "
                    f"treat as UNLABELED for now."
                )


def plot_class_counts(dataset_name: str, counts: Counter[str], out_path: Path) -> None:
    """Bar chart of flows per class; log-scale y-axis keeps rare classes visible."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.6 * len(names)), 6))
    colors = ["#7f7f7f" if name == "benign" else "#1f77b4" for name in names]
    ax.bar(range(len(names)), values, color=colors)
    ax.set_yscale("log")  # crucial: benign >> rare attack classes
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("training flows (log scale)")
    ax.set_title(f"{dataset_name}: flows per class (benign highlighted)")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_class_counts_csv(rows: list[tuple[str, str, int]], out_path: Path) -> None:
    """Write dataset,class,count rows (already sorted rarest-first per dataset)."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "class", "count"])
        writer.writerows(rows)


def print_class_summary(dataset_name: str, counts: Counter[str], total: int) -> None:
    print(
        f"\n=== {dataset_name}: class distribution "
        f"({total:,} flows, {len(counts)} classes) ==="
    )
    for cls, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * cnt / total if total else 0.0
        print(f"  {cls:>20s}: {cnt:>9,d}  ({pct:5.2f}%)")


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    failures = 0
    csv_rows: list[tuple[str, str, int]] = []

    for dataset_name in ("netml2020", "cicids2017"):
        ds_cfg = getattr(cfg.data, dataset_name)
        try:
            counts, total = count_training_classes(dataset_name, ds_cfg)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"\n[ERROR] {dataset_name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        check_test_label_availability(dataset_name, ds_cfg)
        print_class_summary(dataset_name, counts, total)

        chart_path = REPORTS_DIR / f"{dataset_name}_data_report.png"
        plot_class_counts(dataset_name, counts, chart_path)
        print(f"[OK] wrote {chart_path}")

        # CSV sorted rarest-first: zero_day_classes candidates are the top rows.
        csv_rows.extend(
            (dataset_name, cls, cnt)
            for cls, cnt in sorted(counts.items(), key=lambda kv: kv[1])
        )

    if csv_rows:
        csv_path = REPORTS_DIR / "class_counts.csv"
        write_class_counts_csv(csv_rows, csv_path)
        print(f"[OK] wrote {csv_path} (rarest classes first)")

    print(
        "\nNext step: pick 3-5 lower-frequency, non-benign classes per dataset "
        "from the CSV and fill splitting.zero_day_classes in "
        "configs/base_config.yaml (see docs/CONFIG.md). Do not guess — use "
        "these real counts."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
