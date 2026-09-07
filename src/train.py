"""CLI entrypoint for the model-comparison Runner: ``python -m src.train``.

Subcommands (v1): ``train``, ``show-config``, ``list-models``, ``list-datasets``,
``builder`` (interactive REPL shell).
The ``train`` subcommand merges three config layers (base -> <model_dir>/model.yaml
-> ``--config``) in memory; base_config.yaml on disk is never modified.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.runner import (
    VALID_DATASETS,
    RunnerError,
    discover_models,
    resolve_dataset,
    resolve_model,
)
from src.utils.config import load_merged_config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with the ``train``, ``show-config``, ``list-models``
    and ``list-datasets`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Model-comparison Runner (netml-cl).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate a model on a dataset.")
    p_train.add_argument(
        "--model",
        required=True,
        help="Path to a model directory (must contain main.py). The directory "
        "path is the model's identity/display name.",
    )
    p_train.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Dataset to run on.",
    )
    p_train.add_argument(
        "--config",
        default=None,
        help="Optional ad-hoc YAML override file (highest config layer).",
    )
    p_train.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One Run per seed, aggregated into mean ± std "
        "(defaults to training.random_seed from config).",
    )
    # Back-compat alias: --seed N is treated as --seeds N.
    p_train.add_argument(
        "--seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p_train.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debugging/smoke-test only: read only the first N rows of the "
        "training data before fitting. Defaults to off (full dataset); "
        "never use for real experiments.",
    )

    p_show = sub.add_parser(
        "show-config",
        help="Print the fully merged config for a model/dataset "
        "(base -> model.yaml -> --config) without running anything.",
    )
    p_show.add_argument(
        "--model",
        required=True,
        help="Path to a model directory (its model.yaml is layer 2 if present).",
    )
    p_show.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Dataset context for the merged config.",
    )
    p_show.add_argument(
        "--config",
        default=None,
        help="Optional ad-hoc YAML override file (highest config layer).",
    )

    sub.add_parser("list-models", help="List runnable model directories.")
    sub.add_parser("list-datasets", help="List the valid datasets.")

    p_builder = sub.add_parser(
        "builder",
        aliases=["interactive"],
        help="Launch the interactive REPL shell (completion, history, rich UI).",
    )
    p_builder.set_defaults(func=None)
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the ``train`` subcommand: load data, run the model, write reports."""
    model_dir = resolve_model(args.model)
    dataset = resolve_dataset(args.dataset)  # argparse choices already gate this

    # Three-layer config merge; model.yaml is optional (layer 2).
    model_yaml = Path(model_dir) / "model.yaml"
    cfg = load_merged_config(
        model_config_path=model_yaml if model_yaml.is_file() else None,
        override_path=args.config,
    )
    seeds: list[int] = []
    if getattr(args, "seeds", None):
        seeds = list(args.seeds)
    elif getattr(args, "seed", None) is not None:
        seeds = [args.seed]
    else:
        seeds = [int(cfg.training.random_seed)]
    seed = seeds[0]

    from src.utils.ui import print_plan

    model_yaml = Path(model_dir) / "model.yaml"
    print_plan(
        model_dir=str(model_dir),
        dataset=dataset,
        seeds=seeds,
        config_layers=[
            ("configs/base_config.yaml", "always"),
            ("model.yaml (model dir)", "yes" if model_yaml.is_file() else ""),
            ("--config", args.config if args.config else ""),
        ],
    )

    # Test data is blocked (labels_available: false), so evaluation falls back
    # to the val split held out by the seam.
    import json as _json

    from src.data.loader import get_training_data
    from src.data.seam import assemble_run_arrays
    from src.runner import run_seeds

    ds_cfg = getattr(cfg.data, dataset)
    with open("configs/feature_meta.json") as fh:
        feature_dict = _json.load(fh)
    # read_dataset() walks a folder of .json.gz files; the config stores the
    # training file path, so use its parent dir (raw_dir would also sweep the
    # test sets, which must stay out of training).
    training_folder = str(Path(ds_cfg.training_set).parent)
    # Tracer-bullet mode: the raw file is class-grouped, so a head cap would
    # miss classes; load full and subsample deterministically per seed.
    _sample_rows = 50000
    limit = getattr(args, "limit", None)
    if limit is not None:
        X, y, _, _ = get_training_data(
            training_folder,
            ds_cfg.training_annotations,
            feature_dict,
            max_rows=limit,
        )
        # The raw file is class-grouped, so a plain head cap can yield a single
        # class (unfittable). Re-read with a growing cap until at least 2 classes
        # appear, then keep the first `limit` rows per class.
        retries = 0
        while y is not None and len(set(y.tolist())) < 2 and retries < 6:
            retries += 1
            _cap = min(limit * (4**retries), 10**9)
            print(
                f"--limit: first {limit} rows contain a single class "
                f"(file is class-grouped); re-reading with cap {_cap} ..."
            )
            X, y, _, _ = get_training_data(
                training_folder,
                ds_cfg.training_annotations,
                feature_dict,
                max_rows=_cap,
            )
        if y is not None and len(set(y.tolist())) >= 2:
            import numpy as _np

            _keep = _np.zeros(len(y), dtype=bool)
            for _cls in set(y.tolist()):
                _idx = _np.flatnonzero(y == _cls)[:limit]
                _keep[_idx] = True
            X, y = X[_keep], y[_keep]
        print(
            f"--limit: using {len(X)} row(s) of training data (debug run, first {limit}/class)"
        )
    else:
        X, y, _, _ = get_training_data(
            training_folder,
            ds_cfg.training_annotations,
            feature_dict,
        )
        if len(X) > _sample_rows:
            import numpy as _np

            _rng = _np.random.default_rng(seed)
            _idx = _np.sort(_rng.choice(len(X), size=_sample_rows, replace=False))
            X, y = X[_idx], y[_idx]
            print(
                f"tracer run: sampled {_sample_rows} of 387268 loaded rows (seed {seed})"
            )

    arrays = assemble_run_arrays(X, y, cfg.splitting.val_split, seed)

    run_seeds(
        model_dir=model_dir,
        dataset=dataset,
        seeds=seeds,
        cfg=cfg,
        X_train=arrays.X_train,
        y_train=arrays.y_train,
        X_val=arrays.X_val,
        y_val=arrays.y_val,
    )
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    """Execute the ``show-config`` subcommand: print the merged config."""
    from src.utils.config import (
        DEFAULT_CONFIG_PATH,
        _load_yaml_mapping,
        deep_merge,
        deep_merge_with_sources,
    )

    model_dir = resolve_model(args.model)
    dataset = resolve_dataset(args.dataset)

    model_yaml = Path(model_dir) / "model.yaml"
    layer2 = model_yaml if model_yaml.is_file() else None

    base = _load_yaml_mapping(DEFAULT_CONFIG_PATH)
    merged = dict(base)
    sources: dict[str, str] = {}

    layers: list[tuple[dict | None, str]] = [
        (base, f"base: {DEFAULT_CONFIG_PATH}"),
    ]
    if layer2 is not None:
        layer_cfg = _load_yaml_mapping(layer2)
        merged = deep_merge(merged, layer_cfg)
        deep_merge_with_sources(base, layer_cfg, "model.yaml", sources)
        layers.append((layer_cfg, f"model.yaml: {layer2}"))
    if args.config:
        layer_cfg = _load_yaml_mapping(args.config)
        merged = deep_merge(merged, layer_cfg)
        deep_merge_with_sources(merged, layer_cfg, "--config", sources)
        layers.append((layer_cfg, f"--config: {args.config}"))
    # Base layer contributes everything not later-overridden.
    _record_base_sources(base, sources, "base")

    print(f"# Effective merged config (dataset: {dataset})")
    for _, label in layers:
        print(f"# layer: {label}")
    print("# markers:  # <- model.yaml   # <- --config   (no marker = base)")
    print(_yaml_with_sources(merged, sources))

    overridden = {p for p, s in sources.items() if s in ("model.yaml", "--config")}
    if overridden:
        print("\n# Overridden keys:")
        for path in sorted(overridden):
            print(f"#   {path} <- {sources[path]}")
    else:
        print("\n# No keys overridden; all values inherited from base.")
    return 0


def _record_base_sources(
    mapping: Mapping, sources: dict[str, str], label: str, prefix: str = ""
) -> None:
    """Record the source layer for every leaf key of a merged-config mapping.

    Only adds paths not already present, so later layers win.
    """
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            _record_base_sources(value, sources, label, prefix=f"{path}.")
        elif path not in sources:
            sources[path] = label


def _yaml_with_sources(merged: dict, sources: dict[str, str]) -> str:
    """Dump merged config as YAML, annotating leaf keys with their source."""
    import yaml as _yaml

    lines: list[str] = []

    def walk(node: Any, prefix: str, indent: int) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            pad = "  " * indent
            if isinstance(value, Mapping):
                lines.append(f"{pad}{key}:")
                walk(value, f"{path}.", indent + 1)
            else:
                src = sources.get(path, "base")
                marker = "" if src == "base" else f"   # <- {src}"
                lines.append(
                    f"{pad}{key}: {_yaml.safe_dump(value, default_flow_style=True).strip()}{marker}"
                )

    walk(merged, "", 0)
    return "\n".join(lines)


def cmd_list_models() -> int:
    """Print every runnable model directory found under ``models/``."""
    models = discover_models()
    if not models:
        print("No runnable model directories found under models/ (need main.py).")
        return 0
    for path in models:
        print(str(path))
    return 0


def cmd_list_datasets() -> int:
    """Print the valid dataset names, one per line."""
    for name in VALID_DATASETS:
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, dispatch the chosen subcommand, and format errors.

    Returns 0 on success, 2 on RunnerError, 1 if dispatch falls through.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return cmd_train(args)
        if args.command == "show-config":
            return cmd_show_config(args)
        if args.command == "list-models":
            return cmd_list_models()
        if args.command == "list-datasets":
            return cmd_list_datasets()
        if args.command in ("builder", "interactive"):
            from src.utils.shell import run_builder

            return run_builder()
    except RunnerError as exc:
        from src.utils.ui import format_error

        print(format_error(str(exc)), file=sys.stderr)
        return 2
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
