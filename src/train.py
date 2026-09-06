"""CLI entrypoint for the model-comparison Runner: ``python -m src.train``.

Subcommands (v1): ``train``, ``list-models``, ``list-datasets``.
Per CONTEXT.md Q7/Q9 the ``train`` subcommand merges three config layers
(base -> <model_dir>/model.yaml -> ``--config``) in memory; the actual run
logic is wired up in later tickets and currently prints a stub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.runner import VALID_DATASETS, RunnerError, discover_models, resolve_dataset, resolve_model
from src.utils.config import load_merged_config


def build_parser() -> argparse.ArgumentParser:
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
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to training.random_seed from config).",
    )

    sub.add_parser("list-models", help="List runnable model directories.")
    sub.add_parser("list-datasets", help="List the valid datasets.")
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    model_dir = resolve_model(args.model)
    dataset = resolve_dataset(args.dataset)  # argparse choices already gate this

    # Three-layer config merge (ticket 01); base on disk is never modified.
    # model.yaml is optional per layer-2 semantics — only pass it if present.
    model_yaml = Path(model_dir) / "model.yaml"
    cfg = load_merged_config(
        model_config_path=model_yaml if model_yaml.is_file() else None,
        override_path=args.config,
    )
    seed = args.seed if args.seed is not None else cfg.training.random_seed

    print(f"Model:   {model_dir}")
    print(f"Dataset: {dataset}")
    print(f"Seed:    {seed}")
    print("Run not yet wired (ticket 05): config merged successfully, stopping here.")
    return 0


def cmd_list_models() -> int:
    models = discover_models()
    if not models:
        print("No runnable model directories found under models/ (need main.py).")
        return 0
    for path in models:
        print(str(path))
    return 0


def cmd_list_datasets() -> int:
    for name in VALID_DATASETS:
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return cmd_train(args)
        if args.command == "list-models":
            return cmd_list_models()
        if args.command == "list-datasets":
            return cmd_list_datasets()
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
