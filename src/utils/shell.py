"""Interactive REPL shell ("builder") for the model-comparison Runner.

Launched via ``python -m src.train builder``. Built on ``prompt_toolkit``
(completion + persistent history) and ``rich`` (banner panel + styled output).

User input is parsed with the same argparse parser as the one-shot CLI and
routed **in-process** to the existing handlers in ``src/train.py`` (which call
into ``src/runner.py`` and ``src/utils/config.py``) — no subprocesses.

Typing ``train`` with no flags opens the interactive InquirerPy form
(``src.utils.interactive_train``); explicit flags keep working unchanged.

Built-in shell commands: ``help``, ``clear``, ``exit``/``quit`` (Ctrl+D also
exits; Ctrl+C cancels the current line).
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.runner import RunnerError
from src.utils.config import ConfigError

_HISTORY_PATH = Path(__file__).resolve().parents[2] / "logs" / "builder_history"

# Claude Code-inspired palette: soft gray chrome, soft teal highlights,
# amber for warnings/smoke-test notices.
_BORDER = "#5C6370"
_TEAL = "#56B6C2"
_AMBER = "#E5C07B"

_BANNER = (
    f"[bold {_TEAL}]netml-cl builder[/bold {_TEAL}]\n"
    "Interactive shell for the model-comparison Runner.\n"
    f"Type [bold {_TEAL}]help[/bold {_TEAL}] for commands, "
    f"[bold {_TEAL}]train[/bold {_TEAL}] for the interactive form, "
    "or [bold]exit[/bold] (Ctrl+D) to quit."
)

_SHELL_COMMANDS = ("train", "list-models", "list-datasets", "show-config", "help", "clear", "exit")

_HELP_ROWS = [
    ("train", "Open the interactive training form (model / dataset / seeds / limit)."),
    ("train --model DIR --dataset DS [--config F] [--seeds N ...] [--limit N]",
     "Train and evaluate a model (same flags as the CLI)."),
    ("show-config --model DIR --dataset DS [--config F]",
     "Print the fully merged config without running anything."),
    ("list-models", "List runnable model directories."),
    ("list-datasets", "List the valid datasets."),
    ("help", "Show this command table."),
    ("clear", "Clear the screen."),
    ("exit | quit | Ctrl+D", "Leave the builder shell."),
]


def _print_banner(console: Console) -> None:
    """Render the startup banner as a soft-gray borderless-feel panel."""
    console.print(
        Panel(
            _BANNER,
            border_style=_BORDER,
            title=f"[{_TEAL}]builder[/{_TEAL}]",
            title_align="left",
            expand=False,
            padding=(0, 2),
        )
    )


def _print_help(console: Console) -> None:
    """Print the command reference as a borderless key-value grid."""
    table = Table(
        show_header=False,
        box=None,
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("command", style=_TEAL, overflow="fold")
    table.add_column("description", style="default", overflow="fold")
    for cmd, desc in _HELP_ROWS:
        table.add_row(cmd, desc)
    console.print(
        Panel(
            table,
            border_style=_BORDER,
            title=f"[{_TEAL}]commands[/{_TEAL}]",
            title_align="left",
            expand=False,
            padding=(0, 1),
        )
    )


def _route(tokens: list[str], console: Console) -> int:
    """Parse ``tokens`` with the CLI parser and dispatch in-process.

    Reuses ``src.train.build_parser()`` and the ``cmd_*`` handlers so the shell
    and the one-shot CLI can never drift apart. Returns a process-style exit
    code; argparse errors are reported and treated as "keep the shell alive".
    """
    import src.train as train_cli

    if not tokens:
        return 0

    # Bare ``train`` (no flags): fall back to the interactive form instead of
    # failing on the required --model/--dataset arguments.
    if tokens == ["train"]:
        from src.utils.interactive_train import run_interactive_train

        interactive_tokens = run_interactive_train(console)
        if interactive_tokens is None:
            return 0
        tokens = interactive_tokens

    try:
        args = train_cli.build_parser().parse_args(tokens)
    except SystemExit:
        # argparse calls exit() on bad usage / -h; keep the REPL alive instead.
        return 0
    try:
        if args.command == "train":
            with console.status(f"[{_TEAL}]training…[/{_TEAL}]", spinner="dots"):
                return train_cli.cmd_train(args)
        if args.command == "show-config":
            return train_cli.cmd_show_config(args)
        if args.command == "list-models":
            return train_cli.cmd_list_models()
        if args.command == "list-datasets":
            return train_cli.cmd_list_datasets()
    except (RunnerError, ConfigError) as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        return 2
    return 0


def _builtin(tokens: list[str], console: Console) -> int | None:
    """Handle shell built-ins; return None if ``tokens`` is not a built-in."""
    name = tokens[0]
    if name in ("exit", "quit"):
        return 0
    if name == "clear":
        console.clear()
        return 0
    if name == "help":
        _print_help(console)
        return 0
    if name == "-h" or name == "--help":
        _print_help(console)
        return 0
    return None


def run_builder() -> int:
    """Run the interactive builder REPL. Returns a process-style exit code."""
    console = Console()
    _print_banner(console)

    try:
        history: FileHistory | None = FileHistory(str(_HISTORY_PATH))
    except OSError:
        history = None  # history is a convenience, never a hard requirement

    session: PromptSession[str] = PromptSession(
        history=history,
        completer=WordCompleter(_SHELL_COMMANDS, sentence=True),
        enable_history_search=True,  # Ctrl+R / Up-arrow history
    )

    while True:
        try:
            line = session.prompt("builder > ").strip()
        except KeyboardInterrupt:
            console.print("[dim](^C — type 'exit' to quit)[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]bye[/dim]")
            return 0

        if not line:
            continue

        tokens = line.split()
        handled = _builtin(tokens, console)
        if handled is not None:
            if handled == 0 and tokens[0] in ("exit", "quit"):
                console.print("[dim]bye[/dim]")
                return 0
            continue

        _route(tokens, console)


if __name__ == "__main__":
    raise SystemExit(run_builder())
