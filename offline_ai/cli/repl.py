"""Interactive Persian chat — GUI by default (terminal RTL is unreliable)."""

from __future__ import annotations

from pathlib import Path

from offline_ai.core.engine import LocalAI
from offline_ai.utils.console import configure_stdio, print_text


_EXIT = {"exit", "quit", "q", "خروج", "/exit", "/quit"}
_HELP = {"help", "h", "?", "/help", "راهنما"}

_DEFAULT_SEED = "علی رفت به اسپانیا در روز 18 دی 1404 و یک جلسه مهم با رضا پهلوی داشته است."


def run_chat(workspace: str | Path = "./workspace", *, seed: str | None = None) -> None:
    """Plain terminal loop (fallback). Prefer run_persian_ui for Persian typing."""
    configure_stdio()
    ai = LocalAI(workspace)
    if seed and ai.stats().get("documents", 0) == 0:
        ai.add(seed)

    print("----------------------------------------")
    print("offline-ai terminal mode (Persian typing may look wrong here)")
    print("For correct Persian use:  python -m offline_ai")
    print("add <text>  |  stats  |  help  |  exit")
    print(f"workspace: {ai.workspace}")
    print("----------------------------------------")

    while True:
        try:
            raw = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not raw:
            continue
        low = raw.lower()
        if low in _EXIT:
            break
        if low in _HELP:
            print("add <text>  store knowledge")
            print("stats       memory size")
            print("exit        quit")
            continue
        if low == "stats":
            print(f"documents: {ai.stats().get('documents', 0)}")
            continue
        if low.startswith("add ") or raw.startswith("+ "):
            text = raw.split(" ", 1)[1].strip()
            if not text:
                print("add <text>")
                continue
            result = ai.add(text)
            print(f"stored documents_added={result.get('documents_added', 0)}")
            continue
        try:
            answer = ai.ask(raw, text_only=True)
        except Exception as exc:
            print(f"error: {exc}")
            continue
        print("")
        print_text(str(answer or ""))
        print("")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ai — پرسش و پاسخ فارسی با حافظهٔ آفلاین",
    )
    parser.add_argument("-w", "--workspace", default="./workspace")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use terminal loop instead of the Persian window",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not seed sample Persian knowledge into an empty workspace",
    )
    args = parser.parse_args(argv)
    seed = None if args.no_seed else _DEFAULT_SEED
    if args.cli:
        run_chat(args.workspace, seed=seed)
        return
    from offline_ai.cli.persian_ui import run_persian_ui

    run_persian_ui(args.workspace, seed=seed)
