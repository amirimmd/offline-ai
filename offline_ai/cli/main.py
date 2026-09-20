"""CLI entry point for offline-ai."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON

from offline_ai import __version__
from offline_ai.core.engine import LocalAI

app = typer.Typer(
    name="offline-ai",
    help="Fully offline AI system with persistent memory and grounded citations.",
    add_completion=False,
    no_args_is_help=True,
)
model_app = typer.Typer(help="Model management")
adapter_app = typer.Typer(help="Adapter / LoRA management")
app.add_typer(model_app, name="model")
app.add_typer(adapter_app, name="adapter")

console = Console()


def _ai(workspace: Path) -> LocalAI:
    return LocalAI(workspace)


@app.callback()
def main_callback(
    ctx: typer.Context,
    workspace: Path = typer.Option(
        Path("./workspace"),
        "--workspace",
        "-w",
        help="Workspace directory for persistent memory",
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    console.print(__version__)


@app.command("stats")
def stats_cmd(ctx: typer.Context) -> None:
    """Show workspace / component stats."""
    ai = _ai(ctx.obj["workspace"])
    console.print(JSON(json.dumps(ai.stats())))


@app.command("doctor")
def doctor_cmd(
    ctx: typer.Context,
    offline: bool = typer.Option(False, "--offline", help="Verify offline-ready policies"),
) -> None:
    """Environment and health checks."""
    ai = _ai(ctx.obj["workspace"])
    report = ai.doctor(offline=offline)
    console.print(JSON(json.dumps(report)))
    raise typer.Exit(code=0 if report.get("ok") else 1)


@app.command("ingest")
def ingest_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="File to ingest"),
) -> None:
    """Ingest a file (JSON/JSONL/CSV/TXT/MD)."""
    ai = _ai(ctx.obj["workspace"])
    stats = ai.ingest_file(path)
    console.print(JSON(json.dumps(stats)))


@app.command("search")
def search_cmd(ctx: typer.Context, query: str = typer.Argument(...)) -> None:
    """Hybrid search."""
    ai = _ai(ctx.obj["workspace"])
    console.print(JSON(json.dumps(ai.search(query))))


@app.command("rebuild")
def rebuild_cmd(ctx: typer.Context) -> None:
    """Re-extract claims and relationships from all stored documents."""
    ai = _ai(ctx.obj["workspace"])
    result = ai.rebuild_knowledge()
    console.print(JSON(json.dumps(result, ensure_ascii=False)))


@app.command("chat")
def chat_cmd(
    ctx: typer.Context,
    cli: bool = typer.Option(False, "--cli", help="Use terminal loop instead of Persian window"),
) -> None:
    """Persian ask/add UI (window by default; --cli for terminal)."""
    if cli:
        from offline_ai.cli.repl import run_chat

        run_chat(ctx.obj["workspace"])
        return
    from offline_ai.cli.persian_ui import run_persian_ui

    run_persian_ui(ctx.obj["workspace"])


@app.command("ask")
def ask_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json", help="Print full JSON result"),
) -> None:
    """Grounded ask with citations."""
    from offline_ai.utils.console import print_text

    ai = _ai(ctx.obj["workspace"])
    result = ai.ask(query)
    if json_out:
        console.print(JSON(json.dumps(result, ensure_ascii=False)))
        return
    print_text(str(result.get("answer") or ""))


@app.command("document")
def document_cmd(ctx: typer.Context, document_id: str = typer.Argument(...)) -> None:
    ai = _ai(ctx.obj["workspace"])
    doc = ai.get_document(document_id)
    if doc is None:
        console.print(f"[red]Not found:[/red] {document_id}")
        raise typer.Exit(code=1)
    console.print(JSON(json.dumps(doc)))


@app.command("entity")
def entity_cmd(ctx: typer.Context, entity_id: str = typer.Argument(...)) -> None:
    ai = _ai(ctx.obj["workspace"])
    ent = ai.get_entity(entity_id)
    if ent is None:
        console.print(f"[red]Not found:[/red] {entity_id}")
        raise typer.Exit(code=1)
    console.print(JSON(json.dumps(ent)))


@app.command("claim")
def claim_cmd(ctx: typer.Context, claim_id: str = typer.Argument(...)) -> None:
    ai = _ai(ctx.obj["workspace"])
    claim = ai.get_claim(claim_id)
    if claim is None:
        console.print(f"[red]Not found:[/red] {claim_id}")
        raise typer.Exit(code=1)
    console.print(JSON(json.dumps(claim)))


@app.command("backup")
def backup_cmd(ctx: typer.Context, dest: Path = typer.Argument(...)) -> None:
    from offline_ai.utils.backup import BackupManager

    ai = _ai(ctx.obj["workspace"])
    result = BackupManager(ai.workspace).backup(dest)
    console.print(JSON(json.dumps(result)))


@app.command("restore")
def restore_cmd(ctx: typer.Context, src: Path = typer.Argument(...)) -> None:
    from offline_ai.utils.backup import BackupManager

    ai = _ai(ctx.obj["workspace"])
    result = BackupManager(ai.workspace).restore(src)
    console.print(JSON(json.dumps(result)))


@model_app.command("list")
def model_list(ctx: typer.Context) -> None:
    ai = _ai(ctx.obj["workspace"])
    console.print(JSON(json.dumps(ai.settings.models_raw.get("models", {}))))


@model_app.command("status")
def model_status(ctx: typer.Context) -> None:
    ai = _ai(ctx.obj["workspace"])
    hw = ai.detect_hardware()
    console.print(JSON(json.dumps(hw)))


@adapter_app.command("list")
def adapter_list(ctx: typer.Context) -> None:
    from offline_ai.training.adapters import AdapterManager

    ai = _ai(ctx.obj["workspace"])
    mgr = AdapterManager(ai.settings.adapters_dir)
    console.print(JSON(json.dumps(mgr.list_adapters())))


@adapter_app.command("train")
def adapter_train(ctx: typer.Context) -> None:
    from offline_ai.training.adapters import AdapterManager

    ai = _ai(ctx.obj["workspace"])
    mgr = AdapterManager(ai.settings.adapters_dir)
    console.print(JSON(json.dumps(mgr.train_adapter())))


@adapter_app.command("activate")
def adapter_activate(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None),
) -> None:
    from offline_ai.training.adapters import AdapterManager

    if not name:
        console.print("[red]adapter name required[/red]")
        raise typer.Exit(code=1)
    ai = _ai(ctx.obj["workspace"])
    mgr = AdapterManager(ai.settings.adapters_dir)
    console.print(JSON(json.dumps(mgr.activate_adapter(name))))


if __name__ == "__main__":
    app()
