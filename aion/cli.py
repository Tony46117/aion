"""aion CLI - the terminal app.

Rich-powered opencode-style session. Commands:

  /chat   -> menu: talk (conversation) or deep (deep analysis session)
  /voice  -> toggle Jung's voice on/off
  /stats  -> what aion has learned
  /forget -> wipe memory
  /help   -> command list
  /quit   -> leave
"""
from __future__ import annotations

import random
import sys
import threading

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .engine import Engine
from . import voice

console = Console()

BANNER = r"""
    ▄▄▄· .▄▄ ·  ▄▄· ▄ •▄ ▪   ▄▄· ▄▄▄ .▄▄▄
   ▐█ ▀█ ▐█ ▀. ▐█ ▌▪█▌▄▌▪██  ▐█ ▌▪▀▄.▀·▀▄ █·
   ▄█▀▀█ ▄▀▀▀█▄██ ▄▄▐▀▀▄.▀▐█·██ ▄▄▐▀▀▀▄▐▀▀▄
   ▐█ ▪▐▌▐█▄▪▐█▐███▌▐█.█▌▐█▌▐███▌▐█▄▪▐█▐█•█▌
    ▀  ▀  ▀▀▀▀ ·▀▀▀ ·▀  ▀▀▀▀·▀▀▀ ▀▀▀▀ .▀  ▀
"""

TAGLINES = [
    "the psyche is the world's pivot",
    "who looks outside, dreams; who looks inside, awakes",
    "until you make the unconscious conscious, it will direct you",
]


def _panel(text: str, title: str, border: str) -> Panel:
    return Panel(
        Text(text, style="white"),
        title=f"[bold]{title}[/bold]",
        border_style=border,
        padding=(0, 1),
        expand=False,
    )


def _print_help() -> None:
    t = Table.grid(padding=(0, 2))
    t.add_row("[bold cyan]/chat[/]", "start a session - talk (conversation) or deep (analysis)")
    t.add_row("[bold cyan]/voice[/]", "toggle Jung's voice on/off")
    t.add_row("[bold cyan]/stats[/]", "what aion has learned so far")
    t.add_row("[bold cyan]/forget[/]", "wipe everything aion has learned from you")
    t.add_row("[bold cyan]/help[/]", "this list")
    t.add_row("[bold cyan]/quit[/]", "leave (memory is kept)")
    console.print(t)


def _banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(f"  [italic dim]{random.choice(TAGLINES)}[/italic dim]")
    console.print(Rule(style="dim"))
    console.print("[dim]  /chat to begin  ·  /help for commands  ·  /quit to leave[/dim]\n")


def _menu() -> str | None:
    """Return 'talk' | 'deep' | None."""
    console.print("[bold]How shall we speak?[/bold]")
    console.print("  [cyan]1[/] talk - conversation")
    console.print("  [cyan]2[/] deep - deep analysis session")
    try:
        choice = console.input("[bold]aion ›[/bold] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice in {"1", "talk", "t"}:
        return "talk"
    if choice in {"2", "deep", "d"}:
        return "deep"
    console.print("[dim]perhaps another time.[/dim]")
    return None


def _speak_async(text: str, deep: bool, enabled: bool) -> None:
    if enabled:
        threading.Thread(target=voice.speak, args=(text, None, deep), daemon=True).start()


def main() -> int:
    rng = random.Random()
    _banner()

    with console.status("[dim]aion is gathering itself...[/dim]"):
        try:
            engine = Engine()
        except Exception as e:  # pragma: no cover
            console.print(f"[red]failed to load models:[/] {e}")
            console.print("[dim]run: python -m aion.train --epochs 120[/dim]")
            return 1

    voice_on = False
    console.print(
        f"[dim]memory: {engine.stats()['remembered_turns']} turns remembered · "
        f"corpus: {engine.stats()['knowledge_base']:,} paragraphs · "
        f"voice: {'on' if voice_on else 'off'}[/dim]\n"
    )

    mode: str | None = None
    deep = False

    while True:
        try:
            line = console.input("[bold magenta]you ›[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]good night.[/dim]")
            break
        if not line:
            continue

        cmd = line.lower()

        if cmd in {"/quit", "/exit", "/q"}:
            console.print("[dim]\"The meeting of two personalities... both are transformed.\" Good night.[/dim]")
            break

        if cmd == "/help":
            _print_help()
            continue

        if cmd == "/stats":
            s = engine.stats()
            t = Table.grid(padding=(0, 2))
            t.add_row("remembered turns", f"[cyan]{s['remembered_turns']}[/]")
            t.add_row("knowledge base", f"[cyan]{s['knowledge_base']:,}[/] paragraphs")
            t.add_row("teachings head", s["teachings"])
            t.add_row("style head", s["style_model"])
            t.add_row("favorite topic of yours", s.get("mood", {}).get("favorite_topic") or "-")
            t.add_row("memory file", s.get("memory_file", "-"))
            console.print(Panel(t, title="aion's mind", border_style="cyan", expand=False))
            continue

        if cmd == "/forget":
            engine.learner.forget()
            console.print("[dim]the slate is clean. we begin again.[/dim]")
            continue

        if cmd == "/voice":
            if not voice.espeak_available():
                console.print("[red]espeak-ng is not installed.[/] [dim]sudo apt-get install espeak-ng[/dim]")
                continue
            voice_on = not voice_on
            state = "on - you will hear me" if voice_on else "off - silence again"
            console.print(f"[cyan]voice: {state}[/cyan]")
            if voice_on:
                _speak_async("I am here. Let us begin.", False, True)
            continue

        if cmd == "/chat":
            mode = _menu()
            if mode is None:
                continue
            deep = mode == "deep"
            if deep:
                console.print(
                    Panel(
                        "A deep session. Speak of what troubles you; I will not\n"
                        "offer comfort, only the truth as it appears between us.\n"
                        "[dim](type /chat again to switch, /quit to leave)[/dim]",
                        title="deep analysis",
                        border_style="red",
                        expand=False,
                    )
                )
                greeting = (
                    "Sit with me as long as you need. What brings you to this hour?"
                )
            else:
                console.print(
                    Panel(
                        "A conversation. Speak freely; I will answer as myself.\n"
                        "[dim](type /chat again to switch, /quit to leave)[/dim]",
                        title="talk",
                        border_style="green",
                        expand=False,
                    )
                )
                greeting = "So - what shall we speak of?"
            console.print(_panel(greeting, "aion", "magenta"))
            _speak_async(greeting, deep, voice_on)
            continue

        # free-form line: if no mode chosen yet, treat as talk
        if mode is None:
            mode, deep = "talk", False
            console.print("[dim](talk mode - use /chat for deep analysis)[/dim]")

        with console.status("[dim]aion considers...[/dim]"):
            reply = engine.reply(line, deep=deep)

        console.print()
        console.print(_panel(reply, "aion", "red" if deep else "magenta"))
        _speak_async(reply, deep, voice_on)
        console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
