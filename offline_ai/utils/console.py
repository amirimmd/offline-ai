"""UTF-8 console setup. Print Persian in logical Unicode so letters join."""

from __future__ import annotations

import os
import sys


_CONFIGURED = False


def configure_stdio() -> None:
    """Force UTF-8 on stdin/stdout/stderr (Windows cp1252 otherwise breaks Persian)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint(0)
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004 | 0x0008)
        except Exception:
            pass
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def visual_line(line: str) -> str:
    """Keep logical order so the terminal can shape and join Persian letters."""
    return line


def format_console_text(text: str, *, align: bool | None = None) -> str:
    configure_stdio()
    return text or ""


def print_text(text: str, *, file=None, end: str = "\n", align: bool | None = None) -> None:
    """Print UTF-8 Persian without reversing letters or words."""
    configure_stdio()
    stream = file or sys.stdout
    stream.write(f"{format_console_text(text)}{end}")
    try:
        stream.flush()
    except Exception:
        pass
