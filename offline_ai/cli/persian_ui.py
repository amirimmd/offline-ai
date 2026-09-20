"""ai — Persian offline chat window (reliable tkinter layout)."""

from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox

from offline_ai.core.engine import LocalAI
from offline_ai.utils.console import configure_stdio

_PERSIAN_FONTS = (
    "Vazirmatn",
    "Vazir",
    "IRANSansX",
    "IRANSans",
    "B Nazanin",
    "Tahoma",
    "Segoe UI",
    "Arial",
)

_BG = "#111827"
_PANEL = "#1f2937"
_INPUT_BG = "#ffffff"
_INPUT_FG = "#111827"
_TEXT = "#f3f4f6"
_MUTED = "#9ca3af"
_ACCENT = "#2563eb"
_ACCENT_HOVER = "#1d4ed8"
_OK = "#10b981"
_BORDER = "#374151"


def _font(size: int = 12, weight: str = "normal") -> tuple:
    families = {f.lower() for f in tkfont.families()}
    family = "Tahoma"
    for name in _PERSIAN_FONTS:
        if name.lower() in families:
            family = name
            break
    return (family, size, weight)


class AIChatApp:
    def __init__(self, workspace: str | Path, *, seed: str | None = None) -> None:
        configure_stdio()
        self.ai = LocalAI(workspace)
        if seed and self.ai.stats().get("documents", 0) == 0:
            self.ai.add(seed)
        # Strengthen relations for existing workspaces (idempotent re-extract).
        if self.ai.stats().get("documents", 0) > 0:
            try:
                self.ai.rebuild_knowledge()
            except Exception:
                pass

        self.root = tk.Tk()
        self.root.title("ai")
        self.root.geometry("880x640")
        self.root.minsize(700, 520)
        self.root.configure(bg=_BG)

        self.mode = tk.StringVar(value="ask")
        self._busy = False

        # Root grid: header / chat / composer — composer always visible
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_chat()
        self._build_composer()

        self._set_status(self._status_text())
        # Warm up deep LLM in background so first answer is faster
        if self.ai.stats().get("llm_deep"):
            threading.Thread(target=self._warmup_llm, daemon=True).start()
        self.root.after(100, self._ready)

    def _status_text(self) -> str:
        st = self.ai.stats()
        llm = st.get("llm") or "extractive"
        mode = "عمیق" if st.get("llm_deep") else "استخراجی"
        return f"{st.get('documents', 0)} سند · {mode} · {llm}"

    def _warmup_llm(self) -> None:
        try:
            self.ai.llm.load()
            self.root.after(0, lambda: self._set_status(self._status_text() + " · آماده"))
        except Exception:
            pass

    def _ready(self) -> None:
        self.entry.configure(state=tk.NORMAL)
        self.entry.focus_set()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=_BG, height=64)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        tk.Label(header, text="ai", font=_font(26, "bold"), fg=_TEXT, bg=_BG).grid(
            row=0, column=0, padx=(20, 8), pady=12, sticky="w"
        )
        tk.Label(
            header,
            text="حافظه آفلاین · پاسخ مبتنی بر شواهد",
            font=_font(10),
            fg=_MUTED,
            bg=_BG,
        ).grid(row=0, column=1, sticky="w", pady=18)

        self.status_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.status_var, font=_font(10), fg=_MUTED, bg=_BG).grid(
            row=0, column=2, padx=20, sticky="e"
        )

    def _build_chat(self) -> None:
        wrap = tk.Frame(self.root, bg=_BG)
        wrap.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        panel = tk.Frame(wrap, bg=_PANEL, highlightbackground=_BORDER, highlightthickness=1)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.history = tk.Text(
            panel,
            wrap=tk.WORD,
            font=_font(12),
            bg=_PANEL,
            fg=_TEXT,
            relief=tk.FLAT,
            padx=16,
            pady=14,
            state=tk.DISABLED,
            cursor="arrow",
            highlightthickness=0,
            borderwidth=0,
        )
        scroll = tk.Scrollbar(panel, command=self.history.yview)
        self.history.configure(yscrollcommand=scroll.set)
        self.history.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self.history.tag_configure("user_h", foreground="#93c5fd", font=_font(10, "bold"), justify=tk.RIGHT)
        self.history.tag_configure("user", foreground=_TEXT, justify=tk.RIGHT, lmargin1=24, rmargin=8)
        self.history.tag_configure("ai_h", foreground=_OK, font=_font(10, "bold"), justify=tk.RIGHT)
        self.history.tag_configure("answer", foreground=_TEXT, justify=tk.RIGHT, lmargin1=24, rmargin=8)
        self.history.tag_configure("system", foreground=_MUTED, justify=tk.RIGHT, lmargin1=16, rmargin=8)
        self.history.tag_configure("meta", foreground=_OK, justify=tk.RIGHT, lmargin1=16, rmargin=8)

        self._write(
            "system",
            "سلام. من ai هستم.\nفقط از دانش ذخیره‌شده شما جواب می‌دهم.\n"
            "در کادر سفید پایین بنویسید و دکمه ارسال را بزنید.",
        )

    def _build_composer(self) -> None:
        # Fixed-height bottom bar so buttons never get crushed
        bar = tk.Frame(self.root, bg=_BG, height=168)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(bar, bg=_BG)
        inner.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 14))
        inner.grid_columnconfigure(0, weight=1)

        # Mode buttons — always visible, high contrast
        modes = tk.Frame(inner, bg=_BG)
        modes.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.btn_ask = tk.Button(
            modes,
            text="سؤال",
            font=_font(11, "bold"),
            width=10,
            height=1,
            command=lambda: self._set_mode("ask"),
            relief=tk.RAISED,
            bd=1,
            cursor="hand2",
        )
        self.btn_add = tk.Button(
            modes,
            text="ذخیره دانش",
            font=_font(11, "bold"),
            width=12,
            height=1,
            command=lambda: self._set_mode("add"),
            relief=tk.RAISED,
            bd=1,
            cursor="hand2",
        )
        self.btn_ask.pack(side=tk.RIGHT, padx=(6, 0))
        self.btn_add.pack(side=tk.RIGHT)
        self._paint_modes()

        # White input box — always enabled, no placeholder hacks
        input_row = tk.Frame(inner, bg=_INPUT_BG, highlightbackground=_ACCENT, highlightthickness=2)
        input_row.grid(row=1, column=0, sticky="ew")
        input_row.grid_columnconfigure(0, weight=1)

        self.entry = tk.Text(
            input_row,
            height=3,
            wrap=tk.WORD,
            font=_font(13),
            bg=_INPUT_BG,
            fg=_INPUT_FG,
            insertbackground=_ACCENT,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            undo=True,
            exportselection=False,
            highlightthickness=0,
            borderwidth=0,
        )
        self.entry.grid(row=0, column=0, sticky="nsew")
        self.entry.bind("<Control-Return>", self._on_submit)
        # Plain Return submits; Shift+Return = new line
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)

        # Action buttons row — large and obvious
        actions = tk.Frame(inner, bg=_BG)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.btn_send = tk.Button(
            actions,
            text="ارسال",
            font=_font(12, "bold"),
            width=12,
            height=2,
            bg=_ACCENT,
            fg="#ffffff",
            activebackground=_ACCENT_HOVER,
            activeforeground="#ffffff",
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            command=self._on_submit,
        )
        self.btn_send.pack(side=tk.RIGHT)

        self.btn_clear = tk.Button(
            actions,
            text="پاک کردن",
            font=_font(11),
            width=10,
            height=2,
            bg=_PANEL,
            fg=_TEXT,
            activebackground=_BORDER,
            activeforeground=_TEXT,
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            command=self._clear,
        )
        self.btn_clear.pack(side=tk.RIGHT, padx=(0, 8))

        tk.Label(
            actions,
            text="Enter = ارسال   |   Shift+Enter = خط جدید",
            font=_font(9),
            fg=_MUTED,
            bg=_BG,
        ).pack(side=tk.LEFT)

    def _paint_modes(self) -> None:
        ask = self.mode.get() == "ask"
        self.btn_ask.configure(
            bg=_ACCENT if ask else _PANEL,
            fg="#ffffff" if ask else _TEXT,
            activebackground=_ACCENT_HOVER if ask else _BORDER,
            activeforeground="#ffffff",
        )
        self.btn_add.configure(
            bg=_ACCENT if not ask else _PANEL,
            fg="#ffffff" if not ask else _TEXT,
            activebackground=_ACCENT_HOVER if not ask else _BORDER,
            activeforeground="#ffffff",
        )

    def _set_mode(self, mode: str) -> None:
        self.mode.set(mode)
        self._paint_modes()
        self.entry.focus_set()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _write(self, kind: str, text: str) -> None:
        self.history.configure(state=tk.NORMAL)
        if kind == "user":
            self.history.insert(tk.END, "شما\n", "user_h")
            self.history.insert(tk.END, text.rstrip() + "\n\n", "user")
        elif kind == "answer":
            self.history.insert(tk.END, "ai\n", "ai_h")
            self.history.insert(tk.END, text.rstrip() + "\n\n", "answer")
        elif kind == "meta":
            self.history.insert(tk.END, text.rstrip() + "\n\n", "meta")
        else:
            self.history.insert(tk.END, text.rstrip() + "\n\n", "system")
        self.history.configure(state=tk.DISABLED)
        self.history.see(tk.END)

    def _clear(self) -> None:
        self.entry.configure(state=tk.NORMAL)
        self.entry.delete("1.0", tk.END)
        self.entry.focus_set()

    def _read(self) -> str:
        return self.entry.get("1.0", "end-1c").strip()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_send.configure(state=state, text="صبر کنید…" if busy else "ارسال")
        self.btn_clear.configure(state=state)
        self.btn_ask.configure(state=state)
        self.btn_add.configure(state=state)
        # Keep Text editable unless busy — when busy, freeze input
        self.entry.configure(state=state)
        if busy:
            self._set_status("در حال کار…")
        else:
            self._set_status(self._status_text())
            self.entry.focus_set()

    def _on_return(self, event):
        # Shift+Return already allowed through; plain Return submits
        if event.state & 0x0001:  # Shift
            return None
        self._on_submit()
        return "break"

    def _on_submit(self, _event=None):
        if self._busy:
            return "break"
        text = self._read()
        if not text:
            return "break"
        self.entry.delete("1.0", tk.END)
        if self.mode.get() == "add":
            self._do_add(text)
        else:
            self._do_ask(text)
        return "break"

    def _do_add(self, text: str) -> None:
        self._write("user", text)
        self._set_busy(True)

        def work() -> None:
            try:
                result = self.ai.add(text)
                n = int(result.get("documents_added", 0))
                self.root.after(0, lambda: self._done_add(n))
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self._fail(msg))

        threading.Thread(target=work, daemon=True).start()

    def _done_add(self, n: int) -> None:
        self._write("meta", f"ذخیره شد — {n} سند اضافه شد")
        self._set_busy(False)

    def _do_ask(self, query: str) -> None:
        self._write("user", query)
        self._set_busy(True)

        def work() -> None:
            try:
                answer = str(self.ai.ask(query, text_only=True) or "پاسخی تولید نشد.")
                self.root.after(0, lambda: self._done_ask(answer))
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self._fail(msg))

        threading.Thread(target=work, daemon=True).start()

    def _done_ask(self, answer: str) -> None:
        self._write("answer", answer)
        self._set_busy(False)

    def _fail(self, message: str) -> None:
        self._set_busy(False)
        self._write("system", f"خطا: {message}")
        messagebox.showerror("ai", message)

    def run(self) -> None:
        self.root.mainloop()


PersianChatApp = AIChatApp


def run_persian_ui(workspace: str | Path = "./workspace", *, seed: str | None = None) -> None:
    AIChatApp(workspace, seed=seed).run()


def run_ai_ui(workspace: str | Path = "./workspace", *, seed: str | None = None) -> None:
    run_persian_ui(workspace, seed=seed)
