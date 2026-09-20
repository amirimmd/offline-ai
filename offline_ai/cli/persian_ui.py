"""ai — Persian offline chat window (reliable tkinter layout)."""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from offline_ai.core.engine import LocalAI
from offline_ai.ingestion.bulk import should_split_multiline, split_text_to_lines
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
_WARN = "#fbbf24"
_BORDER = "#374151"

# Paste larger than this is written to a temp file (avoids tk/memory crash).
_MAX_SAFE_PASTE_LINES = 400
_MAX_SAFE_PASTE_CHARS = 120_000

_FONT_FAMILY: str | None = None


def _font(size: int = 12, weight: str = "normal") -> tuple:
    global _FONT_FAMILY
    if _FONT_FAMILY is None:
        try:
            families = {f.lower() for f in tkfont.families()}
        except Exception:
            families = set()
        _FONT_FAMILY = "Tahoma"
        for name in _PERSIAN_FONTS:
            if name.lower() in families:
                _FONT_FAMILY = name
                break
    return (_FONT_FAMILY, size, weight)


class AIChatApp:
    def __init__(self, workspace: str | Path, *, seed: str | None = None) -> None:
        configure_stdio()
        self.workspace = Path(workspace)
        self.seed = seed
        self.ai: LocalAI | None = None

        self.root = tk.Tk()
        self.root.title("ai")
        self.root.geometry("880x640")
        self.root.minsize(700, 520)
        self.root.configure(bg=_BG)

        self.mode = tk.StringVar(value="ask")
        self._busy = True
        self._ready = False
        self._pulse_job: str | None = None
        self._pulse_frame = 0
        self._progress_mark: str | None = None
        self._status_base = "در حال آماده‌سازی…"
        self._ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._alive = True
        self._last_history_progress_at = 0.0
        self._pending_progress: str | None = None

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_header()
        self._build_chat()
        self._build_composer()
        self._apply_busy_chrome(True, send_label="صبر کنید…")
        self._set_status(self._status_base)
        self._write(
            "system",
            "سلام. من ai هستم.\nفقط از دانش ذخیره‌شده شما جواب می‌دهم.\n"
            "برای حجم زیاد از «آپلود اکسل/فایل» استفاده کنید "
            "(هر ردیف/خط = یک سند).",
        )
        self._write("progress", "در حال آماده‌سازی حافظه — لطفاً صبر کنید…")
        self._start_pulse("آماده‌سازی")
        self.root.after(50, self._poll_ui_queue)

        threading.Thread(target=self._boot_worker, daemon=True, name="ai-boot").start()

    def _boot_worker(self) -> None:
        try:
            self._post("progress", "بارگذاری موتور آفلاین…")
            ai = LocalAI(self.workspace)
            docs = int(ai.stats().get("documents", 0) or 0)
            if self.seed and docs == 0:
                self._post("progress", "ذخیره دانش نمونه…")
                ai.add(self.seed, on_progress=lambda s: self._post("progress", s))
            # Do NOT rebuild_knowledge or preload GGUF here — that hung/crashed large workspaces.
            self.ai = ai
            self._post("boot_done", None)
        except Exception as exc:
            self._post("boot_fail", str(exc))

    def _boot_done(self) -> None:
        self._ready = True
        self._clear_progress_line()
        self._write("meta", "آماده — سؤال بپرسید، دانش ذخیره کنید، یا فایل آپلود کنید.")
        self._unlock_input()

    def _boot_fail(self, message: str) -> None:
        self._clear_progress_line()
        self._ready = False
        self._unlock_input()
        try:
            self.btn_send.configure(state="disabled", text="خطا")
        except tk.TclError:
            pass
        self._set_status("خطا در راه‌اندازی")
        self._write("system", f"راه‌اندازی ناموفق: {message}")
        try:
            messagebox.showerror("ai", message)
        except Exception:
            pass

    def _post(self, kind: str, payload: Any) -> None:
        if not self._alive:
            return
        self._ui_queue.put((kind, payload))

    def _poll_ui_queue(self) -> None:
        if not self._alive:
            return
        latest_progress: str | None = None
        events: list[tuple[str, Any]] = []
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "progress":
                    latest_progress = str(payload or "")
                else:
                    events.append((kind, payload))
        except queue.Empty:
            pass

        # Completion / errors first so the composer never stays locked.
        for kind, payload in events:
            try:
                if kind == "status":
                    self._set_status(str(payload or ""))
                elif kind == "boot_done":
                    self._boot_done()
                elif kind == "boot_fail":
                    self._boot_fail(str(payload or "unknown"))
                elif kind == "add_done":
                    self._done_add(payload if isinstance(payload, dict) else {})
                elif kind == "ask_done":
                    self._done_ask(str(payload or ""))
                elif kind == "fail":
                    self._fail(str(payload or "unknown"))
            except Exception as exc:
                try:
                    self._unlock_input()
                    self._write("system", f"خطای رابط: {exc}")
                except Exception:
                    self._unlock_input()

        # Progress only while still working (ignore stale progress after unlock).
        if latest_progress is not None and self._busy:
            self._on_progress_ui(latest_progress)

        if self._alive:
            try:
                self.root.after(80, self._poll_ui_queue)
            except tk.TclError:
                self._alive = False

    def _on_entry_click(self, _event=None):
        try:
            self.entry.configure(state="normal")
            self.entry.focus_set()
        except tk.TclError:
            pass

    def _focus_entry(self) -> None:
        try:
            self.entry.configure(state="normal")
            self.entry.focus_force()
            # Place cursor at end for continued typing
            self.entry.mark_set(tk.INSERT, tk.END)
        except tk.TclError:
            pass

    def _redirect_keys_to_entry(self, event):
        """If focus is on the disabled chat history, move typing to the input box."""
        try:
            focused = self.root.focus_get()
        except tk.TclError:
            return None
        if focused is self.entry:
            return None
        if focused is not self.history and focused is not None:
            return None
        if event.keysym in {
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Caps_Lock",
            "Escape",
            "Return",
        }:
            return None
        self._focus_entry()
        ch = event.char
        if ch and ch.isprintable() and not self._busy:
            try:
                self.entry.insert(tk.INSERT, ch)
                return "break"
            except tk.TclError:
                return None
        return None

    def _unlock_input(self) -> None:
        """Always re-enable composer after ask/add — never leave the entry locked."""
        self._busy = False
        self._stop_pulse()
        send_label = "ذخیره" if self.mode.get() == "add" else "ارسال"
        try:
            self.entry.configure(state="normal")
            self.btn_send.configure(state="normal", text=send_label)
            self.btn_clear.configure(state="normal")
            self.btn_upload.configure(state="normal")
            self.btn_ask.configure(state="normal")
            self.btn_add.configure(state="normal")
            self._paint_modes()
            self._set_status(self._status_text())
        except tk.TclError:
            pass
        try:
            self.root.after_idle(self._focus_entry)
        except tk.TclError:
            self._focus_entry()

    def _on_close(self) -> None:
        self._alive = False
        self._stop_pulse()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _status_text(self) -> str:
        if not self.ai:
            return "در حال آماده‌سازی…"
        try:
            st = self.ai.stats()
        except Exception:
            return "آماده"
        llm = st.get("llm") or "extractive"
        mode = "عمیق" if st.get("llm_deep") else "استخراجی"
        return f"{st.get('documents', 0)} سند · {mode} · {llm}"

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
            takefocus=0,
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
        self.history.tag_configure("progress", foreground=_WARN, justify=tk.RIGHT, lmargin1=16, rmargin=8)

    def _build_composer(self) -> None:
        bar = tk.Frame(self.root, bg=_BG, height=188)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(bar, bg=_BG)
        inner.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 14))
        inner.grid_columnconfigure(0, weight=1)

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

        self.hint_var = tk.StringVar(value="حالت سؤال: متن را بنویسید و ارسال کنید")
        tk.Label(modes, textvariable=self.hint_var, font=_font(9), fg=_MUTED, bg=_BG).pack(
            side=tk.LEFT
        )

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
            takefocus=1,
        )
        self.entry.grid(row=0, column=0, sticky="nsew")
        self.entry.bind("<Control-Return>", self._on_submit)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)
        self.entry.bind("<Button-1>", self._on_entry_click)
        # If focus landed on the disabled chat history, typing still goes to the entry.
        self.root.bind_all("<Key>", self._redirect_keys_to_entry, add="+")

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

        self.btn_upload = tk.Button(
            actions,
            text="آپلود اکسل/فایل",
            font=_font(11, "bold"),
            width=14,
            height=2,
            bg=_PANEL,
            fg=_TEXT,
            activebackground=_BORDER,
            activeforeground=_TEXT,
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            command=self._on_upload,
        )
        self.btn_upload.pack(side=tk.RIGHT, padx=(0, 8))

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
            text="حجم زیاد → آپلود فایل   |   Enter = ارسال",
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
        if self._busy:
            return
        self.mode.set(mode)
        self._paint_modes()
        if mode == "add":
            self.hint_var.set("ذخیره: چند خط کوچک OK · حجم زیاد → آپلود اکسل/فایل")
            self.btn_send.configure(text="ذخیره")
        else:
            self.hint_var.set("حالت سؤال: متن را بنویسید و ارسال کنید")
            self.btn_send.configure(text="ارسال")
        self.entry.focus_set()

    def _set_status(self, text: str) -> None:
        try:
            self.status_var.set(text)
        except tk.TclError:
            pass

    def _write(self, kind: str, text: str) -> None:
        try:
            self.history.configure(state=tk.NORMAL)
            if kind == "user":
                self.history.insert(tk.END, "شما\n", "user_h")
                self.history.insert(tk.END, text.rstrip() + "\n\n", "user")
            elif kind == "answer":
                self.history.insert(tk.END, "ai\n", "ai_h")
                self.history.insert(tk.END, text.rstrip() + "\n\n", "answer")
            elif kind == "meta":
                self.history.insert(tk.END, text.rstrip() + "\n\n", "meta")
            elif kind == "progress":
                self._clear_progress_line(keep_mark=False)
                start = self.history.index(tk.END)
                self.history.insert(tk.END, "● " + text.rstrip() + "\n\n", "progress")
                self._progress_mark = start
            else:
                self.history.insert(tk.END, text.rstrip() + "\n\n", "system")
            self.history.configure(state=tk.DISABLED)
            self.history.see(tk.END)
            if not self._busy and kind in {"answer", "meta", "system"}:
                try:
                    self.root.after_idle(self._focus_entry)
                except tk.TclError:
                    pass
        except tk.TclError:
            pass

    def _clear_progress_line(self, *, keep_mark: bool = True) -> None:
        try:
            self.history.configure(state=tk.NORMAL)
            ranges = self.history.tag_ranges("progress")
            pairs = list(zip(ranges[0::2], ranges[1::2]))
            for a, b in reversed(pairs):
                self.history.delete(a, b)
            self.history.configure(state=tk.DISABLED)
        except tk.TclError:
            pass
        if not keep_mark:
            self._progress_mark = None

    def _update_progress_line(self, text: str) -> None:
        """Status always; history at most ~2 Hz to avoid tk hang."""
        now = time.monotonic()
        if now - self._last_history_progress_at < 0.5:
            self._pending_progress = text
            return
        self._last_history_progress_at = now
        self._pending_progress = None
        try:
            self.history.configure(state=tk.NORMAL)
            ranges = self.history.tag_ranges("progress")
            if ranges:
                pairs = list(zip(ranges[0::2], ranges[1::2]))
                for a, b in reversed(pairs):
                    self.history.delete(a, b)
            self.history.insert(tk.END, "● " + text.rstrip() + "\n\n", "progress")
            self._progress_mark = self.history.index("end-2l")
            self.history.configure(state=tk.DISABLED)
            self.history.see(tk.END)
        except tk.TclError:
            pass

    def _on_progress_ui(self, stage: str) -> None:
        msg = (stage or "").strip()
        if not msg:
            return
        self._status_base = msg
        self._set_status(msg)
        self._update_progress_line(msg)

    def _start_pulse(self, label: str) -> None:
        self._stop_pulse()
        self._pulse_frame = 0
        self._status_base = label

        def tick() -> None:
            if not self._alive or not self._busy:
                return
            base = self._pending_progress or self._status_base
            dots = "." * (1 + (self._pulse_frame % 3))
            if "%" in base or "/" in base:
                self._set_status(base)
            else:
                self._set_status(f"{base}{dots}")
            self._pulse_frame += 1
            try:
                self._pulse_job = self.root.after(500, tick)
            except tk.TclError:
                self._pulse_job = None

        tick()

    def _stop_pulse(self) -> None:
        if self._pulse_job is not None:
            try:
                self.root.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None

    def _apply_busy_chrome(self, busy: bool, *, send_label: str | None = None) -> None:
        # Entry must NEVER be disabled — only action buttons while work runs.
        state = "disabled" if busy else "normal"
        if send_label is None:
            if busy:
                send_label = "صبر کنید…"
            else:
                send_label = "ذخیره" if self.mode.get() == "add" else "ارسال"
        try:
            self.btn_send.configure(state=state, text=send_label)
            self.btn_clear.configure(state=state)
            self.btn_upload.configure(state=state)
            self.btn_ask.configure(state=state)
            self.btn_add.configure(state=state)
            self.entry.configure(state="normal")
            if not busy:
                self._paint_modes()
        except tk.TclError:
            pass

    def _set_busy(self, busy: bool, *, label: str | None = None) -> None:
        if busy:
            self._busy = True
            self._apply_busy_chrome(True, send_label="صبر کنید…")
            self._start_pulse(label or "در حال پردازش")
            try:
                self.entry.configure(state="normal")
            except tk.TclError:
                pass
        else:
            self._unlock_input()

    def _clear(self) -> None:
        try:
            self.entry.configure(state="normal")
            self.entry.delete("1.0", tk.END)
            self._focus_entry()
        except tk.TclError:
            pass

    def _read(self) -> str:
        return self.entry.get("1.0", "end-1c").strip()

    def _on_return(self, event):
        if event.state & 0x0001:
            return None
        self._on_submit()
        return "break"

    def _on_submit(self, _event=None):
        if self._busy or not self._ready or self.ai is None:
            return "break"
        text = self._read()
        if not text:
            return "break"
        try:
            self.entry.configure(state="normal")
            self.entry.delete("1.0", tk.END)
        except tk.TclError:
            pass
        if self.mode.get() == "add":
            self._do_add(text)
        else:
            self._do_ask(text)
        try:
            self.entry.configure(state="normal")
            self.root.after_idle(self._focus_entry)
        except tk.TclError:
            pass
        return "break"

    def _on_upload(self) -> None:
        if self._busy or not self._ready or self.ai is None:
            return
        path = filedialog.askopenfilename(
            title="انتخاب فایل دانش",
            filetypes=[
                ("Excel", "*.xlsx *.xlsm"),
                ("CSV", "*.csv"),
                ("Text / Markdown", "*.txt *.md"),
                ("JSON / JSONL", "*.json *.jsonl"),
                ("All supported", "*.xlsx *.xlsm *.csv *.txt *.md *.json *.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.mode.set("add")
        self._paint_modes()
        self._do_ingest_file(path)

    def _do_ingest_file(self, path: str) -> None:
        assert self.ai is not None
        name = Path(path).name
        self._write("user", f"آپلود فایل: {name}")
        self._write("progress", f"شروع خواندن {name}…")
        self._set_busy(True, label="آپلود فایل")

        def work() -> None:
            try:
                result = self.ai.ingest_file(
                    path,
                    on_progress=lambda s: self._post("progress", s),
                )
                self._post("add_done", result)
            except Exception as exc:
                self._post("fail", str(exc))

        threading.Thread(target=work, daemon=True, name="ai-ingest").start()

    def _do_add(self, text: str) -> None:
        assert self.ai is not None
        lines = split_text_to_lines(text)
        if len(lines) > _MAX_SAFE_PASTE_LINES or len(text) > _MAX_SAFE_PASTE_CHARS:
            self._write(
                "system",
                f"متن خیلی بزرگ است ({len(lines):,} خط). "
                "به‌صورت فایل موقت ذخیره و وارد می‌شود…",
            )
            tmp = Path(tempfile.gettempdir()) / f"offline_ai_paste_{os.getpid()}.txt"
            tmp.write_text(text, encoding="utf-8")
            self._do_ingest_file(str(tmp))
            return

        bulk = should_split_multiline(text)
        preview = text if len(text) < 400 else text[:400] + "…"
        if bulk:
            self._write("user", f"{len(lines):,} خط دانش\n{preview}")
            self._write("progress", f"شروع ذخیره دسته‌ای {len(lines):,} سند…")
        else:
            self._write("user", text)
            self._write("progress", "شروع ذخیره دانش…")
        self._set_busy(True, label="ذخیره دانش")

        def work() -> None:
            try:
                result = self.ai.add(
                    text,
                    on_progress=lambda s: self._post("progress", s),
                )
                self._post("add_done", result)
            except Exception as exc:
                self._post("fail", str(exc))

        threading.Thread(target=work, daemon=True, name="ai-add").start()

    def _done_add(self, result: dict[str, Any]) -> None:
        try:
            self._clear_progress_line(keep_mark=False)
            added = int(result.get("documents_added", 0) or 0)
            dupes = int(result.get("duplicates", 0) or 0)
            received = int(result.get("documents_received", 0) or added or 0)
            errors = int(result.get("errors", 0) or 0)
            ms = int(result.get("duration_ms", 0) or 0)
            if added > 0:
                self._write(
                    "meta",
                    f"ذخیره شد — {added:,} سند جدید از {received:,} ردیف"
                    + (f" · تکراری {dupes:,}" if dupes else "")
                    + (f" · خطا {errors:,}" if errors else "")
                    + (f" · {ms:,}ms" if ms else ""),
                )
            elif dupes > 0:
                self._write("meta", f"همه تکراری بودند — {dupes:,} ردیف.")
            else:
                self._write("meta", "ذخیره انجام شد.")
        finally:
            self._unlock_input()

    def _do_ask(self, query: str) -> None:
        assert self.ai is not None
        self._write("user", query)
        self._write("progress", "در حال جست‌وجوی شواهد و تولید پاسخ…")
        self._set_busy(True, label="پاسخ")

        def work() -> None:
            try:
                answer = str(self.ai.ask(query, text_only=True) or "پاسخی تولید نشد.")
                self._post("ask_done", answer)
            except Exception as exc:
                self._post("fail", f"خطا در پاسخ: {exc}")

        threading.Thread(target=work, daemon=True, name="ai-ask").start()

    def _done_ask(self, answer: str) -> None:
        try:
            self._clear_progress_line(keep_mark=False)
            self._write("answer", answer)
        finally:
            self._unlock_input()

    def _fail(self, message: str) -> None:
        try:
            self._clear_progress_line(keep_mark=False)
            self._write("system", f"خطا: {message}")
        finally:
            self._unlock_input()
        try:
            messagebox.showerror("ai", message[:800])
        except Exception:
            pass
        self._focus_entry()

    def run(self) -> None:
        self.root.mainloop()


PersianChatApp = AIChatApp


def run_persian_ui(workspace: str | Path = "./workspace", *, seed: str | None = None) -> None:
    AIChatApp(workspace, seed=seed).run()


def run_ai_ui(workspace: str | Path = "./workspace", *, seed: str | None = None) -> None:
    run_persian_ui(workspace, seed=seed)
