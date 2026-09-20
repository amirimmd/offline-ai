"""ai UI layout smoke checks."""

import time

from offline_ai.cli.persian_ui import AIChatApp, PersianChatApp


def _pump_until_ready(app: AIChatApp, seconds: float = 25.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.root.update()
        if app._ready and not app._busy:
            return
        time.sleep(0.05)


def _pump_until_idle(app: AIChatApp, seconds: float = 25.0) -> None:
    deadline = time.time() + seconds
    while app._busy and time.time() < deadline:
        app.root.update()
        time.sleep(0.05)
    app.root.update()


def test_brand_alias() -> None:
    assert PersianChatApp is AIChatApp


def test_ui_boot_input_and_add_progress(tmp_path) -> None:
    """One Tk root per process — covers boot, typing, and store progress."""
    app = AIChatApp(tmp_path / "ws", seed=None)
    try:
        _pump_until_ready(app)
        assert app._ready, "UI failed to finish background boot"
        assert app.root.title() == "ai"
        assert app.entry.winfo_ismapped()
        assert app.btn_send.winfo_ismapped()
        assert str(app.entry.cget("state")) == "normal"

        app.entry.insert("1.0", "سلام")
        assert "سلام" in app.entry.get("1.0", "end-1c")
        app.entry.delete("1.0", "end")

        app._set_mode("add")
        assert app.btn_send.cget("text") == "ذخیره"
        app.entry.insert("1.0", "علی در تهران جلسه داشت")
        app._on_submit()
        _pump_until_idle(app)
        hist = app.history.get("1.0", "end-1c")
        assert "ذخیره" in hist or "سند" in hist
        assert not app._busy
        assert app.ai is not None
        assert int(app.ai.stats().get("documents", 0)) >= 1
    finally:
        app._on_close()
