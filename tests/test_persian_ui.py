"""ai UI layout smoke checks."""

from offline_ai.cli.persian_ui import AIChatApp, PersianChatApp


def test_brand_alias() -> None:
    assert PersianChatApp is AIChatApp


def test_widgets_visible_and_editable(tmp_path) -> None:
    app = AIChatApp(tmp_path / "ws", seed=None)
    try:
        app.root.update_idletasks()
        app.root.update()
        assert app.root.title() == "ai"
        assert app.entry.winfo_ismapped()
        assert app.btn_send.winfo_ismapped()
        assert app.btn_clear.winfo_ismapped()
        assert app.entry.winfo_height() >= 40
        assert app.btn_send.winfo_height() >= 20
        assert str(app.entry.cget("state")) == "normal"
        app.entry.insert("1.0", "سلام")
        assert "سلام" in app.entry.get("1.0", "end-1c")
    finally:
        app.root.destroy()
