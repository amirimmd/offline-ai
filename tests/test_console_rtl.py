"""Console UTF-8 helpers."""

from offline_ai.utils.console import format_console_text, visual_line


def test_visual_line_keeps_latin() -> None:
    assert visual_line("[DOC-000000006]") == "[DOC-000000006]"


def test_visual_line_keeps_connected_persian() -> None:
    logical = "علی رفت به اسپانیا"
    assert visual_line(logical) == logical
    assert "علی" in visual_line(logical)


def test_format_console_text_unchanged() -> None:
    text = "علی رفت.\n[DOC-1]"
    assert format_console_text(text) == text
