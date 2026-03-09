from __future__ import annotations

import subprocess
from pathlib import Path
import unittest


TEXT_EXTENSIONS = {
    ".py",
    ".ps1",
    ".bat",
    ".json",
    ".yml",
    ".yaml",
    ".md",
    ".toml",
    ".html",
    ".css",
    ".js",
}
SOURCE_SENTINELS = {
    Path("app.py"): [
        "\uc6b4\uc601 \ubaa8\ub2c8\ud130",
        "\ub300\uc2dc\ubcf4\ub4dc",
        "\uc885\ubaa9 \ubd84\uc11d",
        "\ubaa8\uc758\ub9e4\ub9e4",
        "\uc885\ubaa9 \uc2a4\uce94",
        "\ub9e4\uc218",
        "\ub9e4\ub3c4",
        "\uad00\ub9dd",
    ],
    Path("beta_monitor_clone.py"): [
        "\uc6b4\uc601 \ubaa8\ub2c8\ud130",
        "\ubcf4\uc720 \ud604\ud669",
        "\ube0c\ub85c\ucee4 \ub3d9\uae30\ud654",
        "\uc791\uc5c5 \uc774\ub825",
        "\ucd5c\uadfc \uc624\ub958",
    ],
}


def _tracked_text_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8")
    paths = []
    for raw_line in output.splitlines():
        path = Path(raw_line)
        if path.suffix.lower() in TEXT_EXTENSIONS:
            paths.append(path)
    return paths


class SourceEncodingTest(unittest.TestCase):
    def test_tracked_text_files_decode_as_utf8(self) -> None:
        failures: list[str] = []
        for path in _tracked_text_files():
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                failures.append(f"{path}: {exc}")
        self.assertFalse(failures, "Non-UTF-8 tracked text files:\n" + "\n".join(failures))

    def test_core_korean_labels_are_not_lost(self) -> None:
        failures: list[str] = []
        for path, sentinels in SOURCE_SENTINELS.items():
            text = path.read_text(encoding="utf-8")
            for sentinel in sentinels:
                if sentinel not in text:
                    failures.append(f"{path}: missing {sentinel.encode('unicode_escape').decode()}")
        self.assertFalse(failures, "Missing Korean source sentinels:\n" + "\n".join(failures))

    def test_replacement_character_is_not_present(self) -> None:
        failures: list[str] = []
        for path in _tracked_text_files():
            text = path.read_text(encoding="utf-8")
            if "\ufffd" in text:
                failures.append(str(path))
        self.assertFalse(failures, "Replacement character found in:\n" + "\n".join(failures))
