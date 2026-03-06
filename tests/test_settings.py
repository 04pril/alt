from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config.settings import SettingsMergeError, RuntimeSettings, load_settings


class SettingsMergeTest(unittest.TestCase):
    def test_unknown_key_warns_in_non_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"scheduler": {"unknown_key": 1}}), encoding="utf-8")
            with self.assertLogs("config.settings", level="WARNING") as captured:
                settings = load_settings(path, strict=False, warn_unknown=True)
            self.assertIsInstance(settings, RuntimeSettings)
            self.assertTrue(any("scheduler.unknown_key" in line for line in captured.output))

    def test_unknown_key_raises_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"risk": {"not_a_real_field": 1}}), encoding="utf-8")
            with self.assertRaises(SettingsMergeError):
                load_settings(path, strict=True)

    def test_nested_dataclass_override_updates_known_key(self) -> None:
        defaults = RuntimeSettings()
        korean_asset_type = next(
            asset_type
            for asset_type, schedule in defaults.asset_schedules.items()
            if schedule.timezone == "Asia/Seoul"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "asset_schedules": {
                            korean_asset_type: {
                                "lookback_bars": 640,
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            merged = load_settings(path, strict=True)
        self.assertEqual(merged.asset_schedules[korean_asset_type].lookback_bars, 640)


if __name__ == "__main__":
    unittest.main()
