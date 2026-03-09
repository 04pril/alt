from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jobs.tasks import build_task_context
from monitoring.dashboard_hooks import load_dashboard_data


class RuntimeProfileIntegrationTest(unittest.TestCase):
    def test_balanced_profile_flags_and_monitoring_read_model_match_loaded_settings(self) -> None:
        source_path = Path("config/runtime_settings.balanced.json")
        raw = json.loads(source_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            raw.setdefault("storage", {})["db_path"] = f"{tmp}/runtime.sqlite3"
            settings_path = Path(tmp) / "balanced.runtime.json"
            settings_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

            context = build_task_context(str(settings_path))
            repository = context.repository
            dashboard_data = load_dashboard_data(context.settings)

            self.assertEqual(context.settings.profile_name, "balanced")
            self.assertEqual(str(context.settings.profile_source), settings_path.as_posix())
            self.assertEqual(repository.get_control_flag("runtime_profile_name", ""), "balanced")
            self.assertEqual(repository.get_control_flag("runtime_profile_source", ""), settings_path.as_posix())
            self.assertEqual(dashboard_data["runtime_profile"]["name"], "balanced")
            self.assertEqual(dashboard_data["runtime_profile"]["source"], settings_path.as_posix())


if __name__ == "__main__":
    unittest.main()
