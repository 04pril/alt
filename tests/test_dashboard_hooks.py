from __future__ import annotations

import tempfile
import unittest

from config.settings import RuntimeSettings
from monitoring.dashboard_hooks import load_dashboard_data
from storage.repository import TradingRepository


class DashboardHooksTest(unittest.TestCase):
    def test_dashboard_reader_returns_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            data = load_dashboard_data(settings)
            self.assertIn("summary", data)
            self.assertIn("job_health", data)
            self.assertIn("recent_errors", data)


if __name__ == "__main__":
    unittest.main()
