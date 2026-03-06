from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from config.settings import RuntimeSettings
from monitoring.dashboard_hooks import compute_auto_trading_status, load_dashboard_data
from storage.repository import TradingRepository


class DashboardHooksTest(unittest.TestCase):
    def _insert_job_heartbeat(self, repo: TradingRepository, heartbeat_at: str) -> None:
        with repo.connect() as conn:
            conn.execute(
                """
                INSERT INTO job_runs(
                    job_run_id,
                    job_name,
                    run_key,
                    scheduled_at,
                    started_at,
                    finished_at,
                    status,
                    retry_count,
                    lock_owner,
                    error_message,
                    metrics_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?, '', '{}')
                """,
                ("job_test", "scan:한국주식", "2026-03-06T12:00", heartbeat_at, heartbeat_at, heartbeat_at, "completed", "test"),
            )

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
            self.assertIn("auto_trading_status", data)

    def test_auto_trading_status_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            self._insert_job_heartbeat(repo, (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z"))
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Running")
            self.assertIn("KST", status["heartbeat_at_kst"])

    def test_auto_trading_status_uses_worker_heartbeat_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            repo.set_control_flag("worker_heartbeat_at", (now - timedelta(seconds=20)).isoformat().replace("+00:00", "Z"), "test")
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Running")
            self.assertEqual(status["source"], "worker_heartbeat_at")

    def test_auto_trading_status_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            self._insert_job_heartbeat(repo, (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z"))
            repo.set_entry_paused(True, "test")
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Paused")

    def test_auto_trading_status_stopped_when_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            self._insert_job_heartbeat(repo, (now - timedelta(seconds=181)).isoformat().replace("+00:00", "Z"))
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Stopped")

    def test_auto_trading_status_stopped_when_no_job_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Stopped")


if __name__ == "__main__":
    unittest.main()
