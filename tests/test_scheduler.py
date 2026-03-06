from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config.settings import RuntimeSettings
from jobs.scheduler import _run_guarded
from storage.repository import TradingRepository, utc_now_iso


class SchedulerHardeningTest(unittest.TestCase):
    def _context(self, db_path: str):
        settings = RuntimeSettings()
        settings.storage.db_path = db_path
        settings.scheduler.retry_backoff_seconds = 2
        settings.scheduler.max_retry_count = 2
        settings.scheduler.job_lease_timeout_seconds = 60
        repo = TradingRepository(db_path)
        repo.initialize()
        repo.initialize_runtime_flags()
        return SimpleNamespace(settings=settings, repository=repo)

    def test_run_guarded_retries_with_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(f"{tmp}/runtime.sqlite3")
            attempts = {"count": 0}

            def flaky():
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("boom")
                return {"ok": 1}

            with patch("jobs.scheduler.time.sleep") as sleep_mock:
                result = _run_guarded(context, "scan:test", "run-key", flaky)

            self.assertEqual(result, {"ok": 1})
            sleep_mock.assert_called_once_with(2)
            health = context.repository.recent_job_health(limit=1)
            self.assertEqual(str(health.iloc[0]["status"]), "completed")
            self.assertEqual(int(health.iloc[0]["retry_count"]), 1)

    def test_begin_job_run_recovers_stale_running_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = self._context(f"{tmp}/runtime.sqlite3")
            first = context.repository.begin_job_run(
                job_name="entry:test",
                run_key="same-key",
                scheduled_at=utc_now_iso(),
                lock_owner="test",
                lease_timeout_seconds=1,
                max_retry_count=3,
            )
            self.assertIsNotNone(first)
            with context.repository.connect() as conn:
                conn.execute(
                    "UPDATE job_runs SET started_at = ?, status = 'running' WHERE job_run_id = ?",
                    ("2026-03-05T00:00:00Z", first["job_run_id"]),
                )
            recovered = context.repository.begin_job_run(
                job_name="entry:test",
                run_key="same-key",
                scheduled_at=utc_now_iso(),
                lock_owner="test",
                lease_timeout_seconds=1,
                max_retry_count=3,
            )
            self.assertIsNotNone(recovered)
            self.assertTrue(bool(recovered["recovered_stale"]))


if __name__ == "__main__":
    unittest.main()
