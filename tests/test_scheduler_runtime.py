from __future__ import annotations

import tempfile
import time
import unittest
from types import SimpleNamespace

from config.settings import RuntimeSettings
from jobs.scheduler import _run_guarded
from storage.repository import TradingRepository


class SchedulerRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.settings.scheduler.retry_backoff_seconds = 1
        self.settings.scheduler.max_retry_count = 2
        self.settings.scheduler.job_lease_seconds = 30
        self.settings.scheduler.lock_owner = "test-worker"
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.context = SimpleNamespace(repository=self.repo, settings=self.settings)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_failed_job_backoff_does_not_block_other_jobs(self) -> None:
        attempts = {"scan": 0, "exit": 0}

        def scan_job():
            attempts["scan"] += 1
            raise RuntimeError("scan failed")

        def exit_job():
            attempts["exit"] += 1
            return {"ok": True}

        result_scan = _run_guarded(self.context, "scan:crypto", "2026-03-08T10:00", scan_job)
        result_exit = _run_guarded(self.context, "exit_management", "2026-03-08T10:00", exit_job)
        blocked_scan = _run_guarded(self.context, "scan:crypto", "2026-03-08T10:00", scan_job)

        self.assertIsNone(result_scan)
        self.assertEqual(result_exit, {"ok": True})
        self.assertIsNone(blocked_scan)
        self.assertEqual(attempts["scan"], 1)
        self.assertEqual(attempts["exit"], 1)

        scan_row = self.repo.get_job_run_by_key("scan:crypto", "2026-03-08T10:00")
        exit_row = self.repo.get_job_run_by_key("exit_management", "2026-03-08T10:00")
        self.assertEqual(scan_row["status"], "failed")
        self.assertIsNotNone(scan_row["next_retry_at"])
        self.assertEqual(exit_row["status"], "completed")

    def test_retry_count_increases_after_backoff_window(self) -> None:
        attempts = {"job": 0}

        def flaky():
            attempts["job"] += 1
            if attempts["job"] == 1:
                raise RuntimeError("first failure")
            return {"attempt": attempts["job"]}

        first = _run_guarded(self.context, "outcome_resolution", "2026-03-08T11:00", flaky)
        self.assertIsNone(first)

        blocked = _run_guarded(self.context, "outcome_resolution", "2026-03-08T11:00", flaky)
        self.assertIsNone(blocked)
        self.assertEqual(attempts["job"], 1)

        time.sleep(1.1)
        second = _run_guarded(self.context, "outcome_resolution", "2026-03-08T11:00", flaky)
        self.assertEqual(second, {"attempt": 2})

        row = self.repo.get_job_run_by_key("outcome_resolution", "2026-03-08T11:00")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(int(row["retry_count"]), 1)


if __name__ == "__main__":
    unittest.main()
