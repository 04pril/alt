from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from config.settings import RuntimeSettings
from monitoring.dashboard_hooks import build_asset_overview, compute_auto_trading_status, load_dashboard_data
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
            self.assertIn("asset_overview", data)
            self.assertIn("execution_summary", data)
            self.assertIn("broker_sync_status", data)
            self.assertIn("broker_sync_errors", data)
            self.assertIn("runtime_profile", data)
            self.assertIn("kis_runtime", data)

    def test_asset_overview_contains_expected_broker_modes(self) -> None:
        settings = RuntimeSettings()
        overview = build_asset_overview(settings, kis_enabled=True)
        self.assertFalse(overview.empty)
        self.assertEqual(set(overview["자산유형"]), {"코인", "미국주식", "한국주식"})
        self.assertIn("대표 종목", overview.columns)
        self.assertIn("실행브로커", overview.columns)
        broker_modes = dict(zip(overview["자산유형"], overview["실행브로커"]))
        self.assertEqual(broker_modes["한국주식"], "kis_mock")
        self.assertEqual(broker_modes["미국주식"], "sim")
        self.assertEqual(broker_modes["코인"], "sim")

    def test_auto_trading_status_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            now = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
            self._insert_job_heartbeat(repo, (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z"))
            status = compute_auto_trading_status(repo, loop_sleep_seconds=30, now=now)
            self.assertEqual(status["label"], "Running")
            self.assertEqual(status["heartbeat_at_kst"], "2026-03-06 20:59:30")

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
            repo.set_control_flag("trading_paused", "1", "test")
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

    def test_dashboard_reader_exposes_execution_counts_sync_status_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            settings.profile_name = "balanced"
            settings.profile_source = "config/runtime_settings.balanced.json"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.set_control_flag("runtime_profile_name", settings.profile_name, "test")
            repo.set_control_flag("runtime_profile_source", settings.profile_source, "test")
            repo.log_event("INFO", "execution_pipeline", "candidate", "candidate", {"symbol": "005930.KS"})
            repo.log_event("INFO", "execution_pipeline", "entry_allowed", "allowed", {"symbol": "005930.KS"})
            repo.log_event("INFO", "execution_pipeline", "submit_requested", "submit", {"symbol": "005930.KS"})
            repo.log_event("INFO", "execution_pipeline", "noop", "noop", {"reason": "outside_preclose_window"})
            lease = repo.begin_job_run(
                job_name="broker_order_sync",
                run_key="2026-03-09T15:20",
                scheduled_at="2026-03-09T15:20:00Z",
                lock_owner="test",
            )
            repo.finish_job_run(lease.job_run_id, status="completed", metrics={"fills": 0})
            repo.set_control_flag("kis_last_order_sync_at", "2026-03-09T06:20:00Z", "test")

            data = load_dashboard_data(settings)
            self.assertEqual(data["execution_summary"]["today_candidate_count"], 1)
            self.assertEqual(data["execution_summary"]["today_entry_allowed_count"], 1)
            self.assertEqual(data["execution_summary"]["today_submit_requested_count"], 1)
            self.assertEqual(data["execution_summary"]["today_noop_count"], 1)
            self.assertFalse(data["broker_sync_status"].empty)
            self.assertEqual(data["kis_runtime"]["last_broker_order_sync"], "2026-03-09T06:20:00Z")
            self.assertEqual(data["runtime_profile"]["name"], "balanced")
            self.assertEqual(data["runtime_profile"]["source"], "config/runtime_settings.balanced.json")

    def test_broker_sync_errors_filters_out_healthy_info_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.log_event("INFO", "broker_sync_job", "broker_order_sync", "broker order sync completed", {"fills": 0})
            repo.log_event("INFO", "kis_execution", "filled", "fill", {"symbol": "005930.KS"})
            data = load_dashboard_data(settings)
            self.assertTrue(data["broker_sync_errors"].empty)


if __name__ == "__main__":
    unittest.main()
