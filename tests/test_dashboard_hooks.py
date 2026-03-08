from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from config.settings import RuntimeSettings
from monitoring.dashboard_hooks import (
    build_account_snapshot_read_model,
    build_asset_overview,
    build_broker_sync_read_model,
    build_broker_sync_status_frame,
    build_kis_monitor_read_model,
    build_recent_job_status_summary,
    compute_auto_trading_status,
    load_dashboard_data,
    load_monitor_open_positions,
    set_trading_pause_state,
)
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
            self.assertIn("broker_sync_status", data)
            self.assertIn("account_snapshot", data)
            self.assertIn("broker_sync_frame", data)
            self.assertIn("broker_sync_errors", data)
            self.assertIn("recent_job_status_summary", data)
            self.assertIn("kis_monitor", data)
            self.assertEqual(set(data["broker_sync_status"].keys()), {"broker_account_sync", "broker_order_sync", "broker_position_sync"})

    def test_dashboard_reader_exposes_broker_sync_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            with repo.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO job_runs(
                        job_run_id, job_name, run_key, scheduled_at, started_at, finished_at,
                        status, retry_count, lock_owner, error_message, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "job_sync_1",
                        "broker_order_sync",
                        "2026-03-09T10:00",
                        "2026-03-09T10:00:00Z",
                        "2026-03-09T10:00:01Z",
                        "2026-03-09T10:00:02Z",
                        "completed",
                        0,
                        "worker",
                        "",
                        "{}",
                    ),
                )
            repo.log_event("INFO", "broker_sync", "broker_order_sync", "broker_order_sync completed", {})
            repo.log_event("ERROR", "scheduler", "job_failed", "broker_order_sync failed", {"error": "sync boom"})

            data = load_dashboard_data(settings)
            self.assertIn("broker_order_sync", data["broker_sync_status"])
            self.assertEqual(data["broker_sync_status"]["broker_order_sync"]["status"], "completed")
            self.assertFalse(data["broker_sync_errors"].empty)
            self.assertTrue((data["broker_sync_errors"]["level"] == "ERROR").all())

    def test_build_broker_sync_read_model_covers_expected_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            with repo.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO job_runs(
                        job_run_id, job_name, run_key, scheduled_at, started_at, finished_at,
                        status, retry_count, lock_owner, error_message, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "job_account_sync",
                        "broker_account_sync",
                        "2026-03-09T10:00",
                        "2026-03-09T10:00:00.000000Z",
                        "2026-03-09T10:00:01.000000Z",
                        "2026-03-09T10:00:05.000000Z",
                        "completed",
                        0,
                        "worker",
                        "",
                        "{}",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO job_runs(
                        job_run_id, job_name, run_key, scheduled_at, started_at, finished_at,
                        status, retry_count, lock_owner, error_message, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "job_position_sync",
                        "broker_position_sync",
                        "2026-03-09T09:00",
                        "2026-03-09T09:00:00.000000Z",
                        "2026-03-09T09:00:01.000000Z",
                        "2026-03-09T09:00:02.000000Z",
                        "failed",
                        1,
                        "worker",
                        "sync failed",
                        "{}",
                    ),
                )
            repo.log_event("ERROR", "broker_sync", "broker_position_sync_failed", "broker_position_sync failed", {"error": "boom"})

            status, errors = build_broker_sync_read_model(repo.recent_job_health(limit=20), repo.recent_system_events(limit=20))

            self.assertEqual(status["broker_account_sync"]["status"], "completed")
            self.assertEqual(status["broker_position_sync"]["status"], "failed")
            self.assertEqual(status["broker_order_sync"]["status"], "never")
            self.assertFalse(errors.empty)

    def test_build_account_snapshot_read_model_localizes_created_at(self) -> None:
        snapshot = build_account_snapshot_read_model(
            {
                "latest_account": {
                    "created_at": "2026-03-09T01:15:00Z",
                    "equity": 30000000.0,
                    "cash": 28000000.0,
                }
            }
        )
        self.assertEqual(snapshot["created_at_kst"], "2026-03-09 10:15:00")
        self.assertEqual(snapshot["equity"], 30000000.0)

    def test_build_broker_sync_status_frame_preserves_expected_job_order(self) -> None:
        frame = build_broker_sync_status_frame(
            {
                "broker_order_sync": {"status": "completed", "retry_count": 1, "finished_at": "2026-03-09T01:00:00Z"},
                "broker_account_sync": {"status": "failed", "retry_count": 2, "finished_at": "2026-03-09T02:00:00Z"},
            }
        )
        self.assertEqual(frame["job"].tolist(), ["broker_account_sync", "broker_order_sync", "broker_position_sync"])
        self.assertEqual(frame.loc[frame["job"] == "broker_account_sync", "status"].iloc[0], "failed")
        self.assertEqual(frame.loc[frame["job"] == "broker_position_sync", "status"].iloc[0], "never")

    def test_build_recent_job_status_summary_marks_error_on_failed_job(self) -> None:
        job_health = build_broker_sync_status_frame(
            {
                "broker_account_sync": {"job_name": "broker_account_sync", "status": "failed", "retry_count": 1, "finished_at": "2026-03-09T01:00:00Z"},
                "broker_order_sync": {"job_name": "broker_order_sync", "status": "completed", "retry_count": 0, "finished_at": "2026-03-09T01:01:00Z"},
            }
        )
        summary = build_recent_job_status_summary(
            job_health,
            {"state": "running"},
            loop_sleep_seconds=30,
            now=datetime(2026, 3, 9, 1, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(summary["label"], "오류")

    def test_build_recent_job_status_summary_marks_delayed_when_latest_job_is_stale(self) -> None:
        job_health = build_broker_sync_status_frame(
            {
                "broker_account_sync": {"job_name": "broker_account_sync", "status": "completed", "retry_count": 0, "finished_at": "2026-03-09T01:00:00Z"},
            }
        )
        summary = build_recent_job_status_summary(
            job_health,
            {"state": "running"},
            loop_sleep_seconds=30,
            now=datetime(2026, 3, 9, 1, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(summary["label"], "지연")

    def test_build_recent_job_status_summary_marks_normal_when_recent_jobs_are_healthy(self) -> None:
        job_health = build_broker_sync_status_frame(
            {
                "broker_account_sync": {"job_name": "broker_account_sync", "status": "completed", "retry_count": 0, "finished_at": "2026-03-09T01:00:00Z"},
                "broker_order_sync": {"job_name": "broker_order_sync", "status": "completed", "retry_count": 0, "finished_at": "2026-03-09T01:00:30Z"},
            }
        )
        summary = build_recent_job_status_summary(
            job_health,
            {"state": "running"},
            loop_sleep_seconds=30,
            now=datetime(2026, 3, 9, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(summary["label"], "정상")

    def test_build_recent_job_status_summary_ignores_older_failed_rows_when_latest_completed(self) -> None:
        job_health = build_broker_sync_status_frame(
            {
                "broker_account_sync": {"job_name": "broker_account_sync", "status": "completed", "retry_count": 0, "finished_at": "2026-03-09T01:01:00Z"},
                "broker_order_sync": {"job_name": "broker_order_sync", "status": "failed", "retry_count": 1, "finished_at": "2026-03-09T01:00:00Z"},
            }
        )
        summary = build_recent_job_status_summary(
            job_health,
            {"state": "running"},
            loop_sleep_seconds=30,
            now=datetime(2026, 3, 9, 1, 1, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(summary["label"], "정상")

    def test_build_kis_monitor_read_model_marks_normal_when_recent_sync_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.set_control_flag("kis_last_account_sync_at", "2026-03-09T01:00:00Z", "test")
            repo.set_control_flag("kis_last_account_sync_status", "ok", "test")
            repo.set_control_flag("kis_last_buying_power_success_at", "2026-03-09T01:01:00Z", "test")
            repo.set_control_flag("kis_last_buying_power_symbol", "005930.KS", "test")

            monitor = build_kis_monitor_read_model(
                repo,
                settings,
                now=datetime(2026, 3, 9, 1, 5, tzinfo=timezone.utc),
            )
            self.assertEqual(monitor["connection_status"], "정상")
            self.assertEqual(monitor["last_buying_power_symbol"], "005930.KS")

    def test_build_kis_monitor_read_model_marks_auth_error_when_newer_than_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.set_control_flag("kis_last_account_sync_at", "2026-03-09T01:00:00Z", "test")
            repo.set_control_flag("kis_last_auth_error_at", "2026-03-09T01:02:00Z", "test")
            repo.set_control_flag("kis_last_auth_error", "KIS token expired", "test")

            monitor = build_kis_monitor_read_model(
                repo,
                settings,
                now=datetime(2026, 3, 9, 1, 5, tzinfo=timezone.utc),
            )
            self.assertEqual(monitor["connection_status"], "인증오류")

    def test_load_monitor_open_positions_localizes_timestamp_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            with repo.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO positions(
                        position_id, created_at, updated_at, closed_at, prediction_id, symbol, asset_type, timeframe, side, status,
                        quantity, entry_price, mark_price, stop_loss, take_profit, trailing_stop, highest_price, lowest_price,
                        unrealized_pnl, realized_pnl, expected_risk, exposure_value, max_holding_until, strategy_version, cooldown_until, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "pos_1",
                        "2026-03-09T01:00:00Z",
                        "2026-03-09T01:05:00Z",
                        None,
                        None,
                        "005930.KS",
                        "한국주식",
                        "1d",
                        "LONG",
                        "open",
                        3,
                        70000.0,
                        71000.0,
                        None,
                        None,
                        None,
                        71000.0,
                        70000.0,
                        3000.0,
                        0.0,
                        None,
                        213000.0,
                        None,
                        "v1",
                        None,
                        "",
                    ),
                )
            frame = load_monitor_open_positions(settings)
            self.assertEqual(len(frame), 1)
            self.assertEqual(str(frame.iloc[0]["symbol"]), "005930.KS")
            self.assertEqual(str(frame.iloc[0]["updated_at"].strftime("%Y-%m-%d %H:%M:%S")), "2026-03-09 10:05:00")

    def test_set_trading_pause_state_updates_repository_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()

            set_trading_pause_state(settings, paused=True, notes="pause test")
            self.assertEqual(repo.get_control_flag("trading_paused", "0"), "1")

            set_trading_pause_state(settings, paused=False, notes="resume test")
            self.assertEqual(repo.get_control_flag("trading_paused", "1"), "0")

    def test_asset_overview_contains_all_asset_types(self) -> None:
        settings = RuntimeSettings()
        overview = build_asset_overview(settings)
        self.assertFalse(overview.empty)
        self.assertEqual(set(overview["자산유형"]), {"코인", "미국주식", "한국주식"})
        self.assertIn("대표 심볼", overview.columns)

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


if __name__ == "__main__":
    unittest.main()
