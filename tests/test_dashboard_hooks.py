from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from config.settings import RuntimeSettings
from monitoring.dashboard_hooks import (
    build_asset_overview,
    compute_auto_trading_status,
    load_dashboard_data,
    load_monitor_open_positions,
)
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.models import AccountSnapshotRecord, PositionRecord
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
            self.assertIn("accounts_overview", data)
            self.assertIn("total_portfolio_overview", data)
            self.assertIn("recent_realized_trades", data)
            self.assertIn("kr_strategy_overview", data)
            self.assertIn("kr_strategy_recent_events", data)

    def test_asset_overview_contains_expected_broker_modes(self) -> None:
        settings = RuntimeSettings()
        overview = build_asset_overview(settings, kis_enabled=True)
        self.assertFalse(overview.empty)
        self.assertEqual(set(overview["자산유형"]), {"코인", "미국주식", "한국주식"})
        self.assertIn("전략", overview.columns)
        self.assertIn("대표 종목", overview.columns)
        self.assertIn("실행브로커", overview.columns)
        broker_modes = dict(zip(overview["자산유형"], overview["실행브로커"]))
        self.assertEqual(broker_modes["한국주식"], "kis_mock")
        self.assertEqual(broker_modes["미국주식"], "sim")
        self.assertEqual(broker_modes["코인"], "sim")
        kr_row = overview.loc[overview["자산유형"] == "한국주식"].iloc[0]
        self.assertEqual(str(kr_row["전략ID"]), "kr_intraday_1h_v1")
        self.assertEqual(str(kr_row["타임프레임"]), "1h")

    def test_asset_overview_reflects_active_kr_15m_strategy(self) -> None:
        settings = RuntimeSettings()
        settings.kr_default_strategy_id = "kr_intraday_15m_v1"
        settings.kr_strategies["kr_intraday_1h_v1"].enabled = False
        settings.kr_strategies["kr_intraday_15m_v1"].enabled = True

        overview = build_asset_overview(settings, kis_enabled=True)

        kr_row = overview.loc[overview["자산유형"] == "한국주식"].iloc[0]
        self.assertEqual(str(kr_row["전략ID"]), "kr_intraday_15m_v1")
        self.assertEqual(str(kr_row["타임프레임"]), "15m")
        self.assertEqual(int(kr_row["스캔주기(분)"]), 15)
        self.assertEqual(str(kr_row["실험전략"]), "예")
        self.assertEqual(str(kr_row["세션모드"]), "regular")

    def test_dashboard_reader_lists_regular_and_after_hours_kr_15m_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()

            data = load_dashboard_data(settings)
            rows = data["kr_strategy_overview"]
            session_modes = dict(zip(rows["strategy_id"].astype(str), rows["session_mode"].astype(str)))

            self.assertEqual(session_modes["kr_intraday_15m_v1"], "regular")
            self.assertEqual(session_modes["kr_intraday_15m_v1_after_close_close"], "after_close_close")
            self.assertEqual(session_modes["kr_intraday_15m_v1_after_close_single"], "after_close_single")

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
            repo.set_control_flag("kis_last_websocket_quote_at", "2026-03-09T06:21:00Z", "test")
            repo.set_control_flag("kis_quote_stream_status", "connected", "test")

            data = load_dashboard_data(settings)
            self.assertEqual(data["execution_summary"]["today_candidate_count"], 1)
            self.assertEqual(data["execution_summary"]["today_entry_allowed_count"], 1)
            self.assertEqual(data["execution_summary"]["today_submit_requested_count"], 1)
            self.assertEqual(data["execution_summary"]["today_noop_count"], 1)
            self.assertFalse(data["broker_sync_status"].empty)
            self.assertEqual(data["kis_runtime"]["last_broker_order_sync"], "2026-03-09T06:20:00Z")
            self.assertEqual(data["kis_runtime"]["last_websocket_quote_at"], "2026-03-09T06:21:00Z")
            self.assertEqual(data["kis_runtime"]["quote_stream_status"], "connected")
            self.assertEqual(data["runtime_profile"]["name"], "balanced")
            self.assertEqual(data["runtime_profile"]["source"], "config/runtime_settings.balanced.json")
            self.assertEqual(data["runtime_profile"]["mode"], "recommended")
            self.assertEqual(data["runtime_profile"]["recommended_default"], "true")
            self.assertEqual(data["runtime_profile"]["experimental"], "false")
            self.assertEqual(data["runtime_profile"]["kr_default_strategy_id"], "kr_intraday_1h_v1")
            self.assertEqual(data["runtime_profile"]["kr_default_strategy_session_mode"], "regular")
            self.assertIn("kr_intraday_1h_v1", data["runtime_profile"]["kr_active_strategies"])
            self.assertIn("regular", data["runtime_profile"]["kr_active_strategy_session_modes"])

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
                        1,
                        70000.0,
                        71000.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        1000.0,
                        0.0,
                        0.02,
                        71000.0,
                        None,
                        "test",
                        None,
                        "",
                    ),
                )
            frame = load_monitor_open_positions(settings)
            self.assertEqual(len(frame), 1)
            self.assertEqual(str(frame.iloc[0]["symbol"]), "005930.KS")
            self.assertEqual(frame.iloc[0]["updated_at"].strftime("%Y-%m-%d %H:%M:%S"), "2026-03-09 10:05:00")

    def test_dashboard_reader_builds_account_cards_and_read_only_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            rows = [
                AccountSnapshotRecord(
                    snapshot_id="snap_kis",
                    created_at="2026-03-09T09:00:00Z",
                    cash=30_000_000.0,
                    equity=30_500_000.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=20_000.0,
                    daily_pnl=20_000.0,
                    drawdown_pct=-0.5,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="kis_account_sync",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                ),
                AccountSnapshotRecord(
                    snapshot_id="snap_us",
                    created_at="2026-03-09T09:01:00Z",
                    cash=10_000.0,
                    equity=10_500.0,
                    gross_exposure=500.0,
                    net_exposure=500.0,
                    realized_pnl=0.0,
                    unrealized_pnl=100.0,
                    daily_pnl=100.0,
                    drawdown_pct=-1.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                ),
                AccountSnapshotRecord(
                    snapshot_id="snap_crypto",
                    created_at="2026-03-09T09:02:00Z",
                    cash=5_000.0,
                    equity=5_250.0,
                    gross_exposure=250.0,
                    net_exposure=250.0,
                    realized_pnl=0.0,
                    unrealized_pnl=50.0,
                    daily_pnl=50.0,
                    drawdown_pct=-2.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    account_id=ACCOUNT_SIM_CRYPTO,
                ),
            ]
            for row in rows:
                repo.insert_account_snapshot(row)
            repo.upsert_position(
                PositionRecord(
                    position_id="pos_us_closed",
                    created_at="2026-03-09T08:00:00Z",
                    updated_at="2026-03-09T09:05:00Z",
                    closed_at="2026-03-09T09:05:00Z",
                    prediction_id=None,
                    symbol="AAPL",
                    asset_type="미국주식",
                    timeframe="1d",
                    side="LONG",
                    status="closed",
                    quantity=0,
                    entry_price=180.0,
                    mark_price=185.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    trailing_stop=0.0,
                    highest_price=185.0,
                    lowest_price=180.0,
                    unrealized_pnl=0.0,
                    realized_pnl=5.0,
                    expected_risk=0.01,
                    exposure_value=0.0,
                    max_holding_until="2026-03-10T00:00:00Z",
                    strategy_version="test",
                    cooldown_until=None,
                    notes="closed_by_take_profit",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                )
            )

            data = load_dashboard_data(settings)
            accounts = data["accounts_overview"]
            total = data["total_portfolio_overview"]

            self.assertEqual(set(accounts.keys()), {ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO})
            self.assertEqual(accounts[ACCOUNT_KIS_KR_PAPER]["currency"], "KRW")
            self.assertEqual(accounts[ACCOUNT_SIM_US_EQUITY]["currency"], "USD")
            self.assertEqual(accounts[ACCOUNT_SIM_CRYPTO]["currency"], "USD")
            self.assertEqual(accounts[ACCOUNT_SIM_US_EQUITY]["open_positions"], 0)
            self.assertTrue(str(total["warning"]).startswith("전체 합산 뷰"))
            self.assertTrue(str(total["display_currency"]) in {"", "KRW", "USD"})
            self.assertEqual(total["equity_by_currency"]["KRW"], 30_500_000.0)
            self.assertEqual(total["equity_by_currency"]["USD"], 15_750.0)
            self.assertEqual(str(data["recent_realized_trades"].iloc[0]["position_id"]), "pos_us_closed")
            self.assertFalse(data["kr_strategy_overview"].empty)
            strategy_ids = set(data["kr_strategy_overview"]["strategy_id"].astype(str))
            self.assertEqual(
                strategy_ids,
                {
                    "kr_daily_preclose_v1",
                    "kr_intraday_1h_v1",
                    "kr_intraday_15m_v1",
                    "kr_intraday_15m_v1_after_close_close",
                    "kr_intraday_15m_v1_after_close_single",
                },
            )
            intraday_15m = data["kr_strategy_overview"].loc[data["kr_strategy_overview"]["strategy_id"].astype(str) == "kr_intraday_15m_v1"].iloc[0]
            self.assertFalse(bool(intraday_15m["enabled"]))
            self.assertTrue(bool(intraday_15m["experimental"]))
            self.assertEqual(str(intraday_15m["session_mode"]), "regular")
            after_close = data["kr_strategy_overview"].loc[data["kr_strategy_overview"]["strategy_id"].astype(str) == "kr_intraday_15m_v1_after_close_close"].iloc[0]
            self.assertEqual(str(after_close["session_mode"]), "after_close_close")


if __name__ == "__main__":
    unittest.main()
