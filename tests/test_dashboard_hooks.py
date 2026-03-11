from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import strategy_schedule
from monitoring.dashboard_hooks import (
    build_asset_overview,
    compute_auto_trading_status,
    load_dashboard_data,
    load_monitor_open_positions,
)
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.models import AccountSnapshotRecord, CandidateScanRecord, OrderRecord, PositionRecord
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

    def test_asset_overview_reflects_active_us_strategy(self) -> None:
        settings = RuntimeSettings()
        settings.us_default_strategy_id = "us_combo_15m_ahc_regular_v1"
        settings.kr_strategies["us_intraday_1h_v1"].enabled = False
        settings.kr_strategies["us_combo_15m_ahc_regular_v1"].enabled = True

        overview = build_asset_overview(settings, kis_enabled=True)

        us_row = overview.loc[(overview["자산유형"] == "미국주식") & (overview["전략ID"].astype(str) == "us_combo_15m_ahc_regular_v1")].iloc[0]
        self.assertEqual(str(us_row["타임프레임"]), "15m")
        self.assertEqual(int(us_row["스캔주기(분)"]), 15)
        self.assertEqual(str(us_row["세션모드"]), "regular")
        self.assertEqual(str(us_row["실행브로커"]), "sim")

    def test_dashboard_reader_lists_regular_and_after_hours_kr_15m_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()

            data = load_dashboard_data(settings)
            rows = data["kr_strategy_overview"]
            session_modes = dict(zip(rows["strategy_id"].astype(str), rows["session_mode"].astype(str)))
            auto_schedule = strategy_schedule(settings, "kr_intraday_15m_v1_auto")
            expected_auto_mode = {
                "regular": "regular",
                "after_close_close_price": "after_close_close",
                "after_close_single_price": "after_close_single",
            }[auto_schedule.session_mode]

            self.assertEqual(session_modes["kr_intraday_15m_v1"], "regular")
            self.assertEqual(session_modes["kr_intraday_15m_v1_auto"], expected_auto_mode)
            self.assertNotIn("kr_intraday_15m_v1_after_close_close", session_modes)
            self.assertNotIn("kr_intraday_15m_v1_after_close_single", session_modes)

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
            self.assertEqual(data["runtime_profile"]["kr_recommended_strategy_id"], "kr_intraday_1h_v1")
            self.assertIn("kr_intraday_1h_v1", data["runtime_profile"]["kr_active_strategies"])
            self.assertIn("regular", data["runtime_profile"]["kr_active_strategy_session_modes"])
            self.assertEqual(data["runtime_profile"]["us_default_strategy_id"], "us_intraday_1h_v1")
            self.assertEqual(data["runtime_profile"]["us_recommended_strategy_id"], "us_intraday_1h_v1")

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
            self.assertEqual(float(accounts[ACCOUNT_KIS_KR_PAPER]["trade_performance"]["samples"]), 1.0)
            self.assertEqual(float(accounts[ACCOUNT_SIM_US_EQUITY]["trade_performance"]["samples"]), 1.0)
            self.assertEqual(float(data["trade_performance"]["samples"]), 3.0)
            self.assertEqual(str(data["recent_realized_trades"].iloc[0]["position_id"]), "pos_us_closed")
            self.assertFalse(data["kr_strategy_overview"].empty)
            strategy_ids = set(data["kr_strategy_overview"]["strategy_id"].astype(str))
            self.assertTrue(
                {
                    "kr_daily_preclose_v1",
                    "kr_intraday_1h_v1",
                    "kr_intraday_15m_v1",
                    "kr_intraday_15m_v1_auto",
                }.issubset(strategy_ids)
            )
            intraday_15m = data["kr_strategy_overview"].loc[data["kr_strategy_overview"]["strategy_id"].astype(str) == "kr_intraday_15m_v1"].iloc[0]
            self.assertFalse(bool(intraday_15m["enabled"]))
            self.assertTrue(bool(intraday_15m["experimental"]))
            self.assertEqual(str(intraday_15m["session_mode"]), "regular")
            self.assertEqual(str(intraday_15m["broker_mode"]), "kis_mock")
            self.assertEqual(str(intraday_15m["execution_account_id"]), ACCOUNT_KIS_KR_PAPER)
            self.assertEqual(str(intraday_15m["intended_use"]), "regular intraday only")
            auto_15m = data["kr_strategy_overview"].loc[data["kr_strategy_overview"]["strategy_id"].astype(str) == "kr_intraday_15m_v1_auto"].iloc[0]
            expected_auto = strategy_schedule(settings, "kr_intraday_15m_v1_auto")
            expected_auto_mode = {
                "regular": "regular",
                "after_close_close_price": "after_close_close",
                "after_close_single_price": "after_close_single",
            }[expected_auto.session_mode]
            expected_auto_window = {
                "regular": "09:15~14:45",
                "after_close_close_price": "15:40~16:00",
                "after_close_single_price": "16:00~18:00",
            }[expected_auto.session_mode]
            self.assertEqual(str(auto_15m["session_mode"]), expected_auto_mode)
            self.assertEqual(str(auto_15m["session_window"]), expected_auto_window)
            self.assertEqual(str(auto_15m["intended_use"]), "auto regular + after-close")

    def test_dashboard_reader_uses_today_candidate_scans_not_latest_300_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            us_start = pd.Timestamp.now(tz="America/New_York").normalize()
            crypto_start = pd.Timestamp.now(tz="UTC").normalize()

            records = []
            for index in range(320):
                created_at = (us_start + pd.Timedelta(minutes=index)).tz_convert("UTC").isoformat()
                records.append(
                    CandidateScanRecord(
                        scan_id=f"us_{index}",
                        created_at=created_at,
                        symbol=f"US{index}",
                        asset_type="미국주식",
                        timeframe="15m",
                        score=float(index),
                        rank=index + 1,
                        status="flat",
                        reason="flat_signal",
                        expected_return=0.01,
                        expected_risk=0.01,
                        confidence=0.5,
                        threshold=0.1,
                        volatility=0.1,
                        liquidity_score=1.0,
                        cost_bps=8.0,
                        recent_performance=0.0,
                        signal="FLAT",
                        model_version="m",
                        feature_version="f",
                        strategy_version="us_intraday_1h_v1",
                        execution_account_id=ACCOUNT_SIM_US_EQUITY,
                    )
                )
            records.append(
                CandidateScanRecord(
                    scan_id="coin_1",
                    created_at=(crypto_start + pd.Timedelta(hours=1)).isoformat(),
                    symbol="BTC-USD",
                    asset_type="코인",
                    timeframe="1h",
                    score=10.0,
                    rank=1,
                    status="candidate",
                    reason="signal_ready",
                    expected_return=0.02,
                    expected_risk=0.01,
                    confidence=0.7,
                    threshold=0.1,
                    volatility=0.2,
                    liquidity_score=1.0,
                    cost_bps=10.0,
                    recent_performance=0.0,
                    signal="LONG",
                    model_version="m",
                    feature_version="f",
                    strategy_version="crypto_intraday",
                    execution_account_id=ACCOUNT_SIM_CRYPTO,
                )
            )
            repo.insert_candidate_scans(records)

            data = load_dashboard_data(settings)
            candidate_scans = data["candidate_scans"]

            self.assertGreater(len(candidate_scans), 300)
            self.assertIn("BTC-USD", set(candidate_scans["symbol"].astype(str)))

    def test_dashboard_reader_preserves_event_session_mode_and_surfaces_hidden_strategy_with_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            created_at = pd.Timestamp.now(tz="Asia/Seoul").tz_convert("UTC").isoformat()
            repo.insert_order(
                OrderRecord(
                    order_id="ord_hidden_strategy",
                    created_at=created_at,
                    updated_at=created_at,
                    prediction_id="pred_hidden",
                    scan_id="scan_hidden",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe="15m",
                    side="buy",
                    order_type="after_close_close",
                    requested_qty=1,
                    filled_qty=0,
                    remaining_qty=1,
                    requested_price=70000.0,
                    limit_price=70000.0,
                    status="submitted",
                    fees_estimate=0.0,
                    slippage_bps=0.0,
                    retry_count=0,
                    broker_order_id="b_hidden",
                    strategy_version="kr_combo_15m_ahc_afterclose_v2",
                    reason="entry",
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            repo.log_event(
                "INFO",
                "execution_pipeline",
                "submitted",
                "hidden strategy submitted",
                {
                    "symbol": "005930.KS",
                    "strategy_version": "kr_combo_15m_ahc_afterclose_v2",
                    "session_mode": "after_close_close_price",
                    "account_id": ACCOUNT_KIS_KR_PAPER,
                },
                account_id=ACCOUNT_KIS_KR_PAPER,
            )

            data = load_dashboard_data(settings)

            strategy_ids = set(data["kr_strategy_overview"]["strategy_id"].astype(str))
            self.assertIn("kr_combo_15m_ahc_afterclose_v2", strategy_ids)
            recent = data["kr_strategy_recent_events"]
            hidden_row = recent.loc[recent["strategy_id"].astype(str) == "kr_combo_15m_ahc_afterclose_v2"].iloc[0]
            self.assertEqual(str(hidden_row["session_mode"]), "after_close_close")


if __name__ == "__main__":
    unittest.main()
