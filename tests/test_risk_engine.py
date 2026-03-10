from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import strategy_schedule
from runtime_accounts import (
    ACCOUNT_KIS_KR_PAPER,
    ACCOUNT_SIM_CRYPTO,
    ACCOUNT_SIM_LEGACY_MIXED,
    ACCOUNT_SIM_US_EQUITY,
)
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord, OrderRecord, PositionRecord
from storage.repository import TradingRepository


class RiskEngineTest(unittest.TestCase):
    def _build_signal(self, *, symbol: str = "BTC-USD", asset_type: str = "코인", timeframe: str = "1h") -> SignalDecision:
        return SignalDecision(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            prediction_id="pred1",
            scan_id="scan1",
            score=1.0,
            signal="LONG",
            expected_return=0.02,
            expected_risk=0.01,
            confidence=0.9,
            threshold=0.003,
            position_size=0.5,
            current_price=100.0,
            predicted_price=102.0,
            predicted_return=0.02,
            stop_level=98.0,
            take_level=105.0,
            model_version="v1",
            feature_version="f1",
            strategy_version="s1",
            validation_mode="holdout",
            result=None,
        )

    def test_paused_flag_blocks_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.set_control_flag("trading_paused", "1", "test")
            engine = RiskEngine(settings, repo)
            decision = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "paused")

    def test_pending_new_entry_blocks_duplicate_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_order(
                OrderRecord(
                    order_id="ord_pending",
                    created_at="2026-03-08T09:00:00Z",
                    updated_at="2026-03-08T09:00:00Z",
                    prediction_id="pred-existing",
                    scan_id="scan-existing",
                    symbol="BTC-USD",
                    asset_type="코인",
                    timeframe="1h",
                    side="buy",
                    order_type="market",
                    requested_qty=1,
                    filled_qty=0,
                    remaining_qty=1,
                    requested_price=100.0,
                    limit_price=0.0,
                    status="new",
                    fees_estimate=0.0,
                    slippage_bps=0.0,
                    retry_count=0,
                    strategy_version="s1",
                    reason="entry",
                    raw_json='{"expected_risk": 0.01}',
                )
            )
            engine = RiskEngine(settings, repo)
            decision = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "duplicate_pending_entry")

    def test_kr_strategy_conflict_blocks_cross_strategy_pending_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_order(
                OrderRecord(
                    order_id="ord_kr_pending",
                    created_at="2026-03-08T09:00:00Z",
                    updated_at="2026-03-08T09:00:00Z",
                    prediction_id="pred-existing",
                    scan_id="scan-existing",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe=strategy_schedule(settings, "kr_intraday_1h_v1").timeframe,
                    side="buy",
                    order_type="market",
                    requested_qty=1,
                    filled_qty=0,
                    remaining_qty=1,
                    requested_price=70000.0,
                    limit_price=0.0,
                    status="submitted",
                    fees_estimate=0.0,
                    slippage_bps=0.0,
                    retry_count=0,
                    strategy_version="kr_intraday_1h_v1",
                    reason="entry",
                    raw_json='{"expected_risk": 0.01}',
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)
            signal = replace(
                self._build_signal(symbol="005930.KS", asset_type="한국주식", timeframe="15m"),
                strategy_version="kr_intraday_15m_v1",
            )
            decision = engine.evaluate_entry(signal, correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "kr_strategy_conflict_pending")

    def test_kr_strategy_conflict_blocks_cross_strategy_open_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.upsert_position(
                PositionRecord(
                    position_id="pos_kr_open",
                    created_at="2026-03-08T09:00:00Z",
                    updated_at="2026-03-08T09:05:00Z",
                    closed_at=None,
                    prediction_id="pred-existing",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe=strategy_schedule(settings, "kr_intraday_1h_v1").timeframe,
                    side="LONG",
                    status="open",
                    quantity=1,
                    entry_price=70000.0,
                    mark_price=70500.0,
                    stop_loss=68000.0,
                    take_profit=73000.0,
                    trailing_stop=69000.0,
                    highest_price=70500.0,
                    lowest_price=70000.0,
                    unrealized_pnl=500.0,
                    realized_pnl=0.0,
                    expected_risk=0.01,
                    exposure_value=70500.0,
                    max_holding_until="2026-03-08T12:00:00Z",
                    strategy_version="kr_intraday_1h_v1",
                    cooldown_until=None,
                    notes="open",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)
            signal = replace(
                self._build_signal(symbol="005930.KS", asset_type="한국주식", timeframe="15m"),
                strategy_version="kr_intraday_15m_v1",
            )
            decision = engine.evaluate_entry(signal, correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "kr_strategy_conflict_active")

    def test_after_close_strategy_conflicts_with_regular_15m_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.upsert_position(
                PositionRecord(
                    position_id="pos_kr_regular_open",
                    created_at="2026-03-08T09:00:00Z",
                    updated_at="2026-03-08T09:05:00Z",
                    closed_at=None,
                    prediction_id="pred-existing",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe="15m",
                    side="LONG",
                    status="open",
                    quantity=1,
                    entry_price=70000.0,
                    mark_price=70500.0,
                    stop_loss=68000.0,
                    take_profit=73000.0,
                    trailing_stop=69000.0,
                    highest_price=70500.0,
                    lowest_price=70000.0,
                    unrealized_pnl=500.0,
                    realized_pnl=0.0,
                    expected_risk=0.01,
                    exposure_value=70500.0,
                    max_holding_until="2026-03-09T12:00:00Z",
                    strategy_version="kr_intraday_15m_v1",
                    cooldown_until=None,
                    notes="open",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)
            signal = replace(
                self._build_signal(symbol="005930.KS", asset_type="한국주식", timeframe="15m"),
                strategy_version="kr_intraday_15m_v1_after_close_close",
            )

            decision = engine.evaluate_entry(signal, correlation_matrix=pd.DataFrame(), market_is_open=True)

            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "kr_strategy_conflict_active")

    def test_partially_filled_entry_blocks_until_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_order(
                OrderRecord(
                    order_id="ord_partial",
                    created_at="2026-03-08T09:00:00Z",
                    updated_at="2026-03-08T09:05:00Z",
                    prediction_id="pred-existing",
                    scan_id="scan-existing",
                    symbol="BTC-USD",
                    asset_type="코인",
                    timeframe="1h",
                    side="buy",
                    order_type="market",
                    requested_qty=2,
                    filled_qty=1,
                    remaining_qty=1,
                    requested_price=100.0,
                    limit_price=0.0,
                    status="partially_filled",
                    fees_estimate=0.0,
                    slippage_bps=0.0,
                    retry_count=0,
                    strategy_version="s1",
                    reason="entry",
                    raw_json='{"expected_risk": 0.01}',
                )
            )
            engine = RiskEngine(settings, repo)
            blocked = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(blocked.allowed)
            self.assertEqual(blocked.reason, "duplicate_pending_entry")

            repo.update_order("ord_partial", status="cancelled", filled_qty=1, remaining_qty=1)
            allowed = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertTrue(allowed.allowed)

    def test_cooldown_blocks_then_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            base_position = PositionRecord(
                position_id="pos1",
                created_at="2026-03-08T08:00:00Z",
                updated_at="2026-03-08T09:00:00Z",
                closed_at="2026-03-08T09:00:00Z",
                prediction_id="pred-old",
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="LONG",
                status="closed",
                quantity=0,
                entry_price=100.0,
                mark_price=101.0,
                stop_loss=95.0,
                take_profit=110.0,
                trailing_stop=98.0,
                highest_price=105.0,
                lowest_price=99.0,
                unrealized_pnl=0.0,
                realized_pnl=1.0,
                expected_risk=0.02,
                exposure_value=0.0,
                max_holding_until="2026-03-08T12:00:00Z",
                strategy_version="s1",
                cooldown_until="2099-01-01T00:00:00Z",
            )
            repo.upsert_position(base_position)
            engine = RiskEngine(settings, repo)
            blocked = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(blocked.allowed)
            self.assertEqual(blocked.reason, "cooldown_active")

            repo.upsert_position(
                PositionRecord(
                    **{
                        **base_position.__dict__,
                        "position_id": "pos2",
                        "updated_at": "2026-03-08T09:01:00Z",
                        "cooldown_until": "2020-01-01T00:00:00Z",
                    }
                )
            )
            allowed = engine.evaluate_entry(self._build_signal(), correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertTrue(allowed.allowed)

    def test_non_kr_assets_ignore_kis_account_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_sim",
                    created_at="2026-03-08T09:00:00Z",
                    cash=20_000_000.0,
                    equity=20_500_000.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=100_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-1.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                )
            )
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_kis",
                    created_at="2026-03-08T09:01:00Z",
                    cash=30_000_000.0,
                    equity=31_000_000.0,
                    gross_exposure=1_000_000.0,
                    net_exposure=1_000_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=200_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-2.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="kis_account_sync",
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)
            state = engine._latest_account_state(asset_type="미국주식", symbol="AAPL")
            self.assertEqual(state["cash"], 20_000_000.0)
            self.assertEqual(state["equity"], 20_500_000.0)
            self.assertEqual(state["drawdown_pct"], -1.0)

    def test_kr_assets_prefer_kis_account_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_sim",
                    created_at="2026-03-08T09:00:00Z",
                    cash=12_000_000.0,
                    equity=12_500_000.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=50_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-1.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                )
            )
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_kis",
                    created_at="2026-03-08T09:01:00Z",
                    cash=27_000_000.0,
                    equity=27_500_000.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=60_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-0.5,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="kis_account_sync",
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)
            state = engine._latest_account_state(asset_type="한국주식", symbol="005930.KS")
            self.assertEqual(state["cash"], 27_000_000.0)
            self.assertEqual(state["equity"], 27_500_000.0)
            self.assertEqual(state["drawdown_pct"], -0.5)

    def test_kr_assets_fall_back_to_legacy_snapshot_when_kis_snapshot_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_legacy_sim_only",
                    created_at="2026-03-08T09:00:00Z",
                    cash=19_000_000.0,
                    equity=19_200_000.0,
                    gross_exposure=200_000.0,
                    net_exposure=200_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=20_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-0.8,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_LEGACY_MIXED,
                )
            )
            engine = RiskEngine(settings, repo)

            state = engine._latest_account_state(asset_type="한국주식", symbol="005930.KS")

            self.assertEqual(state["cash"], 19_000_000.0)
            self.assertEqual(state["equity"], 19_200_000.0)
            self.assertEqual(state["drawdown_pct"], -0.8)

    def test_crypto_assets_ignore_kis_account_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_sim_crypto",
                    created_at="2026-03-08T09:00:00Z",
                    cash=8_000_000.0,
                    equity=8_300_000.0,
                    gross_exposure=300_000.0,
                    net_exposure=300_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=30_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-1.5,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_CRYPTO,
                )
            )
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_kis_crypto_noise",
                    created_at="2026-03-08T09:01:00Z",
                    cash=40_000_000.0,
                    equity=41_000_000.0,
                    gross_exposure=1_000_000.0,
                    net_exposure=1_000_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=100_000.0,
                    daily_pnl=0.0,
                    drawdown_pct=-0.2,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="kis_account_sync",
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)

            state = engine._latest_account_state(asset_type="코인", symbol="BTC-USD")

            self.assertEqual(state["cash"], 8_000_000.0)
            self.assertEqual(state["equity"], 8_300_000.0)
            self.assertEqual(state["drawdown_pct"], -1.5)

    def test_kr_cash_shortage_does_not_block_us_equity_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_kis_low_cash",
                    created_at="2026-03-08T09:01:00Z",
                    cash=0.0,
                    equity=500_000.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    daily_pnl=0.0,
                    drawdown_pct=-2.0,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="kis_account_sync",
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_us_ok",
                    created_at="2026-03-08T09:02:00Z",
                    cash=2_000_000.0,
                    equity=2_200_000.0,
                    gross_exposure=0.0,
                    net_exposure=0.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    daily_pnl=0.0,
                    drawdown_pct=-1.0,
                    open_positions=0,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                )
            )
            engine = RiskEngine(settings, repo)
            decision = engine.evaluate_entry(
                self._build_signal(symbol="AAPL", asset_type="미국주식", timeframe="1d"),
                correlation_matrix=pd.DataFrame(),
                market_is_open=True,
            )

            self.assertTrue(decision.allowed)

    def test_us_drawdown_does_not_leak_into_kis_account_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_us_drawdown",
                    created_at="2026-03-08T09:00:00Z",
                    cash=12_000_000.0,
                    equity=12_200_000.0,
                    gross_exposure=300_000.0,
                    net_exposure=300_000.0,
                    realized_pnl=0.0,
                    unrealized_pnl=-500_000.0,
                    daily_pnl=-500_000.0,
                    drawdown_pct=-12.5,
                    open_positions=1,
                    open_orders=0,
                    paused=0,
                    source="paper_broker",
                    raw_json="{}",
                    account_id=ACCOUNT_SIM_US_EQUITY,
                )
            )
            repo.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id="snap_kis_safe",
                    created_at="2026-03-08T09:01:00Z",
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
                    raw_json="{}",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            )
            engine = RiskEngine(settings, repo)

            state = engine._latest_account_state(asset_type="한국주식", symbol="005930.KS")

            self.assertEqual(state["account_id"], ACCOUNT_KIS_KR_PAPER)
            self.assertEqual(state["drawdown_pct"], -0.5)


if __name__ == "__main__":
    unittest.main()
