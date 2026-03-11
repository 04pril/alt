from __future__ import annotations

import json
import tempfile
import unittest

import pandas as pd

from config.settings import RuntimeSettings
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_LEGACY_MIXED, ACCOUNT_SIM_US_EQUITY
from services.market_data_service import MarketQuote
from services.paper_broker import PaperBroker
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord, PositionRecord
from storage.repository import TradingRepository


class _StubMarketDataService:
    def __init__(self, quote: MarketQuote):
        self.quote = quote

    def is_market_open(self, asset_type: str) -> bool:
        return True

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str) -> MarketQuote:
        return self.quote


class PaperBrokerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.settings.risk.starting_cash = 200.0
        self.settings.broker.fee_bps = 0.0
        self.settings.broker.base_slippage_bps = 0.0
        self.settings.broker.max_volume_participation = 0.05
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.broker = PaperBroker(self.settings, self.repo)
        self.broker.ensure_account_initialized()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _signal(self) -> SignalDecision:
        return SignalDecision(
            symbol="BTC-USD",
            asset_type="코인",
            timeframe="1h",
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

    def test_snapshot_account_does_not_double_count_unrealized_for_long(self) -> None:
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos1",
                created_at="2026-03-08T09:00:00Z",
                updated_at="2026-03-08T09:10:00Z",
                closed_at=None,
                prediction_id="pred1",
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="LONG",
                status="open",
                quantity=1,
                entry_price=100.0,
                mark_price=110.0,
                stop_loss=95.0,
                take_profit=120.0,
                trailing_stop=100.0,
                highest_price=110.0,
                lowest_price=100.0,
                unrealized_pnl=10.0,
                realized_pnl=0.0,
                expected_risk=0.02,
                exposure_value=110.0,
                max_holding_until="2026-03-08T12:00:00Z",
                strategy_version="s1",
            )
        )

        self.broker.snapshot_account(cash_override=90.0)
        latest = self.repo.latest_account_snapshot(account_id=ACCOUNT_SIM_CRYPTO)

        self.assertIsNotNone(latest)
        self.assertEqual(float(latest["equity"]), 200.0)
        self.assertEqual(float(latest["gross_exposure"]), 110.0)
        self.assertEqual(float(latest["net_exposure"]), 110.0)
        self.assertEqual(float(latest["unrealized_pnl"]), 10.0)

    def test_snapshot_account_uses_signed_exposure_for_short_and_drawdown(self) -> None:
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_peak",
                created_at="2026-03-08T08:00:00Z",
                cash=150.0,
                equity=150.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="test",
                account_id=ACCOUNT_SIM_CRYPTO,
            )
        )
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos_short",
                created_at="2026-03-08T09:00:00Z",
                updated_at="2026-03-08T09:10:00Z",
                closed_at=None,
                prediction_id="pred2",
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="SHORT",
                status="open",
                quantity=1,
                entry_price=100.0,
                mark_price=90.0,
                stop_loss=105.0,
                take_profit=80.0,
                trailing_stop=95.0,
                highest_price=100.0,
                lowest_price=90.0,
                unrealized_pnl=10.0,
                realized_pnl=0.0,
                expected_risk=0.02,
                exposure_value=-90.0,
                max_holding_until="2026-03-08T12:00:00Z",
                strategy_version="s1",
                account_id=ACCOUNT_SIM_CRYPTO,
            )
        )

        self.broker.snapshot_account(cash_override=210.0)
        latest = self.repo.latest_account_snapshot(account_id=ACCOUNT_SIM_CRYPTO)

        self.assertIsNotNone(latest)
        self.assertEqual(float(latest["gross_exposure"]), 90.0)
        self.assertEqual(float(latest["net_exposure"]), -90.0)
        self.assertEqual(float(latest["equity"]), 120.0)
        self.assertAlmostEqual(float(latest["drawdown_pct"]), -20.0)

    def test_sim_snapshot_drawdown_ignores_kis_peak(self) -> None:
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis_peak",
                created_at="2026-03-08T08:10:00Z",
                cash=1000.0,
                equity=1000.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )

        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_sim_crypto_peak",
                created_at="2026-03-08T08:00:00Z",
                cash=200.0,
                equity=200.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="paper_broker",
                account_id=ACCOUNT_SIM_CRYPTO,
            )
        )

        self.broker.snapshot_account(cash_override=150.0, account_id=ACCOUNT_SIM_CRYPTO)
        latest = self.repo.latest_account_snapshot(account_id=ACCOUNT_SIM_CRYPTO)

        self.assertIsNotNone(latest)
        self.assertEqual(str(latest["source"]), "paper_broker")
        self.assertAlmostEqual(float(latest["drawdown_pct"]), -25.0)

    def test_kis_snapshot_drawdown_ignores_sim_peak(self) -> None:
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_sim_peak",
                created_at="2026-03-08T08:00:00Z",
                cash=500.0,
                equity=500.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="paper_broker",
                account_id=ACCOUNT_SIM_CRYPTO,
            )
        )
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis_peak",
                created_at="2026-03-08T08:05:00Z",
                cash=300.0,
                equity=300.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )

        snapshot = self.broker.record_external_account_snapshot(
            cash=240.0,
            equity=240.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            unrealized_pnl=0.0,
            open_positions=0,
            source="kis_account_sync",
        )

        self.assertEqual(str(snapshot["source"]), "kis_account_sync")
        self.assertAlmostEqual(float(snapshot["drawdown_pct"]), -20.0)

    def test_kis_snapshot_drawdown_ignores_legacy_paper_peak_on_same_account(self) -> None:
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis_legacy_paper_peak",
                created_at="2026-03-08T08:00:00Z",
                cash=500.0,
                equity=500.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="paper_broker",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis_true_peak",
                created_at="2026-03-08T08:05:00Z",
                cash=300.0,
                equity=300.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )

        snapshot = self.broker.record_external_account_snapshot(
            cash=240.0,
            equity=240.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            unrealized_pnl=0.0,
            open_positions=0,
            source="kis_account_sync",
        )

        self.assertEqual(str(snapshot["source"]), "kis_account_sync")
        self.assertAlmostEqual(float(snapshot["drawdown_pct"]), -20.0)

    def test_fill_recalculates_long_exit_levels_from_actual_fill_price(self) -> None:
        signal = SignalDecision(
            **{
                **self._signal().__dict__,
                "symbol": "005930.KS",
                "asset_type": "한국주식",
                "timeframe": "15m",
                "strategy_version": "kr_intraday_15m_v1",
                "current_price": 194500.0,
                "stop_level": 194070.0,
                "take_level": 195353.0,
                "expected_risk": 0.003302578834825514,
                "atr_value": 643.0,
            }
        )
        order_id = self.broker.submit_entry_order(signal, quantity=1, account_id=ACCOUNT_KIS_KR_PAPER)

        self.broker.apply_external_fill(
            order_id,
            fill_qty=1,
            fill_price=190100.0,
            raw_json={"broker": "kis_mock", "account_id": ACCOUNT_KIS_KR_PAPER},
            final_status="filled",
        )

        position = self.repo.open_positions(account_id=ACCOUNT_KIS_KR_PAPER).iloc[0]
        self.assertLess(float(position["stop_loss"]), float(position["entry_price"]))
        self.assertGreater(float(position["take_profit"]), float(position["entry_price"]))
        self.assertAlmostEqual(float(position["trailing_stop"]), float(position["stop_loss"]))
        self.assertEqual(str(position["strategy_family"]), "kr_intraday_15m")
        self.assertEqual(str(position["session_mode"]), "regular")
        self.assertEqual(str(position["price_policy"]), "market_best_effort")
        with self.repo.connect() as conn:
            fills = pd.read_sql_query(
                "SELECT * FROM fills WHERE account_id = ? ORDER BY created_at DESC, rowid DESC",
                conn,
                params=[ACCOUNT_KIS_KR_PAPER],
            )
        self.assertFalse(fills.empty)
        fill_meta = json.loads(str(fills.iloc[0]["raw_json"] or "{}"))
        self.assertEqual(str(fill_meta.get("strategy_family") or ""), "kr_intraday_15m")
        self.assertEqual(str(fill_meta.get("session_mode") or ""), "regular")
        self.assertEqual(str(fill_meta.get("price_policy") or ""), "market_best_effort")

    def test_external_fill_for_kis_order_does_not_write_paper_snapshot(self) -> None:
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis_only",
                created_at="2026-03-08T08:00:00Z",
                cash=30000000.0,
                equity=30000000.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        signal = SignalDecision(
            **{
                **self._signal().__dict__,
                "symbol": "005930.KS",
                "asset_type": "한국주식",
                "timeframe": "15m",
                "strategy_version": "kr_intraday_15m_v1",
                "atr_value": 643.0,
            }
        )
        order_id = self.broker.submit_entry_order(signal, quantity=1, account_id=ACCOUNT_KIS_KR_PAPER)

        self.broker.apply_external_fill(
            order_id,
            fill_qty=1,
            fill_price=190100.0,
            raw_json={"broker": "kis_mock", "account_id": ACCOUNT_KIS_KR_PAPER},
            final_status="filled",
        )

        latest = self.repo.latest_account_snapshot(account_id=ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(str(latest["source"]), "kis_account_sync")

    def test_submit_entry_order_records_crypto_account_id(self) -> None:
        order_id = self.broker.submit_entry_order(self._signal(), quantity=1)
        order = self.repo.get_order(order_id)

        self.assertIsNotNone(order)
        self.assertEqual(str(order["account_id"]), ACCOUNT_SIM_CRYPTO)

    def test_submit_entry_order_records_us_equity_account_id(self) -> None:
        signal = SignalDecision(
            **{
                **self._signal().__dict__,
                "symbol": "AAPL",
                "asset_type": "미국주식",
                "timeframe": "15m",
            }
        )
        order_id = self.broker.submit_entry_order(signal, quantity=1)
        order = self.repo.get_order(order_id)

        self.assertIsNotNone(order)
        self.assertEqual(str(order["account_id"]), ACCOUNT_SIM_US_EQUITY)

    def test_snapshot_account_never_creates_legacy_mixed_runtime_rows(self) -> None:
        self.broker.snapshot_account()
        snapshots = self.repo.load_account_snapshots(limit=20)
        self.assertFalse(snapshots.empty)
        self.assertNotIn(ACCOUNT_SIM_LEGACY_MIXED, set(snapshots["account_id"].astype(str)))

    def test_allow_partial_fills_false_keeps_order_open(self) -> None:
        self.settings.broker.allow_partial_fills = False
        order_id = self.broker.submit_entry_order(self._signal(), quantity=10)
        quote = MarketQuote(
            symbol="BTC-USD",
            asset_type="코인",
            timeframe="1h",
            price=100.0,
            high=101.0,
            low=99.0,
            open=100.0,
            volume=50.0,
            timestamp=pd.Timestamp("2026-03-08T09:00:00Z"),
        )

        fills = self.broker.process_open_orders(_StubMarketDataService(quote))
        order = self.repo.get_order(order_id)

        self.assertEqual(fills, 0)
        self.assertEqual(order["status"], "new")
        self.assertEqual(int(order["filled_qty"]), 0)

    def test_allow_partial_fills_true_allows_partial_fill(self) -> None:
        self.settings.broker.allow_partial_fills = True
        order_id = self.broker.submit_entry_order(self._signal(), quantity=10)
        quote = MarketQuote(
            symbol="BTC-USD",
            asset_type="코인",
            timeframe="1h",
            price=100.0,
            high=101.0,
            low=99.0,
            open=100.0,
            volume=50.0,
            timestamp=pd.Timestamp("2026-03-08T09:00:00Z"),
        )

        fills = self.broker.process_open_orders(_StubMarketDataService(quote))
        order = self.repo.get_order(order_id)

        self.assertEqual(fills, 1)
        self.assertEqual(order["status"], "partially_filled")
        self.assertEqual(int(order["filled_qty"]), 2)
        self.assertEqual(int(order["remaining_qty"]), 8)


if __name__ == "__main__":
    unittest.main()
