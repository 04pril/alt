from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from config.settings import RuntimeSettings
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision
from storage.models import OrderRecord, PositionRecord
from storage.repository import TradingRepository


class RiskEngineTest(unittest.TestCase):
    def _build_signal(self) -> SignalDecision:
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
            self.assertEqual(decision.reason, "already_holding_symbol")

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
            self.assertEqual(blocked.reason, "already_holding_symbol")

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


if __name__ == "__main__":
    unittest.main()
