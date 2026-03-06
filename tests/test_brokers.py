from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from config.settings import RuntimeSettings
from services.kis_paper_broker import KISPaperBroker
from services.market_data_service import MarketQuote
from services.paper_broker import SimBroker
from services.signal_engine import SignalDecision
from storage.repository import TradingRepository


class _AlwaysOpenMarketData:
    def is_market_open(self, asset_type: str) -> bool:
        return True

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str, purpose: str = "execution") -> MarketQuote:
        return MarketQuote(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            price=101.0,
            high=102.0,
            low=99.0,
            open=100.0,
            volume=10000.0,
            timestamp=pd.Timestamp("2026-03-06T00:00:00Z"),
        )


class _FakeKISClient:
    def __init__(self) -> None:
        self.quote_calls = 0
        self.order_calls = 0

    def get_quote(self, symbol_or_code: str):
        self.quote_calls += 1
        return {"current_price": 100.0, "market_name": "KOSPI"}

    def place_cash_order(self, symbol_or_code: str, side: str, quantity: int, order_type: str = "market", price: float | None = None):
        self.order_calls += 1
        return {"order_no": "KIS-001", "side": side, "quantity": quantity}

    def get_account_snapshot(self):
        return type(
            "Snapshot",
            (),
            {
                "summary": {"cash": 1000000.0, "stock_eval": 0.0, "total_eval": 1000000.0, "pnl": 0.0},
                "holdings": pd.DataFrame(columns=["symbol_code", "holding_qty", "avg_price", "market_price", "pnl"]),
            },
        )()


class _FakeKISHoldingClient(_FakeKISClient):
    def get_account_snapshot(self):
        return type(
            "Snapshot",
            (),
            {
                "summary": {"cash": 900000.0, "stock_eval": 100000.0, "total_eval": 1000000.0, "pnl": 5000.0},
                "holdings": pd.DataFrame(
                    [
                        {
                            "symbol_code": "005930",
                            "holding_qty": 2,
                            "avg_price": 50000.0,
                            "market_price": 52000.0,
                            "pnl": 4000.0,
                        }
                    ]
                ),
            },
        )()


class BrokerTest(unittest.TestCase):
    def test_sim_broker_process_open_orders_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.initialize_runtime_flags()
            broker = SimBroker(settings, repo)
            broker.ensure_account_initialized()
            coin_asset_type = next(asset for asset, schedule in settings.asset_schedules.items() if schedule.session_mode == "always")
            signal = SignalDecision(
                symbol="BTC-USD",
                asset_type=coin_asset_type,
                timeframe="1h",
                prediction_id="pred1",
                scan_id="scan1",
                score=1.0,
                signal="LONG",
                expected_return=0.02,
                expected_risk=0.01,
                confidence=0.8,
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
            order_id = broker.submit_entry_order(signal, quantity=2, scan_id="scan1")
            market_data = _AlwaysOpenMarketData()
            first = broker.process_open_orders(market_data)
            second = broker.process_open_orders(market_data)
            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            fills = repo.fills_for_order(order_id)
            self.assertEqual(len(fills), 1)

    def test_kis_paper_broker_rejects_short_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            settings.strategy.allow_short = True
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.initialize_runtime_flags()
            korean_asset_type = next(asset for asset, mode in settings.broker.asset_broker_mode.items() if mode == "kis_paper")
            broker = KISPaperBroker(settings, repo, client=_FakeKISClient())
            short_signal = SignalDecision(
                symbol="005930.KS",
                asset_type=korean_asset_type,
                timeframe="1d",
                prediction_id="pred-kr",
                scan_id="scan-kr",
                score=1.0,
                signal="SHORT",
                expected_return=-0.01,
                expected_risk=0.02,
                confidence=0.9,
                threshold=0.002,
                position_size=0.5,
                current_price=100.0,
                predicted_price=99.0,
                predicted_return=-0.01,
                stop_level=101.0,
                take_level=97.0,
                model_version="v1",
                feature_version="f1",
                strategy_version="s1",
                validation_mode="holdout",
                result=None,
            )
            with self.assertRaisesRegex(Exception, "short"):
                broker.submit_entry_order(short_signal, quantity=1)

    def test_kis_sync_is_idempotent_for_same_holdings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.initialize_runtime_flags()
            broker = KISPaperBroker(settings, repo, client=_FakeKISHoldingClient())
            broker.sync_state(force=True)
            broker.sync_state(force=True)
            open_positions = repo.open_positions()
            self.assertEqual(len(open_positions), 1)
            self.assertEqual(int(open_positions.iloc[0]["quantity"]), 2)


if __name__ == "__main__":
    unittest.main()
