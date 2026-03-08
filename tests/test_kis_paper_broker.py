from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperSnapshot
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from services.signal_engine import SignalDecision
from storage.repository import TradingRepository


class _FakeKISClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_paper=True)
        self.order_no = 0
        self.buying_power = {
            "cash_buy_qty": 10,
            "max_buy_qty": 10,
            "cash_buy_amount": 1000000.0,
        }
        self.holdings = pd.DataFrame(
            columns=[
                "symbol_code",
                "quantity",
                "avg_price",
                "보유수량",
                "매입평균가",
            ]
        )

    def get_account_snapshot(self) -> KISPaperSnapshot:
        return KISPaperSnapshot(summary={}, holdings=self.holdings.copy(), raw_summary={})

    def place_cash_order(self, symbol_or_code: str, side: str, quantity: int, order_type: str = "market", price: float | None = None):
        self.order_no += 1
        return {
            "order_no": f"ODR{self.order_no}",
            "requested_at": "2026-03-08T09:00:00+09:00",
            "message": "accepted",
        }

    def get_quote(self, symbol: str):
        return {"current_price": 70000.0}

    def get_buying_power(
        self,
        symbol_or_code: str,
        order_price: float | None = None,
        order_type: str = "market",
    ):
        return dict(self.buying_power)


class KISPaperBrokerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.settings.broker.fee_bps = 0.0
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.sim_broker = PaperBroker(self.settings, self.repo)
        self.sim_broker.ensure_account_initialized()
        self.client = _FakeKISClient()
        self.broker = KISPaperBroker(
            self.settings,
            self.repo,
            self.sim_broker,
            client_factory=lambda: self.client,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _signal(self) -> SignalDecision:
        return SignalDecision(
            symbol="005930.KS",
            asset_type="한국주식",
            timeframe="1d",
            prediction_id="pred-kr-1",
            scan_id="scan-kr-1",
            score=1.0,
            signal="LONG",
            expected_return=0.02,
            expected_risk=0.01,
            confidence=0.9,
            threshold=0.003,
            position_size=0.5,
            current_price=70000.0,
            predicted_price=71400.0,
            predicted_return=0.02,
            stop_level=68000.0,
            take_level=73000.0,
            model_version="v1",
            feature_version="f1",
            strategy_version="s1",
            validation_mode="holdout",
            result=None,
        )

    def test_kis_order_is_acknowledged_before_fill_and_reconciles_idempotently(self) -> None:
        order_id = self.broker.submit_entry_order(self._signal(), quantity=2)
        created = self.repo.get_order(order_id)

        self.assertEqual(created["status"], "acknowledged")
        self.assertEqual(int(created["filled_qty"]), 0)

        first_sync = self.broker.process_open_orders(market_data_service=None)
        pending = self.repo.get_order(order_id)
        self.assertEqual(first_sync, 0)
        self.assertEqual(pending["status"], "pending_fill")

        self.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 2,
                    "avg_price": 70000.0,
                    "보유수량": 2,
                    "매입평균가": 70000.0,
                }
            ]
        )
        second_sync = self.broker.process_open_orders(market_data_service=None)
        filled = self.repo.get_order(order_id)
        positions = self.repo.open_positions()
        fills = self.repo.open_orders(statuses=("filled",))

        self.assertEqual(second_sync, 1)
        self.assertEqual(filled["status"], "filled")
        self.assertEqual(int(filled["filled_qty"]), 2)
        self.assertFalse(positions.empty)
        self.assertEqual(int(positions.iloc[0]["quantity"]), 2)
        self.assertEqual(len(fills), 1)

        third_sync = self.broker.process_open_orders(market_data_service=None)
        self.assertEqual(third_sync, 0)
        still_filled = self.repo.get_order(order_id)
        self.assertEqual(still_filled["status"], "filled")

    def test_sync_account_records_monitor_flags_and_buying_power_probe(self) -> None:
        result = self.broker.sync_account()

        self.assertTrue(result["enabled"])
        self.assertNotEqual(self.repo.get_control_flag("kis_last_account_sync_at", ""), "")
        self.assertEqual(self.repo.get_control_flag("kis_last_account_sync_status", ""), "ok")
        self.assertNotEqual(self.repo.get_control_flag("kis_last_buying_power_success_at", ""), "")
        self.assertEqual(self.repo.get_control_flag("kis_last_buying_power_symbol", ""), "005930.KS")


if __name__ == "__main__":
    unittest.main()
