from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperSnapshot
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision
from storage.models import OrderRecord
from storage.repository import TradingRepository, utc_now_iso


class _FakeMarketDataService:
    def __init__(self, *, market_open: bool = True, pre_close: bool = True):
        self.market_open = market_open
        self.pre_close = pre_close

    def is_market_open(self, asset_type: str, when=None) -> bool:
        return self.market_open

    def is_pre_close_window(self, asset_type: str, when=None) -> bool:
        return self.pre_close


class _FakeKISClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_paper=True, hts_id="demo-user")
        self.order_no = 0
        self.place_calls = 0
        self.holdings = pd.DataFrame(columns=["symbol_code", "quantity", "avg_price"])
        self.buying_power = {"cash_buy_qty": 10, "max_buy_qty": 10, "cash_buy_amount": 1_000_000.0}
        self.sellable = {"sellable_qty": 10}
        self.daily_rows = pd.DataFrame()

    def get_account_snapshot(self) -> KISPaperSnapshot:
        market_value = float(pd.to_numeric(self.holdings.get("market_value"), errors="coerce").fillna(0.0).sum()) if not self.holdings.empty else 0.0
        unrealized = float(pd.to_numeric(self.holdings.get("unrealized_pnl"), errors="coerce").fillna(0.0).sum()) if not self.holdings.empty else 0.0
        cash = 30_000_000.0
        return KISPaperSnapshot(
            summary={
                "cash": cash,
                "stock_eval": market_value,
                "total_eval": cash + market_value,
                "pnl": unrealized,
                "holding_count": int(len(self.holdings)),
            },
            holdings=self.holdings.copy(),
            raw_summary={},
        )

    def get_quote(self, symbol: str):
        return {"symbol_code": "005930", "current_price": 70000.0, "raw": {}}

    def get_orderbook(self, symbol: str):
        return {"expected_price": 70000.0, "best_ask": 70000.0, "best_bid": 69900.0}

    def get_market_status(self, symbol: str):
        return {"is_halted": False, "phase_code": "open"}

    def get_buying_power(self, symbol: str, *, order_price: float, order_division: str = "01", include_cma: str = "N", include_overseas: str = "N"):
        return dict(self.buying_power)

    def get_sellable_quantity(self, symbol: str):
        return dict(self.sellable)

    def get_daily_order_fills(self, **kwargs):
        return self.daily_rows.copy()

    def get_websocket_approval_key(self) -> str:
        return "approval"

    def place_cash_order(self, symbol_or_code: str, side: str, quantity: int, order_type: str = "market", price: float | None = None):
        self.place_calls += 1
        self.order_no += 1
        return {
            "order_no": f"ODR{self.order_no}",
            "requested_at": utc_now_iso(),
            "message": "accepted",
        }


class KISPaperBrokerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.settings.broker.fee_bps = 0.0
        self.settings.broker.stale_submitted_order_timeout_minutes = 1
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.sim_broker = PaperBroker(self.settings, self.repo)
        self.sim_broker.ensure_account_initialized()
        self.client = _FakeKISClient()
        self.market = _FakeMarketDataService()
        self.broker = KISPaperBroker(self.settings, self.repo, self.sim_broker, client_factory=lambda: self.client)
        self.kr_asset_type = next(
            asset_type for asset_type, schedule in self.settings.asset_schedules.items() if schedule.timeframe == "1d" and schedule.timezone == "Asia/Seoul"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _signal(self) -> SignalDecision:
        return SignalDecision(
            symbol="005930.KS",
            asset_type=self.kr_asset_type,
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

    def test_market_open_preclose_with_buying_power_submits_order(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=2, market_data_service=self.market)
        order = self.repo.get_order(result["order_id"])
        self.assertTrue(result["submitted"])
        self.assertEqual(result["status"], "acknowledged")
        self.assertEqual(self.client.place_calls, 1)
        self.assertEqual(order["status"], "acknowledged")

    def test_market_closed_does_not_submit(self) -> None:
        market = _FakeMarketDataService(market_open=False, pre_close=False)
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "market_closed")
        self.assertEqual(self.client.place_calls, 0)

    def test_outside_preclose_window_does_not_submit(self) -> None:
        market = _FakeMarketDataService(market_open=True, pre_close=False)
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "outside_preclose_window")
        self.assertEqual(self.client.place_calls, 0)

    def test_insufficient_buying_power_does_not_submit(self) -> None:
        self.client.buying_power = {"cash_buy_qty": 0, "max_buy_qty": 0, "cash_buy_amount": 0.0}
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "insufficient_buying_power")

    def test_duplicate_pending_entry_does_not_submit(self) -> None:
        self.repo.insert_order(
            OrderRecord(
                order_id="ord_pending",
                created_at="2026-03-09T15:15:00Z",
                updated_at="2026-03-09T15:15:00Z",
                prediction_id="pred-old",
                scan_id="scan-old",
                symbol="005930.KS",
                asset_type=self.kr_asset_type,
                timeframe="1d",
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
                strategy_version="s1",
                reason="entry",
                raw_json='{"broker":"kis_mock"}',
            )
        )
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "duplicate_pending_entry")

    def test_order_sync_moves_acknowledged_to_pending_then_filled_idempotently(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=2, market_data_service=self.market)
        order_id = result["order_id"]

        first = self.broker.sync_orders(self.market)
        self.assertEqual(first["fills"], 0)
        self.assertEqual(self.repo.get_order(order_id)["status"], "pending_fill")

        self.client.daily_rows = pd.DataFrame([{"broker_order_id": "ODR1", "tot_ccld_qty": 2, "avg_prvs": 70000.0}])
        second = self.broker.sync_orders(self.market)
        self.assertEqual(second["fills"], 1)
        self.assertEqual(self.repo.get_order(order_id)["status"], "filled")
        self.assertEqual(int(self.repo.open_positions().iloc[0]["quantity"]), 2)

        third = self.broker.sync_orders(self.market)
        self.assertEqual(third["fills"], 0)
        self.assertEqual(self.repo.get_order(order_id)["status"], "filled")

    def test_order_lifecycle_rejected_and_cancelled(self) -> None:
        rejected = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.client.daily_rows = pd.DataFrame([{"broker_order_id": "ODR1", "rfus_yn": "Y", "status_text": "reject"}])
        reject_sync = self.broker.sync_orders(self.market)
        self.assertEqual(reject_sync["rejected"], 1)
        self.assertEqual(self.repo.get_order(rejected["order_id"])["status"], "rejected")

        self.client.daily_rows = pd.DataFrame()
        cancelled = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.client.daily_rows = pd.DataFrame([{"broker_order_id": "ODR2", "status_text": "cancel"}])
        cancel_sync = self.broker.sync_orders(self.market)
        self.assertEqual(cancel_sync["cancelled"], 1)
        self.assertEqual(self.repo.get_order(cancelled["order_id"])["status"], "cancelled")

    def test_websocket_execution_event_updates_internal_state(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=2, market_data_service=self.market)
        updated = self.broker.handle_websocket_execution_event(
            {"broker_order_id": "ODR1", "status": "filled", "filled_qty": 2, "fill_price": 70000.0}
        )
        self.assertTrue(updated)
        self.assertEqual(self.repo.get_order(result["order_id"])["status"], "filled")
        self.assertFalse(self.repo.open_positions().empty)

    def test_rest_reconcile_is_fallback_when_websocket_missing(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.client.daily_rows = pd.DataFrame([{"broker_order_id": "ODR1", "tot_ccld_qty": 1, "avg_prvs": 70000.0}])
        sync = self.broker.sync_orders(self.market)
        self.assertEqual(sync["fills"], 1)
        self.assertEqual(self.repo.get_order(result["order_id"])["status"], "filled")

    def test_sync_account_writes_canonical_snapshot_used_by_risk_engine(self) -> None:
        self.sim_broker.snapshot_account(cash_override=1_000_000.0)
        self.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 10,
                    "avg_price": 70000.0,
                    "current_price": 72000.0,
                    "unrealized_pnl": 20_000.0,
                    "market_value": 720_000.0,
                }
            ]
        )
        sync = self.broker.sync_account()
        latest = self.repo.latest_account_snapshot()
        engine = RiskEngine(self.settings, self.repo)
        state = engine._latest_account_state(asset_type="한국주식", symbol="005930.KS")

        self.assertTrue(sync["enabled"])
        self.assertEqual(str(latest["source"]), "kis_account_sync")
        self.assertEqual(float(latest["cash"]), 30_000_000.0)
        self.assertEqual(float(latest["equity"]), 30_720_000.0)
        self.assertEqual(float(latest["gross_exposure"]), 720_000.0)
        self.assertEqual(float(latest["net_exposure"]), 720_000.0)
        self.assertEqual(state["cash"], 30_000_000.0)
        self.assertEqual(state["equity"], 30_720_000.0)


if __name__ == "__main__":
    unittest.main()
