from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import default_kr_strategy, strategy_schedule
from kis_paper import KISPaperSnapshot
from runtime_accounts import ACCOUNT_KIS_KR_PAPER
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision
from storage.models import OrderRecord, PositionRecord
from storage.repository import TradingRepository, utc_now_iso


class _FakeMarketDataService:
    def __init__(self, *, market_open: bool = True, pre_close: bool = True):
        self.market_open = market_open
        self.pre_close = pre_close
        self.current_times: dict[str, datetime] = {}

    def is_market_open(self, asset_type: str, when=None) -> bool:
        return self.market_open

    def is_pre_close_window(self, asset_type: str, when=None) -> bool:
        return self.pre_close

    def current_time(self, asset_type: str) -> datetime:
        if asset_type in self.current_times:
            return self.current_times[asset_type]
        return datetime(2026, 3, 9, 14, 10, tzinfo=ZoneInfo("Asia/Seoul"))


class _FakeKISClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_paper=True, hts_id="demo-user")
        self.order_no = 0
        self.place_calls = 0
        self.place_orders: list[dict[str, object]] = []
        self.holdings = pd.DataFrame(columns=["symbol_code", "quantity", "avg_price"])
        self.buying_power = {"cash_buy_qty": 10, "max_buy_qty": 10, "cash_buy_amount": 1_000_000.0}
        self.sellable = {"sellable_qty": 10}
        self.daily_rows = pd.DataFrame()

    def get_account_snapshot(self) -> KISPaperSnapshot:
        market_value = (
            float(pd.to_numeric(self.holdings["market_value"], errors="coerce").fillna(0.0).sum())
            if not self.holdings.empty and "market_value" in self.holdings.columns
            else 0.0
        )
        unrealized = (
            float(pd.to_numeric(self.holdings["unrealized_pnl"], errors="coerce").fillna(0.0).sum())
            if not self.holdings.empty and "unrealized_pnl" in self.holdings.columns
            else 0.0
        )
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

    def get_overtime_price(self, symbol: str):
        return {
            "symbol_code": "005930",
            "current_price": 70000.0,
            "expected_price": 70100.0,
            "close_price": 70000.0,
            "upper_limit": 77000.0,
            "lower_limit": 63000.0,
            "best_bid": 70000.0,
            "best_ask": 70000.0,
            "raw": {},
        }

    def get_overtime_asking_price(self, symbol: str):
        return {"expected_price": 70100.0, "best_ask": 70100.0, "best_bid": 70000.0}

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

    def place_cash_order(self, symbol_or_code: str, side: str, quantity: int, order_type: str = "market", price: float | None = None, order_division: str | None = None):
        self.place_calls += 1
        self.order_no += 1
        self.place_orders.append(
            {
                "symbol_or_code": symbol_or_code,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "order_division": order_division,
            }
        )
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
        self.kr_strategy = default_kr_strategy(self.settings)
        self.assertIsNotNone(self.kr_strategy)
        self.kr_timeframe = str(strategy_schedule(self.settings, str(self.kr_strategy.strategy_id)).timeframe)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _signal(self) -> SignalDecision:
        return SignalDecision(
            symbol="005930.KS",
            asset_type=self.kr_asset_type,
            timeframe=self.kr_timeframe,
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
            strategy_version=str(self.kr_strategy.strategy_id),
            validation_mode=str(self.kr_strategy.validation_mode),
            result=None,
        )

    def test_market_open_preclose_with_buying_power_submits_order(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=2, market_data_service=self.market)
        order = self.repo.get_order(result["order_id"])
        self.assertTrue(result["submitted"])
        self.assertEqual(result["status"], "acknowledged")
        self.assertEqual(result["account_id"], ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(self.client.place_calls, 1)
        self.assertEqual(order["status"], "acknowledged")
        self.assertEqual(order["account_id"], ACCOUNT_KIS_KR_PAPER)

    def test_market_closed_does_not_submit(self) -> None:
        market = _FakeMarketDataService(market_open=False, pre_close=False)
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "market_closed")
        self.assertEqual(self.client.place_calls, 0)

    def test_outside_preclose_window_does_not_submit(self) -> None:
        market = _FakeMarketDataService(market_open=True, pre_close=False)
        market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 15, 5, tzinfo=ZoneInfo("Asia/Seoul"))
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "outside_intraday_entry_window")
        self.assertEqual(self.client.place_calls, 0)

    def test_insufficient_buying_power_does_not_submit(self) -> None:
        self.client.buying_power = {"cash_buy_qty": 0, "max_buy_qty": 0, "cash_buy_amount": 0.0}
        result = self.broker.submit_entry_order_result(self._signal(), quantity=1, market_data_service=self.market)
        self.assertFalse(result["submitted"])
        self.assertEqual(result["reason"], "insufficient_buying_power")

    def test_after_close_close_preflight_uses_close_price_order_division(self) -> None:
        self.settings.kr_strategies["kr_intraday_15m_v1_after_close_close"].enabled = True
        signal = self._signal()
        signal = SignalDecision(**{**signal.__dict__, "timeframe": "15m", "strategy_version": "kr_intraday_15m_v1_after_close_close"})
        self.market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul"))

        result = self.broker.submit_entry_order_result(signal, quantity=1, market_data_service=self.market)

        self.assertTrue(result["submitted"])
        self.assertEqual(self.client.place_orders[-1]["order_division"], "06")
        self.assertEqual(self.client.place_orders[-1]["order_type"], "after_close_close")
        order = self.repo.get_order(str(result["order_id"]))
        payload = json.loads(str(order.get("raw_json") or "{}"))
        self.assertEqual(str(payload.get("strategy_family") or ""), "kr_intraday_15m")
        self.assertEqual(str(payload.get("session_mode") or ""), "after_close_close_price")
        self.assertEqual(str(payload.get("price_policy") or ""), "close_price")

    def test_after_close_single_gate_and_order_division(self) -> None:
        self.settings.kr_strategies["kr_intraday_15m_v1_after_close_single"].enabled = True
        signal = self._signal()
        signal = SignalDecision(**{**signal.__dict__, "timeframe": "15m", "strategy_version": "kr_intraday_15m_v1_after_close_single"})
        self.market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 16, 5, tzinfo=ZoneInfo("Asia/Seoul"))

        waiting = self.broker.submit_entry_order_result(signal, quantity=1, market_data_service=self.market)
        self.assertFalse(waiting["submitted"])
        self.assertEqual(waiting["reason"], "after_close_single_waiting_auction")

        self.market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul"))
        submitted = self.broker.submit_entry_order_result(signal, quantity=1, market_data_service=self.market)
        self.assertTrue(submitted["submitted"])
        self.assertEqual(self.client.place_orders[-1]["order_division"], "07")
        self.assertEqual(self.client.place_orders[-1]["order_type"], "after_close_single")

    def test_auto_15m_profile_switches_order_division_by_session(self) -> None:
        self.settings.kr_strategies["kr_intraday_15m_v1_auto"].enabled = True
        signal = self._signal()
        signal = SignalDecision(**{**signal.__dict__, "timeframe": "15m", "strategy_version": "kr_intraday_15m_v1_auto"})

        self.market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul"))
        close_submitted = self.broker.submit_entry_order_result(signal, quantity=1, market_data_service=self.market)
        self.assertTrue(close_submitted["submitted"])
        self.assertEqual(self.client.place_orders[-1]["order_division"], "06")
        self.assertEqual(self.client.place_orders[-1]["order_type"], "after_close_close")

        signal_2 = SignalDecision(
            **{
                **signal.__dict__,
                "symbol": "000660.KS",
                "prediction_id": "pred-kr-2",
                "scan_id": "scan-kr-2",
            }
        )
        self.market.current_times[self.kr_asset_type] = datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul"))
        single_submitted = self.broker.submit_entry_order_result(signal_2, quantity=1, market_data_service=self.market)
        self.assertTrue(single_submitted["submitted"])
        self.assertEqual(self.client.place_orders[-1]["order_division"], "07")
        self.assertEqual(self.client.place_orders[-1]["order_type"], "after_close_single")
        order = self.repo.get_order(str(single_submitted["order_id"]))
        payload = json.loads(str(order.get("raw_json") or "{}"))
        self.assertEqual(str(payload.get("strategy_family") or ""), "kr_intraday_15m")
        self.assertEqual(str(payload.get("session_mode") or ""), "after_close_single_price")
        self.assertEqual(str(payload.get("price_policy") or ""), "auction_expected_price")

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
                timeframe=self.kr_timeframe,
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
                strategy_version=str(self.kr_strategy.strategy_id),
                reason="entry",
                raw_json='{"broker":"kis_mock"}',
                account_id=ACCOUNT_KIS_KR_PAPER,
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
        self.assertEqual(int(self.repo.open_positions(account_id=ACCOUNT_KIS_KR_PAPER).iloc[0]["quantity"]), 2)

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

    def test_holdings_reconcile_fills_acknowledged_order_when_rest_lookup_missing(self) -> None:
        result = self.broker.submit_entry_order_result(self._signal(), quantity=2, market_data_service=self.market)
        self.client.daily_rows = pd.DataFrame()
        self.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 2,
                    "avg_price": 70000.0,
                    "current_price": 70500.0,
                    "unrealized_pnl": 1000.0,
                    "market_value": 141000.0,
                }
            ]
        )

        sync = self.broker.sync_orders(self.market)
        order = self.repo.get_order(result["order_id"])
        position = self.repo.open_positions(account_id=ACCOUNT_KIS_KR_PAPER).iloc[0]

        self.assertEqual(sync["fills"], 1)
        self.assertEqual(order["status"], "filled")
        self.assertEqual(int(position["quantity"]), 2)
        self.assertLess(float(position["stop_loss"]), float(position["entry_price"]))
        self.assertGreater(float(position["take_profit"]), float(position["entry_price"]))

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
        latest = self.repo.latest_account_snapshot(account_id=ACCOUNT_KIS_KR_PAPER)
        engine = RiskEngine(self.settings, self.repo)
        state = engine._latest_account_state(asset_type="한국주식", symbol="005930.KS")

        self.assertTrue(sync["enabled"])
        self.assertEqual(sync["account_id"], ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(str(latest["source"]), "kis_account_sync")
        self.assertEqual(str(latest["account_id"]), ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(float(latest["cash"]), 30_000_000.0)
        self.assertEqual(float(latest["equity"]), 30_720_000.0)
        self.assertEqual(float(latest["gross_exposure"]), 720_000.0)
        self.assertEqual(float(latest["net_exposure"]), 720_000.0)
        self.assertEqual(state["cash"], 30_000_000.0)
        self.assertEqual(state["equity"], 30_720_000.0)

    def test_sync_account_restores_and_repairs_position_from_holdings(self) -> None:
        self.repo.insert_order(
            OrderRecord(
                order_id="ord_restore",
                created_at="2026-03-09T05:55:46Z",
                updated_at="2026-03-09T05:55:48Z",
                prediction_id="pred_restore",
                scan_id="scan_restore",
                symbol="005930.KS",
                asset_type=self.kr_asset_type,
                timeframe="15m",
                side="buy",
                order_type="market",
                requested_qty=27,
                filled_qty=27,
                remaining_qty=0,
                requested_price=190100.0,
                limit_price=0.0,
                status="filled",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="kr_intraday_15m_v1",
                reason="entry",
                raw_json=json.dumps(
                    {
                        "broker": "kis_mock",
                        "account_id": ACCOUNT_KIS_KR_PAPER,
                        "symbol_code": "005930",
                        "expected_risk": 0.003302578834825514,
                        "stop_level": 194070.0,
                        "take_level": 195353.0,
                        "atr_14": 643.0,
                        "max_holding_until": "2026-03-10T00:00:00Z",
                        "strategy_version": "kr_intraday_15m_v1",
                    },
                    ensure_ascii=False,
                ),
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos_wrong",
                created_at="2026-03-09T05:55:49Z",
                updated_at="2026-03-09T05:55:49Z",
                closed_at=None,
                prediction_id="pred_restore",
                symbol="005930.KS",
                asset_type=self.kr_asset_type,
                timeframe="15m",
                side="LONG",
                status="open",
                quantity=27,
                entry_price=190003.0,
                mark_price=190003.0,
                stop_loss=194070.0,
                take_profit=195353.0,
                trailing_stop=194070.0,
                highest_price=190003.0,
                lowest_price=190003.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.003302578834825514,
                exposure_value=5130081.0,
                max_holding_until="2026-03-10T00:00:00Z",
                strategy_version="kr_intraday_15m_v1",
                cooldown_until=None,
                notes="opened_by_fill",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        self.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 27,
                    "avg_price": 190003.703,
                    "current_price": 190100.0,
                    "unrealized_pnl": 2600.0,
                    "market_value": 5132700.0,
                }
            ]
        )

        sync = self.broker.sync_account()
        position = self.repo.open_positions(account_id=ACCOUNT_KIS_KR_PAPER).iloc[0]

        self.assertEqual(sync["reconcile_totals"]["updated"], 1)
        self.assertAlmostEqual(float(position["entry_price"]), 190003.703, places=3)
        self.assertLess(float(position["stop_loss"]), float(position["entry_price"]))
        self.assertGreater(float(position["take_profit"]), float(position["entry_price"]))

    def test_exit_preflight_falls_back_to_snapshot_when_sellable_api_fails_in_paper(self) -> None:
        self.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 2,
                    "avg_price": 70000.0,
                }
            ]
        )

        def _raise_sellable(symbol: str):
            raise RuntimeError("paper sellable unavailable")

        self.client.get_sellable_quantity = _raise_sellable  # type: ignore[method-assign]
        now_iso = utc_now_iso()
        position = pd.Series(
            PositionRecord(
                position_id="pos_exit_1",
                created_at=now_iso,
                updated_at=now_iso,
                closed_at=None,
                prediction_id="pred_exit_1",
                symbol="005930.KS",
                asset_type=self.kr_asset_type,
                timeframe=self.kr_timeframe,
                side="LONG",
                status="open",
                quantity=2,
                entry_price=70000.0,
                mark_price=70000.0,
                stop_loss=68000.0,
                take_profit=73000.0,
                trailing_stop=69000.0,
                highest_price=70000.0,
                lowest_price=70000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.01,
                exposure_value=140000.0,
                max_holding_until="2026-03-10T00:00:00Z",
                strategy_version=str(self.kr_strategy.strategy_id),
                cooldown_until=None,
                notes="opened_by_fill",
                account_id=ACCOUNT_KIS_KR_PAPER,
            ).__dict__
        )

        result = self.broker.submit_exit_order_result(position, reason="take_profit", market_data_service=self.market)

        self.assertTrue(result["submitted"])
        self.assertEqual(result["status"], "acknowledged")
        self.assertEqual(self.client.place_orders[-1]["side"], "sell")


if __name__ == "__main__":
    unittest.main()
