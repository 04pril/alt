from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from services.broker_router import BrokerRouter, resolve_broker_mode


class _StubSimBroker:
    def __init__(self) -> None:
        self.entry_calls = 0
        self.exit_calls = 0
        self.preflight_calls = 0
        self.entry_result_calls = 0
        self.exit_result_calls = 0

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "sim-entry"

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None):
        self.entry_result_calls += 1
        return {"broker": "sim", "order_id": "sim-entry"}

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "sim-exit"

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None):
        self.exit_result_calls += 1
        return {"broker": "sim", "order_id": "sim-exit"}

    def preflight_entry(self, signal, quantity: int, market_data_service=None):
        self.preflight_calls += 1
        return {"allowed": True, "reason": "ok", "broker": "sim"}


class _StubKISBroker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entry_calls = 0
        self.exit_calls = 0
        self.preflight_calls = 0
        self.entry_result_calls = 0
        self.exit_result_calls = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "kis-entry"

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None):
        self.entry_result_calls += 1
        return {"broker": "kis_mock", "order_id": "kis-entry"}

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "kis-exit"

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None):
        self.exit_result_calls += 1
        return {"broker": "kis_mock", "order_id": "kis-exit"}

    def preflight_entry(self, signal, quantity: int, market_data_service=None):
        self.preflight_calls += 1
        return {"allowed": True, "reason": "ok", "broker": "kis_mock"}


class BrokerRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _StubSimBroker()
        self.kis = _StubKISBroker(enabled=True)
        self.router = BrokerRouter(sim_broker=self.sim, kis_broker=self.kis)

    def test_kr_equity_routes_to_kis(self) -> None:
        signal = SimpleNamespace(symbol="005930.KS", asset_type="한국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "kis-entry")
        self.assertEqual(self.kis.entry_calls, 1)
        self.assertEqual(self.sim.entry_calls, 0)

    def test_bare_kr_code_routes_to_kis(self) -> None:
        signal = SimpleNamespace(symbol="005930", asset_type="한국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "kis-entry")
        self.assertEqual(self.kis.entry_calls, 1)
        self.assertEqual(self.sim.entry_calls, 0)

    def test_us_equity_routes_to_sim(self) -> None:
        signal = SimpleNamespace(symbol="AAPL", asset_type="미국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)

    def test_crypto_routes_to_sim(self) -> None:
        signal = SimpleNamespace(symbol="BTC-USD", asset_type="코인")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)

    def test_kr_symbol_does_not_override_us_asset_type(self) -> None:
        self.assertEqual(
            resolve_broker_mode(symbol="005930.KS", asset_type="미국주식", kis_enabled=True),
            "sim",
        )
        signal = SimpleNamespace(symbol="005930.KS", asset_type="미국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)

    def test_non_kr_symbol_does_not_route_to_kis_from_kr_asset_label(self) -> None:
        self.assertEqual(
            resolve_broker_mode(symbol="AAPL", asset_type="한국주식", kis_enabled=True),
            "sim",
        )
        signal = SimpleNamespace(symbol="AAPL", asset_type="한국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)

    def test_blank_asset_type_falls_back_to_kr_symbol(self) -> None:
        self.assertEqual(resolve_broker_mode(symbol="005930.KS", asset_type="", kis_enabled=True), "kis_mock")
        signal = SimpleNamespace(symbol="005930.KS", asset_type="")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "kis-entry")
        self.assertEqual(self.kis.entry_calls, 1)

    def test_blank_asset_type_with_non_kr_symbol_stays_sim(self) -> None:
        self.assertEqual(resolve_broker_mode(symbol="AAPL", asset_type="", kis_enabled=True), "sim")
        signal = SimpleNamespace(symbol="AAPL", asset_type="")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)

    def test_disabled_kis_falls_back_to_sim_for_kr_symbols(self) -> None:
        router = BrokerRouter(sim_broker=self.sim, kis_broker=_StubKISBroker(enabled=False))
        signal = SimpleNamespace(symbol="005930.KS", asset_type="한국주식")
        order_id = router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)

    def test_preflight_and_exit_use_same_routing_rule(self) -> None:
        signal = SimpleNamespace(symbol="AAPL", asset_type="한국주식")
        preflight = self.router.preflight_entry(signal, quantity=1, market_data_service=object())
        self.assertEqual(preflight["broker"], "sim")
        self.assertEqual(self.sim.preflight_calls, 1)
        self.assertEqual(self.kis.preflight_calls, 0)

        position = pd.Series({"symbol": "AAPL", "asset_type": "한국주식"})
        result = self.router.submit_exit_order_result(position, "manual_exit", market_data_service=object())
        self.assertEqual(result["broker"], "sim")
        self.assertEqual(self.sim.exit_result_calls, 1)
        self.assertEqual(self.kis.exit_result_calls, 0)


if __name__ == "__main__":
    unittest.main()
