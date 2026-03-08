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

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "sim-entry"

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "sim-exit"

    def preflight_entry(self, signal, quantity: int, market_data_service=None):
        self.preflight_calls += 1
        return {"allowed": True, "reason": "ok", "broker": "sim"}


class _StubKISBroker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entry_calls = 0
        self.exit_calls = 0
        self.preflight_calls = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "kis-entry"

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "kis-exit"

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
        self.assertEqual(resolve_broker_mode(symbol="005930.KS", asset_type="한국주식", kis_enabled=True), "kis_mock")

    def test_us_equity_routes_to_sim(self) -> None:
        signal = SimpleNamespace(symbol="AAPL", asset_type="미국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)

        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)
        self.assertEqual(resolve_broker_mode(symbol="AAPL", asset_type="미국주식", kis_enabled=True), "sim")

    def test_crypto_routes_to_sim(self) -> None:
        signal = SimpleNamespace(symbol="BTC-USD", asset_type="코인")
        order_id = self.router.submit_entry_order(signal, quantity=1)

        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)
        self.assertEqual(resolve_broker_mode(symbol="BTC-USD", asset_type="코인", kis_enabled=True), "sim")

    def test_asset_type_wins_on_symbol_conflict(self) -> None:
        self.assertEqual(
            resolve_broker_mode(symbol="005930.KS", asset_type="미국주식", kis_enabled=True),
            "sim",
        )
        signal = SimpleNamespace(symbol="005930.KS", asset_type="미국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)

        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)

    def test_blank_asset_type_falls_back_to_kr_suffix(self) -> None:
        self.assertEqual(resolve_broker_mode(symbol="005930.KS", asset_type="", kis_enabled=True), "kis_mock")
        self.assertEqual(resolve_broker_mode(symbol="068270.KQ", asset_type=None, kis_enabled=True), "kis_mock")

    def test_non_kr_symbol_does_not_route_to_kis_from_asset_label_only(self) -> None:
        self.assertEqual(resolve_broker_mode(symbol="AAPL", asset_type="한국주식", kis_enabled=True), "kis_mock")
        signal = SimpleNamespace(symbol="AAPL", asset_type="한국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)

        self.assertEqual(order_id, "kis-entry")
        self.assertEqual(self.kis.entry_calls, 1)

    def test_kis_disabled_falls_back_to_sim_even_for_kr(self) -> None:
        router = BrokerRouter(sim_broker=self.sim, kis_broker=_StubKISBroker(enabled=False))
        signal = SimpleNamespace(symbol="005930.KS", asset_type="한국주식")
        order_id = router.submit_entry_order(signal, quantity=1)

        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)

    def test_preflight_entry_uses_same_routing_rule(self) -> None:
        kr_signal = SimpleNamespace(symbol="005930.KS", asset_type="한국주식")
        us_signal = SimpleNamespace(symbol="AAPL", asset_type="미국주식")

        kr_result = self.router.preflight_entry(kr_signal, quantity=1, market_data_service=object())
        us_result = self.router.preflight_entry(us_signal, quantity=1, market_data_service=object())

        self.assertEqual(kr_result["broker"], "kis_mock")
        self.assertEqual(us_result["broker"], "sim")
        self.assertEqual(self.kis.preflight_calls, 1)
        self.assertEqual(self.sim.preflight_calls, 0)

    def test_submit_exit_order_uses_same_routing_rule(self) -> None:
        kr_position = pd.Series({"symbol": "005930.KS", "asset_type": "한국주식"})
        us_position = pd.Series({"symbol": "AAPL", "asset_type": "미국주식"})

        kr_order_id = self.router.submit_exit_order(kr_position, reason="manual_exit")
        us_order_id = self.router.submit_exit_order(us_position, reason="manual_exit")

        self.assertEqual(kr_order_id, "kis-exit")
        self.assertEqual(us_order_id, "sim-exit")
        self.assertEqual(self.kis.exit_calls, 1)
        self.assertEqual(self.sim.exit_calls, 1)


if __name__ == "__main__":
    unittest.main()
