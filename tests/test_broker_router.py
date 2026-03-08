from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from services.broker_router import BrokerRouter, resolve_broker_mode


class _StubSimBroker:
    def __init__(self) -> None:
        self.entry_calls = 0
        self.exit_calls = 0

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "sim-entry"

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "sim-exit"


class _StubKISBroker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entry_calls = 0
        self.exit_calls = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        self.entry_calls += 1
        return "kis-entry"

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        self.exit_calls += 1
        return "kis-exit"


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

    def test_kr_symbol_suffix_wins_on_asset_type_mismatch(self) -> None:
        self.assertEqual(
            resolve_broker_mode(symbol="005930.KS", asset_type="미국주식", kis_enabled=True),
            "kis_mock",
        )
        signal = SimpleNamespace(symbol="005930.KS", asset_type="미국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "kis-entry")
        self.assertEqual(self.kis.entry_calls, 1)

    def test_non_kr_symbol_does_not_route_to_kis_from_asset_label_only(self) -> None:
        self.assertEqual(
            resolve_broker_mode(symbol="AAPL", asset_type="한국주식", kis_enabled=True),
            "sim",
        )
        signal = SimpleNamespace(symbol="AAPL", asset_type="한국주식")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)


if __name__ == "__main__":
    unittest.main()
