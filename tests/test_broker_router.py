from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from services.broker_router import BrokerRouter, resolve_broker_mode


class _StubSimBroker:
    def __init__(self) -> None:
        self.entry_calls = 0
        self.exit_calls = 0
        self.preflight_calls = 0
        self.entry_result_calls = 0
        self.exit_result_calls = 0
        self.last_account_id = ""

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None, *, account_id: str | None = None) -> str:
        self.entry_calls += 1
        self.last_account_id = str(account_id or "")
        return "sim-entry"

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None, account_id: str | None = None):
        self.entry_result_calls += 1
        self.last_account_id = str(account_id or "")
        return {"broker": "sim", "order_id": "sim-entry", "account_id": self.last_account_id}

    def submit_exit_order(self, position: pd.Series, reason: str, *, account_id: str | None = None) -> str:
        self.exit_calls += 1
        self.last_account_id = str(account_id or "")
        return "sim-exit"

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None, account_id: str | None = None):
        self.exit_result_calls += 1
        self.last_account_id = str(account_id or "")
        return {"broker": "sim", "order_id": "sim-exit", "account_id": self.last_account_id}

    def preflight_entry(self, signal, quantity: int, market_data_service=None, account_id: str | None = None):
        self.preflight_calls += 1
        self.last_account_id = str(account_id or "")
        return {"allowed": True, "reason": "ok", "broker": "sim", "account_id": self.last_account_id}


class _StubKISBroker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entry_calls = 0
        self.exit_calls = 0
        self.preflight_calls = 0
        self.entry_result_calls = 0
        self.exit_result_calls = 0
        self.last_account_id = ""

    def is_enabled(self) -> bool:
        return self.enabled

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None, *, account_id: str | None = None) -> str:
        self.entry_calls += 1
        self.last_account_id = str(account_id or "")
        return "kis-entry"

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None, account_id: str | None = None):
        self.entry_result_calls += 1
        self.last_account_id = str(account_id or "")
        return {"broker": "kis_mock", "order_id": "kis-entry", "account_id": self.last_account_id}

    def submit_exit_order(self, position: pd.Series, reason: str, *, account_id: str | None = None) -> str:
        self.exit_calls += 1
        self.last_account_id = str(account_id or "")
        return "kis-exit"

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None, account_id: str | None = None):
        self.exit_result_calls += 1
        self.last_account_id = str(account_id or "")
        return {"broker": "kis_mock", "order_id": "kis-exit", "account_id": self.last_account_id}

    def preflight_entry(self, signal, quantity: int, market_data_service=None, account_id: str | None = None):
        self.preflight_calls += 1
        self.last_account_id = str(account_id or "")
        return {"allowed": True, "reason": "ok", "broker": "kis_mock", "account_id": self.last_account_id}


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
        self.assertEqual(self.kis.last_account_id, ACCOUNT_KIS_KR_PAPER)

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
        self.assertEqual(self.sim.last_account_id, ACCOUNT_SIM_US_EQUITY)

    def test_crypto_routes_to_sim(self) -> None:
        signal = SimpleNamespace(symbol="BTC-USD", asset_type="코인")
        order_id = self.router.submit_entry_order(signal, quantity=1)
        self.assertEqual(order_id, "sim-entry")
        self.assertEqual(self.sim.entry_calls, 1)
        self.assertEqual(self.kis.entry_calls, 0)
        self.assertEqual(self.sim.last_account_id, ACCOUNT_SIM_CRYPTO)

    def test_truth_table_resolves_execution_account_id(self) -> None:
        self.assertEqual(self.router.resolve_execution_account_id("005930.KS", "한국주식"), ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(self.router.resolve_execution_account_id("AAPL", "미국주식"), ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(self.router.resolve_execution_account_id("BTC-USD", "코인"), ACCOUNT_SIM_CRYPTO)

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

    def test_blank_symbol_with_kr_asset_type_defaults_to_kis(self) -> None:
        self.assertEqual(resolve_broker_mode(symbol="", asset_type="한국주식", kis_enabled=True), "kis_mock")
        self.assertEqual(self.router.resolve_execution_account_id("", "한국주식"), ACCOUNT_KIS_KR_PAPER)

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
        self.assertEqual(preflight["account_id"], ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(self.sim.preflight_calls, 1)
        self.assertEqual(self.kis.preflight_calls, 0)

        position = pd.Series({"symbol": "AAPL", "asset_type": "한국주식", "account_id": ACCOUNT_SIM_US_EQUITY})
        result = self.router.submit_exit_order_result(position, "manual_exit", market_data_service=object())
        self.assertEqual(result["broker"], "sim")
        self.assertEqual(result["account_id"], ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(self.sim.exit_result_calls, 1)
        self.assertEqual(self.kis.exit_result_calls, 0)


if __name__ == "__main__":
    unittest.main()
