from __future__ import annotations

from typing import Any

import pandas as pd

from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker


class BrokerRouter:
    def __init__(self, sim_broker: PaperBroker, kis_broker: KISPaperBroker | None = None):
        self.sim_broker = sim_broker
        self.kis_broker = kis_broker

    def ensure_account_initialized(self) -> None:
        self.sim_broker.ensure_account_initialized()

    def snapshot_account(self, cash_override: float | None = None) -> None:
        self.sim_broker.snapshot_account(cash_override=cash_override)

    def _use_kis(self, symbol: str, asset_type: str) -> bool:
        if self.kis_broker is None or not self.kis_broker.is_enabled():
            return False
        normalized_symbol = str(symbol).upper().strip()
        normalized_asset_type = str(asset_type).strip()
        return normalized_symbol.endswith((".KS", ".KQ")) or normalized_asset_type.endswith("주식")

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        if self._use_kis(signal.symbol, signal.asset_type):
            return self.kis_broker.submit_entry_order(signal=signal, quantity=quantity, scan_id=scan_id)
        return self.sim_broker.submit_entry_order(signal=signal, quantity=quantity, scan_id=scan_id)

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None):
        if self._use_kis(signal.symbol, signal.asset_type):
            return self.kis_broker.submit_entry_order_result(
                signal=signal,
                quantity=quantity,
                scan_id=scan_id,
                market_data_service=market_data_service,
            )
        return self.sim_broker.submit_entry_order_result(
            signal=signal,
            quantity=quantity,
            scan_id=scan_id,
            market_data_service=market_data_service,
        )

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        if self._use_kis(str(position["symbol"]), str(position["asset_type"])):
            return self.kis_broker.submit_exit_order(position=position, reason=reason)
        return self.sim_broker.submit_exit_order(position=position, reason=reason)

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None):
        if self._use_kis(str(position["symbol"]), str(position["asset_type"])):
            return self.kis_broker.submit_exit_order_result(
                position=position,
                reason=reason,
                market_data_service=market_data_service,
            )
        return self.sim_broker.submit_exit_order_result(
            position=position,
            reason=reason,
            market_data_service=market_data_service,
        )

    def preflight_entry(self, signal, quantity: int, *, market_data_service):
        if self._use_kis(signal.symbol, signal.asset_type):
            return self.kis_broker.preflight_entry(signal, quantity, market_data_service)
        return {"allowed": True, "reason": "ok", "broker": "sim"}

    def sync_account(self) -> dict[str, Any]:
        sim_summary = self.sim_broker.sync_account()
        result = {"sim": sim_summary}
        if self.kis_broker is not None and self.kis_broker.is_enabled():
            result["kis"] = self.kis_broker.sync_account()
        return result

    def sync_orders(self, market_data_service: Any, touch=None) -> dict[str, int]:
        result = self.sim_broker.sync_orders(market_data_service, touch=touch)
        if self.kis_broker is not None and self.kis_broker.is_enabled():
            kis_result = self.kis_broker.sync_orders(market_data_service, touch=touch)
            for key, value in kis_result.items():
                result[key] = int(result.get(key, 0)) + int(value)
        return result

    def process_open_orders(self, market_data_service: Any) -> int:
        return int(self.sync_orders(market_data_service).get("fills", 0))

    def handle_websocket_execution_event(self, event: dict[str, Any]) -> bool:
        if self.kis_broker is None or not self.kis_broker.is_enabled():
            return False
        return bool(self.kis_broker.handle_websocket_execution_event(event))
