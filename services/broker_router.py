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

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        if self._use_kis(str(position["symbol"]), str(position["asset_type"])):
            return self.kis_broker.submit_exit_order(position=position, reason=reason)
        return self.sim_broker.submit_exit_order(position=position, reason=reason)

    def process_open_orders(self, market_data_service: Any) -> int:
        filled = self.sim_broker.process_open_orders(market_data_service)
        if self.kis_broker is not None and self.kis_broker.is_enabled():
            filled += self.kis_broker.process_open_orders(market_data_service)
        return filled
