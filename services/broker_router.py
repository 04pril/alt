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

    def submit_manual_kis_order(
        self,
        *,
        symbol: str,
        asset_type: str,
        timeframe: str,
        side: str,
        quantity: int,
        order_type: str,
        requested_price: float,
        prediction_id: str | None = None,
        scan_id: str | None = None,
        strategy_version: str = "manual_kis",
        reason: str = "manual_entry",
        raw_metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.kis_broker is None or not self._use_kis(symbol, asset_type):
            raise RuntimeError("KIS mock broker is not enabled for this symbol")
        return self.kis_broker.submit_manual_order(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            side=side,
            quantity=quantity,
            order_type=order_type,
            requested_price=requested_price,
            prediction_id=prediction_id,
            scan_id=scan_id,
            strategy_version=strategy_version,
            reason=reason,
            raw_metadata=raw_metadata,
        )

    def sync_account(self, touch=None) -> dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        touch("broker_account_sync_start", {"broker": "router"})
        self.sim_broker.snapshot_account(
            touch=lambda stage=None, details=None: touch(stage or "sim_account_snapshot", details),
        )
        payload: dict[str, Any] = {"sim_snapshot_written": 1}
        if self.kis_broker is not None and self.kis_broker.is_enabled():
            touch("broker_account_sync_kis", {"broker": "kis_mock"})
            payload["kis"] = self.kis_broker.sync_account(
                touch=lambda stage=None, details=None: touch(stage or "kis_account_sync", details),
            )
        else:
            payload["kis"] = {"broker": "kis_mock", "enabled": False}
        touch(
            "broker_account_sync_complete",
            {"sim_snapshot_written": payload["sim_snapshot_written"], "kis_enabled": bool(payload["kis"].get("enabled", False))},
        )
        return payload

    def sync_orders(self, market_data_service: Any, touch=None) -> dict[str, int]:
        touch = touch or (lambda *args, **kwargs: None)
        filled = self.sim_broker.process_open_orders(market_data_service, touch=touch)
        if self.kis_broker is not None and self.kis_broker.is_enabled():
            filled += self.kis_broker.process_open_orders(market_data_service, touch=touch)
        return {"fills": int(filled)}

    def process_open_orders(self, market_data_service: Any, touch=None) -> int:
        return int(self.sync_orders(market_data_service, touch=touch)["fills"])
