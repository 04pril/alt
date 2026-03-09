from __future__ import annotations

from typing import Any

import pandas as pd

from runtime_accounts import (
    ACCOUNT_KIS_KR_PAPER,
    BROKER_MODE_KIS,
    BROKER_MODE_SIM,
    CRYPTO_ASSET_TYPE,
    ExecutionAccount,
    KR_EQUITY_ASSET_TYPE,
    US_EQUITY_ASSET_TYPE,
    is_kis_routable_kr_equity,
    resolve_execution_account,
)
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker


def resolve_broker_mode(symbol: str = "", asset_type: str = "", *, kis_enabled: bool) -> str:
    return resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=kis_enabled).broker_mode


class BrokerRouter:
    def __init__(self, sim_broker: PaperBroker, kis_broker: KISPaperBroker | None = None):
        self.sim_broker = sim_broker
        self.kis_broker = kis_broker

    def _kis_enabled(self) -> bool:
        return self.kis_broker is not None and self.kis_broker.is_enabled()

    def resolve_execution_context(self, symbol: str = "", asset_type: str = "") -> ExecutionAccount:
        return resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=self._kis_enabled())

    def resolve_execution_account_id(self, symbol: str = "", asset_type: str = "") -> str:
        return self.resolve_execution_context(symbol=symbol, asset_type=asset_type).account_id

    def ensure_account_initialized(self) -> None:
        self.sim_broker.ensure_account_initialized()

    def snapshot_account(self, cash_override: float | None = None) -> None:
        self.sim_broker.snapshot_account(cash_override=cash_override)

    def _use_kis(self, symbol: str, asset_type: str) -> bool:
        return self.resolve_execution_context(symbol=symbol, asset_type=asset_type).broker_mode == BROKER_MODE_KIS

    def broker_mode_for_asset(self, asset_type: str, symbol: str = "") -> str:
        return self.resolve_execution_context(symbol=symbol, asset_type=asset_type).broker_mode

    def _invoke_with_account(self, fn, *args, account_id: str, **kwargs):
        try:
            return fn(*args, account_id=account_id, **kwargs)
        except TypeError as exc:
            if "account_id" not in str(exc):
                raise
            return fn(*args, **kwargs)

    def submit_entry_order(self, signal, quantity: int, scan_id: str | None = None) -> str:
        context = self.resolve_execution_context(signal.symbol, signal.asset_type)
        if context.broker_mode == BROKER_MODE_KIS:
            return self._invoke_with_account(
                self.kis_broker.submit_entry_order,
                signal=signal,
                quantity=quantity,
                scan_id=scan_id,
                account_id=context.account_id,
            )
        return self._invoke_with_account(
            self.sim_broker.submit_entry_order,
            signal=signal,
            quantity=quantity,
            scan_id=scan_id,
            account_id=context.account_id,
        )

    def submit_entry_order_result(self, signal, quantity: int, scan_id: str | None = None, *, market_data_service=None):
        context = self.resolve_execution_context(signal.symbol, signal.asset_type)
        if context.broker_mode == BROKER_MODE_KIS:
            return self._invoke_with_account(
                self.kis_broker.submit_entry_order_result,
                signal=signal,
                quantity=quantity,
                scan_id=scan_id,
                market_data_service=market_data_service,
                account_id=context.account_id,
            )
        return self._invoke_with_account(
            self.sim_broker.submit_entry_order_result,
            signal=signal,
            quantity=quantity,
            scan_id=scan_id,
            market_data_service=market_data_service,
            account_id=context.account_id,
        )

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        context = self.resolve_execution_context(str(position["symbol"]), str(position["asset_type"]))
        account_id = str(position.get("account_id") or context.account_id)
        if context.broker_mode == BROKER_MODE_KIS:
            return self._invoke_with_account(self.kis_broker.submit_exit_order, position=position, reason=reason, account_id=account_id)
        return self._invoke_with_account(self.sim_broker.submit_exit_order, position=position, reason=reason, account_id=account_id)

    def submit_exit_order_result(self, position: pd.Series, reason: str, *, market_data_service=None):
        context = self.resolve_execution_context(str(position["symbol"]), str(position["asset_type"]))
        account_id = str(position.get("account_id") or context.account_id)
        if context.broker_mode == BROKER_MODE_KIS:
            return self._invoke_with_account(
                self.kis_broker.submit_exit_order_result,
                position=position,
                reason=reason,
                market_data_service=market_data_service,
                account_id=account_id,
            )
        return self._invoke_with_account(
            self.sim_broker.submit_exit_order_result,
            position=position,
            reason=reason,
            market_data_service=market_data_service,
            account_id=account_id,
        )

    def preflight_entry(self, signal, quantity: int, *, market_data_service):
        context = self.resolve_execution_context(signal.symbol, signal.asset_type)
        if context.broker_mode == BROKER_MODE_KIS:
            return self._invoke_with_account(self.kis_broker.preflight_entry, signal, quantity, market_data_service, account_id=context.account_id)
        if hasattr(self.sim_broker, "preflight_entry"):
            return self._invoke_with_account(self.sim_broker.preflight_entry, signal, quantity, market_data_service, account_id=context.account_id)
        return {"allowed": True, "reason": "ok", "broker": BROKER_MODE_SIM, "account_id": context.account_id}

    def sync_account(self, touch=None) -> dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        touch("broker_account_sync_start", {"broker": "router"})
        sim_summary = self.sim_broker.sync_account(
            touch=lambda stage=None, details=None: touch(stage or "sim_account_sync", details),
        )
        result = {"sim": sim_summary}
        if self._kis_enabled():
            touch("broker_account_sync_kis", {"broker": BROKER_MODE_KIS})
            result["kis"] = self.kis_broker.sync_account(
                touch=lambda stage=None, details=None: touch(stage or "kis_account_sync", details),
            )
        else:
            result["kis"] = {"broker": BROKER_MODE_KIS, "enabled": False}
        touch(
            "broker_account_sync_complete",
            {
                "sim_accounts": list((result["sim"].get("accounts") or {}).keys()),
                "kis_enabled": bool(result["kis"].get("enabled", False)),
            },
        )
        return result

    def sync_orders(self, market_data_service: Any, touch=None) -> dict[str, int]:
        result = self.sim_broker.sync_orders(market_data_service, touch=touch)
        if self._kis_enabled():
            kis_result = self.kis_broker.sync_orders(market_data_service, touch=touch)
            for key, value in kis_result.items():
                result[key] = int(result.get(key, 0)) + int(value)
        return result

    def process_open_orders(self, market_data_service: Any) -> int:
        return int(self.sync_orders(market_data_service).get("fills", 0))

    def handle_websocket_execution_event(self, event: dict[str, Any]) -> bool:
        if not self._kis_enabled():
            return False
        return bool(self.kis_broker.handle_websocket_execution_event(event))


__all__ = [
    "ACCOUNT_KIS_KR_PAPER",
    "BROKER_MODE_KIS",
    "BROKER_MODE_SIM",
    "BrokerRouter",
    "CRYPTO_ASSET_TYPE",
    "KR_EQUITY_ASSET_TYPE",
    "US_EQUITY_ASSET_TYPE",
    "is_kis_routable_kr_equity",
    "resolve_broker_mode",
]
