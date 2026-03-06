from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

import pandas as pd

from config.settings import RuntimeSettings
from services.market_data_service import MarketDataService
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


@runtime_checkable
class BrokerProtocol(Protocol):
    mode: str
    name: str

    def supports_asset_type(self, asset_type: str) -> bool: ...
    def supports_short(self, asset_type: str) -> bool: ...
    def ensure_account_initialized(self) -> None: ...
    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str: ...
    def submit_exit_order(self, position: pd.Series, reason: str) -> str: ...
    def process_open_orders(self, market_data_service: MarketDataService) -> int: ...
    def snapshot_account(self) -> Dict[str, Any]: ...
    def sync_state(self, force: bool = False) -> Dict[str, Any]: ...
    def reconcile_order(self, order_id: str) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class BrokerStatus:
    mode: str
    name: str
    synced_at: str
    status: str
    message: str


class BrokerRouter:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository, brokers: Dict[str, BrokerProtocol]):
        self.settings = settings
        self.repository = repository
        self.brokers = brokers

    def broker_for_asset_type(self, asset_type: str) -> BrokerProtocol:
        mode = self.settings.broker_mode_for(asset_type)
        broker = self.brokers.get(mode)
        if broker is None:
            raise ValueError(f"unknown broker mode for {asset_type}: {mode}")
        return broker

    def supports_short(self, asset_type: str) -> bool:
        return self.broker_for_asset_type(asset_type).supports_short(asset_type)

    def ensure_account_initialized(self) -> None:
        for broker in self.brokers.values():
            broker.ensure_account_initialized()
        self.snapshot_account()

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str:
        return self.broker_for_asset_type(signal.asset_type).submit_entry_order(signal, quantity, scan_id=scan_id)

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        return self.broker_for_asset_type(str(position["asset_type"])).submit_exit_order(position, reason)

    def process_open_orders(self, market_data_service: MarketDataService) -> int:
        processed = 0
        for broker in self.brokers.values():
            processed += int(broker.process_open_orders(market_data_service))
        self.snapshot_account()
        return processed

    def sync_state(self, force: bool = False) -> Dict[str, Any]:
        status_map: Dict[str, Any] = {}
        for mode, broker in self.brokers.items():
            status_map[mode] = broker.sync_state(force=force)
        self.snapshot_account()
        return status_map

    def reconcile_order(self, order_id: str) -> Dict[str, Any]:
        order = self.repository.get_order(order_id)
        if not order:
            return {"status": "missing", "order_id": order_id}
        broker = self.broker_for_asset_type(str(order["asset_type"]))
        result = broker.reconcile_order(order_id)
        self.snapshot_account()
        return result

    def broker_modes(self) -> Dict[str, str]:
        return dict(self.settings.broker.asset_broker_mode)

    def snapshot_account(self) -> Dict[str, Any]:
        per_broker: Dict[str, Dict[str, Any]] = {}
        cash = 0.0
        equity = 0.0
        gross = 0.0
        net = 0.0
        realized = 0.0
        unrealized = 0.0
        open_positions = 0
        open_orders = len(self.repository.open_orders())
        paused = int(self.repository.get_control_flag_bool("entry_paused") or self.repository.get_control_flag_bool("worker_paused"))
        for mode, broker in self.brokers.items():
            snapshot = broker.snapshot_account()
            if not snapshot:
                continue
            per_broker[mode] = snapshot
            cash += float(snapshot.get("cash", 0.0) or 0.0)
            equity += float(snapshot.get("equity", 0.0) or 0.0)
            gross += abs(float(snapshot.get("gross_exposure", 0.0) or 0.0))
            net += float(snapshot.get("net_exposure", 0.0) or 0.0)
            realized += float(snapshot.get("realized_pnl", 0.0) or 0.0)
            unrealized += float(snapshot.get("unrealized_pnl", 0.0) or 0.0)
            open_positions += int(snapshot.get("open_positions", 0) or 0)

        latest_history = self.repository.load_account_snapshots(limit=2000)
        history = latest_history[latest_history["source"] == "broker_router"] if not latest_history.empty else latest_history
        peak = float(pd.to_numeric(history.get("equity"), errors="coerce").dropna().max()) if not history.empty else equity
        peak = max(peak, equity)
        drawdown_pct = (equity / peak - 1.0) * 100.0 if peak > 0 else 0.0
        account_snapshot = {
            "cash": cash,
            "equity": equity,
            "gross_exposure": gross,
            "net_exposure": net,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "daily_pnl": self.repository.recent_closed_realized_pnl(str(pd.Timestamp.utcnow().date())),
            "drawdown_pct": drawdown_pct,
            "open_positions": open_positions,
            "open_orders": open_orders,
            "paused": paused,
        }
        self.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id=make_id("snap"),
                created_at=utc_now_iso(),
                cash=float(cash),
                equity=float(equity),
                gross_exposure=float(gross),
                net_exposure=float(net),
                realized_pnl=float(realized),
                unrealized_pnl=float(unrealized),
                daily_pnl=float(account_snapshot["daily_pnl"]),
                drawdown_pct=float(drawdown_pct),
                open_positions=int(open_positions),
                open_orders=int(open_orders),
                paused=int(paused),
                source="broker_router",
                raw_json=json_dumps(per_broker),
            )
        )
        return account_snapshot


def json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)
