from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings, load_settings
from kis_paper import KISPaperClient
from prediction_memory import attach_order_to_prediction
from services.broker_router import BrokerRouter
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from storage.repository import TradingRepository


@dataclass(frozen=True)
class ManualKISRuntime:
    settings: RuntimeSettings
    repository: TradingRepository
    router: BrokerRouter


def _open_repository(settings: RuntimeSettings) -> TradingRepository:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    return repository


def _build_router(settings: RuntimeSettings, repository: TradingRepository) -> BrokerRouter:
    sim_broker = PaperBroker(settings, repository)
    kis_broker = KISPaperBroker(settings, repository, sim_broker)
    router = BrokerRouter(sim_broker=sim_broker, kis_broker=kis_broker)
    router.ensure_account_initialized()
    return router


def _build_manual_runtime(settings_path: str | None = None) -> ManualKISRuntime:
    settings = load_settings(settings_path)
    repository = _open_repository(settings)
    return ManualKISRuntime(
        settings=settings,
        repository=repository,
        router=_build_router(settings, repository),
    )


def _load_client(settings_path: str | None = None) -> KISPaperClient:
    _ = load_settings(settings_path)
    return KISPaperClient()


def _attach_prediction_order_link(prediction_id: str | None, order_id: str) -> None:
    if prediction_id:
        attach_order_to_prediction(prediction_id=prediction_id, order_id=order_id)


def load_kis_config(settings_path: str | None = None):
    return _load_client(settings_path).config


def load_kis_account_snapshot(settings_path: str | None = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    snapshot = _load_client(settings_path).get_account_snapshot()
    return dict(snapshot.summary), snapshot.holdings.copy()


def load_kis_quote(symbol: str, settings_path: str | None = None) -> Dict[str, Any]:
    return _load_client(settings_path).get_quote(symbol)


def submit_manual_kis_order(
    *,
    symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    requested_price: float,
    prediction_id: str | None = None,
    settings_path: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    runtime = _build_manual_runtime(settings_path)
    order_id = runtime.router.submit_manual_kis_order(
        symbol=symbol,
        asset_type="한국주식",
        timeframe="1d",
        side=side,
        quantity=quantity,
        order_type=order_type,
        requested_price=requested_price,
        prediction_id=prediction_id,
        strategy_version="manual_kis",
        reason="manual_entry" if side == "buy" else "manual_exit",
        raw_metadata=metadata or {},
    )
    order = runtime.repository.get_order(order_id) or {}
    _attach_prediction_order_link(prediction_id, str(order_id))
    return {
        "order_id": order_id,
        "broker_order_id": str(order.get("broker_order_id") or ""),
        "status": str(order.get("status") or ""),
        "message": str(order.get("error_message") or ""),
    }


def load_manual_order_history(limit: int = 200, settings_path: str | None = None) -> pd.DataFrame:
    repository = _open_repository(load_settings(settings_path))
    frame = repository.recent_orders(limit=limit)
    if frame.empty:
        return frame
    payloads = []
    for payload in frame["raw_json"].fillna("{}").astype(str).tolist():
        try:
            payloads.append(json.loads(payload))
        except Exception:
            payloads.append({})
    expanded = pd.DataFrame(payloads)
    if not expanded.empty:
        for column in expanded.columns:
            if column not in frame.columns:
                frame[column] = expanded[column]
    broker_mask = frame["raw_json"].fillna("{}").astype(str).map(lambda payload: str(json.loads(payload).get("broker", "")) if payload else "")
    frame = frame.loc[broker_mask == "kis_mock"].copy()
    return frame.reset_index(drop=True)


def load_manual_equity_curve(limit: int = 500, settings_path: str | None = None) -> pd.DataFrame:
    repository = _open_repository(load_settings(settings_path))
    frame = repository.load_account_snapshots(limit=limit)
    if frame.empty:
        return frame
    return frame.sort_values("created_at").reset_index(drop=True)


def compute_manual_equity_metrics(settings_path: str | None = None) -> Dict[str, float]:
    repository = _open_repository(load_settings(settings_path))
    trade_metrics = repository.trade_performance_report()
    curve = repository.load_account_snapshots(limit=2000)
    latest_equity = float(curve.sort_values("created_at").iloc[-1]["equity"]) if not curve.empty else float("nan")
    return {
        "samples": float(len(curve)),
        "latest_equity": latest_equity,
        "total_return_pct": float(trade_metrics.get("total_return_pct", np.nan)),
        "max_drawdown_pct": float(trade_metrics.get("max_drawdown_pct", np.nan)),
        "today_pnl": float(trade_metrics.get("today_pnl", 0.0)),
        "sharpe": np.nan,
        "sortino": np.nan,
        "calmar": np.nan,
        "win_rate_pct": np.nan,
        "profit_factor": np.nan,
        "exposure_pct": np.nan,
        "max_consecutive_losses": np.nan,
    }
