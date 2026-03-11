from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timedelta
from math import floor
from typing import Any, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import get_kr_strategy, strategy_runtime_metadata
from runtime_accounts import (
    ACCOUNT_KIS_KR_PAPER,
    ACCOUNT_SIM_CRYPTO,
    ACCOUNT_SIM_LEGACY_MIXED,
    ACCOUNT_SIM_US_EQUITY,
    BROKER_MODE_SIM,
    get_account_metadata,
    resolve_execution_account,
)
from services.market_data_service import MarketDataService
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord, FillRecord, OrderRecord, PositionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class PaperBroker:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository):
        self.settings = settings
        self.repository = repository

    def _sim_account_ids(self) -> list[str]:
        accounts = self.repository.load_broker_accounts(active_only=True)
        if accounts.empty:
            return [ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO]
        filtered = accounts.loc[
            (accounts["broker_mode"].astype(str) == BROKER_MODE_SIM)
            & (accounts["account_id"].astype(str) != ACCOUNT_SIM_LEGACY_MIXED)
        ]
        values = filtered["account_id"].astype(str).tolist()
        return values or [ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO]

    def _allocation_weight(self, account_id: str) -> float:
        account = self.repository.get_broker_account(account_id) or {}
        metadata = self.repository._parse_json_payload(account.get("metadata_json") or get_account_metadata(account_id))
        weight = float(pd.to_numeric(pd.Series([metadata.get("bootstrap_cash_allocation")]), errors="coerce").iloc[0] or 0.0)
        return weight if np.isfinite(weight) and weight > 0 else 0.0

    def _bootstrap_account_cash(self, account_id: str) -> float:
        legacy = self.repository.latest_account_snapshot(account_id=ACCOUNT_SIM_LEGACY_MIXED) or {}
        pool_cash = float(legacy.get("cash", self.settings.risk.starting_cash))
        account_ids = self._sim_account_ids()
        weights = {item: self._allocation_weight(item) for item in account_ids}
        total_weight = sum(value for value in weights.values() if value > 0)
        if total_weight > 0 and account_id in weights:
            return float(pool_cash * (weights[account_id] / total_weight))
        divisor = max(len(account_ids), 1)
        return float(pool_cash / divisor)

    def _resolve_account_id(self, *, symbol: str = "", asset_type: str = "", account_id: str | None = None) -> str:
        if account_id:
            return str(account_id)
        return resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=False).account_id

    def ensure_account_initialized(self, account_id: str | None = None) -> None:
        target_ids = [str(account_id)] if account_id else self._sim_account_ids()
        for target_id in target_ids:
            if self.repository.latest_account_snapshot(account_id=target_id) is not None:
                continue
            bootstrap_cash = self._bootstrap_account_cash(target_id)
            self.repository.insert_account_snapshot(
                AccountSnapshotRecord(
                    snapshot_id=make_id("snap"),
                    created_at=utc_now_iso(),
                    cash=bootstrap_cash,
                    equity=bootstrap_cash,
                    gross_exposure=0.0,
                    net_exposure=0.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    daily_pnl=0.0,
                    drawdown_pct=0.0,
                    open_positions=0,
                    open_orders=0,
                    paused=int(self.repository.get_control_flag("trading_paused", "0") == "1"),
                    source="bootstrap",
                    raw_json=json.dumps({"account_id": target_id, "broker": BROKER_MODE_SIM}, ensure_ascii=False),
                    account_id=target_id,
                )
            )

    def _latest_account(self, account_id: str) -> Dict[str, Any]:
        self.ensure_account_initialized(account_id)
        return self.repository.latest_account_snapshot(account_id=account_id) or {}

    def _store_account_snapshot(
        self,
        *,
        account_id: str,
        cash: float,
        equity: float,
        gross_exposure: float,
        net_exposure: float,
        realized_pnl: float,
        unrealized_pnl: float,
        daily_pnl: float,
        open_positions: int,
        open_orders: int,
        source: str,
        raw_json: str = "{}",
        touch=None,
    ) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        payload = self.repository._parse_json_payload(raw_json)
        payload.setdefault("account_id", account_id)
        if str(payload.get("broker") or "").strip() == "":
            payload["broker"] = BROKER_MODE_SIM
        normalized_source = str(source or "").strip()
        peak_kwargs: Dict[str, Any] = {"account_id": account_id}
        if account_id == ACCOUNT_KIS_KR_PAPER:
            if normalized_source == "kis_account_sync":
                peak_kwargs["source"] = "kis_account_sync"
            else:
                peak_kwargs["exclude_sources"] = ("kis_account_sync",)
        peak_source = float(self.repository.max_account_equity(**peak_kwargs))
        if peak_source <= 0.0 and account_id != ACCOUNT_SIM_LEGACY_MIXED:
            peak_source = float(self.repository.max_account_equity(account_id=ACCOUNT_SIM_LEGACY_MIXED))
        peak = max(peak_source, float(equity))
        drawdown_pct = (equity / peak - 1.0) * 100.0 if peak > 0 else 0.0
        record = AccountSnapshotRecord(
            snapshot_id=make_id("snap"),
            created_at=utc_now_iso(),
            cash=float(cash),
            equity=float(equity),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            realized_pnl=float(realized_pnl),
            unrealized_pnl=float(unrealized_pnl),
            daily_pnl=float(daily_pnl),
            drawdown_pct=float(drawdown_pct),
            open_positions=int(open_positions),
            open_orders=int(open_orders),
            paused=int(self.repository.get_control_flag("trading_paused", "0") == "1"),
            source=source,
            raw_json=json.dumps(payload, ensure_ascii=False),
            account_id=account_id,
        )
        self.repository.insert_account_snapshot(record)
        touch(
            "account_snapshot_stored",
            {
                "account_id": account_id,
                "source": source,
                "equity": float(equity),
                "cash": float(cash),
                "gross_exposure": float(gross_exposure),
                "net_exposure": float(net_exposure),
            },
        )
        return {
            "snapshot_id": record.snapshot_id,
            "created_at": record.created_at,
            "cash": record.cash,
            "equity": record.equity,
            "gross_exposure": record.gross_exposure,
            "net_exposure": record.net_exposure,
            "realized_pnl": record.realized_pnl,
            "unrealized_pnl": record.unrealized_pnl,
            "daily_pnl": record.daily_pnl,
            "drawdown_pct": record.drawdown_pct,
            "open_positions": record.open_positions,
            "open_orders": record.open_orders,
            "source": record.source,
            "account_id": record.account_id,
        }

    def record_external_account_snapshot(
        self,
        *,
        account_id: str = "",
        cash: float,
        equity: float,
        gross_exposure: float,
        net_exposure: float,
        unrealized_pnl: float,
        open_positions: int,
        source: str,
        raw_json: str = "{}",
        touch=None,
    ) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        resolved_account_id = str(account_id or (ACCOUNT_KIS_KR_PAPER if str(source) == "kis_account_sync" else ACCOUNT_SIM_US_EQUITY))
        touch("external_account_snapshot_start", {"source": source, "account_id": resolved_account_id})
        realized = self.repository.total_realized_pnl(account_id=resolved_account_id)
        today = str(pd.Timestamp.utcnow().date())
        daily_pnl = self.repository.recent_closed_realized_pnl(today, account_id=resolved_account_id)
        open_order_count = int(len(self.repository.open_orders(account_id=resolved_account_id)))
        return self._store_account_snapshot(
            account_id=resolved_account_id,
            cash=float(cash),
            equity=float(equity),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            realized_pnl=float(realized),
            unrealized_pnl=float(unrealized_pnl),
            daily_pnl=float(daily_pnl),
            open_positions=int(open_positions),
            open_orders=open_order_count,
            source=source,
            raw_json=raw_json,
            touch=touch,
        )

    def _position_side_from_order(self, order_side: str) -> str:
        return "LONG" if order_side == "buy" else "SHORT"

    def _signed_market_value(self, side: str, quantity: int, price: float) -> float:
        notional = float(quantity) * float(price)
        return notional if side == "LONG" else -notional

    def _compute_unrealized(self, side: str, entry_price: float, mark_price: float, quantity: int) -> float:
        if side == "LONG":
            return (mark_price - entry_price) * quantity
        return (entry_price - mark_price) * quantity

    def _timeframe_delta(self, timeframe: str) -> timedelta:
        if timeframe == "15m":
            return timedelta(minutes=15)
        if timeframe == "1h":
            return timedelta(hours=1)
        return timedelta(days=1)

    def _strategy_config(self, strategy_version: str = ""):
        return get_kr_strategy(self.settings, str(strategy_version or ""))

    def _max_holding_bars(self, strategy_version: str = "") -> int:
        strategy_cfg = self._strategy_config(strategy_version)
        return max(int(strategy_cfg.max_holding_bars if strategy_cfg is not None else self.settings.strategy.max_holding_bars), 0)

    def _trailing_stop_mult(self, strategy_version: str = "") -> float:
        strategy_cfg = self._strategy_config(strategy_version)
        return float(strategy_cfg.trailing_stop_atr_mult if strategy_cfg is not None else self.settings.strategy.trailing_stop_atr_mult)

    def _score_decay_threshold(self, strategy_version: str = "") -> float:
        strategy_cfg = self._strategy_config(strategy_version)
        return float(strategy_cfg.score_decay_exit_threshold if strategy_cfg is not None else self.settings.strategy.score_decay_exit_threshold)

    def _stop_loss_mult(self, strategy_version: str = "") -> float:
        strategy_cfg = self._strategy_config(strategy_version)
        return float(strategy_cfg.stop_loss_atr_mult if strategy_cfg is not None else self.settings.strategy.stop_loss_atr_mult)

    def _take_profit_mult(self, strategy_version: str = "") -> float:
        strategy_cfg = self._strategy_config(strategy_version)
        return float(strategy_cfg.take_profit_atr_mult if strategy_cfg is not None else self.settings.strategy.take_profit_atr_mult)

    def _atr_value_from_order_meta(self, order_meta: Dict[str, Any], entry_price: float) -> float:
        atr_value = float(pd.to_numeric(pd.Series([order_meta.get("atr_14")]), errors="coerce").iloc[0])
        if np.isfinite(atr_value) and atr_value > 0:
            return atr_value
        expected_risk = float(pd.to_numeric(pd.Series([order_meta.get("expected_risk")]), errors="coerce").iloc[0])
        if np.isfinite(expected_risk) and expected_risk > 0 and np.isfinite(entry_price) and entry_price > 0:
            return float(entry_price) * float(expected_risk)
        return float("nan")

    def _recalculate_exit_levels(
        self,
        *,
        entry_price: float,
        side: str,
        order_meta: Dict[str, Any],
        strategy_version: str = "",
        fallback_stop: float = float("nan"),
        fallback_take: float = float("nan"),
        fallback_trailing: float = float("nan"),
    ) -> Dict[str, float]:
        def _valid_exit(value: float, *, is_stop: bool) -> float:
            if not np.isfinite(value):
                return float("nan")
            if side == "LONG":
                return float(value) if (value < entry_price if is_stop else value > entry_price) else float("nan")
            return float(value) if (value > entry_price if is_stop else value < entry_price) else float("nan")

        atr_value = self._atr_value_from_order_meta(order_meta, entry_price)
        if np.isfinite(atr_value) and atr_value > 0:
            stop_mult = max(self._stop_loss_mult(strategy_version), 0.0)
            take_mult = max(self._take_profit_mult(strategy_version), 0.0)
            if side == "LONG":
                stop_loss = entry_price - atr_value * stop_mult if stop_mult > 0 else float("nan")
                take_profit = entry_price + atr_value * take_mult if take_mult > 0 else float("nan")
            else:
                stop_loss = entry_price + atr_value * stop_mult if stop_mult > 0 else float("nan")
                take_profit = entry_price - atr_value * take_mult if take_mult > 0 else float("nan")
            return {
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "trailing_stop": float(stop_loss),
            }
        stop_loss = _valid_exit(float(fallback_stop), is_stop=True)
        take_profit = _valid_exit(float(fallback_take), is_stop=False)
        trailing_stop = _valid_exit(float(fallback_trailing), is_stop=True)
        if not np.isfinite(trailing_stop):
            trailing_stop = stop_loss
        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "trailing_stop": float(trailing_stop),
        }

    def _compute_cooldown_until(self, asset_type: str, timeframe: str, now_iso: str, *, strategy_version: str = "") -> str:
        strategy_cfg = self._strategy_config(strategy_version)
        bars = max(int(strategy_cfg.cooldown_bars_after_exit if strategy_cfg is not None else self.settings.risk.cooldown_bars_after_exit), 0)
        current = pd.Timestamp(now_iso)
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        if bars <= 0:
            return current.isoformat()
        if timeframe == "1h":
            return (current + pd.Timedelta(hours=bars)).isoformat()

        schedule = self.settings.asset_schedules.get(asset_type)
        if schedule is None:
            return (current + pd.Timedelta(days=bars)).isoformat()

        local = current.tz_convert(ZoneInfo(schedule.timezone))
        cursor = local
        remaining = bars
        while remaining > 0:
            cursor = cursor + pd.Timedelta(days=1)
            if cursor.weekday() >= 5:
                continue
            remaining -= 1
        return cursor.tz_convert("UTC").isoformat()

    def _execution_metadata(self, strategy_version: str) -> Dict[str, str]:
        return strategy_runtime_metadata(self.settings, str(strategy_version or ""))

    def _position_metadata(self, source: Dict[str, Any]) -> Dict[str, str]:
        return {
            "strategy_family": str(source.get("strategy_family") or ""),
            "session_mode": str(source.get("session_mode") or ""),
            "price_policy": str(source.get("price_policy") or ""),
        }

    def preflight_entry(
        self,
        signal: SignalDecision,
        quantity: int,
        market_data_service: MarketDataService | None = None,
        *,
        account_id: str | None = None,
    ) -> Dict[str, Any]:
        resolved_account_id = self._resolve_account_id(symbol=signal.symbol, asset_type=signal.asset_type, account_id=account_id)
        account = self._latest_account(resolved_account_id)
        return {
            "allowed": True,
            "reason": "ok",
            "broker": BROKER_MODE_SIM,
            "account_id": resolved_account_id,
            "cash": float(account.get("cash", self._bootstrap_account_cash(resolved_account_id))),
            "requested_qty": int(quantity),
        }

    def submit_entry_order(
        self,
        signal: SignalDecision,
        quantity: int,
        scan_id: str | None = None,
        *,
        account_id: str | None = None,
    ) -> str:
        now_iso = utc_now_iso()
        order_id = make_id("ord")
        side = "buy" if signal.signal == "LONG" else "sell"
        resolved_account_id = self._resolve_account_id(symbol=signal.symbol, asset_type=signal.asset_type, account_id=account_id)
        max_holding_until = (
            pd.Timestamp.now(tz="UTC") + self._timeframe_delta(signal.timeframe) * int(self._max_holding_bars(signal.strategy_version))
        ).isoformat()
        record = OrderRecord(
            order_id=order_id,
            created_at=now_iso,
            updated_at=now_iso,
            prediction_id=signal.prediction_id,
            scan_id=scan_id or signal.scan_id,
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            timeframe=signal.timeframe,
            side=side,
            order_type=self.settings.broker.default_order_type,
            requested_qty=int(quantity),
            filled_qty=0,
            remaining_qty=int(quantity),
            requested_price=signal.current_price,
            limit_price=np.nan,
            status="new",
            fees_estimate=signal.current_price * quantity * self.settings.broker.fee_bps / 10000.0,
            slippage_bps=self.settings.broker.base_slippage_bps,
            retry_count=0,
            strategy_version=signal.strategy_version,
            reason="entry",
            raw_json=json.dumps(
                {
                    "broker": BROKER_MODE_SIM,
                    "account_id": resolved_account_id,
                    "prediction_id": signal.prediction_id,
                    "stop_level": signal.stop_level,
                    "take_level": signal.take_level,
                    "atr_14": signal.atr_value,
                    "expected_risk": signal.expected_risk,
                    "max_holding_until": max_holding_until,
                    "strategy_version": signal.strategy_version,
                    **self._execution_metadata(signal.strategy_version),
                },
                ensure_ascii=False,
            ),
            account_id=resolved_account_id,
        )
        self.repository.insert_order(record)
        return order_id

    def submit_entry_order_result(
        self,
        signal: SignalDecision,
        quantity: int,
        scan_id: str | None = None,
        *,
        account_id: str | None = None,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        resolved_account_id = self._resolve_account_id(symbol=signal.symbol, asset_type=signal.asset_type, account_id=account_id)
        order_id = self.submit_entry_order(signal, quantity, scan_id, account_id=resolved_account_id)
        self.repository.log_event(
            "INFO",
            "execution_pipeline",
            "submit_requested",
            "sim entry submit requested",
            {"order_id": order_id, "symbol": signal.symbol, "account_id": resolved_account_id, "strategy_version": signal.strategy_version},
            account_id=resolved_account_id,
        )
        self.repository.log_event(
            "INFO",
            "execution_pipeline",
            "submitted",
            "sim entry submitted",
            {"order_id": order_id, "symbol": signal.symbol, "account_id": resolved_account_id, "strategy_version": signal.strategy_version},
            account_id=resolved_account_id,
        )
        return {
            "submitted": True,
            "status": "new",
            "reason": "ok",
            "broker": BROKER_MODE_SIM,
            "order_id": order_id,
            "account_id": resolved_account_id,
        }

    def submit_exit_order(self, position: pd.Series, reason: str, *, account_id: str | None = None) -> str:
        now_iso = utc_now_iso()
        order_id = make_id("ord")
        side = "sell" if str(position["side"]) == "LONG" else "buy"
        resolved_account_id = self._resolve_account_id(
            symbol=str(position["symbol"]),
            asset_type=str(position["asset_type"]),
            account_id=account_id or str(position.get("account_id") or ""),
        )
        record = OrderRecord(
            order_id=order_id,
            created_at=now_iso,
            updated_at=now_iso,
            prediction_id=str(position.get("prediction_id") or ""),
            scan_id=None,
            symbol=str(position["symbol"]),
            asset_type=str(position["asset_type"]),
            timeframe=str(position["timeframe"]),
            side=side,
            order_type=self.settings.broker.default_order_type,
            requested_qty=int(position["quantity"]),
            filled_qty=0,
            remaining_qty=int(position["quantity"]),
            requested_price=float(position["mark_price"]),
            limit_price=np.nan,
            status="new",
            fees_estimate=float(position["mark_price"]) * int(position["quantity"]) * self.settings.broker.fee_bps / 10000.0,
            slippage_bps=self.settings.broker.base_slippage_bps,
            retry_count=0,
            strategy_version=str(position["strategy_version"]),
            reason=reason,
            raw_json=json.dumps(
                {
                    "broker": BROKER_MODE_SIM,
                    "account_id": resolved_account_id,
                    "position_id": str(position["position_id"]),
                    "strategy_version": str(position["strategy_version"]),
                    **self._position_metadata(position),
                },
                ensure_ascii=False,
            ),
            account_id=resolved_account_id,
        )
        self.repository.insert_order(record)
        return order_id

    def submit_exit_order_result(
        self,
        position: pd.Series,
        reason: str,
        *,
        account_id: str | None = None,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        resolved_account_id = self._resolve_account_id(
            symbol=str(position["symbol"]),
            asset_type=str(position["asset_type"]),
            account_id=account_id or str(position.get("account_id") or ""),
        )
        order_id = self.submit_exit_order(position, reason, account_id=resolved_account_id)
        self.repository.log_event(
            "INFO",
            "execution_pipeline",
            "submit_requested",
            "sim exit submit requested",
            {"order_id": order_id, "symbol": str(position["symbol"]), "account_id": resolved_account_id},
            account_id=resolved_account_id,
        )
        self.repository.log_event(
            "INFO",
            "execution_pipeline",
            "submitted",
            "sim exit submitted",
            {"order_id": order_id, "symbol": str(position["symbol"]), "account_id": resolved_account_id},
            account_id=resolved_account_id,
        )
        return {
            "submitted": True,
            "status": "new",
            "reason": "ok",
            "broker": BROKER_MODE_SIM,
            "order_id": order_id,
            "account_id": resolved_account_id,
        }

    def _slippage_price(self, side: str, price: float, volatility: float) -> float:
        bps = self.settings.broker.base_slippage_bps + max(volatility, 0.0) * 10000.0 * self.settings.broker.volatility_slippage_mult
        if side == "buy":
            return price * (1.0 + bps / 10000.0)
        return price * (1.0 - bps / 10000.0)

    def _max_fill_qty(self, requested_qty: int, volume: float) -> int:
        if not np.isfinite(volume) or volume <= 0:
            return requested_qty
        max_qty = max(1, int(floor(volume * self.settings.broker.max_volume_participation)))
        return min(requested_qty, max_qty)

    def _update_position_from_fill(self, order: pd.Series, fill_qty: int, fill_price: float, fees: float) -> None:
        account_id = str(order.get("account_id") or self._resolve_account_id(symbol=str(order["symbol"]), asset_type=str(order["asset_type"])))
        latest = self.repository.latest_position_by_symbol(
            symbol=str(order["symbol"]),
            timeframe=str(order["timeframe"]),
            account_id=account_id,
        )
        now_iso = utc_now_iso()
        side = self._position_side_from_order(str(order["side"]))
        order_meta = json.loads(str(order.get("raw_json") or "{}"))
        strategy_version = str(order_meta.get("strategy_version") or order["strategy_version"])
        metadata = self._execution_metadata(strategy_version)
        if latest.empty or str(latest.iloc[0]["status"]) != "open":
            exit_levels = self._recalculate_exit_levels(
                entry_price=float(fill_price),
                side=side,
                order_meta=order_meta,
                strategy_version=strategy_version,
                fallback_stop=float(order_meta.get("stop_level", np.nan)),
                fallback_take=float(order_meta.get("take_level", np.nan)),
                fallback_trailing=float(order_meta.get("stop_level", np.nan)),
            )
            self.repository.upsert_position(
                PositionRecord(
                    position_id=make_id("pos"),
                    created_at=now_iso,
                    updated_at=now_iso,
                    closed_at=None,
                    prediction_id=str(order.get("prediction_id") or ""),
                    symbol=str(order["symbol"]),
                    asset_type=str(order["asset_type"]),
                    timeframe=str(order["timeframe"]),
                    side=side,
                    status="open",
                    quantity=int(fill_qty),
                    entry_price=float(fill_price),
                    mark_price=float(fill_price),
                    stop_loss=float(exit_levels["stop_loss"]),
                    take_profit=float(exit_levels["take_profit"]),
                    trailing_stop=float(exit_levels["trailing_stop"]),
                    highest_price=float(fill_price),
                    lowest_price=float(fill_price),
                    unrealized_pnl=-fees,
                    realized_pnl=0.0,
                    expected_risk=float(order_meta.get("expected_risk", np.nan)),
                    exposure_value=self._signed_market_value(side, int(fill_qty), float(fill_price)),
                    max_holding_until=str(order_meta.get("max_holding_until") or now_iso),
                    strategy_version=strategy_version,
                    cooldown_until=None,
                    notes="opened_by_fill",
                    account_id=account_id,
                    strategy_family=str(order_meta.get("strategy_family") or metadata["strategy_family"]),
                    session_mode=str(order_meta.get("session_mode") or metadata["session_mode"]),
                    price_policy=str(order_meta.get("price_policy") or metadata["price_policy"]),
                )
            )
            return

        position = latest.iloc[0]
        current_qty = int(position["quantity"])
        entry_price = float(position["entry_price"])
        position_side = str(position["side"])
        same_direction = (position_side == "LONG" and str(order["side"]) == "buy") or (
            position_side == "SHORT" and str(order["side"]) == "sell"
        )
        if same_direction:
            new_qty = current_qty + fill_qty
            avg_price = ((entry_price * current_qty) + (fill_price * fill_qty)) / max(new_qty, 1)
            exit_levels = self._recalculate_exit_levels(
                entry_price=float(avg_price),
                side=position_side,
                order_meta=order_meta,
                strategy_version=strategy_version,
                fallback_stop=float(position["stop_loss"]),
                fallback_take=float(position["take_profit"]),
                fallback_trailing=float(position["trailing_stop"]),
            )
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position.to_dict(),
                        "updated_at": now_iso,
                        "quantity": int(new_qty),
                        "entry_price": float(avg_price),
                        "mark_price": float(fill_price),
                        "stop_loss": float(exit_levels["stop_loss"]),
                        "take_profit": float(exit_levels["take_profit"]),
                        "trailing_stop": float(exit_levels["trailing_stop"]),
                        "highest_price": max(float(position["highest_price"]), float(fill_price)),
                        "lowest_price": min(float(position["lowest_price"]), float(fill_price)),
                        "unrealized_pnl": self._compute_unrealized(position_side, float(avg_price), float(fill_price), int(new_qty)),
                        "exposure_value": self._signed_market_value(position_side, int(new_qty), float(fill_price)),
                        "notes": "scaled_position",
                        "account_id": account_id,
                        "strategy_family": str(position.get("strategy_family") or order_meta.get("strategy_family") or metadata["strategy_family"]),
                        "session_mode": str(position.get("session_mode") or order_meta.get("session_mode") or metadata["session_mode"]),
                        "price_policy": str(position.get("price_policy") or order_meta.get("price_policy") or metadata["price_policy"]),
                    }
                )
            )
            return

        closing_qty = min(current_qty, fill_qty)
        if position_side == "LONG":
            realized = (fill_price - entry_price) * closing_qty - fees
        else:
            realized = (entry_price - fill_price) * closing_qty - fees
        remaining = current_qty - closing_qty
        if remaining <= 0:
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position.to_dict(),
                        "updated_at": now_iso,
                        "closed_at": now_iso,
                        "status": "closed",
                        "quantity": 0,
                        "mark_price": float(fill_price),
                        "unrealized_pnl": 0.0,
                        "realized_pnl": float(position["realized_pnl"]) + realized,
                        "exposure_value": 0.0,
                        "cooldown_until": self._compute_cooldown_until(
                            asset_type=str(position["asset_type"]),
                            timeframe=str(position["timeframe"]),
                            now_iso=now_iso,
                            strategy_version=str(position.get("strategy_version") or ""),
                        ),
                        "notes": f"closed_by_{order['reason']}",
                        "account_id": account_id,
                        "strategy_family": str(position.get("strategy_family") or order_meta.get("strategy_family") or metadata["strategy_family"]),
                        "session_mode": str(position.get("session_mode") or order_meta.get("session_mode") or metadata["session_mode"]),
                        "price_policy": str(position.get("price_policy") or order_meta.get("price_policy") or metadata["price_policy"]),
                    }
                )
            )
        else:
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position.to_dict(),
                        "updated_at": now_iso,
                        "quantity": int(remaining),
                        "mark_price": float(fill_price),
                        "unrealized_pnl": self._compute_unrealized(position_side, entry_price, float(fill_price), int(remaining)),
                        "realized_pnl": float(position["realized_pnl"]) + realized,
                        "exposure_value": self._signed_market_value(position_side, int(remaining), float(fill_price)),
                        "notes": "partial_close",
                        "account_id": account_id,
                        "strategy_family": str(position.get("strategy_family") or order_meta.get("strategy_family") or metadata["strategy_family"]),
                        "session_mode": str(position.get("session_mode") or order_meta.get("session_mode") or metadata["session_mode"]),
                        "price_policy": str(position.get("price_policy") or order_meta.get("price_policy") or metadata["price_policy"]),
                    }
                )
            )

    def apply_external_fill(
        self,
        order_id: str,
        *,
        fill_qty: int,
        fill_price: float,
        fees: float | None = None,
        raw_json: Dict[str, Any] | None = None,
        final_status: str | None = None,
    ) -> bool:
        order_record = self.repository.get_order(order_id)
        if not order_record:
            return False
        order = pd.Series(order_record)
        account_id = str(order.get("account_id") or self._resolve_account_id(symbol=str(order["symbol"]), asset_type=str(order["asset_type"])))
        remaining_qty = int(order["remaining_qty"])
        applied_qty = min(max(int(fill_qty), 0), remaining_qty)
        if applied_qty <= 0:
            return False
        effective_fees = float(fees if fees is not None else applied_qty * fill_price * self.settings.broker.fee_bps / 10000.0)
        self.repository.insert_fill(
            FillRecord(
                fill_id=make_id("fill"),
                created_at=utc_now_iso(),
                order_id=str(order["order_id"]),
                symbol=str(order["symbol"]),
                side=str(order["side"]),
                quantity=int(applied_qty),
                fill_price=float(fill_price),
                fees=float(effective_fees),
                slippage_bps=float(order["slippage_bps"]),
                status="filled",
                raw_json=json.dumps(
                    {
                        **self._execution_metadata(str(order.get("strategy_version") or "")),
                        **(raw_json or {}),
                    },
                    default=str,
                    ensure_ascii=False,
                ),
                account_id=account_id,
            )
        )
        new_remaining = remaining_qty - applied_qty
        new_filled = int(order["filled_qty"]) + applied_qty
        self.repository.update_order(
            str(order["order_id"]),
            status=final_status or ("filled" if new_remaining <= 0 else "partially_filled"),
            filled_qty=new_filled,
            remaining_qty=new_remaining,
            raw_json={"last_fill_price": float(fill_price), **(raw_json or {})},
        )
        account = self._latest_account(account_id)
        cash_value = float(account.get("cash", self._bootstrap_account_cash(account_id)))
        if str(order["side"]) == "buy":
            cash_value -= applied_qty * fill_price + effective_fees
        else:
            cash_value += applied_qty * fill_price - effective_fees
        self._update_position_from_fill(order=order, fill_qty=applied_qty, fill_price=float(fill_price), fees=float(effective_fees))
        if account_id != ACCOUNT_KIS_KR_PAPER:
            self.snapshot_account(cash_override=cash_value, account_id=account_id)
        return True

    def process_open_orders(self, market_data_service: MarketDataService, touch: Any | None = None) -> int:
        orders = self.repository.open_orders(statuses=("new", "partially_filled"))
        if not orders.empty and "raw_json" in orders.columns:
            broker_name = orders["raw_json"].fillna("{}").astype(str).map(
                lambda payload: str(json.loads(payload).get("broker", "sim")) if payload else "sim"
            )
            orders = orders.loc[broker_name.isin({"", "sim"})].copy()

        filled_count = 0
        cash_overrides: Dict[str, float] = {}
        touched_accounts: set[str] = set()
        for _, order in orders.iterrows():
            account_id = str(order.get("account_id") or self._resolve_account_id(symbol=str(order["symbol"]), asset_type=str(order["asset_type"])))
            touched_accounts.add(account_id)
            if callable(touch):
                touch("sim_order_sync", {"order_id": str(order["order_id"]), "symbol": str(order["symbol"]), "account_id": account_id})
            asset_type = str(order["asset_type"])
            if not market_data_service.is_market_open(asset_type):
                continue
            try:
                quote = market_data_service.latest_quote(
                    symbol=str(order["symbol"]),
                    asset_type=asset_type,
                    timeframe=str(order["timeframe"]),
                )
            except Exception as exc:
                self.repository.update_order(str(order["order_id"]), status=str(order["status"]), error_message=str(exc))
                continue
            requested_qty = int(order["remaining_qty"])
            fill_qty = self._max_fill_qty(requested_qty=requested_qty, volume=quote.volume)
            if not self.settings.broker.allow_partial_fills and fill_qty < requested_qty:
                continue
            if fill_qty <= 0:
                continue
            fill_price = self._slippage_price(str(order["side"]), float(quote.price), float(0.0))
            fees = fill_qty * fill_price * self.settings.broker.fee_bps / 10000.0
            self.repository.insert_fill(
                FillRecord(
                    fill_id=make_id("fill"),
                    created_at=utc_now_iso(),
                    order_id=str(order["order_id"]),
                    symbol=str(order["symbol"]),
                    side=str(order["side"]),
                    quantity=int(fill_qty),
                    fill_price=float(fill_price),
                    fees=float(fees),
                    slippage_bps=float(order["slippage_bps"]),
                    status="filled",
                    raw_json=json.dumps(
                        {
                            **self._execution_metadata(str(order.get("strategy_version") or "")),
                            **asdict(quote),
                        },
                        default=str,
                        ensure_ascii=False,
                    ),
                    account_id=account_id,
                )
            )
            remaining_qty = requested_qty - fill_qty
            self.repository.update_order(
                str(order["order_id"]),
                status="filled" if remaining_qty <= 0 else "partially_filled",
                filled_qty=int(order["filled_qty"]) + fill_qty,
                remaining_qty=remaining_qty,
                raw_json={"last_fill_price": fill_price},
            )
            current_cash = cash_overrides.get(account_id)
            if current_cash is None:
                current_cash = float(self._latest_account(account_id).get("cash", self._bootstrap_account_cash(account_id)))
            if str(order["side"]) == "buy":
                current_cash -= fill_qty * fill_price + fees
            else:
                current_cash += fill_qty * fill_price - fees
            cash_overrides[account_id] = current_cash
            self._update_position_from_fill(order=order, fill_qty=fill_qty, fill_price=fill_price, fees=fees)
            filled_count += 1
            self.repository.log_event(
                "INFO",
                "execution_pipeline",
                "filled" if remaining_qty <= 0 else "partially_filled",
                "sim order filled",
                {"order_id": str(order["order_id"]), "symbol": str(order["symbol"]), "fill_qty": int(fill_qty), "account_id": account_id, "strategy_version": str(order.get("strategy_version") or "")},
                account_id=account_id,
            )
        for account_id in touched_accounts:
            self.snapshot_account(cash_override=cash_overrides.get(account_id), account_id=account_id)
        return filled_count

    def sync_orders(self, market_data_service: MarketDataService, touch: Any | None = None) -> Dict[str, int]:
        return {"fills": int(self.process_open_orders(market_data_service, touch=touch))}

    def _snapshot_single_account(self, account_id: str, cash_override: float | None = None, touch=None) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        touch("sim_account_snapshot_start", {"source": "paper_broker", "account_id": account_id})
        account = self._latest_account(account_id)
        open_positions = self.repository.open_positions(account_id=account_id)
        open_orders = self.repository.open_orders(account_id=account_id)
        cash = float(cash_override if cash_override is not None else account.get("cash", self._bootstrap_account_cash(account_id)))
        realized = self.repository.total_realized_pnl(account_id=account_id)
        unrealized = (
            float(pd.to_numeric(open_positions.get("unrealized_pnl"), errors="coerce").fillna(0.0).sum())
            if not open_positions.empty
            else 0.0
        )
        exposures = (
            pd.to_numeric(open_positions.get("exposure_value"), errors="coerce").fillna(0.0)
            if not open_positions.empty
            else pd.Series(dtype=float)
        )
        gross_exposure = float(exposures.abs().sum()) if not exposures.empty else 0.0
        net_exposure = float(exposures.sum()) if not exposures.empty else 0.0
        equity = cash + net_exposure
        today = str(pd.Timestamp.utcnow().date())
        daily_pnl = self.repository.recent_closed_realized_pnl(today, account_id=account_id)
        touch(
            "sim_account_snapshot_metrics",
            {
                "account_id": account_id,
                "equity": float(equity),
                "cash": float(cash),
                "gross_exposure": float(gross_exposure),
                "net_exposure": float(net_exposure),
                "open_positions": int(len(open_positions)),
                "open_orders": int(len(open_orders)),
            },
        )
        return self._store_account_snapshot(
            account_id=account_id,
            cash=float(cash),
            equity=float(equity),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            realized_pnl=float(realized),
            unrealized_pnl=float(unrealized),
            daily_pnl=float(daily_pnl),
            open_positions=int(len(open_positions)),
            open_orders=int(len(open_orders)),
            source="paper_broker",
            raw_json=json.dumps({"account_id": account_id, "broker": BROKER_MODE_SIM}, ensure_ascii=False),
            touch=touch,
        )

    def snapshot_account(
        self,
        cash_override: float | Dict[str, float] | None = None,
        *,
        account_id: str | None = None,
        touch=None,
    ) -> Dict[str, Any]:
        if account_id:
            override = float(cash_override) if isinstance(cash_override, (int, float)) else None
            return self._snapshot_single_account(account_id, cash_override=override, touch=touch)
        if isinstance(cash_override, (int, float)):
            candidate_ids = []
            open_positions = self.repository.open_positions()
            if not open_positions.empty and "account_id" in open_positions.columns:
                candidate_ids.extend([str(value) for value in open_positions["account_id"].dropna().astype(str).unique().tolist() if str(value).strip()])
            open_orders = self.repository.open_orders()
            if not open_orders.empty and "account_id" in open_orders.columns:
                candidate_ids.extend([str(value) for value in open_orders["account_id"].dropna().astype(str).unique().tolist() if str(value).strip()])
            unique_ids = list(dict.fromkeys(candidate_ids))
            if len(unique_ids) == 1:
                return self._snapshot_single_account(unique_ids[0], cash_override=float(cash_override), touch=touch)
        snapshots: Dict[str, Any] = {}
        cash_map = cash_override if isinstance(cash_override, dict) else {}
        for target_id in self._sim_account_ids():
            override = cash_map.get(target_id) if isinstance(cash_map, dict) else None
            snapshots[target_id] = self._snapshot_single_account(target_id, cash_override=override, touch=touch)
        return snapshots

    def sync_account(self, touch=None) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        touch("sim_account_sync_start", {"broker": "sim"})
        snapshots = self.snapshot_account(touch=touch)
        total_cash = sum(float((summary or {}).get("cash", 0.0) or 0.0) for summary in snapshots.values())
        touch("sim_account_sync_complete", {"accounts": list(snapshots.keys())})
        return {
            "broker": "sim",
            "enabled": True,
            "accounts": snapshots,
            "cash": float(total_cash),
        }
