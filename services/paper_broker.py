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
from services.market_data_service import MarketDataService
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord, FillRecord, OrderRecord, PositionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class PaperBroker:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository):
        self.settings = settings
        self.repository = repository

    def ensure_account_initialized(self) -> None:
        if self.repository.latest_account_snapshot() is not None:
            return
        self.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id=make_id("snap"),
                created_at=utc_now_iso(),
                cash=self.settings.risk.starting_cash,
                equity=self.settings.risk.starting_cash,
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
                raw_json="{}",
            )
        )

    def _latest_account(self) -> Dict[str, Any]:
        self.ensure_account_initialized()
        return self.repository.latest_account_snapshot() or {}

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
        if timeframe == "1h":
            return timedelta(hours=1)
        return timedelta(days=1)

    def _compute_cooldown_until(self, asset_type: str, timeframe: str, now_iso: str) -> str:
        bars = max(int(self.settings.risk.cooldown_bars_after_exit), 0)
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

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str:
        now_iso = utc_now_iso()
        order_id = make_id("ord")
        side = "buy" if signal.signal == "LONG" else "sell"
        max_holding_until = (
            pd.Timestamp.now(tz="UTC") + self._timeframe_delta(signal.timeframe) * int(self.settings.strategy.max_holding_bars)
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
                    "broker": "sim",
                    "prediction_id": signal.prediction_id,
                    "stop_level": signal.stop_level,
                    "take_level": signal.take_level,
                    "expected_risk": signal.expected_risk,
                    "max_holding_until": max_holding_until,
                    "strategy_version": signal.strategy_version,
                },
                ensure_ascii=False,
            ),
        )
        self.repository.insert_order(record)
        return order_id

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        now_iso = utc_now_iso()
        order_id = make_id("ord")
        side = "sell" if str(position["side"]) == "LONG" else "buy"
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
                    "broker": "sim",
                    "position_id": str(position["position_id"]),
                },
                ensure_ascii=False,
            ),
        )
        self.repository.insert_order(record)
        return order_id

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
        latest = self.repository.latest_position_by_symbol(symbol=str(order["symbol"]), timeframe=str(order["timeframe"]))
        now_iso = utc_now_iso()
        side = self._position_side_from_order(str(order["side"]))
        order_meta = json.loads(str(order.get("raw_json") or "{}"))
        if latest.empty or str(latest.iloc[0]["status"]) != "open":
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
                    stop_loss=float(order_meta.get("stop_level", np.nan)),
                    take_profit=float(order_meta.get("take_level", np.nan)),
                    trailing_stop=float(order_meta.get("stop_level", np.nan)),
                    highest_price=float(fill_price),
                    lowest_price=float(fill_price),
                    unrealized_pnl=-fees,
                    realized_pnl=0.0,
                    expected_risk=float(order_meta.get("expected_risk", np.nan)),
                    exposure_value=self._signed_market_value(side, int(fill_qty), float(fill_price)),
                    max_holding_until=str(order_meta.get("max_holding_until") or now_iso),
                    strategy_version=str(order_meta.get("strategy_version") or order["strategy_version"]),
                    cooldown_until=None,
                    notes="opened_by_fill",
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
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position.to_dict(),
                        "updated_at": now_iso,
                        "quantity": int(new_qty),
                        "entry_price": float(avg_price),
                        "mark_price": float(fill_price),
                        "stop_loss": float(order_meta.get("stop_level", position["stop_loss"])),
                        "take_profit": float(order_meta.get("take_level", position["take_profit"])),
                        "highest_price": max(float(position["highest_price"]), float(fill_price)),
                        "lowest_price": min(float(position["lowest_price"]), float(fill_price)),
                        "unrealized_pnl": self._compute_unrealized(position_side, float(avg_price), float(fill_price), int(new_qty)),
                        "exposure_value": self._signed_market_value(position_side, int(new_qty), float(fill_price)),
                        "notes": "scaled_position",
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
                        ),
                        "notes": f"closed_by_{order['reason']}",
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
                raw_json=json.dumps(raw_json or {}, default=str, ensure_ascii=False),
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
        account = self._latest_account()
        cash_value = float(account.get("cash", self.settings.risk.starting_cash))
        if str(order["side"]) == "buy":
            cash_value -= applied_qty * fill_price + effective_fees
        else:
            cash_value += applied_qty * fill_price - effective_fees
        self._update_position_from_fill(order=order, fill_qty=applied_qty, fill_price=float(fill_price), fees=float(effective_fees))
        self.snapshot_account(cash_override=cash_value)
        return True

    def process_open_orders(self, market_data_service: MarketDataService, touch=None) -> int:
        touch = touch or (lambda *args, **kwargs: None)
        orders = self.repository.open_orders(statuses=("new", "partially_filled"))
        if not orders.empty and "raw_json" in orders.columns:
            broker_name = orders["raw_json"].fillna("{}").astype(str).map(
                lambda payload: str(json.loads(payload).get("broker", "sim")) if payload else "sim"
            )
            orders = orders.loc[broker_name.isin({"", "sim"})].copy()

        filled_count = 0
        account = self._latest_account()
        cash_value = float(account.get("cash", self.settings.risk.starting_cash))
        for _, order in orders.iterrows():
            touch("sim_order_sync", {"order_id": str(order["order_id"]), "symbol": str(order["symbol"])})
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
                    raw_json=json.dumps(asdict(quote), default=str, ensure_ascii=False),
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
            if str(order["side"]) == "buy":
                cash_value -= fill_qty * fill_price + fees
            else:
                cash_value += fill_qty * fill_price - fees
            self._update_position_from_fill(order=order, fill_qty=fill_qty, fill_price=fill_price, fees=fees)
            filled_count += 1
        self.snapshot_account(cash_override=cash_value)
        return filled_count

    def snapshot_account(self, cash_override: float | None = None) -> None:
        account = self._latest_account()
        open_positions = self.repository.open_positions()
        open_orders = self.repository.open_orders()
        cash = float(cash_override if cash_override is not None else account.get("cash", self.settings.risk.starting_cash))
        realized = self.repository.total_realized_pnl()
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
        peak = max(float(self.repository.max_account_equity()), float(equity))
        drawdown_pct = (equity / peak - 1.0) * 100.0 if peak > 0 else 0.0
        today = str(pd.Timestamp.utcnow().date())
        daily_pnl = self.repository.recent_closed_realized_pnl(today)
        self.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id=make_id("snap"),
                created_at=utc_now_iso(),
                cash=float(cash),
                equity=float(equity),
                gross_exposure=float(gross_exposure),
                net_exposure=float(net_exposure),
                realized_pnl=float(realized),
                unrealized_pnl=float(unrealized),
                daily_pnl=float(daily_pnl),
                drawdown_pct=float(drawdown_pct),
                open_positions=int(len(open_positions)),
                open_orders=int(len(open_orders)),
                paused=int(self.repository.get_control_flag("trading_paused", "0") == "1"),
                source="paper_broker",
                raw_json="{}",
            )
        )
