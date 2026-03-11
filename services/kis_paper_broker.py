from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperClient, extract_kis_code
from kr_strategy import entry_gate_reason, get_kr_strategy, strategy_runtime_config, strategy_runtime_metadata, strategy_session_is_open, strategy_session_label, strategy_session_mode
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, BROKER_MODE_KIS
from services.market_data_service import MarketDataService
from services.paper_broker import PaperBroker
from services.signal_engine import SignalDecision
from storage.models import OrderRecord, PositionRecord
from storage.repository import TradingRepository, make_id, parse_utc_timestamp, utc_now_iso


class KISPaperBroker:
    def __init__(
        self,
        settings: RuntimeSettings,
        repository: TradingRepository,
        sim_broker: PaperBroker,
        client_factory: Callable[[], KISPaperClient] = KISPaperClient,
    ):
        self.settings = settings
        self.repository = repository
        self.sim_broker = sim_broker
        self.client_factory = client_factory
        self._enabled: bool | None = None

    def is_enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        try:
            self._enabled = bool(self._client().config.is_paper)
        except Exception:
            self._enabled = False
        return self._enabled

    def _client(self) -> KISPaperClient:
        return self.client_factory()

    def _parse_payload(self, payload: object) -> Dict[str, Any]:
        try:
            return json.loads(str(payload or "{}"))
        except Exception:
            return {}

    def _log_state(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        *,
        account_id: str = ACCOUNT_KIS_KR_PAPER,
        level: str = "INFO",
    ) -> None:
        payload = dict(details)
        payload.setdefault("account_id", account_id)
        self.repository.log_event(level, "kis_execution", event_type, message, payload, account_id=account_id)

    def _holding_baseline(self, client: KISPaperClient, symbol_code: str) -> Dict[str, float]:
        snapshot = client.get_account_snapshot()
        holdings = snapshot.holdings
        if holdings.empty or "symbol_code" not in holdings.columns:
            return {"quantity": 0.0, "avg_price": float("nan")}
        matched = holdings.loc[holdings["symbol_code"].astype(str) == str(symbol_code)]
        if matched.empty:
            return {"quantity": 0.0, "avg_price": float("nan")}
        row = matched.iloc[0]
        return {
            "quantity": float(pd.to_numeric(row.get("quantity", row.get("보유수량")), errors="coerce") or 0.0),
            "avg_price": float(pd.to_numeric(row.get("avg_price", row.get("매입평균가")), errors="coerce")),
        }

    def _holding_frame(self, snapshot) -> pd.DataFrame:
        holdings = snapshot.holdings.copy()
        if holdings.empty:
            return pd.DataFrame(columns=["symbol_code", "quantity", "avg_price", "current_price", "unrealized_pnl", "market_value"])
        if "symbol_code" not in holdings.columns:
            if "pdno" in holdings.columns:
                holdings["symbol_code"] = holdings["pdno"]
            else:
                holdings["symbol_code"] = ""
        for column, aliases in {
            "quantity": ("quantity", "보유수량"),
            "avg_price": ("avg_price", "매입평균가"),
            "current_price": ("current_price", "현재가"),
            "unrealized_pnl": ("unrealized_pnl", "평가손익"),
            "market_value": ("market_value", "평가금액"),
        }.items():
            if column in holdings.columns:
                continue
            for alias in aliases:
                if alias in holdings.columns:
                    holdings[column] = holdings[alias]
                    break
            if column not in holdings.columns:
                holdings[column] = np.nan if column != "quantity" else 0.0
        holdings["symbol_code"] = holdings["symbol_code"].astype(str).str.strip()
        for column in ("quantity", "avg_price", "current_price", "unrealized_pnl", "market_value"):
            holdings[column] = pd.to_numeric(holdings[column], errors="coerce")
        return holdings[["symbol_code", "quantity", "avg_price", "current_price", "unrealized_pnl", "market_value"]].copy()

    def _resolve_runtime_symbol(self, symbol_code: str) -> str:
        code = str(symbol_code or "").strip()
        if not code:
            return code
        frames = [
            self.repository.open_positions(account_id=ACCOUNT_KIS_KR_PAPER),
            self.repository.recent_orders(limit=500, account_id=ACCOUNT_KIS_KR_PAPER),
            self.repository.latest_candidates(execution_account_id=ACCOUNT_KIS_KR_PAPER, limit=500),
        ]
        for frame in frames:
            if frame.empty or "symbol" not in frame.columns:
                continue
            for symbol in frame["symbol"].dropna().astype(str).tolist():
                try:
                    if extract_kis_code(symbol) == code:
                        return symbol
                except Exception:
                    continue
        universe = self.settings.universes.get("한국주식")
        if universe is not None:
            for symbol in list(getattr(universe, "watchlist", []) or []) + list(getattr(universe, "top_universe", []) or []):
                try:
                    if extract_kis_code(symbol) == code:
                        return str(symbol)
                except Exception:
                    continue
        return f"{code}.KS"

    def _canonical_account_values(self, snapshot) -> Dict[str, float]:
        holdings = snapshot.holdings.copy()
        market_value = 0.0
        unrealized_pnl = float(snapshot.summary.get("pnl", 0.0) or 0.0)
        if not holdings.empty:
            if "market_value" in holdings.columns:
                market_value = float(pd.to_numeric(holdings["market_value"], errors="coerce").fillna(0.0).sum())
            if not np.isfinite(market_value) or market_value <= 0:
                market_value = float(pd.to_numeric(holdings.get("평가금액"), errors="coerce").fillna(0.0).sum())
            if "unrealized_pnl" in holdings.columns:
                unrealized_pnl = float(pd.to_numeric(holdings["unrealized_pnl"], errors="coerce").fillna(0.0).sum())
            elif "평가손익" in holdings.columns:
                unrealized_pnl = float(pd.to_numeric(holdings["평가손익"], errors="coerce").fillna(0.0).sum())

        cash = float(snapshot.summary.get("cash", 0.0) or 0.0)
        stock_eval = float(snapshot.summary.get("stock_eval", market_value) or market_value)
        total_eval = float(snapshot.summary.get("total_eval", cash + stock_eval) or (cash + stock_eval))
        gross_exposure = max(stock_eval, market_value, 0.0)
        net_exposure = gross_exposure
        equity = total_eval if np.isfinite(total_eval) and total_eval > 0 else cash + net_exposure
        return {
            "cash": float(cash),
            "equity": float(equity),
            "gross_exposure": float(gross_exposure),
            "net_exposure": float(net_exposure),
            "unrealized_pnl": float(unrealized_pnl),
            "open_positions": int(len(holdings)),
        }

    def _pick_execution_price(self, quote: Dict[str, Any], orderbook: Dict[str, Any], side: str) -> float:
        if side == "buy":
            for key in ("best_ask", "expected_price"):
                value = float(orderbook.get(key, np.nan))
                if np.isfinite(value) and value > 0:
                    return value
        else:
            for key in ("best_bid", "expected_price"):
                value = float(orderbook.get(key, np.nan))
                if np.isfinite(value) and value > 0:
                    return value
        value = float(quote.get("current_price", np.nan))
        return value if np.isfinite(value) and value > 0 else float("nan")

    def _strategy_context(self, strategy_version: str, *, when: datetime | None = None) -> Dict[str, Any]:
        strategy = get_kr_strategy(self.settings, str(strategy_version or ""))
        strategy_view = strategy_runtime_config(strategy, when=when)
        return {
            "strategy": strategy_view if strategy_view is not None else strategy,
            "session_mode": strategy_session_mode(strategy, when=when),
            "session_label": strategy_session_label(strategy, when=when),
            "order_division": str((strategy_view.order_division if strategy_view is not None else strategy.order_division)) if strategy is not None else ("01" if self.settings.broker.default_order_type == "market" else "00"),
        }

    def _execution_metadata(self, strategy_version: str, *, when: datetime | None = None) -> Dict[str, str]:
        return strategy_runtime_metadata(self.settings, str(strategy_version or ""), when=when)

    def _resolve_quote_bundle(self, client: KISPaperClient, symbol: str, side: str, *, strategy_version: str, when: datetime | None = None) -> Dict[str, Any]:
        strategy_ctx = self._strategy_context(strategy_version, when=when)
        session_mode = str(strategy_ctx["session_mode"])
        market_status = client.get_market_status(symbol)
        if session_mode == "after_close_close_price":
            quote = client.get_overtime_price(symbol)
            orderbook = client.get_overtime_asking_price(symbol)
            close_price = float(quote.get("close_price", np.nan))
            execution_price = close_price if np.isfinite(close_price) and close_price > 0 else float("nan")
            return {
                "quote": quote,
                "orderbook": orderbook,
                "market_status": market_status,
                "execution_price": execution_price,
                "price_error_reason": "after_close_price_unavailable",
                "order_division": "06",
                "order_type": "after_close_close",
            }
        if session_mode == "after_close_single_price":
            quote = client.get_overtime_price(symbol)
            orderbook = client.get_overtime_asking_price(symbol)
            execution_price = self._pick_execution_price(quote, orderbook, side)
            lower_limit = float(quote.get("lower_limit", np.nan))
            upper_limit = float(quote.get("upper_limit", np.nan))
            if np.isfinite(execution_price) and execution_price > 0 and np.isfinite(lower_limit) and np.isfinite(upper_limit):
                if execution_price < lower_limit or execution_price > upper_limit:
                    execution_price = float("nan")
            return {
                "quote": quote,
                "orderbook": orderbook,
                "market_status": market_status,
                "execution_price": execution_price,
                "price_error_reason": "after_close_single_quote_unavailable",
                "order_division": "07",
                "order_type": "after_close_single",
            }
        quote = client.get_quote(symbol)
        orderbook = client.get_orderbook(symbol)
        return {
            "quote": quote,
            "orderbook": orderbook,
            "market_status": market_status,
            "execution_price": self._pick_execution_price(quote, orderbook, side),
            "price_error_reason": "no_quote",
            "order_division": str(strategy_ctx["order_division"]),
            "order_type": self.settings.broker.default_order_type,
        }

    def _max_holding_until_iso(self, *, strategy_version: str, timeframe: str, when: datetime | None = None) -> str:
        strategy = get_kr_strategy(self.settings, str(strategy_version or ""))
        strategy_view = strategy_runtime_config(strategy, when=when)
        bar_delta = self.sim_broker._timeframe_delta(timeframe)
        if strategy is None:
            return (pd.Timestamp.now(tz="UTC") + bar_delta * int(self.settings.strategy.max_holding_bars)).isoformat()
        if strategy_session_mode(strategy, when=when) == "regular":
            active_strategy = strategy_view if strategy_view is not None else strategy
            return (pd.Timestamp.now(tz="UTC") + bar_delta * int(active_strategy.max_holding_bars)).isoformat()
        next_session = pd.Timestamp.now(tz="Asia/Seoul").normalize() + pd.Timedelta(days=1, hours=9)
        return (next_session.tz_convert("UTC") + bar_delta * int(strategy.max_holding_bars)).isoformat()

    def _holding_row_for_code(self, holdings: pd.DataFrame, symbol_code: str) -> pd.Series | None:
        if holdings.empty or "symbol_code" not in holdings.columns:
            return None
        matched = holdings.loc[holdings["symbol_code"].astype(str) == str(symbol_code)]
        return matched.iloc[0] if not matched.empty else None

    def _infer_sync_state_from_holdings(self, order: pd.Series, holdings: pd.DataFrame) -> Dict[str, Any] | None:
        payload = self._parse_payload(order.get("raw_json"))
        symbol_code = str(payload.get("symbol_code") or extract_kis_code(str(order["symbol"]))).strip()
        holding = self._holding_row_for_code(holdings, symbol_code)
        held_qty = int(round(float(pd.to_numeric(pd.Series([holding.get("quantity") if holding is not None else 0.0]), errors="coerce").iloc[0] or 0.0)))
        avg_price = float(pd.to_numeric(pd.Series([holding.get("avg_price") if holding is not None else np.nan]), errors="coerce").iloc[0])
        baseline_qty = int(round(float(pd.to_numeric(pd.Series([payload.get("baseline_qty", payload.get("last_seen_broker_qty", 0.0))]), errors="coerce").iloc[0] or 0.0)))
        existing_filled = int(order["filled_qty"])
        requested_qty = int(order["requested_qty"])
        if str(order["side"]) == "buy":
            inferred_total = max(existing_filled, max(held_qty - baseline_qty, 0))
            fill_price = avg_price if np.isfinite(avg_price) and avg_price > 0 else float(order["requested_price"])
        else:
            inferred_total = max(existing_filled, max(baseline_qty - held_qty, 0))
            fill_price = float(order["requested_price"])
        inferred_total = min(int(inferred_total), requested_qty)
        if inferred_total <= existing_filled:
            return None
        return {
            "status": "filled" if inferred_total >= requested_qty else "partially_filled",
            "filled_total": int(inferred_total),
            "fill_price": float(fill_price),
            "source": "holdings_snapshot",
            "details": {
                "symbol_code": symbol_code,
                "baseline_qty": baseline_qty,
                "held_qty": held_qty,
                "avg_price": avg_price,
            },
        }

    def _reconcile_positions_from_holdings(self, snapshot, touch=None) -> Dict[str, int]:
        touch = touch or (lambda *args, **kwargs: None)
        holdings = self._holding_frame(snapshot)
        if holdings.empty:
            return {"restored": 0, "updated": 0}
        open_positions = self.repository.open_positions(account_id=ACCOUNT_KIS_KR_PAPER)
        recent_orders = self.repository.recent_orders(limit=500, account_id=ACCOUNT_KIS_KR_PAPER)
        counts = {"restored": 0, "updated": 0}
        now_iso = utc_now_iso()
        for _, holding in holdings.iterrows():
            symbol_code = str(holding.get("symbol_code") or "").strip()
            quantity = int(round(float(holding.get("quantity") or 0.0)))
            if not symbol_code or quantity <= 0:
                continue
            symbol = self._resolve_runtime_symbol(symbol_code)
            current_price = float(pd.to_numeric(pd.Series([holding.get("current_price")]), errors="coerce").iloc[0])
            avg_price = float(pd.to_numeric(pd.Series([holding.get("avg_price")]), errors="coerce").iloc[0])
            market_value = float(pd.to_numeric(pd.Series([holding.get("market_value")]), errors="coerce").iloc[0])
            unrealized_pnl = float(pd.to_numeric(pd.Series([holding.get("unrealized_pnl")]), errors="coerce").iloc[0])
            related_orders = recent_orders.loc[
                recent_orders["symbol"].astype(str).map(lambda value: extract_kis_code(str(value)) == symbol_code)
                & (recent_orders["side"].astype(str) == "buy")
                & (~recent_orders["status"].astype(str).isin({"rejected", "cancelled", "expired"}))
            ] if not recent_orders.empty else pd.DataFrame()
            related_order = related_orders.iloc[0] if not related_orders.empty else None
            order_meta = self._parse_payload(related_order.get("raw_json")) if related_order is not None else {}
            strategy_version = str((related_order.get("strategy_version") if related_order is not None else "") or "")
            timeframe = str((related_order.get("timeframe") if related_order is not None else "") or "")
            if not timeframe and strategy_version:
                try:
                    timeframe = str(get_kr_strategy(self.settings, strategy_version).timeframe)
                except Exception:
                    timeframe = ""
            timeframe = timeframe or str(self.settings.asset_schedules["한국주식"].timeframe)
            if not np.isfinite(avg_price) or avg_price <= 0:
                avg_price = float(pd.to_numeric(pd.Series([related_order.get("requested_price") if related_order is not None else np.nan]), errors="coerce").iloc[0])
            if not np.isfinite(current_price) or current_price <= 0:
                current_price = avg_price
            if not np.isfinite(market_value) or market_value <= 0:
                market_value = float(current_price) * int(quantity)
            if not np.isfinite(unrealized_pnl):
                unrealized_pnl = self.sim_broker._compute_unrealized("LONG", float(avg_price), float(current_price), int(quantity))
            exit_levels = self.sim_broker._recalculate_exit_levels(
                entry_price=float(avg_price),
                side="LONG",
                order_meta=order_meta,
                strategy_version=strategy_version,
            )
            matched = open_positions.loc[
                open_positions["symbol"].astype(str).map(lambda value: extract_kis_code(str(value)) == symbol_code)
            ] if not open_positions.empty else pd.DataFrame()
            touch("kis_holding_reconcile", {"symbol": symbol, "quantity": quantity})
            if not matched.empty:
                position = matched.iloc[0]
                position_dict = {key: value for key, value in position.to_dict().items() if key != "rowid"}
                self.repository.upsert_position(
                    PositionRecord(
                        **{
                            **position_dict,
                            "updated_at": now_iso,
                            "symbol": symbol,
                            "status": "open",
                            "quantity": int(quantity),
                            "entry_price": float(avg_price),
                            "mark_price": float(current_price),
                            "stop_loss": float(exit_levels["stop_loss"]),
                            "take_profit": float(exit_levels["take_profit"]),
                            "trailing_stop": float(exit_levels["trailing_stop"]),
                            "highest_price": max(float(position.get("highest_price", current_price) or current_price), float(current_price), float(avg_price)),
                            "lowest_price": min(float(position.get("lowest_price", current_price) or current_price), float(current_price), float(avg_price)),
                            "unrealized_pnl": float(unrealized_pnl),
                            "exposure_value": float(market_value),
                            "expected_risk": float(pd.to_numeric(pd.Series([order_meta.get("expected_risk", position.get("expected_risk"))]), errors="coerce").iloc[0]),
                            "strategy_version": str(strategy_version or position.get("strategy_version") or ""),
                            "max_holding_until": str(order_meta.get("max_holding_until") or position.get("max_holding_until") or now_iso),
                            "notes": "synced_from_kis_holdings",
                            "account_id": ACCOUNT_KIS_KR_PAPER,
                            "strategy_family": str(order_meta.get("strategy_family") or position.get("strategy_family") or ""),
                            "session_mode": str(order_meta.get("session_mode") or position.get("session_mode") or ""),
                            "price_policy": str(order_meta.get("price_policy") or position.get("price_policy") or ""),
                        }
                    )
                )
                counts["updated"] += 1
                continue
            self.repository.upsert_position(
                PositionRecord(
                    position_id=make_id("pos"),
                    created_at=str((related_order.get("created_at") if related_order is not None else now_iso) or now_iso),
                    updated_at=now_iso,
                    closed_at=None,
                    prediction_id=str((related_order.get("prediction_id") if related_order is not None else "") or ""),
                    symbol=symbol,
                    asset_type="한국주식",
                    timeframe=timeframe,
                    side="LONG",
                    status="open",
                    quantity=int(quantity),
                    entry_price=float(avg_price),
                    mark_price=float(current_price),
                    stop_loss=float(exit_levels["stop_loss"]),
                    take_profit=float(exit_levels["take_profit"]),
                    trailing_stop=float(exit_levels["trailing_stop"]),
                    highest_price=max(float(current_price), float(avg_price)),
                    lowest_price=min(float(current_price), float(avg_price)),
                    unrealized_pnl=float(unrealized_pnl),
                    realized_pnl=0.0,
                    expected_risk=float(pd.to_numeric(pd.Series([order_meta.get("expected_risk")]), errors="coerce").iloc[0]),
                    exposure_value=float(market_value),
                    max_holding_until=str(order_meta.get("max_holding_until") or now_iso),
                    strategy_version=str(strategy_version),
                    cooldown_until=None,
                    notes="restored_from_kis_holdings",
                    account_id=ACCOUNT_KIS_KR_PAPER,
                    strategy_family=str(order_meta.get("strategy_family") or ""),
                    session_mode=str(order_meta.get("session_mode") or ""),
                    price_policy=str(order_meta.get("price_policy") or ""),
                )
            )
            counts["restored"] += 1
        return counts

    def _insert_submitted_order(
        self,
        *,
        account_id: str,
        symbol: str,
        asset_type: str,
        timeframe: str,
        prediction_id: str | None,
        scan_id: str | None,
        side: str,
        quantity: int,
        order_type: str,
        requested_price: float,
        strategy_version: str,
        reason: str,
        payload: Dict[str, Any],
    ) -> str:
        order_id = make_id("ord")
        now_iso = utc_now_iso()
        self.repository.insert_order(
            OrderRecord(
                order_id=order_id,
                created_at=now_iso,
                updated_at=now_iso,
                prediction_id=prediction_id,
                scan_id=scan_id,
                symbol=symbol,
                asset_type=asset_type,
                timeframe=timeframe,
                side=side,
                order_type=str(order_type),
                requested_qty=int(quantity),
                filled_qty=0,
                remaining_qty=int(quantity),
                requested_price=float(requested_price),
                limit_price=np.nan,
                status="submitted",
                fees_estimate=float(requested_price) * int(quantity) * self.settings.broker.fee_bps / 10000.0,
                slippage_bps=self.settings.broker.base_slippage_bps,
                retry_count=0,
                strategy_version=strategy_version,
                reason=reason,
                raw_json=json.dumps(payload, ensure_ascii=False),
                account_id=account_id,
            )
        )
        return order_id

    def preflight_entry(
        self,
        signal: SignalDecision,
        quantity: int,
        market_data_service: MarketDataService,
        *,
        account_id: str = ACCOUNT_KIS_KR_PAPER,
    ) -> Dict[str, Any]:
        asset_type = str(signal.asset_type)
        if not self.is_enabled():
            return {"allowed": False, "reason": "kis_disabled", "broker": BROKER_MODE_KIS, "account_id": account_id}
        current_time = (
            market_data_service.current_time(asset_type)
            if market_data_service is not None and callable(getattr(market_data_service, "current_time", None))
            else None
        )
        regular_market_open = bool(market_data_service.is_market_open(asset_type)) if market_data_service is not None else False
        session_open = strategy_session_is_open(
            self.settings,
            str(signal.strategy_version or ""),
            when=current_time,
            market_is_open=regular_market_open,
        )
        if not session_open:
            gate_reason = entry_gate_reason(
                self.settings,
                str(signal.strategy_version or ""),
                when=current_time,
                market_is_open=regular_market_open,
            )
            return {
                "allowed": False,
                "reason": str(gate_reason or "market_closed"),
                "broker": BROKER_MODE_KIS,
                "account_id": account_id,
            }
        gate_reason = entry_gate_reason(
            self.settings,
            str(signal.strategy_version or ""),
            when=current_time,
            market_is_open=regular_market_open,
        )
        if gate_reason:
            return {"allowed": False, "reason": gate_reason, "broker": BROKER_MODE_KIS, "account_id": account_id}

        client = self._client()
        side = "buy" if signal.signal == "LONG" else "sell"
        quote_bundle = self._resolve_quote_bundle(client, signal.symbol, side, strategy_version=str(signal.strategy_version or ""), when=current_time)
        quote = dict(quote_bundle["quote"])
        orderbook = dict(quote_bundle["orderbook"])
        market_status = dict(quote_bundle["market_status"])
        if bool(market_status.get("is_halted")):
            return {"allowed": False, "reason": "market_halted", "broker": BROKER_MODE_KIS, "market_status": market_status, "account_id": account_id}

        exec_price = float(quote_bundle.get("execution_price", np.nan))
        if not np.isfinite(exec_price) or exec_price <= 0:
            return {
                "allowed": False,
                "reason": str(quote_bundle.get("price_error_reason") or "no_quote"),
                "broker": BROKER_MODE_KIS,
                "quote": quote,
                "orderbook": orderbook,
                "account_id": account_id,
            }

        if not self.repository.active_entry_orders(symbol=signal.symbol, timeframe=signal.timeframe, asset_type=asset_type, account_id=account_id).empty:
            return {"allowed": False, "reason": "duplicate_pending_entry", "broker": BROKER_MODE_KIS, "account_id": account_id}
        latest_position = self.repository.latest_position_by_symbol(signal.symbol, signal.timeframe, account_id=account_id)
        if not latest_position.empty and str(latest_position.iloc[0].get("status")) == "open":
            return {"allowed": False, "reason": "already_holding_symbol", "broker": BROKER_MODE_KIS, "account_id": account_id}

        order_division = str(quote_bundle.get("order_division") or "01")
        buying_power = client.get_buying_power(signal.symbol, order_price=exec_price, order_division=order_division)
        max_buy_qty = int(buying_power.get("cash_buy_qty") or buying_power.get("max_buy_qty") or 0)
        if max_buy_qty < int(quantity):
            strategy_ctx = self._strategy_context(str(signal.strategy_version or ""), when=current_time)
            insufficient_reason = "after_close_buying_power_insufficient" if str(strategy_ctx["session_mode"]).startswith("after_close_") else "insufficient_buying_power"
            return {"allowed": False, "reason": insufficient_reason, "broker": BROKER_MODE_KIS, "buying_power": buying_power, "account_id": account_id}

        metadata = self._execution_metadata(str(signal.strategy_version or ""), when=current_time)
        return {
            "allowed": True,
            "reason": "ok",
            "broker": BROKER_MODE_KIS,
            "account_id": account_id,
            "symbol_code": extract_kis_code(signal.symbol),
            "execution_price": float(exec_price),
            "order_division": order_division,
            "order_type": str(quote_bundle.get("order_type") or self.settings.broker.default_order_type),
            "session_mode": str(self._strategy_context(str(signal.strategy_version or ""), when=current_time)["session_mode"]),
            "strategy_family": str(metadata.get("strategy_family") or ""),
            "price_policy": str(metadata.get("price_policy") or ""),
            "quote": quote,
            "orderbook": orderbook,
            "market_status": market_status,
            "buying_power": buying_power,
            "account_summary": client.get_account_snapshot().summary,
        }

    def preflight_exit(
        self,
        position: pd.Series,
        market_data_service: MarketDataService,
        *,
        account_id: str = ACCOUNT_KIS_KR_PAPER,
    ) -> Dict[str, Any]:
        asset_type = str(position["asset_type"])
        if not self.is_enabled():
            return {"allowed": False, "reason": "kis_disabled", "broker": BROKER_MODE_KIS, "account_id": account_id}
        current_time = (
            market_data_service.current_time(asset_type)
            if market_data_service is not None and callable(getattr(market_data_service, "current_time", None))
            else None
        )
        regular_market_open = bool(market_data_service.is_market_open(asset_type)) if market_data_service is not None else False
        strategy_version = str(position.get("strategy_version") or "")
        session_open = strategy_session_is_open(
            self.settings,
            strategy_version,
            when=current_time,
            market_is_open=regular_market_open,
        )
        if not session_open and not regular_market_open:
            return {"allowed": False, "reason": "market_closed", "broker": BROKER_MODE_KIS, "account_id": account_id}
        client = self._client()
        side = "sell" if str(position["side"]) == "LONG" else "buy"
        symbol = str(position["symbol"])
        symbol_code = extract_kis_code(symbol)
        quote_bundle = self._resolve_quote_bundle(client, str(position["symbol"]), side, strategy_version=strategy_version, when=current_time)
        quote = dict(quote_bundle["quote"])
        orderbook = dict(quote_bundle["orderbook"])
        exec_price = float(quote_bundle.get("execution_price", np.nan))
        if not np.isfinite(exec_price) or exec_price <= 0:
            return {"allowed": False, "reason": str(quote_bundle.get("price_error_reason") or "no_quote"), "broker": BROKER_MODE_KIS, "account_id": account_id}
        sellable_fallback = ""
        sellable_error = ""
        try:
            sellable = client.get_sellable_quantity(symbol)
        except Exception as exc:
            sellable_error = str(exc)
            if not bool(client.config.is_paper):
                raise
            baseline = self._holding_baseline(client, symbol_code)
            snapshot_qty = max(int(round(float(baseline.get("quantity", 0.0) or 0.0))), 0)
            position_qty = max(int(position.get("quantity") or 0), 0)
            inferred_qty = max(snapshot_qty, position_qty)
            sellable = {
                "symbol_code": symbol_code,
                "sellable_qty": inferred_qty,
                "held_qty": snapshot_qty,
                "raw": {"fallback": "account_snapshot", "error": sellable_error},
            }
            sellable_fallback = "account_snapshot"
        if str(position["side"]) == "LONG" and int(sellable.get("sellable_qty", 0)) < int(position["quantity"]):
            return {"allowed": False, "reason": "no_sellable_qty", "broker": BROKER_MODE_KIS, "sellable": sellable, "account_id": account_id}
        return {
            "allowed": True,
            "reason": "ok",
            "broker": BROKER_MODE_KIS,
            "execution_price": float(exec_price),
            "sellable": sellable,
            "account_id": account_id,
            "order_division": str(quote_bundle.get("order_division") or "01"),
            "order_type": str(quote_bundle.get("order_type") or self.settings.broker.default_order_type),
            "sellable_fallback": sellable_fallback,
            "sellable_error": sellable_error,
        }

    def submit_entry_order_result(
        self,
        signal: SignalDecision,
        quantity: int,
        scan_id: str | None = None,
        *,
        account_id: str = ACCOUNT_KIS_KR_PAPER,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        preflight = (
            self.preflight_entry(signal, quantity, market_data_service, account_id=account_id)
            if market_data_service is not None
            else {"allowed": True, "account_id": account_id}
        )
        if not preflight.get("allowed", False):
            return {
                "submitted": False,
                "status": "rejected",
                "reason": str(preflight.get("reason") or "rejected"),
                "broker": BROKER_MODE_KIS,
                "order_id": "",
                "account_id": account_id,
            }

        client = self._client()
        current_time = (
            market_data_service.current_time(signal.asset_type)
            if market_data_service is not None and callable(getattr(market_data_service, "current_time", None))
            else None
        )
        metadata = self._execution_metadata(str(signal.strategy_version or ""), when=current_time)
        symbol_code = str(preflight.get("symbol_code") or extract_kis_code(signal.symbol))
        baseline = self._holding_baseline(client, symbol_code)
        requested_price = float(preflight.get("execution_price") or signal.current_price)
        requested_order_type = str(preflight.get("order_type") or self.settings.broker.default_order_type)
        requested_order_division = str(preflight.get("order_division") or "01")
        order_id = self._insert_submitted_order(
            account_id=account_id,
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            timeframe=signal.timeframe,
            prediction_id=signal.prediction_id,
            scan_id=scan_id or signal.scan_id,
            side="buy" if signal.signal == "LONG" else "sell",
            quantity=int(quantity),
            order_type=requested_order_type,
            requested_price=requested_price,
            strategy_version=signal.strategy_version,
            reason="entry",
            payload={
                "broker": BROKER_MODE_KIS,
                "account_id": account_id,
                "symbol_code": symbol_code,
                "baseline_qty": baseline["quantity"],
                "baseline_avg_price": baseline["avg_price"],
                "last_seen_broker_qty": baseline["quantity"],
                "last_seen_broker_avg_price": baseline["avg_price"],
                "session_mode": str(preflight.get("session_mode") or self._strategy_context(str(signal.strategy_version or ""), when=current_time)["session_mode"]),
                "strategy_family": str(preflight.get("strategy_family") or metadata.get("strategy_family") or ""),
                "price_policy": str(preflight.get("price_policy") or metadata.get("price_policy") or ""),
                "order_division": requested_order_division,
                "order_type": requested_order_type,
                "expected_risk": signal.expected_risk,
                "atr_14": signal.atr_value,
                "stop_level": signal.stop_level,
                "take_level": signal.take_level,
                "max_holding_until": self._max_holding_until_iso(strategy_version=str(signal.strategy_version or ""), timeframe=signal.timeframe, when=current_time),
                "strategy_version": signal.strategy_version,
                "broker_state": "submit_requested",
                "preflight": preflight,
                "hts_id": client.config.hts_id,
            },
        )
        self._log_state(
            "submit_requested",
            "KIS entry submit requested",
            {"order_id": order_id, "symbol": signal.symbol, "quantity": int(quantity)},
            account_id=account_id,
        )
        try:
            result = client.place_cash_order(
                symbol_or_code=signal.symbol,
                side="buy" if signal.signal == "LONG" else "sell",
                quantity=int(quantity),
                order_type=requested_order_type,
                price=requested_price,
                order_division=requested_order_division,
            )
        except Exception as exc:
            strategy_ctx = self._strategy_context(str(signal.strategy_version or ""), when=current_time)
            rejected_reason = "after_close_order_rejected" if str(strategy_ctx["session_mode"]) == "after_close_close_price" else "after_close_single_order_rejected" if str(strategy_ctx["session_mode"]) == "after_close_single_price" else "broker_rejected"
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": BROKER_MODE_KIS, "broker_state": "rejected", "error_stage": "submit", "account_id": account_id},
            )
            self._log_state(
                "rejected",
                "KIS broker rejected entry",
                {"order_id": order_id, "symbol": signal.symbol, "reason": rejected_reason, "error": str(exc)},
                account_id=account_id,
                level="ERROR",
            )
            return {"submitted": False, "status": "rejected", "reason": rejected_reason, "broker": BROKER_MODE_KIS, "order_id": order_id, "account_id": account_id}

        status = "acknowledged" if result.get("order_no") else "submitted"
        self.repository.update_order(
            order_id,
            status=status,
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": BROKER_MODE_KIS,
                "account_id": account_id,
                "broker_state": status,
                "order_division": requested_order_division,
                "order_type": requested_order_type,
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        self._log_state("submitted", "KIS entry submitted", {"order_id": order_id, "symbol": signal.symbol, "broker_order_id": str(result.get("order_no") or "")}, account_id=account_id)
        if status == "acknowledged":
            self._log_state("acknowledged", "KIS entry acknowledged", {"order_id": order_id, "symbol": signal.symbol, "broker_order_id": str(result.get("order_no") or "")}, account_id=account_id)
        return {"submitted": True, "status": status, "reason": "ok", "broker": BROKER_MODE_KIS, "order_id": order_id, "broker_order_id": str(result.get("order_no") or ""), "account_id": account_id}

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None, *, account_id: str = ACCOUNT_KIS_KR_PAPER) -> str:
        return str(self.submit_entry_order_result(signal, quantity, scan_id, account_id=account_id).get("order_id") or "")

    def submit_exit_order_result(
        self,
        position: pd.Series,
        reason: str,
        *,
        account_id: str = ACCOUNT_KIS_KR_PAPER,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        preflight = (
            self.preflight_exit(position, market_data_service, account_id=account_id)
            if market_data_service is not None
            else {"allowed": True, "account_id": account_id}
        )
        if not preflight.get("allowed", False):
            return {"submitted": False, "status": "rejected", "reason": str(preflight.get("reason") or "rejected"), "order_id": "", "account_id": account_id}

        client = self._client()
        symbol = str(position["symbol"])
        symbol_code = extract_kis_code(symbol)
        baseline = self._holding_baseline(client, symbol_code)
        requested_order_type = str(preflight.get("order_type") or self.settings.broker.default_order_type)
        requested_order_division = str(preflight.get("order_division") or "01")
        order_id = self._insert_submitted_order(
            account_id=account_id,
            symbol=symbol,
            asset_type=str(position["asset_type"]),
            timeframe=str(position["timeframe"]),
            prediction_id=str(position.get("prediction_id") or ""),
            scan_id=None,
            side="sell" if str(position["side"]) == "LONG" else "buy",
            quantity=int(position["quantity"]),
            order_type=requested_order_type,
            requested_price=float(preflight.get("execution_price") or position["mark_price"]),
            strategy_version=str(position["strategy_version"]),
            reason=reason,
            payload={
                "broker": BROKER_MODE_KIS,
                "account_id": account_id,
                "symbol_code": symbol_code,
                "baseline_qty": baseline["quantity"],
                "baseline_avg_price": baseline["avg_price"],
                "last_seen_broker_qty": baseline["quantity"],
                "last_seen_broker_avg_price": baseline["avg_price"],
                "strategy_family": str(position.get("strategy_family") or ""),
                "session_mode": str(position.get("session_mode") or ""),
                "price_policy": str(position.get("price_policy") or ""),
                "order_division": requested_order_division,
                "order_type": requested_order_type,
                "position_id": str(position.get("position_id") or ""),
                "broker_state": "submit_requested",
                "preflight": preflight,
                "hts_id": client.config.hts_id,
            },
        )
        self._log_state("submit_requested", "KIS exit submit requested", {"order_id": order_id, "symbol": symbol, "quantity": int(position["quantity"])}, account_id=account_id)
        try:
            result = client.place_cash_order(
                symbol_or_code=symbol,
                side="sell" if str(position["side"]) == "LONG" else "buy",
                quantity=int(position["quantity"]),
                order_type=requested_order_type,
                price=float(preflight.get("execution_price") or position["mark_price"]),
                order_division=requested_order_division,
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": BROKER_MODE_KIS, "account_id": account_id, "broker_state": "rejected", "error_stage": "submit"},
            )
            self._log_state("rejected", "KIS broker rejected exit", {"order_id": order_id, "symbol": symbol, "reason": "broker_rejected", "error": str(exc)}, account_id=account_id, level="ERROR")
            return {"submitted": False, "status": "rejected", "reason": "broker_rejected", "order_id": order_id, "account_id": account_id}

        status = "acknowledged" if result.get("order_no") else "submitted"
        self.repository.update_order(
            order_id,
            status=status,
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": BROKER_MODE_KIS,
                "account_id": account_id,
                "broker_state": status,
                "order_division": requested_order_division,
                "order_type": requested_order_type,
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        self._log_state("submitted", "KIS exit submitted", {"order_id": order_id, "symbol": symbol, "broker_order_id": str(result.get("order_no") or "")}, account_id=account_id)
        if status == "acknowledged":
            self._log_state("acknowledged", "KIS exit acknowledged", {"order_id": order_id, "symbol": symbol, "broker_order_id": str(result.get("order_no") or "")}, account_id=account_id)
        return {"submitted": True, "status": status, "reason": "ok", "order_id": order_id, "broker_order_id": str(result.get("order_no") or ""), "account_id": account_id}

    def submit_exit_order(self, position: pd.Series, reason: str, *, account_id: str = ACCOUNT_KIS_KR_PAPER) -> str:
        return str(self.submit_exit_order_result(position, reason, account_id=account_id).get("order_id") or "")

    def sync_account(self, touch=None) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        if not self.is_enabled():
            touch("kis_account_sync_skipped", {"enabled": False})
            return {"broker": "kis_mock", "enabled": False}
        touch("kis_account_sync_request", {"enabled": True})
        client = self._client()
        snapshot = client.get_account_snapshot()
        touch(
            "kis_account_sync_snapshot",
            {
                "holding_count": int(snapshot.summary.get("holding_count", 0) or 0),
                "cash": float(snapshot.summary.get("cash", 0.0) or 0.0),
            },
        )
        approval_key_available = False
        try:
            approval_key_available = bool(client.get_websocket_approval_key())
        except Exception:
            approval_key_available = False
        touch("kis_account_sync_ws", {"approval_key_available": approval_key_available, "hts_id_configured": bool(client.config.hts_id)})
        canonical = self._canonical_account_values(snapshot)
        canonical_snapshot = self.sim_broker.record_external_account_snapshot(
            account_id=ACCOUNT_KIS_KR_PAPER,
            cash=canonical["cash"],
            equity=canonical["equity"],
            gross_exposure=canonical["gross_exposure"],
            net_exposure=canonical["net_exposure"],
            unrealized_pnl=canonical["unrealized_pnl"],
            open_positions=canonical["open_positions"],
            source="kis_account_sync",
            raw_json=json.dumps(
                {
                    "broker": BROKER_MODE_KIS,
                    "account_id": ACCOUNT_KIS_KR_PAPER,
                    "summary": snapshot.summary,
                    "holding_count": canonical["open_positions"],
                },
                ensure_ascii=False,
                default=str,
            ),
            touch=lambda stage=None, details=None: touch(stage or "kis_account_snapshot", details),
        )
        reconcile_totals = {"fills": 0, "restored": 0, "updated": 0}
        active = self.repository.open_orders(
            statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"),
            account_id=ACCOUNT_KIS_KR_PAPER,
        )
        if not active.empty:
            active = active.loc[
                active["raw_json"].fillna("{}").astype(str).map(lambda payload: self._parse_payload(payload).get("broker") == "kis_mock")
            ].copy()
            holdings = self._holding_frame(snapshot)
            for _, order in active.iterrows():
                holdings_sync_state = self._infer_sync_state_from_holdings(order, holdings)
                if holdings_sync_state is None:
                    continue
                applied = self._apply_state_transition(order, holdings_sync_state)
                for key, value in applied.items():
                    reconcile_totals[key] = int(reconcile_totals.get(key, 0)) + int(value)
        position_totals = self._reconcile_positions_from_holdings(snapshot, touch=touch)
        for key, value in position_totals.items():
            reconcile_totals[key] = int(reconcile_totals.get(key, 0)) + int(value)
        self.repository.set_control_flag("kis_last_account_sync_at", utc_now_iso(), "broker_account_sync")
        self._log_state(
            "broker_account_sync",
            "KIS broker account sync complete",
            {
                "holding_count": int(snapshot.summary.get("holding_count", 0)),
                "approval_key_available": approval_key_available,
                "hts_id_configured": bool(client.config.hts_id),
                "canonical_snapshot_source": str(canonical_snapshot.get("source") or "kis_account_sync"),
                "reconcile_totals": reconcile_totals,
            },
            account_id=ACCOUNT_KIS_KR_PAPER,
        )
        touch("kis_account_sync_complete", {"canonical_source": str(canonical_snapshot.get("source") or "kis_account_sync")})
        return {
            "broker": "kis_mock",
            "enabled": True,
            "summary": snapshot.summary,
            "holding_count": int(snapshot.summary.get("holding_count", 0)),
            "approval_key_available": approval_key_available,
            "hts_id_configured": bool(client.config.hts_id),
            "canonical_snapshot": canonical_snapshot,
            "reconcile_totals": reconcile_totals,
            "account_id": ACCOUNT_KIS_KR_PAPER,
        }

    def _today_order_frame(self, client: KISPaperClient, order: pd.Series) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d")
        try:
            return client.get_daily_order_fills(
                start_date=today,
                end_date=today,
                product_code=extract_kis_code(str(order["symbol"])),
                order_no=str(order.get("broker_order_id") or ""),
            )
        except Exception:
            return pd.DataFrame()

    def _normalize_order_sync_state(self, order: pd.Series, rest_rows: pd.DataFrame) -> Dict[str, Any]:
        existing_filled = int(order["filled_qty"])
        requested_qty = int(order["requested_qty"])
        broker_order_id = str(order.get("broker_order_id") or "")
        if rest_rows.empty:
            return {"status": str(order["status"]), "filled_total": existing_filled, "fill_price": float(order["requested_price"]), "source": "none", "details": {}}

        matched = rest_rows
        if broker_order_id and "broker_order_id" in matched.columns:
            broker_match = matched.loc[matched["broker_order_id"].astype(str) == broker_order_id]
            if not broker_match.empty:
                matched = broker_match
        row = matched.iloc[0]
        fill_candidates = [
            pd.to_numeric(row.get("tot_ccld_qty"), errors="coerce"),
            pd.to_numeric(row.get("ccld_qty"), errors="coerce"),
            pd.to_numeric(row.get("tot_ccld_qty_sum"), errors="coerce"),
        ]
        filled_total = max([existing_filled] + [int(value) for value in fill_candidates if np.isfinite(value)])
        price_candidates = [
            pd.to_numeric(row.get("avg_prvs"), errors="coerce"),
            pd.to_numeric(row.get("ord_unpr"), errors="coerce"),
            pd.to_numeric(row.get("avg_cntr_prc"), errors="coerce"),
        ]
        fill_price = next((float(value) for value in price_candidates if np.isfinite(value) and float(value) > 0), float(order["requested_price"]))

        raw_text = " ".join(str(row.get(col, "")) for col in row.index).lower()
        if any(token in raw_text for token in ("거부", "reject", "rfus")):
            status = "rejected"
        elif any(token in raw_text for token in ("취소", "cancel")):
            status = "cancelled"
        elif filled_total >= requested_qty and requested_qty > 0:
            status = "filled"
        elif 0 < filled_total < requested_qty:
            status = "partially_filled"
        else:
            status = "acknowledged" if broker_order_id else "submitted"
        return {"status": status, "filled_total": filled_total, "fill_price": fill_price, "source": "rest_daily_ccld", "details": row.to_dict()}

    def _apply_state_transition(self, order: pd.Series, sync_state: Dict[str, Any]) -> Dict[str, int]:
        requested_qty = int(order["requested_qty"])
        existing_filled = int(order["filled_qty"])
        filled_total = min(int(sync_state.get("filled_total", existing_filled)), requested_qty)
        delta_fill = max(0, filled_total - existing_filled)
        status = str(sync_state.get("status") or order["status"])
        account_id = str(order.get("account_id") or ACCOUNT_KIS_KR_PAPER)
        totals = {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}

        if delta_fill > 0:
            final_status = "filled" if filled_total >= requested_qty else "partially_filled"
            if self.sim_broker.apply_external_fill(
                str(order["order_id"]),
                fill_qty=delta_fill,
                fill_price=float(sync_state.get("fill_price") or order["requested_price"]),
                raw_json={
                    **self._execution_metadata(str(order.get("strategy_version") or "")),
                    "broker": BROKER_MODE_KIS,
                    "account_id": account_id,
                    "broker_state": final_status,
                    "sync_source": sync_state.get("source"),
                    "sync_details": sync_state.get("details", {}),
                },
                final_status=final_status,
            ):
                totals["fills"] += 1
                self._log_state(final_status, "KIS fill confirmed", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, account_id=account_id)

        payload = {"broker": BROKER_MODE_KIS, "account_id": account_id, "broker_state": status, "sync_source": sync_state.get("source"), "sync_details": sync_state.get("details", {})}
        if status == "acknowledged":
            totals["acknowledged"] += 1
            self.repository.update_order(str(order["order_id"]), status="acknowledged", raw_json=payload)
        elif status == "rejected":
            totals["rejected"] += 1
            self.repository.update_order(str(order["order_id"]), status="rejected", error_message="broker_rejected", raw_json=payload)
            self._log_state("rejected", "KIS order rejected", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, account_id=account_id, level="ERROR")
        elif status == "cancelled":
            totals["cancelled"] += 1
            self.repository.update_order(str(order["order_id"]), status="cancelled", raw_json=payload)
            self._log_state("cancelled", "KIS order cancelled", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, account_id=account_id)
        elif status in {"submitted", "pending_fill", "filled", "partially_filled"} and delta_fill <= 0:
            self.repository.update_order(str(order["order_id"]), status=status, raw_json=payload)
            if status == "pending_fill":
                self._log_state("pending_fill", "KIS order pending fill", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, account_id=account_id)

        submitted_at = parse_utc_timestamp(str(self._parse_payload(order.get("raw_json")).get("submitted_at") or ""))
        if status in {"submitted", "acknowledged", "pending_fill"} and submitted_at is not None:
            timeout = timedelta(minutes=max(int(self.settings.broker.stale_submitted_order_timeout_minutes), 1))
            if datetime.now(timezone.utc) - submitted_at > timeout:
                totals["expired"] += 1
                self.repository.update_order(str(order["order_id"]), status="expired", raw_json={**payload, "broker_state": "expired"})
                self._log_state("expired", "KIS order expired while waiting for fill", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, account_id=account_id, level="ERROR")
        return totals

    def handle_websocket_execution_event(self, event: Dict[str, Any]) -> bool:
        broker_order_id = str(event.get("broker_order_id") or event.get("order_no") or "").strip()
        if not broker_order_id:
            return False
        orders = self.repository.open_orders(statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"), account_id=ACCOUNT_KIS_KR_PAPER)
        if orders.empty:
            return False
        matched = orders.loc[orders["broker_order_id"].astype(str) == broker_order_id]
        if matched.empty:
            return False
        order = matched.iloc[0]
        filled_qty = int(float(event.get("filled_qty") or event.get("fill_qty") or order["filled_qty"]))
        fill_price = float(event.get("fill_price") or order["requested_price"])
        status = str(event.get("status") or "pending_fill")
        self.repository.set_control_flag("kis_last_websocket_execution_at", utc_now_iso(), "websocket execution event")
        sync_state = {
            "status": status,
            "filled_total": filled_qty,
            "fill_price": fill_price,
            "source": "websocket",
            "details": dict(event),
        }
        self._apply_state_transition(order, sync_state)
        self._log_state(status, "KIS websocket execution event received", {"order_id": str(order["order_id"]), "broker_order_id": broker_order_id}, account_id=ACCOUNT_KIS_KR_PAPER)
        return True

    def sync_orders(
        self,
        market_data_service: MarketDataService | None = None,
        touch: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> Dict[str, int]:
        if not self.is_enabled():
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        active = self.repository.open_orders(statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"), account_id=ACCOUNT_KIS_KR_PAPER)
        if active.empty:
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        active = active.loc[
            active["raw_json"].fillna("{}").astype(str).map(lambda payload: self._parse_payload(payload).get("broker") == "kis_mock")
        ].copy()
        if active.empty:
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}

        client = self._client()
        holdings_snapshot = client.get_account_snapshot()
        holdings = self._holding_frame(holdings_snapshot)
        totals = {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        for _, order in active.iterrows():
            if touch is not None:
                touch("kis_order_sync", {"order_id": str(order["order_id"]), "symbol": str(order["symbol"])})
            sync_state = self._normalize_order_sync_state(order, self._today_order_frame(client, order))
            holdings_sync_state = self._infer_sync_state_from_holdings(order, holdings)
            if holdings_sync_state is not None and int(holdings_sync_state.get("filled_total", 0)) > int(sync_state.get("filled_total", 0)):
                sync_state = holdings_sync_state
            if sync_state["source"] == "none":
                sync_state["status"] = "pending_fill" if str(order["status"]) != "submitted" else "submitted"
            applied = self._apply_state_transition(order, sync_state)
            for key, value in applied.items():
                totals[key] += int(value)
        self.repository.set_control_flag("kis_last_order_sync_at", utc_now_iso(), "broker_order_sync")
        self._log_state("broker_order_sync", "KIS broker order sync complete", totals, account_id=ACCOUNT_KIS_KR_PAPER)
        return totals

    def process_open_orders(self, market_data_service: MarketDataService | None) -> int:
        return int(self.sync_orders(market_data_service).get("fills", 0))
