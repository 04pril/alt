from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperClient, extract_kis_code
from services.market_data_service import MarketDataService
from services.paper_broker import PaperBroker
from services.signal_engine import SignalDecision
from storage.models import OrderRecord
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

    def _log_state(self, event_type: str, message: str, details: Dict[str, Any], *, level: str = "INFO") -> None:
        self.repository.log_event(level, "kis_execution", event_type, message, details)

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

    def _insert_submitted_order(
        self,
        *,
        symbol: str,
        asset_type: str,
        timeframe: str,
        prediction_id: str | None,
        scan_id: str | None,
        side: str,
        quantity: int,
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
                order_type=self.settings.broker.default_order_type,
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
            )
        )
        return order_id

    def preflight_entry(self, signal: SignalDecision, quantity: int, market_data_service: MarketDataService) -> Dict[str, Any]:
        asset_type = str(signal.asset_type)
        if not self.is_enabled():
            return {"allowed": False, "reason": "kis_disabled", "broker": "kis_mock"}
        if not market_data_service.is_market_open(asset_type):
            return {"allowed": False, "reason": "market_closed", "broker": "kis_mock"}
        if self.settings.asset_schedules[asset_type].timeframe == "1d" and not market_data_service.is_pre_close_window(asset_type):
            return {"allowed": False, "reason": "outside_preclose_window", "broker": "kis_mock"}

        client = self._client()
        quote = client.get_quote(signal.symbol)
        orderbook = client.get_orderbook(signal.symbol)
        market_status = client.get_market_status(signal.symbol)
        if bool(market_status.get("is_halted")):
            return {"allowed": False, "reason": "market_halted", "broker": "kis_mock", "market_status": market_status}

        side = "buy" if signal.signal == "LONG" else "sell"
        exec_price = self._pick_execution_price(quote, orderbook, side)
        if not np.isfinite(exec_price) or exec_price <= 0:
            return {"allowed": False, "reason": "no_quote", "broker": "kis_mock", "quote": quote, "orderbook": orderbook}

        if not self.repository.active_entry_orders(symbol=signal.symbol, timeframe=signal.timeframe, asset_type=asset_type).empty:
            return {"allowed": False, "reason": "duplicate_pending_entry", "broker": "kis_mock"}
        latest_position = self.repository.latest_position_by_symbol(signal.symbol, signal.timeframe)
        if not latest_position.empty and str(latest_position.iloc[0].get("status")) == "open":
            return {"allowed": False, "reason": "already_holding_symbol", "broker": "kis_mock"}

        buying_power = client.get_buying_power(signal.symbol, order_price=exec_price)
        max_buy_qty = int(buying_power.get("cash_buy_qty") or buying_power.get("max_buy_qty") or 0)
        if max_buy_qty < int(quantity):
            return {"allowed": False, "reason": "insufficient_buying_power", "broker": "kis_mock", "buying_power": buying_power}

        return {
            "allowed": True,
            "reason": "ok",
            "broker": "kis_mock",
            "symbol_code": extract_kis_code(signal.symbol),
            "execution_price": float(exec_price),
            "quote": quote,
            "orderbook": orderbook,
            "market_status": market_status,
            "buying_power": buying_power,
            "account_summary": client.get_account_snapshot().summary,
        }

    def preflight_exit(self, position: pd.Series, market_data_service: MarketDataService) -> Dict[str, Any]:
        asset_type = str(position["asset_type"])
        if not self.is_enabled():
            return {"allowed": False, "reason": "kis_disabled", "broker": "kis_mock"}
        if not market_data_service.is_market_open(asset_type):
            return {"allowed": False, "reason": "market_closed", "broker": "kis_mock"}
        client = self._client()
        quote = client.get_quote(str(position["symbol"]))
        orderbook = client.get_orderbook(str(position["symbol"]))
        exec_price = self._pick_execution_price(quote, orderbook, "sell")
        if not np.isfinite(exec_price) or exec_price <= 0:
            return {"allowed": False, "reason": "no_quote", "broker": "kis_mock"}
        sellable = client.get_sellable_quantity(str(position["symbol"]))
        if str(position["side"]) == "LONG" and int(sellable.get("sellable_qty", 0)) < int(position["quantity"]):
            return {"allowed": False, "reason": "no_sellable_qty", "broker": "kis_mock", "sellable": sellable}
        return {"allowed": True, "reason": "ok", "broker": "kis_mock", "execution_price": float(exec_price), "sellable": sellable}

    def submit_entry_order_result(
        self,
        signal: SignalDecision,
        quantity: int,
        scan_id: str | None = None,
        *,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        preflight = self.preflight_entry(signal, quantity, market_data_service) if market_data_service is not None else {"allowed": True}
        if not preflight.get("allowed", False):
            return {
                "submitted": False,
                "status": "rejected",
                "reason": str(preflight.get("reason") or "rejected"),
                "broker": "kis_mock",
                "order_id": "",
            }

        client = self._client()
        symbol_code = str(preflight.get("symbol_code") or extract_kis_code(signal.symbol))
        baseline = self._holding_baseline(client, symbol_code)
        requested_price = float(preflight.get("execution_price") or signal.current_price)
        order_id = self._insert_submitted_order(
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            timeframe=signal.timeframe,
            prediction_id=signal.prediction_id,
            scan_id=scan_id or signal.scan_id,
            side="buy" if signal.signal == "LONG" else "sell",
            quantity=int(quantity),
            requested_price=requested_price,
            strategy_version=signal.strategy_version,
            reason="entry",
            payload={
                "broker": "kis_mock",
                "symbol_code": symbol_code,
                "baseline_qty": baseline["quantity"],
                "baseline_avg_price": baseline["avg_price"],
                "last_seen_broker_qty": baseline["quantity"],
                "last_seen_broker_avg_price": baseline["avg_price"],
                "expected_risk": signal.expected_risk,
                "stop_level": signal.stop_level,
                "take_level": signal.take_level,
                "max_holding_until": (
                    pd.Timestamp.now(tz="UTC")
                    + self.sim_broker._timeframe_delta(signal.timeframe) * int(self.settings.strategy.max_holding_bars)
                ).isoformat(),
                "strategy_version": signal.strategy_version,
                "broker_state": "submit_requested",
                "preflight": preflight,
                "hts_id": client.config.hts_id,
            },
        )
        self._log_state("submit_requested", "KIS entry submit requested", {"order_id": order_id, "symbol": signal.symbol, "quantity": int(quantity)})
        try:
            result = client.place_cash_order(
                symbol_or_code=signal.symbol,
                side="buy" if signal.signal == "LONG" else "sell",
                quantity=int(quantity),
                order_type=self.settings.broker.default_order_type,
                price=requested_price,
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": "kis_mock", "broker_state": "rejected", "error_stage": "submit"},
            )
            self._log_state("rejected", "KIS broker rejected entry", {"order_id": order_id, "symbol": signal.symbol, "reason": "broker_rejected", "error": str(exc)}, level="ERROR")
            return {"submitted": False, "status": "rejected", "reason": "broker_rejected", "broker": "kis_mock", "order_id": order_id}

        status = "acknowledged" if result.get("order_no") else "submitted"
        self.repository.update_order(
            order_id,
            status=status,
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": "kis_mock",
                "broker_state": status,
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        self._log_state("submitted", "KIS entry submitted", {"order_id": order_id, "symbol": signal.symbol, "broker_order_id": str(result.get("order_no") or "")})
        if status == "acknowledged":
            self._log_state("acknowledged", "KIS entry acknowledged", {"order_id": order_id, "symbol": signal.symbol, "broker_order_id": str(result.get("order_no") or "")})
        return {"submitted": True, "status": status, "reason": "ok", "broker": "kis_mock", "order_id": order_id, "broker_order_id": str(result.get("order_no") or "")}

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str:
        return str(self.submit_entry_order_result(signal, quantity, scan_id).get("order_id") or "")

    def submit_exit_order_result(
        self,
        position: pd.Series,
        reason: str,
        *,
        market_data_service: MarketDataService | None = None,
    ) -> Dict[str, Any]:
        preflight = self.preflight_exit(position, market_data_service) if market_data_service is not None else {"allowed": True}
        if not preflight.get("allowed", False):
            return {"submitted": False, "status": "rejected", "reason": str(preflight.get("reason") or "rejected"), "order_id": ""}

        client = self._client()
        symbol = str(position["symbol"])
        symbol_code = extract_kis_code(symbol)
        baseline = self._holding_baseline(client, symbol_code)
        order_id = self._insert_submitted_order(
            symbol=symbol,
            asset_type=str(position["asset_type"]),
            timeframe=str(position["timeframe"]),
            prediction_id=str(position.get("prediction_id") or ""),
            scan_id=None,
            side="sell" if str(position["side"]) == "LONG" else "buy",
            quantity=int(position["quantity"]),
            requested_price=float(preflight.get("execution_price") or position["mark_price"]),
            strategy_version=str(position["strategy_version"]),
            reason=reason,
            payload={
                "broker": "kis_mock",
                "symbol_code": symbol_code,
                "baseline_qty": baseline["quantity"],
                "baseline_avg_price": baseline["avg_price"],
                "last_seen_broker_qty": baseline["quantity"],
                "last_seen_broker_avg_price": baseline["avg_price"],
                "position_id": str(position.get("position_id") or ""),
                "broker_state": "submit_requested",
                "preflight": preflight,
                "hts_id": client.config.hts_id,
            },
        )
        self._log_state("submit_requested", "KIS exit submit requested", {"order_id": order_id, "symbol": symbol, "quantity": int(position["quantity"])})
        try:
            result = client.place_cash_order(
                symbol_or_code=symbol,
                side="sell" if str(position["side"]) == "LONG" else "buy",
                quantity=int(position["quantity"]),
                order_type=self.settings.broker.default_order_type,
                price=float(preflight.get("execution_price") or position["mark_price"]),
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": "kis_mock", "broker_state": "rejected", "error_stage": "submit"},
            )
            self._log_state("rejected", "KIS broker rejected exit", {"order_id": order_id, "symbol": symbol, "reason": "broker_rejected", "error": str(exc)}, level="ERROR")
            return {"submitted": False, "status": "rejected", "reason": "broker_rejected", "order_id": order_id}

        status = "acknowledged" if result.get("order_no") else "submitted"
        self.repository.update_order(
            order_id,
            status=status,
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": "kis_mock",
                "broker_state": status,
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        self._log_state("submitted", "KIS exit submitted", {"order_id": order_id, "symbol": symbol, "broker_order_id": str(result.get("order_no") or "")})
        if status == "acknowledged":
            self._log_state("acknowledged", "KIS exit acknowledged", {"order_id": order_id, "symbol": symbol, "broker_order_id": str(result.get("order_no") or "")})
        return {"submitted": True, "status": status, "reason": "ok", "order_id": order_id, "broker_order_id": str(result.get("order_no") or "")}

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        return str(self.submit_exit_order_result(position, reason).get("order_id") or "")

    def sync_account(self) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"broker": "kis_mock", "enabled": False}
        client = self._client()
        snapshot = client.get_account_snapshot()
        approval_key_available = False
        try:
            approval_key_available = bool(client.get_websocket_approval_key())
        except Exception:
            approval_key_available = False
        self.repository.set_control_flag("kis_last_account_sync_at", utc_now_iso(), "broker_account_sync")
        self._log_state(
            "broker_account_sync",
            "KIS broker account sync complete",
            {
                "holding_count": int(snapshot.summary.get("holding_count", 0)),
                "approval_key_available": approval_key_available,
                "hts_id_configured": bool(client.config.hts_id),
            },
        )
        return {
            "broker": "kis_mock",
            "enabled": True,
            "summary": snapshot.summary,
            "holding_count": int(snapshot.summary.get("holding_count", 0)),
            "approval_key_available": approval_key_available,
            "hts_id_configured": bool(client.config.hts_id),
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
        totals = {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}

        if delta_fill > 0:
            final_status = "filled" if filled_total >= requested_qty else "partially_filled"
            if self.sim_broker.apply_external_fill(
                str(order["order_id"]),
                fill_qty=delta_fill,
                fill_price=float(sync_state.get("fill_price") or order["requested_price"]),
                raw_json={"broker": "kis_mock", "broker_state": final_status, "sync_source": sync_state.get("source"), "sync_details": sync_state.get("details", {})},
                final_status=final_status,
            ):
                totals["fills"] += 1
                self._log_state(final_status, "KIS fill confirmed", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")})

        payload = {"broker": "kis_mock", "broker_state": status, "sync_source": sync_state.get("source"), "sync_details": sync_state.get("details", {})}
        if status == "acknowledged":
            totals["acknowledged"] += 1
            self.repository.update_order(str(order["order_id"]), status="acknowledged", raw_json=payload)
        elif status == "rejected":
            totals["rejected"] += 1
            self.repository.update_order(str(order["order_id"]), status="rejected", error_message="broker_rejected", raw_json=payload)
            self._log_state("rejected", "KIS order rejected", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, level="ERROR")
        elif status == "cancelled":
            totals["cancelled"] += 1
            self.repository.update_order(str(order["order_id"]), status="cancelled", raw_json=payload)
            self._log_state("cancelled", "KIS order cancelled", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")})
        elif status in {"submitted", "pending_fill", "filled", "partially_filled"} and delta_fill <= 0:
            self.repository.update_order(str(order["order_id"]), status=status, raw_json=payload)
            if status == "pending_fill":
                self._log_state("pending_fill", "KIS order pending fill", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")})

        submitted_at = parse_utc_timestamp(str(self._parse_payload(order.get("raw_json")).get("submitted_at") or ""))
        if status in {"submitted", "acknowledged", "pending_fill"} and submitted_at is not None:
            timeout = timedelta(minutes=max(int(self.settings.broker.stale_submitted_order_timeout_minutes), 1))
            if datetime.now(timezone.utc) - submitted_at > timeout:
                totals["expired"] += 1
                self.repository.update_order(str(order["order_id"]), status="expired", raw_json={**payload, "broker_state": "expired"})
                self._log_state("expired", "KIS order expired while waiting for fill", {"order_id": str(order["order_id"]), "broker_order_id": str(order.get("broker_order_id") or "")}, level="ERROR")
        return totals

    def handle_websocket_execution_event(self, event: Dict[str, Any]) -> bool:
        broker_order_id = str(event.get("broker_order_id") or event.get("order_no") or "").strip()
        if not broker_order_id:
            return False
        orders = self.repository.open_orders(statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"))
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
        self._log_state(status, "KIS websocket execution event received", {"order_id": str(order["order_id"]), "broker_order_id": broker_order_id})
        return True

    def sync_orders(
        self,
        market_data_service: MarketDataService | None = None,
        touch: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> Dict[str, int]:
        if not self.is_enabled():
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        active = self.repository.open_orders(statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"))
        if active.empty:
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        active = active.loc[
            active["raw_json"].fillna("{}").astype(str).map(lambda payload: self._parse_payload(payload).get("broker") == "kis_mock")
        ].copy()
        if active.empty:
            return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}

        client = self._client()
        totals = {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        for _, order in active.iterrows():
            if touch is not None:
                touch("kis_order_sync", {"order_id": str(order["order_id"]), "symbol": str(order["symbol"])})
            sync_state = self._normalize_order_sync_state(order, self._today_order_frame(client, order))
            if sync_state["source"] == "none":
                sync_state["status"] = "pending_fill" if str(order["status"]) != "submitted" else "submitted"
            applied = self._apply_state_transition(order, sync_state)
            for key, value in applied.items():
                totals[key] += int(value)
        self.repository.set_control_flag("kis_last_order_sync_at", utc_now_iso(), "broker_order_sync")
        self._log_state("broker_order_sync", "KIS broker order sync complete", totals)
        return totals

    def process_open_orders(self, market_data_service: MarketDataService | None) -> int:
        return int(self.sync_orders(market_data_service).get("fills", 0))
