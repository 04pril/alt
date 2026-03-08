from __future__ import annotations

import json
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperClient, KISPaperError, extract_kis_code
from services.paper_broker import PaperBroker
from services.signal_engine import SignalDecision
from storage.models import OrderRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


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

    def _looks_like_auth_error(self, message: str) -> bool:
        text = str(message or "").lower()
        return any(
            token in text
            for token in (
                "token",
                "oauth",
                "authorization",
                "hashkey",
                "appkey",
                "appsecret",
                "401",
                "403",
                "인증",
                "토큰",
            )
        )

    def _default_kr_probe_symbol(self) -> str:
        for asset_type, schedule in self.settings.asset_schedules.items():
            if str(schedule.timezone) != "Asia/Seoul" or str(schedule.timeframe) != "1d":
                continue
            universe = self.settings.universes.get(asset_type)
            if universe is None:
                continue
            for symbol in list(universe.watchlist) + list(universe.top_universe):
                text = str(symbol or "").strip()
                if text:
                    return text
        return "005930.KS"

    def _record_kis_error(self, *, flag_prefix: str, message: str, notes: str) -> None:
        now_iso = utc_now_iso()
        self.repository.set_control_flag(f"{flag_prefix}_error_at", now_iso, notes)
        self.repository.set_control_flag(f"{flag_prefix}_error", str(message), notes)
        if self._looks_like_auth_error(message):
            self.repository.set_control_flag("kis_last_auth_error_at", now_iso, notes)
            self.repository.set_control_flag("kis_last_auth_error", str(message), notes)

    def _clear_kis_error(self, *, flag_prefix: str, notes: str) -> None:
        self.repository.set_control_flag(f"{flag_prefix}_error_at", "", notes)
        self.repository.set_control_flag(f"{flag_prefix}_error", "", notes)

    def _probe_buying_power(self, client: KISPaperClient) -> Dict[str, Any]:
        symbol = self._default_kr_probe_symbol()
        quote = client.get_quote(symbol)
        order_price = float(quote.get("current_price", 0.0) or 0.0)
        if not np.isfinite(order_price) or order_price <= 0:
            raise KISPaperError(f"매수가능조회 probe용 현재가가 유효하지 않습니다: {symbol}")
        buying_power = client.get_buying_power(symbol, order_price=order_price)
        now_iso = utc_now_iso()
        self.repository.set_control_flag("kis_last_buying_power_success_at", now_iso, "broker_account_sync")
        self.repository.set_control_flag("kis_last_buying_power_symbol", symbol, "broker_account_sync")
        self.repository.set_control_flag("kis_last_buying_power_status", "ok", "broker_account_sync")
        self._clear_kis_error(flag_prefix="kis_last_buying_power", notes="broker_account_sync")
        return {"symbol": symbol, "order_price": order_price, **buying_power}

    def is_enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        try:
            client = self.client_factory()
            self._enabled = bool(client.config.is_paper)
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
        record = OrderRecord(
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
        self.repository.insert_order(record)
        return order_id

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str:
        client = self._client()
        symbol_code = extract_kis_code(signal.symbol)
        baseline = self._holding_baseline(client, symbol_code)
        payload = {
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
            "broker_state": "submitted",
        }
        order_id = self._insert_submitted_order(
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            timeframe=signal.timeframe,
            prediction_id=signal.prediction_id,
            scan_id=scan_id or signal.scan_id,
            side="buy" if signal.signal == "LONG" else "sell",
            quantity=int(quantity),
            requested_price=float(signal.current_price),
            strategy_version=signal.strategy_version,
            reason="entry",
            payload=payload,
        )
        try:
            result = client.place_cash_order(
                symbol_or_code=signal.symbol,
                side="buy" if signal.signal == "LONG" else "sell",
                quantity=int(quantity),
                order_type=self.settings.broker.default_order_type,
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": "kis_mock", "broker_state": "rejected"},
            )
            return order_id
        self.repository.update_order(
            order_id,
            status="acknowledged" if result.get("order_no") else "submitted",
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": "kis_mock",
                "broker_state": "acknowledged" if result.get("order_no") else "submitted",
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        return order_id

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        client = self._client()
        symbol = str(position["symbol"])
        symbol_code = extract_kis_code(symbol)
        baseline = self._holding_baseline(client, symbol_code)
        payload = {
            "broker": "kis_mock",
            "symbol_code": symbol_code,
            "baseline_qty": baseline["quantity"],
            "baseline_avg_price": baseline["avg_price"],
            "last_seen_broker_qty": baseline["quantity"],
            "last_seen_broker_avg_price": baseline["avg_price"],
            "position_id": str(position.get("position_id") or ""),
            "broker_state": "submitted",
        }
        side = "sell" if str(position["side"]) == "LONG" else "buy"
        order_id = self._insert_submitted_order(
            symbol=symbol,
            asset_type=str(position["asset_type"]),
            timeframe=str(position["timeframe"]),
            prediction_id=str(position.get("prediction_id") or ""),
            scan_id=None,
            side=side,
            quantity=int(position["quantity"]),
            requested_price=float(position["mark_price"]),
            strategy_version=str(position["strategy_version"]),
            reason=reason,
            payload=payload,
        )
        try:
            result = client.place_cash_order(
                symbol_or_code=symbol,
                side=side,
                quantity=int(position["quantity"]),
                order_type=self.settings.broker.default_order_type,
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": "kis_mock", "broker_state": "rejected"},
            )
            return order_id
        self.repository.update_order(
            order_id,
            status="acknowledged" if result.get("order_no") else "submitted",
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": "kis_mock",
                "broker_state": "acknowledged" if result.get("order_no") else "submitted",
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        return order_id

    def submit_manual_order(
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
        raw_metadata: Dict[str, Any] | None = None,
    ) -> str:
        client = self._client()
        symbol_code = extract_kis_code(symbol)
        baseline = self._holding_baseline(client, symbol_code)
        payload = {
            "broker": "kis_mock",
            "symbol_code": symbol_code,
            "baseline_qty": baseline["quantity"],
            "baseline_avg_price": baseline["avg_price"],
            "last_seen_broker_qty": baseline["quantity"],
            "last_seen_broker_avg_price": baseline["avg_price"],
            "broker_state": "submitted",
            **(raw_metadata or {}),
        }
        order_id = self._insert_submitted_order(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            prediction_id=prediction_id,
            scan_id=scan_id,
            side=side,
            quantity=int(quantity),
            requested_price=float(requested_price),
            strategy_version=strategy_version,
            reason=reason,
            payload=payload,
        )
        try:
            result = client.place_cash_order(
                symbol_or_code=symbol,
                side=side,
                quantity=int(quantity),
                order_type=order_type,
                price=requested_price if order_type == "limit" else None,
            )
        except Exception as exc:
            self.repository.update_order(
                order_id,
                status="rejected",
                error_message=str(exc),
                raw_json={"broker": "kis_mock", "broker_state": "rejected"},
            )
            return order_id
        self.repository.update_order(
            order_id,
            status="acknowledged" if result.get("order_no") else "submitted",
            broker_order_id=str(result.get("order_no") or ""),
            raw_json={
                "broker": "kis_mock",
                "broker_state": "acknowledged" if result.get("order_no") else "submitted",
                "broker_message": str(result.get("message") or ""),
                "submitted_at": str(result.get("requested_at") or utc_now_iso()),
            },
        )
        return order_id

    def sync_account(self, touch=None) -> Dict[str, Any]:
        touch = touch or (lambda *args, **kwargs: None)
        if not self.is_enabled():
            touch("kis_account_sync_skipped", {"enabled": False})
            return {"broker": "kis_mock", "enabled": False, "holding_count": 0}
        touch("kis_account_sync_request", {"enabled": True})
        client = self._client()
        try:
            snapshot = client.get_account_snapshot()
        except Exception as exc:
            self._record_kis_error(flag_prefix="kis_last_account_sync", message=str(exc), notes="broker_account_sync")
            raise
        result = {
            "broker": "kis_mock",
            "enabled": True,
            "cash": float(snapshot.summary.get("cash", 0.0) or 0.0),
            "total_eval": float(snapshot.summary.get("total_eval", 0.0) or 0.0),
            "holding_count": int(snapshot.summary.get("holding_count", 0) or 0),
        }
        now_iso = utc_now_iso()
        self.repository.set_control_flag("kis_last_account_sync_at", now_iso, "broker_account_sync")
        self.repository.set_control_flag("kis_last_account_sync_status", "ok", "broker_account_sync")
        self._clear_kis_error(flag_prefix="kis_last_account_sync", notes="broker_account_sync")
        self.repository.set_control_flag("kis_last_auth_error_at", "", "broker_account_sync")
        self.repository.set_control_flag("kis_last_auth_error", "", "broker_account_sync")
        try:
            probe = self._probe_buying_power(client)
            result["buying_power_probe"] = probe
        except Exception as exc:
            self._record_kis_error(flag_prefix="kis_last_buying_power", message=str(exc), notes="broker_account_sync")
            result["buying_power_probe_error"] = str(exc)
        touch("kis_account_sync_snapshot", result)
        return result

    def process_open_orders(self, market_data_service, touch=None) -> int:
        touch = touch or (lambda *args, **kwargs: None)
        if not self.is_enabled():
            return 0
        active = self.repository.open_orders(statuses=("submitted", "acknowledged", "pending_fill", "partially_filled"))
        if active.empty:
            return 0
        active = active.loc[
            active["raw_json"].fillna("{}").astype(str).map(lambda payload: self._parse_payload(payload).get("broker") == "kis_mock")
        ].copy()
        if active.empty:
            return 0

        client = self._client()
        snapshot = client.get_account_snapshot()
        holdings = snapshot.holdings.copy()
        if holdings.empty:
            holdings = pd.DataFrame(columns=["symbol_code", "quantity", "avg_price"])
        fills_applied = 0
        for _, order in active.iterrows():
            touch("kis_order_sync", {"order_id": str(order["order_id"]), "symbol": str(order["symbol"])})
            payload = self._parse_payload(order.get("raw_json"))
            symbol_code = str(payload.get("symbol_code") or extract_kis_code(str(order["symbol"])))
            matched = holdings.loc[holdings["symbol_code"].astype(str) == symbol_code]
            current_qty = (
                float(pd.to_numeric(matched.iloc[0].get("quantity", matched.iloc[0].get("보유수량")), errors="coerce") or 0.0)
                if not matched.empty
                else 0.0
            )
            current_avg = (
                float(pd.to_numeric(matched.iloc[0].get("avg_price", matched.iloc[0].get("매입평균가")), errors="coerce"))
                if not matched.empty
                else float("nan")
            )
            baseline_qty = float(payload.get("baseline_qty", 0.0) or 0.0)
            last_seen_qty = float(payload.get("last_seen_broker_qty", baseline_qty) or baseline_qty)
            last_seen_avg = float(payload.get("last_seen_broker_avg_price", payload.get("baseline_avg_price", float("nan"))))
            requested_qty = int(order["requested_qty"])
            existing_filled = int(order["filled_qty"])

            if str(order["side"]) == "buy":
                broker_filled_total = max(0.0, current_qty - baseline_qty)
                observed_delta = max(0.0, current_qty - last_seen_qty)
                if observed_delta > 0:
                    if last_seen_qty <= 0 or not np.isfinite(last_seen_avg):
                        inferred_price = current_avg if np.isfinite(current_avg) else float(order["requested_price"])
                    else:
                        inferred_price = ((current_avg * current_qty) - (last_seen_avg * last_seen_qty)) / max(observed_delta, 1e-9)
                else:
                    inferred_price = float(order["requested_price"])
            else:
                broker_filled_total = max(0.0, baseline_qty - current_qty)
                observed_delta = max(0.0, last_seen_qty - current_qty)
                inferred_price = float(order["requested_price"])
                if not np.isfinite(inferred_price):
                    try:
                        inferred_price = float(client.get_quote(str(order["symbol"])).get("current_price"))
                    except Exception:
                        inferred_price = 0.0

            broker_filled_total = min(float(requested_qty), broker_filled_total)
            delta_fill = int(max(0.0, broker_filled_total - existing_filled))
            if delta_fill > 0:
                if self.sim_broker.apply_external_fill(
                    str(order["order_id"]),
                    fill_qty=delta_fill,
                    fill_price=float(inferred_price),
                    raw_json={
                        "broker": "kis_mock",
                        "broker_state": "filled" if broker_filled_total >= requested_qty else "partially_filled",
                    },
                    final_status="filled" if broker_filled_total >= requested_qty else "partially_filled",
                ):
                    fills_applied += 1

            next_status = "filled" if broker_filled_total >= requested_qty else "pending_fill"
            if 0 < broker_filled_total < requested_qty:
                next_status = "partially_filled"
            self.repository.update_order(
                str(order["order_id"]),
                status=next_status,
                broker_order_id=str(order.get("broker_order_id") or ""),
                raw_json={
                    "broker": "kis_mock",
                    "broker_state": next_status,
                    "last_seen_broker_qty": current_qty,
                    "last_seen_broker_avg_price": current_avg,
                },
            )
        return fills_applied
