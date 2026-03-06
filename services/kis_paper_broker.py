from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kis_paper import KISPaperClient, KISPaperError, extract_kis_code
from services.market_data_service import MarketDataService
from services.signal_engine import SignalDecision
from storage.models import AccountSnapshotRecord, FillRecord, OrderRecord, PositionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class KISPaperBroker:
    mode = "kis_paper"
    name = "KISPaperBroker"

    def __init__(self, settings: RuntimeSettings, repository: TradingRepository, client: KISPaperClient | None = None):
        self.settings = settings
        self.repository = repository
        self._client = client
        self.asset_types = tuple(settings.asset_types_for_broker_mode(self.mode))
        self.default_asset_type = self.asset_types[0] if self.asset_types else "한국주식"

    @property
    def client(self) -> KISPaperClient:
        if self._client is None:
            self._client = KISPaperClient(config_path=self.settings.broker.kis_config_path)
        return self._client

    def supports_asset_type(self, asset_type: str) -> bool:
        return asset_type in self.asset_types

    def supports_short(self, asset_type: str) -> bool:
        return self.settings.broker_supports_short(self.mode)

    def ensure_account_initialized(self) -> None:
        self.sync_state(force=True)

    def _resolve_market_symbol(self, symbol_or_code: str) -> str:
        raw = str(symbol_or_code).strip().upper()
        if raw.endswith((".KS", ".KQ")):
            return raw
        code = extract_kis_code(raw)
        existing = self.repository.open_positions()
        if not existing.empty:
            matched = existing[existing["symbol"].astype(str).str.startswith(code)]
            if not matched.empty:
                return str(matched.iloc[0]["symbol"])
        recent_orders = self.repository.recent_orders(limit=500)
        if not recent_orders.empty:
            matched = recent_orders[recent_orders["symbol"].astype(str).str.startswith(code)]
            if not matched.empty:
                return str(matched.iloc[0]["symbol"])
        quote = self.client.get_quote(code)
        market_name = str(quote.get("market_name") or "").upper()
        suffix = ".KQ" if "KOSDAQ" in market_name or "코스닥" in market_name else ".KS"
        return f"{code}{suffix}"

    def _quote_price(self, symbol: str) -> float:
        quote = self.client.get_quote(symbol)
        price = float(quote.get("current_price", np.nan))
        if not np.isfinite(price) or price <= 0:
            raise KISPaperError(f"invalid KIS quote for {symbol}")
        return price

    def submit_entry_order(self, signal: SignalDecision, quantity: int, scan_id: str | None = None) -> str:
        if signal.signal == "SHORT":
            raise KISPaperError("KIS paper broker does not support short orders for Korean equities")
        quote = self.client.get_quote(signal.symbol)
        quote_price = float(quote.get("current_price", np.nan))
        response = self.client.place_cash_order(signal.symbol, side="buy", quantity=int(quantity), order_type=self.settings.broker.default_order_type)
        order_id = make_id("ord")
        broker_order_id = str(response.get("order_no") or "")
        now_iso = utc_now_iso()
        self.repository.insert_order(
            OrderRecord(
                order_id=order_id,
                created_at=now_iso,
                updated_at=now_iso,
                prediction_id=signal.prediction_id,
                scan_id=scan_id or signal.scan_id,
                symbol=signal.symbol,
                asset_type=signal.asset_type,
                timeframe=signal.timeframe,
                side="buy",
                order_type=self.settings.broker.default_order_type,
                requested_qty=int(quantity),
                filled_qty=int(quantity),
                remaining_qty=0,
                requested_price=float(quote_price),
                limit_price=np.nan,
                status="filled",
                fees_estimate=float(quote_price * quantity * self.settings.broker.fee_bps / 10000.0),
                slippage_bps=float(self.settings.broker.base_slippage_bps),
                retry_count=0,
                strategy_version=signal.strategy_version,
                reason="entry",
                broker_order_id=broker_order_id,
                raw_json=json.dumps(
                    {
                        "broker": self.mode,
                        "response": response,
                        "quote": quote,
                        "stop_level": signal.stop_level,
                        "take_level": signal.take_level,
                        "expected_risk": signal.expected_risk,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        self.repository.insert_fill(
            FillRecord(
                fill_id=f"fill_{order_id}",
                created_at=now_iso,
                order_id=order_id,
                symbol=signal.symbol,
                side="buy",
                quantity=int(quantity),
                fill_price=float(quote_price),
                fees=float(quote_price * quantity * self.settings.broker.fee_bps / 10000.0),
                slippage_bps=float(self.settings.broker.base_slippage_bps),
                status="filled",
                raw_json=json.dumps({"broker_response": response, "quote": quote}, ensure_ascii=False),
            )
        )
        self.sync_state(force=True)
        return order_id

    def submit_exit_order(self, position: pd.Series, reason: str) -> str:
        if str(position["side"]) != "LONG":
            raise KISPaperError("KIS paper broker exit path only supports LONG positions")
        quote = self.client.get_quote(str(position["symbol"]))
        quote_price = float(quote.get("current_price", np.nan))
        quantity = int(position["quantity"])
        response = self.client.place_cash_order(str(position["symbol"]), side="sell", quantity=quantity, order_type=self.settings.broker.default_order_type)
        order_id = make_id("ord")
        broker_order_id = str(response.get("order_no") or "")
        now_iso = utc_now_iso()
        self.repository.insert_order(
            OrderRecord(
                order_id=order_id,
                created_at=now_iso,
                updated_at=now_iso,
                prediction_id=str(position.get("prediction_id") or ""),
                scan_id=None,
                symbol=str(position["symbol"]),
                asset_type=str(position["asset_type"]),
                timeframe=str(position["timeframe"]),
                side="sell",
                order_type=self.settings.broker.default_order_type,
                requested_qty=quantity,
                filled_qty=quantity,
                remaining_qty=0,
                requested_price=float(quote_price),
                limit_price=np.nan,
                status="filled",
                fees_estimate=float(quote_price * quantity * self.settings.broker.fee_bps / 10000.0),
                slippage_bps=float(self.settings.broker.base_slippage_bps),
                retry_count=0,
                strategy_version=str(position["strategy_version"]),
                reason=reason,
                broker_order_id=broker_order_id,
                raw_json=json.dumps({"broker": self.mode, "response": response, "quote": quote}, ensure_ascii=False),
            )
        )
        self.repository.insert_fill(
            FillRecord(
                fill_id=f"fill_{order_id}",
                created_at=now_iso,
                order_id=order_id,
                symbol=str(position["symbol"]),
                side="sell",
                quantity=quantity,
                fill_price=float(quote_price),
                fees=float(quote_price * quantity * self.settings.broker.fee_bps / 10000.0),
                slippage_bps=float(self.settings.broker.base_slippage_bps),
                status="filled",
                raw_json=json.dumps({"broker_response": response, "quote": quote}, ensure_ascii=False),
            )
        )
        self.sync_state(force=True)
        return order_id

    def _reconcile_positions_from_snapshot(self, snapshot) -> int:
        holdings = snapshot.holdings.copy()
        now_iso = utc_now_iso()
        existing = self.repository.open_positions()
        existing = existing[existing["asset_type"] == self.default_asset_type] if not existing.empty else existing
        seen_symbols: set[str] = set()
        updated = 0

        if not holdings.empty:
            for row in holdings.to_dict("records"):
                quantity = int(pd.to_numeric(row.get("holding_qty"), errors="coerce") or 0)
                if quantity <= 0:
                    continue
                code = str(row.get("symbol_code") or "").strip()
                symbol = self._resolve_market_symbol(code)
                seen_symbols.add(symbol)
                avg_price = float(pd.to_numeric(row.get("avg_price"), errors="coerce"))
                market_price = float(pd.to_numeric(row.get("market_price"), errors="coerce"))
                pnl = float(pd.to_numeric(row.get("pnl"), errors="coerce"))
                existing_symbol = existing[existing["symbol"].astype(str) == symbol]
                position_id = str(existing_symbol.iloc[0]["position_id"]) if not existing_symbol.empty else make_id("pos")
                highest_price = max(float(existing_symbol.iloc[0]["highest_price"]), market_price) if not existing_symbol.empty else market_price
                lowest_price = min(float(existing_symbol.iloc[0]["lowest_price"]), market_price) if not existing_symbol.empty else market_price
                self.repository.upsert_position(
                    PositionRecord(
                        position_id=position_id,
                        created_at=str(existing_symbol.iloc[0]["created_at"]) if not existing_symbol.empty else now_iso,
                        updated_at=now_iso,
                        closed_at=None,
                        prediction_id=str(existing_symbol.iloc[0].get("prediction_id") or "") if not existing_symbol.empty else None,
                        symbol=symbol,
                        asset_type=self.default_asset_type,
                        timeframe="1d",
                        side="LONG",
                        status="open",
                        quantity=quantity,
                        entry_price=avg_price,
                        mark_price=market_price,
                        stop_loss=float(existing_symbol.iloc[0]["stop_loss"]) if not existing_symbol.empty else np.nan,
                        take_profit=float(existing_symbol.iloc[0]["take_profit"]) if not existing_symbol.empty else np.nan,
                        trailing_stop=float(existing_symbol.iloc[0]["trailing_stop"]) if not existing_symbol.empty else np.nan,
                        highest_price=highest_price,
                        lowest_price=lowest_price,
                        unrealized_pnl=pnl,
                        realized_pnl=float(existing_symbol.iloc[0]["realized_pnl"]) if not existing_symbol.empty else 0.0,
                        expected_risk=float(existing_symbol.iloc[0]["expected_risk"]) if not existing_symbol.empty else np.nan,
                        exposure_value=market_price * quantity,
                        max_holding_until=str(existing_symbol.iloc[0]["max_holding_until"]) if not existing_symbol.empty else now_iso,
                        strategy_version=str(existing_symbol.iloc[0]["strategy_version"]) if not existing_symbol.empty else self.settings.strategy.strategy_version,
                        cooldown_until=None,
                        notes="kis_sync",
                    )
                )
                updated += 1

        if not existing.empty:
            for _, position in existing.iterrows():
                symbol = str(position["symbol"])
                if symbol in seen_symbols:
                    continue
                self.repository.upsert_position(
                    PositionRecord(
                        **{
                            **position.to_dict(),
                            "updated_at": now_iso,
                            "closed_at": now_iso,
                            "status": "closed",
                            "quantity": 0,
                            "mark_price": float(position["mark_price"]),
                            "unrealized_pnl": 0.0,
                            "realized_pnl": float(position["realized_pnl"]) + float(position["unrealized_pnl"]),
                            "exposure_value": 0.0,
                            "cooldown_until": now_iso,
                            "notes": "kis_sync_closed",
                        }
                    )
                )
                updated += 1
        return updated

    def _record_account_snapshot(self, snapshot) -> Dict[str, Any]:
        summary = snapshot.summary
        cash = float(summary.get("cash", 0.0) or 0.0)
        stock_eval = float(summary.get("stock_eval", 0.0) or 0.0)
        total_eval = float(summary.get("total_eval", cash + stock_eval) or (cash + stock_eval))
        pnl = float(summary.get("pnl", 0.0) or 0.0)
        account = {
            "cash": cash,
            "equity": total_eval,
            "gross_exposure": abs(stock_eval),
            "net_exposure": stock_eval,
            "realized_pnl": 0.0,
            "unrealized_pnl": pnl,
            "daily_pnl": self.repository.recent_closed_realized_pnl(str(pd.Timestamp.utcnow().date())),
            "drawdown_pct": 0.0,
            "open_positions": int(len(snapshot.holdings.index)) if snapshot.holdings is not None else 0,
            "open_orders": int(len(self.repository.open_orders().query("asset_type == @self.default_asset_type"))) if not self.repository.open_orders().empty else 0,
        }
        previous = self.repository.load_account_snapshots(limit=1000)
        previous = previous[previous["source"] == "kis_paper"] if not previous.empty else previous
        peak = float(pd.to_numeric(previous.get("equity"), errors="coerce").dropna().max()) if not previous.empty else total_eval
        peak = max(peak, total_eval)
        account["drawdown_pct"] = (total_eval / peak - 1.0) * 100.0 if peak > 0 else 0.0
        self.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id=make_id("snap"),
                created_at=utc_now_iso(),
                cash=account["cash"],
                equity=account["equity"],
                gross_exposure=account["gross_exposure"],
                net_exposure=account["net_exposure"],
                realized_pnl=account["realized_pnl"],
                unrealized_pnl=account["unrealized_pnl"],
                daily_pnl=account["daily_pnl"],
                drawdown_pct=account["drawdown_pct"],
                open_positions=account["open_positions"],
                open_orders=account["open_orders"],
                paused=int(self.repository.get_control_flag_bool("entry_paused") or self.repository.get_control_flag_bool("worker_paused")),
                source="kis_paper",
                raw_json=json.dumps(summary, ensure_ascii=False),
            )
        )
        return account

    def sync_state(self, force: bool = False) -> Dict[str, Any]:
        try:
            snapshot = self.client.get_account_snapshot()
            updated_positions = self._reconcile_positions_from_snapshot(snapshot)
            account = self._record_account_snapshot(snapshot)
            self.repository.set_control_flag("broker_kis_last_sync_at", utc_now_iso(), "KIS broker sync")
            self.repository.set_control_flag("broker_kis_last_sync_status", "ok", "KIS broker sync")
            self.repository.set_control_flag("broker_kis_last_sync_message", f"positions={updated_positions}", "KIS broker sync")
            self.repository.log_event("INFO", "kis_broker", "sync", "KIS paper account synced", {"positions": updated_positions, "forced": force})
            return {"mode": self.mode, "status": "ok", "updated_positions": updated_positions, **account}
        except Exception as exc:
            self.repository.set_control_flag("broker_kis_last_sync_at", utc_now_iso(), "KIS broker sync failed")
            self.repository.set_control_flag("broker_kis_last_sync_status", "error", "KIS broker sync failed")
            self.repository.set_control_flag("broker_kis_last_sync_message", str(exc), "KIS broker sync failed")
            self.repository.log_event("ERROR", "kis_broker", "sync_failed", "KIS paper account sync failed", {"error": str(exc)})
            return {"mode": self.mode, "status": "error", "message": str(exc)}

    def process_open_orders(self, market_data_service: MarketDataService) -> int:
        pending = self.repository.open_orders()
        pending = pending[pending["asset_type"] == self.default_asset_type] if not pending.empty else pending
        self.sync_state(force=False)
        return int(len(pending))

    def snapshot_account(self) -> Dict[str, Any]:
        latest = self.repository.latest_account_snapshot_by_source("kis_paper")
        return latest or {}

    def reconcile_order(self, order_id: str) -> Dict[str, Any]:
        order = self.repository.get_order(order_id)
        if not order:
            return {"status": "missing", "order_id": order_id}
        self.sync_state(force=True)
        refreshed = self.repository.get_order(order_id) or order
        return {
            "status": str(refreshed.get("status", "unknown")),
            "order_id": order_id,
            "broker_order_id": str(refreshed.get("broker_order_id") or ""),
            "broker": self.mode,
        }
