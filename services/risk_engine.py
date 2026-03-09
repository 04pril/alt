from __future__ import annotations

import json
from dataclasses import dataclass
from math import floor
from typing import Callable, Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_LEGACY_MIXED, ExecutionAccount, resolve_execution_account
from services.signal_engine import SignalDecision
from storage.repository import TradingRepository, parse_utc_timestamp


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    quantity: int
    notional: float
    expected_loss: float


class RiskEngine:
    def __init__(
        self,
        settings: RuntimeSettings,
        repository: TradingRepository,
        account_resolver: Callable[[str, str], ExecutionAccount] | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self._account_resolver = account_resolver

    def resolve_execution_account(self, asset_type: str = "", symbol: str = "") -> ExecutionAccount:
        if self._account_resolver is not None:
            return self._account_resolver(symbol, asset_type)
        return resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=True)

    def _latest_account_state(self, asset_type: str = "", symbol: str = "") -> Dict[str, float]:
        account = self.resolve_execution_account(asset_type=asset_type, symbol=symbol)
        latest = self.repository.latest_account_snapshot(account_id=account.account_id) or {}
        if not latest and account.account_id != ACCOUNT_SIM_LEGACY_MIXED:
            if account.account_id == ACCOUNT_KIS_KR_PAPER:
                latest = self.repository.latest_account_snapshot(source="kis_account_sync") or {}
            if not latest:
                latest = self.repository.latest_account_snapshot(account_id=ACCOUNT_SIM_LEGACY_MIXED) or {}
        fallback_cash = float(self.settings.risk.starting_cash)
        equity = float(latest.get("equity", fallback_cash))
        cash = float(latest.get("cash", fallback_cash))
        drawdown_pct = float(latest.get("drawdown_pct", 0.0) or 0.0)
        return {
            "account_id": account.account_id,
            "broker_mode": account.broker_mode,
            "equity": equity,
            "cash": cash,
            "drawdown_pct": drawdown_pct,
        }

    def _pending_entry_frame(self, account_id: str) -> pd.DataFrame:
        frame = self.repository.active_entry_orders(active_only=True, account_id=account_id)
        if frame.empty:
            return frame

        def parse_payload(payload: object) -> Dict[str, float]:
            try:
                return json.loads(str(payload or "{}"))
            except Exception:
                return {}

        payloads = frame["raw_json"].map(parse_payload)
        frame = frame.copy()
        frame["pending_price"] = pd.to_numeric(frame["requested_price"], errors="coerce")
        frame["pending_qty"] = pd.to_numeric(frame["remaining_qty"], errors="coerce").fillna(0.0)
        frame["pending_notional"] = (frame["pending_price"].fillna(0.0) * frame["pending_qty"]).abs()
        frame["pending_expected_risk"] = payloads.map(lambda item: float(item.get("expected_risk", np.nan)))
        frame["pending_expected_risk"] = pd.to_numeric(frame["pending_expected_risk"], errors="coerce").fillna(0.0)
        return frame

    def evaluate_entry(
        self,
        signal: SignalDecision,
        correlation_matrix: pd.DataFrame,
        market_is_open: bool,
    ) -> RiskDecision:
        strategy = self.settings.strategy
        risk = self.settings.risk
        state = self._latest_account_state(asset_type=signal.asset_type, symbol=signal.symbol)
        account_id = str(state["account_id"])
        today = str(pd.Timestamp.utcnow().date())
        if self.repository.get_control_flag("trading_paused", "0") == "1":
            return RiskDecision(False, "paused", 0, 0.0, 0.0)
        if not market_is_open:
            return RiskDecision(False, "market_closed", 0, 0.0, 0.0)
        if signal.signal == "FLAT":
            return RiskDecision(False, "flat_signal", 0, 0.0, 0.0)
        if signal.expected_return * 100.0 < strategy.min_expected_return_pct:
            return RiskDecision(False, "expected_return_too_low", 0, 0.0, 0.0)
        if signal.confidence < strategy.min_confidence:
            return RiskDecision(False, "confidence_too_low", 0, 0.0, 0.0)
        if signal.expected_risk * 100.0 > strategy.max_expected_risk_pct:
            return RiskDecision(False, "risk_too_high", 0, 0.0, 0.0)
        if strategy.round_trip_cost_bps > strategy.max_cost_bps:
            return RiskDecision(False, "cost_too_high", 0, 0.0, 0.0)

        cooldown_until = self.repository.latest_cooldown_until(signal.symbol, signal.timeframe, account_id=account_id)
        cooldown_dt = parse_utc_timestamp(cooldown_until)
        if cooldown_dt is not None and pd.Timestamp.now(tz="UTC").to_pydatetime() < cooldown_dt:
            return RiskDecision(False, "cooldown_active", 0, 0.0, 0.0)

        open_positions = self.repository.open_positions(account_id=account_id)
        pending_entries = self._pending_entry_frame(account_id)
        if self.repository.count_daily_entries(today, account_id=account_id) >= risk.max_daily_new_entries:
            return RiskDecision(False, "max_daily_entries", 0, 0.0, 0.0)
        if state["drawdown_pct"] <= -(risk.max_drawdown_limit_pct * 100.0):
            return RiskDecision(False, "max_drawdown_limit", 0, 0.0, 0.0)
        if self.repository.recent_closed_realized_pnl(today, account_id=account_id) <= -(state["equity"] * risk.daily_loss_limit_pct):
            return RiskDecision(False, "daily_loss_limit", 0, 0.0, 0.0)

        same_symbol = open_positions[
            (open_positions["symbol"].astype(str) == signal.symbol)
            & (open_positions["timeframe"].astype(str) == signal.timeframe)
        ]
        same_symbol_pending = pending_entries[
            (pending_entries["symbol"].astype(str) == signal.symbol)
            & (pending_entries["timeframe"].astype(str) == signal.timeframe)
        ]
        if not same_symbol_pending.empty:
            return RiskDecision(False, "duplicate_pending_entry", 0, 0.0, 0.0)
        if not same_symbol.empty:
            return RiskDecision(False, "already_holding_symbol", 0, 0.0, 0.0)

        reserved_slots = pd.concat(
            [
                open_positions.loc[:, ["symbol", "timeframe"]] if not open_positions.empty else pd.DataFrame(columns=["symbol", "timeframe"]),
                pending_entries.loc[:, ["symbol", "timeframe"]] if not pending_entries.empty else pd.DataFrame(columns=["symbol", "timeframe"]),
            ],
            ignore_index=True,
        ).drop_duplicates()
        if len(reserved_slots) >= risk.max_open_positions:
            return RiskDecision(False, "max_open_positions", 0, 0.0, 0.0)

        if not correlation_matrix.empty and signal.symbol in correlation_matrix.index:
            current_side = signal.signal
            for _, row in open_positions.iterrows():
                if str(row.get("side")) != current_side:
                    continue
                other = str(row.get("symbol"))
                if other == signal.symbol or other not in correlation_matrix.columns:
                    continue
                corr = float(correlation_matrix.loc[signal.symbol, other])
                if np.isfinite(corr) and corr >= risk.max_same_direction_correlation:
                    return RiskDecision(False, f"correlation_limit:{other}", 0, 0.0, 0.0)

        open_exposures = pd.to_numeric(open_positions.get("exposure_value"), errors="coerce").fillna(0.0) if not open_positions.empty else pd.Series(dtype=float)
        current_gross_exposure = float(open_exposures.abs().sum()) if not open_exposures.empty else 0.0
        asset_open_exposure = (
            float(
                pd.to_numeric(open_positions.loc[open_positions["asset_type"] == signal.asset_type, "exposure_value"], errors="coerce")
                .fillna(0.0)
                .abs()
                .sum()
            )
            if not open_positions.empty
            else 0.0
        )
        pending_gross_exposure = float(pd.to_numeric(pending_entries.get("pending_notional"), errors="coerce").fillna(0.0).sum()) if not pending_entries.empty else 0.0
        pending_asset_exposure = (
            float(pd.to_numeric(pending_entries.loc[pending_entries["asset_type"] == signal.asset_type, "pending_notional"], errors="coerce").fillna(0.0).sum())
            if not pending_entries.empty
            else 0.0
        )
        reserved_cash = (
            float(
                pd.to_numeric(
                    pending_entries.loc[pending_entries["side"].astype(str) == "buy", "pending_notional"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )
            if not pending_entries.empty
            else 0.0
        )
        position_risk = (
            float(
                (
                    pd.to_numeric(open_positions.get("exposure_value"), errors="coerce").fillna(0.0).abs()
                    * pd.to_numeric(open_positions.get("expected_risk"), errors="coerce").fillna(0.0)
                ).sum()
            )
            if not open_positions.empty
            else 0.0
        )
        pending_risk = (
            float((pending_entries["pending_notional"] * pending_entries["pending_expected_risk"]).sum())
            if not pending_entries.empty
            else 0.0
        )

        equity = max(state["equity"], 1.0)
        available_cash = max(0.0, state["cash"] - reserved_cash)
        symbol_cap = equity * risk.symbol_max_weight
        asset_cap = equity * risk.asset_type_max_weight.get(signal.asset_type, 0.3) - asset_open_exposure - pending_asset_exposure
        remaining_total_risk = max(0.0, equity * risk.total_risk_budget_pct - position_risk - pending_risk)
        risk_cap = equity * risk.per_trade_risk_budget_pct / max(signal.expected_risk, 1e-6)
        total_exposure_cap = max(0.0, equity * risk.total_risk_budget_pct / max(signal.expected_risk, 1e-6) - current_gross_exposure - pending_gross_exposure)
        notional = max(0.0, min(symbol_cap, asset_cap, remaining_total_risk / max(signal.expected_risk, 1e-6), risk_cap, total_exposure_cap, available_cash))
        if notional <= 0:
            return RiskDecision(False, "no_risk_budget", 0, 0.0, 0.0)
        quantity = int(floor(notional / max(signal.current_price, 1e-9)))
        if quantity <= 0:
            return RiskDecision(False, "notional_too_small", 0, notional, 0.0)
        expected_loss = quantity * signal.current_price * max(signal.expected_risk, 0.0)
        return RiskDecision(True, "ok", quantity, quantity * signal.current_price, expected_loss)
