from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from services.signal_engine import SignalDecision
from storage.repository import TradingRepository


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    quantity: int
    notional: float
    expected_loss: float


class RiskEngine:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository):
        self.settings = settings
        self.repository = repository

    def _latest_account_state(self) -> Dict[str, float]:
        latest = self.repository.latest_account_snapshot() or {}
        equity = float(latest.get("equity", self.settings.risk.starting_cash))
        cash = float(latest.get("cash", self.settings.risk.starting_cash))
        drawdown_pct = float(latest.get("drawdown_pct", 0.0) or 0.0)
        return {"equity": equity, "cash": cash, "drawdown_pct": drawdown_pct}

    def evaluate_entry(
        self,
        signal: SignalDecision,
        correlation_matrix: pd.DataFrame,
        market_is_open: bool,
    ) -> RiskDecision:
        strategy = self.settings.strategy
        risk = self.settings.risk
        state = self._latest_account_state()
        today = str(pd.Timestamp.utcnow().date())
        if self.repository.get_control_flag_bool("worker_paused", False):
            return RiskDecision(False, "worker_paused", 0, 0.0, 0.0)
        if self.repository.get_control_flag_bool("entry_paused", False):
            return RiskDecision(False, "entry_paused", 0, 0.0, 0.0)
        if self.repository.get_control_flag_bool("exit_only_mode", False):
            return RiskDecision(False, "exit_only_mode", 0, 0.0, 0.0)
        if not market_is_open:
            return RiskDecision(False, "market_closed", 0, 0.0, 0.0)
        if signal.signal == "FLAT":
            return RiskDecision(False, "flat_signal", 0, 0.0, 0.0)
        if signal.signal == "SHORT" and not self.settings.short_allowed_for(signal.asset_type):
            return RiskDecision(False, "short_not_supported", 0, 0.0, 0.0)
        if signal.expected_return * 100.0 < strategy.min_expected_return_pct:
            return RiskDecision(False, "expected_return_too_low", 0, 0.0, 0.0)
        if signal.confidence < strategy.min_confidence:
            return RiskDecision(False, "confidence_too_low", 0, 0.0, 0.0)
        if signal.expected_risk * 100.0 > strategy.max_expected_risk_pct:
            return RiskDecision(False, "risk_too_high", 0, 0.0, 0.0)
        if strategy.round_trip_cost_bps > strategy.max_cost_bps:
            return RiskDecision(False, "cost_too_high", 0, 0.0, 0.0)

        open_positions = self.repository.open_positions()
        if len(open_positions) >= risk.max_open_positions:
            return RiskDecision(False, "max_open_positions", 0, 0.0, 0.0)
        if self.repository.count_daily_entries(today) >= risk.max_daily_new_entries:
            return RiskDecision(False, "max_daily_entries", 0, 0.0, 0.0)
        if state["drawdown_pct"] <= -(risk.max_drawdown_limit_pct * 100.0):
            return RiskDecision(False, "max_drawdown_limit", 0, 0.0, 0.0)
        if self.repository.recent_closed_realized_pnl(today) <= -(state["equity"] * risk.daily_loss_limit_pct):
            return RiskDecision(False, "daily_loss_limit", 0, 0.0, 0.0)

        same_symbol = open_positions[
            (open_positions["symbol"].astype(str) == signal.symbol)
            & (open_positions["timeframe"].astype(str) == signal.timeframe)
        ]
        if not same_symbol.empty:
            return RiskDecision(False, "already_holding_symbol", 0, 0.0, 0.0)

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

        current_exposure = (
            pd.to_numeric(open_positions.get("exposure_value"), errors="coerce").abs().sum() if not open_positions.empty else 0.0
        )
        asset_exposure = (
            pd.to_numeric(
                open_positions.loc[open_positions["asset_type"] == signal.asset_type, "exposure_value"],
                errors="coerce",
            ).abs().sum()
            if not open_positions.empty
            else 0.0
        )
        equity = max(state["equity"], 1.0)
        symbol_cap = equity * risk.symbol_max_weight
        asset_cap = equity * risk.asset_type_max_weight.get(signal.asset_type, 0.3) - asset_exposure
        remaining_total_risk = max(0.0, equity * risk.total_risk_budget_pct - current_exposure * signal.expected_risk)
        risk_cap = equity * risk.per_trade_risk_budget_pct / max(signal.expected_risk, 1e-6)
        notional = max(0.0, min(symbol_cap, asset_cap, remaining_total_risk, risk_cap, state["cash"]))
        if notional <= 0:
            return RiskDecision(False, "no_risk_budget", 0, 0.0, 0.0)
        quantity = int(floor(notional / max(signal.current_price, 1e-9)))
        if quantity <= 0:
            return RiskDecision(False, "notional_too_small", 0, notional, 0.0)
        expected_loss = quantity * signal.current_price * max(signal.expected_risk, 0.0)
        return RiskDecision(True, "ok", quantity, quantity * signal.current_price, expected_loss)
