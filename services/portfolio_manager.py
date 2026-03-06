from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from services.market_data_service import MarketDataService
from services.paper_broker import PaperBroker
from storage.models import PositionRecord
from storage.repository import TradingRepository, utc_now_iso


class PortfolioManager:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository, broker: PaperBroker):
        self.settings = settings
        self.repository = repository
        self.broker = broker

    def mark_to_market(self, market_data_service: MarketDataService) -> None:
        positions = self.repository.open_positions()
        for _, position in positions.iterrows():
            try:
                quote = market_data_service.latest_quote(
                    symbol=str(position["symbol"]),
                    asset_type=str(position["asset_type"]),
                    timeframe=str(position["timeframe"]),
                )
            except Exception:
                continue
            side = str(position["side"])
            entry = float(position["entry_price"])
            mark = float(quote.price)
            qty = int(position["quantity"])
            if side == "LONG":
                unrealized = (mark - entry) * qty
                high = max(float(position["highest_price"]), mark)
                low = min(float(position["lowest_price"]), mark)
                candidate_trailing = high * (1.0 - self.settings.strategy.trailing_stop_atr_mult * max(float(position["expected_risk"]), 0.0))
                trailing = max(float(position["trailing_stop"]), candidate_trailing) if np.isfinite(float(position["trailing_stop"])) else candidate_trailing
            else:
                unrealized = (entry - mark) * qty
                high = max(float(position["highest_price"]), mark)
                low = min(float(position["lowest_price"]), mark)
                candidate_trailing = low * (1.0 + self.settings.strategy.trailing_stop_atr_mult * max(float(position["expected_risk"]), 0.0))
                trailing = min(float(position["trailing_stop"]), candidate_trailing) if np.isfinite(float(position["trailing_stop"])) else candidate_trailing
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position.to_dict(),
                        "updated_at": utc_now_iso(),
                        "mark_price": mark,
                        "highest_price": high,
                        "lowest_price": low,
                        "trailing_stop": trailing if np.isfinite(trailing) else float(position["trailing_stop"]),
                        "unrealized_pnl": unrealized,
                        "exposure_value": abs(mark * qty),
                        "notes": "mtm_update",
                    }
                )
            )
        self.broker.snapshot_account()

    def evaluate_exit_orders(self, market_data_service: MarketDataService) -> int:
        exit_orders = 0
        positions = self.repository.open_positions()
        latest_candidates = self.repository.latest_candidates(limit=500)
        latest_candidates = latest_candidates.sort_values("created_at").drop_duplicates(subset=["symbol", "timeframe"], keep="last")
        for _, position in positions.iterrows():
            try:
                quote = market_data_service.latest_quote(
                    symbol=str(position["symbol"]),
                    asset_type=str(position["asset_type"]),
                    timeframe=str(position["timeframe"]),
                )
            except Exception:
                continue
            price = float(quote.price)
            side = str(position["side"])
            stop_loss = float(position["stop_loss"])
            take_profit = float(position["take_profit"])
            trailing_stop = float(position["trailing_stop"])
            max_holding_until = pd.Timestamp(position["max_holding_until"]) if str(position["max_holding_until"]) else None
            reason = ""

            if side == "LONG":
                if np.isfinite(stop_loss) and price <= stop_loss:
                    reason = "stop_loss"
                elif np.isfinite(take_profit) and price >= take_profit:
                    reason = "take_profit"
                elif np.isfinite(trailing_stop) and price <= trailing_stop:
                    reason = "trailing_stop"
            else:
                if np.isfinite(stop_loss) and price >= stop_loss:
                    reason = "stop_loss"
                elif np.isfinite(take_profit) and price <= take_profit:
                    reason = "take_profit"
                elif np.isfinite(trailing_stop) and price >= trailing_stop:
                    reason = "trailing_stop"

            if not reason and max_holding_until is not None:
                if max_holding_until.tzinfo is None:
                    max_holding_until = max_holding_until.tz_localize("UTC")
                else:
                    max_holding_until = max_holding_until.tz_convert("UTC")
                if pd.Timestamp.utcnow().tz_localize("UTC") >= max_holding_until:
                    reason = "time_stop"

            if not reason and not latest_candidates.empty:
                candidate = latest_candidates[
                    (latest_candidates["symbol"].astype(str) == str(position["symbol"]))
                    & (latest_candidates["timeframe"].astype(str) == str(position["timeframe"]))
                ]
                if not candidate.empty:
                    row = candidate.iloc[0]
                    cand_signal = str(row["signal"])
                    score = float(row["score"])
                    if side == "LONG" and cand_signal == "SHORT":
                        reason = "opposite_signal"
                    elif side == "SHORT" and cand_signal == "LONG":
                        reason = "opposite_signal"
                    elif score < self.settings.strategy.score_decay_exit_threshold:
                        reason = "score_decay"

            if reason:
                self.broker.submit_exit_order(position=position, reason=reason)
                exit_orders += 1
        return exit_orders
