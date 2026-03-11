from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import flatten_due, get_kr_strategy
from services.market_data_service import MarketDataService
from storage.models import PositionRecord
from storage.repository import TradingRepository, utc_now_iso


def _float_or_nan(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


class PortfolioManager:
    KR_LIVE_QUOTE_MAX_AGE_SECONDS = 30

    def __init__(self, settings: RuntimeSettings, repository: TradingRepository, broker):
        self.settings = settings
        self.repository = repository
        self.broker = broker

    def _strategy_config(self, strategy_version: str = ""):
        return get_kr_strategy(self.settings, str(strategy_version or ""))

    def _live_kr_price_bundle(self, symbol: str) -> dict[str, float] | None:
        frame = self.repository.latest_live_market_quotes(
            symbols=[symbol],
            max_age_seconds=self.KR_LIVE_QUOTE_MAX_AGE_SECONDS,
        )
        if frame.empty:
            return None
        row = frame.iloc[0]
        current_price = _float_or_nan(row.get("current_price"))
        if not np.isfinite(current_price) or current_price <= 0:
            return None
        bid_price = _float_or_nan(row.get("bid_price"))
        ask_price = _float_or_nan(row.get("ask_price"))
        return {
            "mark_price": current_price,
            "long_exit_price": bid_price if np.isfinite(bid_price) and bid_price > 0 else current_price,
            "short_exit_price": ask_price if np.isfinite(ask_price) and ask_price > 0 else current_price,
        }

    def _price_bundle(self, position: pd.Series, market_data_service: MarketDataService) -> dict[str, float] | None:
        symbol = str(position["symbol"])
        asset_type = str(position["asset_type"])
        if asset_type == "한국주식":
            live_bundle = self._live_kr_price_bundle(symbol)
            if live_bundle is not None:
                return live_bundle
        quote = market_data_service.latest_quote(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=str(position["timeframe"]),
        )
        price = float(quote.price)
        return {
            "mark_price": price,
            "long_exit_price": price,
            "short_exit_price": price,
        }

    def mark_to_market(self, market_data_service: MarketDataService, touch=None) -> None:
        positions = self.repository.open_positions()
        for _, position in positions.iterrows():
            if callable(touch):
                touch("position_mark", {"symbol": str(position["symbol"]), "position_id": str(position.get("position_id") or "")})
            try:
                price_bundle = self._price_bundle(position, market_data_service)
            except Exception:
                continue
            if price_bundle is None:
                continue
            side = str(position["side"])
            entry = float(position["entry_price"])
            mark = float(price_bundle["mark_price"])
            qty = int(position["quantity"])
            strategy_cfg = self._strategy_config(str(position.get("strategy_version") or ""))
            trailing_mult = float(strategy_cfg.trailing_stop_atr_mult if strategy_cfg is not None else self.settings.strategy.trailing_stop_atr_mult)
            if side == "LONG":
                unrealized = (mark - entry) * qty
                high = max(float(position["highest_price"]), mark)
                low = min(float(position["lowest_price"]), mark)
                candidate_trailing = high * (1.0 - trailing_mult * max(float(position["expected_risk"]), 0.0))
                trailing = max(float(position["trailing_stop"]), candidate_trailing) if np.isfinite(float(position["trailing_stop"])) else candidate_trailing
            else:
                unrealized = (entry - mark) * qty
                high = max(float(position["highest_price"]), mark)
                low = min(float(position["lowest_price"]), mark)
                candidate_trailing = low * (1.0 + trailing_mult * max(float(position["expected_risk"]), 0.0))
                trailing = min(float(position["trailing_stop"]), candidate_trailing) if np.isfinite(float(position["trailing_stop"])) else candidate_trailing
            position_dict = {key: value for key, value in position.to_dict().items() if key != "rowid"}
            self.repository.upsert_position(
                PositionRecord(
                    **{
                        **position_dict,
                        "updated_at": utc_now_iso(),
                        "mark_price": mark,
                        "highest_price": high,
                        "lowest_price": low,
                        "trailing_stop": trailing if np.isfinite(trailing) else float(position["trailing_stop"]),
                        "unrealized_pnl": unrealized,
                        "exposure_value": mark * qty if side == "LONG" else -(mark * qty),
                        "notes": "mtm_update",
                    }
                )
            )
        self.broker.snapshot_account()

    def evaluate_exit_orders(self, market_data_service: MarketDataService, touch=None) -> int:
        exit_orders = 0
        positions = self.repository.open_positions()
        active_exit_orders = self.repository.open_orders(statuses=("new", "submitted", "acknowledged", "pending_fill", "partially_filled"))
        latest_candidates = self.repository.latest_candidates(limit=500)
        latest_candidates = latest_candidates.sort_values("created_at").drop_duplicates(
            subset=["symbol", "timeframe", "strategy_version", "execution_account_id"],
            keep="last",
        )
        for _, position in positions.iterrows():
            account_id = str(position.get("account_id") or "")
            if callable(touch):
                touch("exit_candidate", {"symbol": str(position["symbol"]), "position_id": str(position.get("position_id") or ""), "account_id": account_id})
            if not active_exit_orders.empty:
                duplicated = active_exit_orders[
                    (active_exit_orders["reason"].astype(str) != "entry")
                    & (active_exit_orders["symbol"].astype(str) == str(position["symbol"]))
                    & (active_exit_orders["timeframe"].astype(str) == str(position["timeframe"]))
                    & (
                        active_exit_orders.get("account_id", pd.Series([""] * len(active_exit_orders))).astype(str)
                        == account_id
                    )
                ]
                if not duplicated.empty:
                    continue
            try:
                price_bundle = self._price_bundle(position, market_data_service)
            except Exception:
                continue
            if price_bundle is None:
                continue
            side = str(position["side"])
            price = float(price_bundle["long_exit_price"] if side == "LONG" else price_bundle["short_exit_price"])
            stop_loss = float(position["stop_loss"])
            take_profit = float(position["take_profit"])
            trailing_stop = float(position["trailing_stop"])
            max_holding_until = pd.Timestamp(position["max_holding_until"]) if str(position["max_holding_until"]) else None
            reason = ""
            strategy_version = str(position.get("strategy_version") or "")

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
                if pd.Timestamp.now(tz="UTC") >= max_holding_until:
                    reason = "time_stop"

            if not reason and flatten_due(self.settings, strategy_version):
                reason = "session_flatten"

            if not reason and not latest_candidates.empty:
                candidate = latest_candidates[
                    (latest_candidates["symbol"].astype(str) == str(position["symbol"]))
                    & (latest_candidates["timeframe"].astype(str) == str(position["timeframe"]))
                    & (latest_candidates["strategy_version"].astype(str) == strategy_version)
                    & (
                        latest_candidates.get("execution_account_id", pd.Series([""] * len(latest_candidates))).astype(str)
                        == account_id
                    )
                ]
                if not candidate.empty:
                    row = candidate.iloc[0]
                    cand_signal = str(row["signal"])
                    score = float(row["score"])
                    if side == "LONG" and cand_signal == "SHORT":
                        reason = "opposite_signal"
                    elif side == "SHORT" and cand_signal == "LONG":
                        reason = "opposite_signal"
                    elif score < float((self._strategy_config(strategy_version).score_decay_exit_threshold if self._strategy_config(strategy_version) is not None else self.settings.strategy.score_decay_exit_threshold)):
                        reason = "score_decay"

            if reason:
                self.broker.submit_exit_order_result(position=position, reason=reason, market_data_service=market_data_service)
                exit_orders += 1
        return exit_orders
