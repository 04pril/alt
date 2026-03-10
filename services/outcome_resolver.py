from __future__ import annotations

import numpy as np
import pandas as pd

from services.market_data_service import MarketDataService
from storage.models import EvaluationRecord, OutcomeRecord
from storage.repository import TradingRepository, utc_now_iso


class OutcomeResolver:
    def __init__(self, repository: TradingRepository, market_data_service: MarketDataService):
        self.repository = repository
        self.market_data_service = market_data_service

    @staticmethod
    def _coerce_utc_timestamp(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _floor_frequency(timeframe: object) -> str:
        normalized = str(timeframe or "").strip().lower()
        if normalized.endswith("m") and normalized[:-1].isdigit():
            return f"{int(normalized[:-1])}min"
        if normalized.endswith("h") and normalized[:-1].isdigit():
            return f"{int(normalized[:-1])}h"
        return str(timeframe or "")

    def resolve(self, limit: int = 500) -> int:
        unresolved = self.repository.unresolved_predictions(limit=limit)
        if unresolved.empty:
            return 0
        resolved_count = 0
        now_utc = pd.Timestamp.now(tz="UTC")
        for _, row in unresolved.iterrows():
            target_at = self._coerce_utc_timestamp(row["target_at"])
            if target_at > now_utc:
                continue
            try:
                bars = self.market_data_service.get_bars(
                    symbol=str(row["symbol"]),
                    asset_type=str(row["asset_type"]),
                    timeframe=str(row["timeframe"]),
                    lookback_bars=120,
                )
            except Exception:
                continue
            index = pd.to_datetime(bars.index, utc=True)
            if str(row["timeframe"]) == "1d":
                target_key = target_at.tz_convert("UTC").tz_localize(None).normalize()
                close_map = pd.Series(bars["Close"].astype(float).values, index=index.tz_localize(None).normalize())
            else:
                target_key = target_at.tz_convert("UTC").tz_localize(None).floor(self._floor_frequency(row["timeframe"]))
                close_map = pd.Series(bars["Close"].astype(float).values, index=index.tz_localize(None))
            actual_price = close_map.get(target_key)
            if pd.isna(actual_price):
                continue
            current_price = float(row["current_price"])
            predicted_price = float(row["predicted_price"])
            predicted_return = float(row["predicted_return"])
            actual_return = (float(actual_price) / current_price - 1.0) if current_price else np.nan
            error_price = float(actual_price) - predicted_price if np.isfinite(predicted_price) else np.nan
            error_return = actual_return - predicted_return if np.isfinite(predicted_return) else np.nan
            direction_hit = float(np.sign(actual_return) == np.sign(predicted_return)) if np.isfinite(actual_return) else np.nan
            probability = 0.5 + 0.5 * float(row["confidence"]) if str(row["signal"]) == "LONG" else 0.5
            if str(row["signal"]) == "SHORT":
                probability = 0.5 - 0.5 * float(row["confidence"])
            actual_up = 1.0 if actual_return > 0 else 0.0
            paper_trade_return = actual_return * (1.0 if str(row["signal"]) == "LONG" else -1.0 if str(row["signal"]) == "SHORT" else 0.0) * abs(float(row["position_size"]))
            paper_trade_pnl = paper_trade_return * current_price if np.isfinite(paper_trade_return) else np.nan
            self.repository.insert_outcome(
                OutcomeRecord(
                    prediction_id=str(row["prediction_id"]),
                    created_at=utc_now_iso(),
                    resolved_at=utc_now_iso(),
                    symbol=str(row["symbol"]),
                    asset_type=str(row["asset_type"]),
                    timeframe=str(row["timeframe"]),
                    actual_price=float(actual_price),
                    actual_return=float(actual_return),
                    outcome_source="market_data",
                )
            )
            self.repository.insert_evaluation(
                EvaluationRecord(
                    prediction_id=str(row["prediction_id"]),
                    created_at=utc_now_iso(),
                    error_price=float(error_price),
                    abs_error_price=abs(float(error_price)) if np.isfinite(error_price) else np.nan,
                    squared_error_price=float(error_price) ** 2 if np.isfinite(error_price) else np.nan,
                    error_return=float(error_return),
                    abs_error_return=abs(float(error_return)) if np.isfinite(error_return) else np.nan,
                    squared_error_return=float(error_return) ** 2 if np.isfinite(error_return) else np.nan,
                    ape_pct=abs(float(error_price)) / abs(float(actual_price)) * 100.0 if np.isfinite(error_price) and actual_price else np.nan,
                    directional_accuracy=direction_hit,
                    sign_hit_rate=direction_hit,
                    brier_score=(probability - actual_up) ** 2,
                    paper_trade_return=float(paper_trade_return),
                    paper_trade_pnl=float(paper_trade_pnl),
                )
            )
            resolved_count += 1
        return resolved_count
