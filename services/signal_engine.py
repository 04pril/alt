from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from predictor import ForecastResult, run_forecast_on_price_data
from runtime_accounts import resolve_execution_account
from storage.models import PredictionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


@dataclass(frozen=True)
class SignalDecision:
    symbol: str
    asset_type: str
    timeframe: str
    prediction_id: str
    scan_id: str | None
    score: float
    signal: str
    expected_return: float
    expected_risk: float
    confidence: float
    threshold: float
    position_size: float
    current_price: float
    predicted_price: float
    predicted_return: float
    stop_level: float
    take_level: float
    model_version: str
    feature_version: str
    strategy_version: str
    validation_mode: str
    result: ForecastResult | None


class SignalEngine:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository):
        self.settings = settings
        self.repository = repository

    def _score_candidate(
        self,
        *,
        expected_return: float,
        confidence: float,
        cost_bps: float,
        volatility: float,
        recent_performance: float,
    ) -> float:
        weights = self.settings.strategy.scan_score_weights
        return (
            weights["expected_return"] * expected_return * 100.0
            + weights["confidence"] * confidence * 100.0
            + weights["cost"] * cost_bps
            + weights["volatility"] * volatility * 100.0
            + weights["recent_performance"] * recent_performance * 100.0
        )

    def _record_predictions(
        self,
        *,
        asset_type: str,
        timeframe: str,
        scan_id: str | None,
        result: ForecastResult,
        score: float,
        confidence: float,
        expected_risk: float,
    ) -> List[PredictionRecord]:
        now_iso = utc_now_iso()
        run_id = make_id("run")
        rows: List[PredictionRecord] = []
        execution_account_id = resolve_execution_account(symbol=result.symbol, asset_type=asset_type, kis_enabled=True).account_id
        for horizon, (target_at, row) in enumerate(result.future_frame.iterrows(), start=1):
            prediction_id = make_id("pred")
            predicted_return = float(row.get("ensemble_pred_return_pct", np.nan)) / 100.0
            current_price = float(result.latest_close)
            predicted_price = float(row.get("ensemble_pred", np.nan))
            position_size = float(row.get("position_size", np.nan))
            threshold = float(getattr(result, "signal_threshold_pct", np.nan)) / 100.0
            signal_value = float(row.get("planned_signal", 0.0))
            signal = "LONG" if signal_value > 1e-12 else "SHORT" if signal_value < -1e-12 else "FLAT"
            rows.append(
                PredictionRecord(
                    prediction_id=prediction_id,
                    created_at=now_iso,
                    run_id=run_id,
                    symbol=result.symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    market_timezone="UTC" if asset_type == "코인" else ("America/New_York" if asset_type == "미국주식" else "Asia/Seoul"),
                    data_cutoff_at=pd.Timestamp(getattr(result, "data_cutoff_at")).isoformat(),
                    target_at=pd.Timestamp(target_at).isoformat(),
                    forecast_horizon_bars=horizon,
                    target_type="next_close_return" if result.target_mode == "return" else "next_close_price",
                    current_price=current_price,
                    predicted_price=predicted_price,
                    predicted_return=predicted_return,
                    signal=signal,
                    score=score,
                    confidence=confidence,
                    threshold=threshold,
                    expected_return=predicted_return,
                    expected_risk=expected_risk,
                    position_size=position_size,
                    model_name=result.model_name,
                    model_version=result.model_version,
                    feature_version=result.feature_version,
                    strategy_version=self.settings.strategy.strategy_version,
                    validation_mode=result.validation_mode,
                    feature_hash=result.feature_hash,
                    scan_id=scan_id,
                    notes=json.dumps(
                        {
                            "timeframe": timeframe,
                            "stop_level": float(row.get("stop_level", np.nan)),
                            "take_level": float(row.get("take_level", np.nan)),
                            "atr_14": float(row.get("atr_14", np.nan)),
                            "planned_signal": float(row.get("planned_signal", np.nan)),
                        },
                        ensure_ascii=False,
                    ),
                    execution_account_id=execution_account_id,
                )
            )
        self.repository.insert_predictions(rows)
        return rows

    def generate_signal(
        self,
        *,
        symbol: str,
        asset_type: str,
        timeframe: str,
        bars: pd.DataFrame,
        scan_id: str | None = None,
        cost_bps: float | None = None,
        volatility: float = np.nan,
    ) -> SignalDecision:
        strategy = self.settings.strategy
        shortable_assets = {str(value) for value in getattr(strategy, "allow_short_asset_types", [])}
        result = run_forecast_on_price_data(
            symbol=symbol,
            price_data=bars,
            timeframe=timeframe,
            test_days=strategy.test_bars,
            forecast_days=max(1, self.settings.asset_schedules[asset_type].forecast_horizon_bars),
            validation_mode=strategy.validation_mode,
            retrain_every=strategy.retrain_every_bars,
            round_trip_cost_bps=strategy.round_trip_cost_bps,
            min_signal_strength_pct=strategy.min_signal_strength_pct,
            final_holdout_days=strategy.final_holdout_bars,
            purge_days=strategy.purge_bars,
            embargo_days=strategy.embargo_bars,
            target_mode=strategy.target_mode,
            validation_days=strategy.validation_bars,
            allow_short=bool(strategy.allow_short or asset_type in shortable_assets),
            trade_mode=strategy.trade_mode,
            target_daily_vol_pct=strategy.target_daily_vol_pct,
            max_position_size=strategy.max_position_size,
            stop_loss_atr_mult=strategy.stop_loss_atr_mult,
            take_profit_atr_mult=strategy.take_profit_atr_mult,
        )
        primary = result.future_frame.iloc[0]
        expected_return = float(primary.get("ensemble_pred_return_pct", np.nan)) / 100.0
        expected_risk = abs(float(primary.get("atr_14", np.nan)) / max(float(result.latest_close), 1e-9))
        threshold = float(getattr(result, "signal_threshold_pct", np.nan)) / 100.0
        position_size = float(primary.get("position_size", np.nan))
        move_strength = abs(expected_return) / max(abs(threshold), 1e-9) if np.isfinite(expected_return) else 0.0
        confidence = float(np.clip(max(position_size, move_strength), 0.0, 1.0))
        recent_perf = self.repository.recent_prediction_performance(symbol=symbol)["paper_trade_return"]
        score = self._score_candidate(
            expected_return=expected_return,
            confidence=confidence,
            cost_bps=float(cost_bps if cost_bps is not None else strategy.round_trip_cost_bps),
            volatility=float(volatility if np.isfinite(volatility) else 0.0),
            recent_performance=recent_perf,
        )
        self.repository.record_model_version(
            model_version=result.model_version,
            model_name=result.model_name,
            feature_version=result.feature_version,
            strategy_version=self.settings.strategy.strategy_version,
            metrics={"score": score},
            is_champion=False,
            notes="signal_engine observed model version",
        )
        prediction_rows = self._record_predictions(
            asset_type=asset_type,
            timeframe=timeframe,
            scan_id=scan_id,
            result=result,
            score=score,
            confidence=confidence,
            expected_risk=expected_risk,
        )
        head_prediction = prediction_rows[0]
        return SignalDecision(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            prediction_id=head_prediction.prediction_id,
            scan_id=scan_id,
            score=score,
            signal=head_prediction.signal,
            expected_return=expected_return,
            expected_risk=expected_risk,
            confidence=confidence,
            threshold=threshold,
            position_size=position_size,
            current_price=float(result.latest_close),
            predicted_price=float(primary.get("ensemble_pred", np.nan)),
            predicted_return=expected_return,
            stop_level=float(primary.get("stop_level", np.nan)),
            take_level=float(primary.get("take_level", np.nan)),
            model_version=result.model_version,
            feature_version=result.feature_version,
            strategy_version=self.settings.strategy.strategy_version,
            validation_mode=result.validation_mode,
            result=result,
        )
