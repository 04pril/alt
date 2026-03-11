from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from config.settings import RuntimeSettings
from kr_strategy import get_kr_strategy, strategy_asset_schedule_key, strategy_runtime_config, strategy_schedule
from services.market_data_service import MarketDataService
from services.signal_engine import SignalDecision, SignalEngine
from runtime_accounts import resolve_execution_account
from storage.models import CandidateScanRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class UniverseScanner:
    def __init__(
        self,
        settings: RuntimeSettings,
        repository: TradingRepository,
        market_data_service: MarketDataService,
        signal_engine: SignalEngine,
    ):
        self.settings = settings
        self.repository = repository
        self.market_data_service = market_data_service
        self.signal_engine = signal_engine

    def _universe(self, asset_type: str) -> List[str]:
        universe = self.settings.universes[asset_type]
        seen = set()
        ordered: List[str] = []
        for symbol in universe.watchlist + universe.top_universe:
            if symbol not in seen:
                ordered.append(symbol)
                seen.add(symbol)
        return ordered

    def _passes_intraday_liquidity(self, *, metrics: Dict[str, float], bars, strategy_id: str | None) -> tuple[bool, str]:
        strategy = get_kr_strategy(self.settings, strategy_id or "")
        if strategy is None:
            return True, "ok"
        liquidity_score = float(metrics.get("liquidity_score", 0.0) or 0.0)
        if liquidity_score < float(strategy.liquidity_min_score):
            return False, "low_intraday_liquidity"
        closes = np.asarray(bars.get("Close"), dtype=float)
        volumes = np.asarray(bars.get("Volume"), dtype=float)
        if closes.size and volumes.size:
            median_value = float(np.nanmedian(closes[-min(len(closes), 40):] * volumes[-min(len(volumes), 40):]))
            if np.isfinite(median_value) and median_value < float(strategy.liquidity_min_median_value):
                return False, "low_intraday_turnover"
        return True, "ok"

    def scan_asset(self, asset_type: str, touch=None) -> List[CandidateScanRecord]:
        return self._scan(asset_type=asset_type, strategy_id=None, touch=touch)

    def scan_strategy(self, strategy_id: str, touch=None) -> List[CandidateScanRecord]:
        strategy = get_kr_strategy(self.settings, strategy_id)
        return self._scan(asset_type=strategy_asset_schedule_key(strategy), strategy_id=strategy_id, touch=touch)

    def _scan(self, *, asset_type: str, strategy_id: str | None, touch=None) -> List[CandidateScanRecord]:
        current_time = self.market_data_service.current_time(asset_type)
        schedule = strategy_schedule(self.settings, strategy_id, when=current_time) if strategy_id else self.settings.asset_schedules[asset_type]
        strategy = get_kr_strategy(self.settings, strategy_id or "") if strategy_id else None
        strategy_view = strategy_runtime_config(strategy, when=current_time) if strategy is not None else None
        strategy_version = str(strategy.strategy_id if strategy is not None else self.settings.strategy.strategy_version)
        candidates: List[CandidateScanRecord] = []
        for symbol in self._universe(asset_type):
            execution_account_id = resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=True).account_id
            if callable(touch):
                touch("scan_symbol", {"asset_type": asset_type, "symbol": symbol, "strategy_version": strategy_version})
            scan_id = make_id("scan")
            created_at = utc_now_iso()
            try:
                bars = self.market_data_service.get_bars(
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=schedule.timeframe,
                    lookback_bars=schedule.lookback_bars,
                )
                is_valid, reason, metrics = self.market_data_service.validate_bars(
                    bars=bars,
                    min_history_bars=schedule.min_history_bars,
                )
                if not is_valid:
                    candidates.append(
                        CandidateScanRecord(
                            scan_id=scan_id,
                            created_at=created_at,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=schedule.timeframe,
                            score=-999.0,
                            rank=9999,
                            status="rejected",
                            reason=reason,
                            expected_return=np.nan,
                            expected_risk=np.nan,
                            confidence=0.0,
                            threshold=np.nan,
                            volatility=float(metrics.get("volatility", np.nan)),
                            liquidity_score=float(metrics.get("liquidity_score", 0.0)),
                            cost_bps=self.settings.strategy.round_trip_cost_bps,
                            recent_performance=0.0,
                            signal="FLAT",
                            model_version="",
                            feature_version="",
                            strategy_version=strategy_version,
                            raw_json=json.dumps(metrics, ensure_ascii=False),
                            execution_account_id=execution_account_id,
                        )
                    )
                    continue
                passes_liquidity, liquidity_reason = self._passes_intraday_liquidity(metrics=metrics, bars=bars, strategy_id=strategy_id)
                if not passes_liquidity:
                    candidates.append(
                        CandidateScanRecord(
                            scan_id=scan_id,
                            created_at=created_at,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=schedule.timeframe,
                            score=-998.0,
                            rank=9998,
                            status="rejected",
                            reason=liquidity_reason,
                            expected_return=np.nan,
                            expected_risk=np.nan,
                            confidence=0.0,
                            threshold=np.nan,
                            volatility=float(metrics.get("volatility", np.nan)),
                            liquidity_score=float(metrics.get("liquidity_score", 0.0)),
                            cost_bps=float(strategy.round_trip_cost_bps if strategy is not None else self.settings.strategy.round_trip_cost_bps),
                            recent_performance=0.0,
                            signal="FLAT",
                            model_version="",
                            feature_version="",
                            strategy_version=strategy_version,
                            raw_json=json.dumps({**metrics, "strategy_version": strategy_version}, ensure_ascii=False),
                            execution_account_id=execution_account_id,
                        )
                    )
                    continue
                signal = self.signal_engine.generate_signal(
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=schedule.timeframe,
                    strategy_id=strategy_id,
                    bars=bars,
                    scan_id=scan_id,
                    cost_bps=float(strategy.round_trip_cost_bps if strategy is not None else self.settings.strategy.round_trip_cost_bps),
                    volatility=float(metrics.get("volatility", np.nan)),
                    current_time=current_time,
                )
                latest_position = self.repository.latest_position_by_symbol(symbol=symbol, timeframe=schedule.timeframe, account_id=execution_account_id)
                pending_entry = self.repository.active_entry_orders(
                    symbol=symbol,
                    timeframe=schedule.timeframe,
                    asset_type=asset_type,
                    account_id=execution_account_id,
                )
                if strategy is not None and latest_position.empty:
                    latest_position = self.repository.open_positions(account_id=execution_account_id)
                    latest_position = latest_position.loc[latest_position["symbol"].astype(str) == str(symbol)].head(1)
                is_holding = int(
                    (not latest_position.empty and str(latest_position.iloc[0].get("status")) == "open")
                    or not pending_entry.empty
                )
                cooldown_until = self.repository.latest_cooldown_until(symbol=symbol, timeframe=schedule.timeframe, account_id=execution_account_id)
                candidates.append(
                    CandidateScanRecord(
                        scan_id=scan_id,
                        created_at=created_at,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=schedule.timeframe,
                        score=float(signal.score),
                        rank=0,
                        status="candidate" if signal.signal != "FLAT" else "flat",
                        reason="signal_ready" if signal.signal != "FLAT" else "flat_signal",
                        expected_return=float(signal.expected_return),
                        expected_risk=float(signal.expected_risk),
                        confidence=float(signal.confidence),
                        threshold=float(signal.threshold),
                        volatility=float(metrics.get("volatility", np.nan)),
                        liquidity_score=float(metrics.get("liquidity_score", 0.0)),
                        cost_bps=float(strategy.round_trip_cost_bps if strategy is not None else self.settings.strategy.round_trip_cost_bps),
                        recent_performance=self.repository.recent_prediction_performance(symbol=symbol)["paper_trade_return"],
                        signal=signal.signal,
                        model_version=signal.model_version,
                        feature_version=signal.feature_version,
                        strategy_version=signal.strategy_version,
                        cooldown_until=cooldown_until,
                        is_holding=is_holding,
                        raw_json=json.dumps(
                            {
                                "bars": int(metrics.get("bars", 0)),
                                "prediction_id": signal.prediction_id,
                                "market_phase": self.market_data_service.market_phase(asset_type),
                                "strategy_version": signal.strategy_version,
                                "strategy_family": signal.strategy_family,
                                "session_mode": str(strategy_view.session_mode if strategy_view is not None else ""),
                                "price_policy": str(strategy_view.session_price_policy if strategy_view is not None else ""),
                                "decision_horizon_bars": int(signal.decision_horizon_bars),
                                "primary_target": signal.primary_target_type,
                                "secondary_target": signal.secondary_target_type,
                                "analysis_target": signal.analysis_target_type,
                                "experimental": bool(signal.experimental),
                            },
                            ensure_ascii=False,
                        ),
                        execution_account_id=execution_account_id,
                    )
                )
            except Exception as exc:
                candidates.append(
                    CandidateScanRecord(
                        scan_id=scan_id,
                        created_at=created_at,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=schedule.timeframe,
                        score=-999.0,
                        rank=9999,
                        status="error",
                        reason=str(exc),
                        expected_return=np.nan,
                        expected_risk=np.nan,
                        confidence=0.0,
                        threshold=np.nan,
                        volatility=np.nan,
                        liquidity_score=0.0,
                        cost_bps=float(strategy.round_trip_cost_bps if strategy is not None else self.settings.strategy.round_trip_cost_bps),
                        recent_performance=0.0,
                        signal="FLAT",
                        model_version="",
                        feature_version="",
                        strategy_version=strategy_version,
                        raw_json="{}",
                        execution_account_id=execution_account_id,
                    )
                )

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        final_rows: List[CandidateScanRecord] = []
        for rank, row in enumerate(ranked, start=1):
            final_rows.append(
                CandidateScanRecord(
                    **{**row.__dict__, "rank": rank}
                )
            )
        self.repository.insert_candidate_scans(final_rows)
        return final_rows
