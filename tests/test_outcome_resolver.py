from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from services.outcome_resolver import OutcomeResolver
from storage.models import PredictionRecord
from storage.repository import TradingRepository


class _FakeMarketDataService:
    def get_bars(self, symbol: str, asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        self.last_request = {
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "lookback_bars": lookback_bars,
        }
        return pd.DataFrame(
            {"Close": [105.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-08T02:00:00Z")]),
        )


class OutcomeResolverTest(unittest.TestCase):
    def test_resolve_accepts_tz_aware_target_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            repo.insert_predictions(
                [
                    PredictionRecord(
                        prediction_id="pred-aware",
                        created_at="2026-03-08T01:00:00Z",
                        run_id="run-aware",
                        symbol="005930.KS",
                        asset_type="한국주식",
                        timeframe="1h",
                        market_timezone="Asia/Seoul",
                        data_cutoff_at="2026-03-08T01:00:00Z",
                        target_at="2026-03-08T02:00:00+00:00",
                        forecast_horizon_bars=1,
                        target_type="next_close_return",
                        current_price=100.0,
                        predicted_price=104.0,
                        predicted_return=0.04,
                        signal="LONG",
                        score=1.0,
                        confidence=0.7,
                        threshold=0.01,
                        expected_return=0.04,
                        expected_risk=0.02,
                        position_size=1.0,
                        model_name="ensemble",
                        model_version="v1",
                        feature_version="f1",
                        strategy_version="s1",
                        validation_mode="holdout",
                        feature_hash="hash",
                    )
                ]
            )

            resolver = OutcomeResolver(repository=repo, market_data_service=_FakeMarketDataService())

            resolved = resolver.resolve(limit=10)

            self.assertEqual(resolved, 1)
            unresolved = repo.unresolved_predictions(limit=10)
            self.assertTrue(unresolved.empty)
            report = repo.prediction_report(limit=10)
            self.assertEqual(len(report), 1)
            self.assertEqual(report.iloc[0]["status"], "resolved")
            self.assertAlmostEqual(float(report.iloc[0]["actual_price"]), 105.0)

    def test_resolve_accepts_intraday_minute_timeframe(self) -> None:
        class _MinuteMarketDataService(_FakeMarketDataService):
            def get_bars(self, symbol: str, asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
                self.last_request = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "timeframe": timeframe,
                    "lookback_bars": lookback_bars,
                }
                return pd.DataFrame(
                    {"Close": [103.0]},
                    index=pd.DatetimeIndex([pd.Timestamp("2026-03-08T02:15:00Z")]),
                )

        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            repo.insert_predictions(
                [
                    PredictionRecord(
                        prediction_id="pred-15m",
                        created_at="2026-03-08T01:45:00Z",
                        run_id="run-15m",
                        symbol="AAPL",
                        asset_type="미국주식",
                        timeframe="15m",
                        market_timezone="America/New_York",
                        data_cutoff_at="2026-03-08T02:00:00Z",
                        target_at="2026-03-08T02:15:00+00:00",
                        forecast_horizon_bars=1,
                        target_type="next_close_return",
                        current_price=100.0,
                        predicted_price=103.0,
                        predicted_return=0.03,
                        signal="LONG",
                        score=1.0,
                        confidence=0.6,
                        threshold=0.01,
                        expected_return=0.03,
                        expected_risk=0.02,
                        position_size=1.0,
                        model_name="ensemble",
                        model_version="v1",
                        feature_version="f1",
                        strategy_version="s1",
                        validation_mode="holdout",
                        feature_hash="hash",
                    )
                ]
            )

            resolver = OutcomeResolver(repository=repo, market_data_service=_MinuteMarketDataService())

            resolved = resolver.resolve(limit=10)

            self.assertEqual(resolved, 1)
            report = repo.prediction_report(limit=10)
            self.assertEqual(len(report), 1)
            self.assertEqual(report.iloc[0]["status"], "resolved")
            self.assertAlmostEqual(float(report.iloc[0]["actual_price"]), 103.0)


if __name__ == "__main__":
    unittest.main()
