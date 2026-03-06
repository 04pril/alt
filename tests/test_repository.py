from __future__ import annotations

import tempfile
import unittest

from storage.models import EvaluationRecord, OutcomeRecord, PredictionRecord
from storage.repository import TradingRepository


class RepositoryTest(unittest.TestCase):
    def test_prediction_outcome_evaluation_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = TradingRepository(f"{tmp}/runtime.sqlite3")
            repo.initialize()
            repo.insert_predictions(
                [
                    PredictionRecord(
                        prediction_id="pred1",
                        created_at="2026-03-06T00:00:00Z",
                        run_id="run1",
                        symbol="BTC-USD",
                        asset_type="코인",
                        timeframe="1h",
                        market_timezone="UTC",
                        data_cutoff_at="2026-03-06T00:00:00Z",
                        target_at="2026-03-06T01:00:00Z",
                        forecast_horizon_bars=1,
                        target_type="next_close_return",
                        current_price=100.0,
                        predicted_price=101.0,
                        predicted_return=0.01,
                        signal="LONG",
                        score=1.0,
                        confidence=0.7,
                        threshold=0.003,
                        expected_return=0.01,
                        expected_risk=0.02,
                        position_size=0.5,
                        model_name="ensemble",
                        model_version="v1",
                        feature_version="f1",
                        strategy_version="s1",
                        validation_mode="holdout",
                        feature_hash="abc",
                    )
                ]
            )
            self.assertEqual(len(repo.unresolved_predictions()), 1)
            repo.insert_outcome(
                OutcomeRecord(
                    prediction_id="pred1",
                    created_at="2026-03-06T01:00:00Z",
                    resolved_at="2026-03-06T01:00:00Z",
                    symbol="BTC-USD",
                    asset_type="코인",
                    timeframe="1h",
                    actual_price=100.5,
                    actual_return=0.005,
                    outcome_source="test",
                )
            )
            repo.insert_evaluation(
                EvaluationRecord(
                    prediction_id="pred1",
                    created_at="2026-03-06T01:00:00Z",
                    error_price=-0.5,
                    abs_error_price=0.5,
                    squared_error_price=0.25,
                    error_return=-0.005,
                    abs_error_return=0.005,
                    squared_error_return=0.000025,
                    ape_pct=0.5,
                    directional_accuracy=1.0,
                    sign_hit_rate=1.0,
                    brier_score=0.1,
                    paper_trade_return=0.0025,
                    paper_trade_pnl=0.25,
                )
            )
            report = repo.prediction_report(limit=10)
            self.assertEqual(len(report), 1)
            self.assertEqual(report.iloc[0]["status"], "resolved")


if __name__ == "__main__":
    unittest.main()
