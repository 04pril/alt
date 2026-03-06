from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from config.settings import RuntimeSettings
from predictor import ForecastResult
from services.signal_engine import SignalEngine
from storage.repository import TradingRepository


def _forecast_result(symbol: str, price_data: pd.DataFrame) -> ForecastResult:
    index = pd.DatetimeIndex([price_data.index[-1] + pd.Timedelta(days=1)])
    future_frame = pd.DataFrame(
        {
            "ensemble_pred": [102.0],
            "ensemble_pred_return_pct": [2.0],
            "position_size": [0.7],
            "stop_level": [98.0],
            "take_level": [105.0],
            "atr_14": [1.0],
            "planned_signal": [1.0],
        },
        index=index,
    )
    empty_trade = pd.DataFrame()
    metrics = pd.DataFrame([{"model": "Ensemble", "mae": 1.0, "mape_pct": 1.0, "direction_acc_pct": 60.0}])
    return ForecastResult(
        symbol=symbol,
        timeframe="1d",
        price_data=price_data,
        test_frame=pd.DataFrame(),
        future_frame=future_frame,
        metrics=metrics,
        weights={"Ridge": 1.0},
        latest_close=100.0,
        trade_backtest=empty_trade,
        trade_metrics=pd.DataFrame(),
        validation_mode="holdout",
        final_holdout_frame=pd.DataFrame(),
        final_holdout_metrics=metrics,
        final_holdout_trade_backtest=empty_trade,
        final_holdout_trade_metrics=pd.DataFrame(),
        regime_metrics=pd.DataFrame(),
        validation_summary=pd.DataFrame(),
        validation_frame=pd.DataFrame(),
        validation_metrics=metrics,
        target_mode="return",
        trade_mode="close_to_close",
        signal_threshold_pct=0.5,
        model_name="ensemble",
        model_version="test_v1",
        feature_version="fv_test",
        feature_hash="hash1234",
        data_cutoff_at=price_data.index[-1],
    )


class SignalEngineTest(unittest.TestCase):
    def test_generate_signal_persists_prediction_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.initialize_runtime_flags()
            engine = SignalEngine(settings, repo)
            bars = pd.DataFrame(
                {
                    "Open": [99.0, 100.0],
                    "High": [101.0, 102.0],
                    "Low": [98.0, 99.0],
                    "Close": [100.0, 101.0],
                    "Volume": [1000.0, 1100.0],
                },
                index=pd.date_range("2026-03-04", periods=2, freq="B"),
            )
            coin_asset_type = next(asset for asset, schedule in settings.asset_schedules.items() if schedule.session_mode == "always")
            with patch("services.signal_engine.run_forecast_on_price_data", return_value=_forecast_result("BTC-USD", bars)):
                signal = engine.generate_signal(symbol="BTC-USD", asset_type=coin_asset_type, timeframe="1h", bars=bars, scan_id="scan1")

            self.assertEqual(signal.signal, "LONG")
            report = repo.prediction_report(limit=10)
            self.assertEqual(len(report), 1)
            self.assertEqual(str(report.iloc[0]["scan_id"]), "scan1")
            self.assertEqual(str(report.iloc[0]["model_version"]), "test_v1")


if __name__ == "__main__":
    unittest.main()
