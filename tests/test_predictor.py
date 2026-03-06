from __future__ import annotations

import unittest

import pandas as pd

from predictor import _infer_future_index, _simulate_trade_backtest


class PredictorRegressionTest(unittest.TestCase):
    def test_infer_future_index_keeps_business_day_frequency(self) -> None:
        index = pd.date_range("2026-03-02", periods=10, freq="B")
        future = _infer_future_index(index, periods=3)
        self.assertEqual(list(future), list(pd.date_range(index[-1] + pd.offsets.BDay(), periods=3, freq="B")))

    def test_infer_future_index_keeps_intraday_step_without_inferred_freq(self) -> None:
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-03-06 09:00:00"),
                pd.Timestamp("2026-03-06 10:00:00"),
                pd.Timestamp("2026-03-06 11:00:00"),
            ]
        )
        future = _infer_future_index(index, periods=2)
        self.assertEqual(list(future), [pd.Timestamp("2026-03-06 12:00:00"), pd.Timestamp("2026-03-06 13:00:00")])

    def test_close_to_close_execution_uses_next_close(self) -> None:
        price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 106.0, 107.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.0, 105.0, 104.0],
                "Volume": [1000.0, 1200.0, 1100.0],
            },
            index=pd.date_range("2026-03-02", periods=3, freq="B"),
        )
        eval_frame = pd.DataFrame(
            {
                "current_close": [100.0],
                "ensemble_pred": [102.0],
                "predicted_move_pct": [2.0],
            },
            index=price_data.index[:1],
        )
        backtest, metrics = _simulate_trade_backtest(
            price_data=price_data,
            eval_frame=eval_frame,
            round_trip_cost_bps=0.0,
            signal_threshold_pct=0.1,
            allow_short=False,
            trade_mode="close_to_close",
            target_daily_vol_pct=0.0,
            max_position_size=1.0,
            stop_loss_atr_mult=0.0,
            take_profit_atr_mult=0.0,
        )
        self.assertAlmostEqual(float(backtest.iloc[0]["entry_price"]), 100.0)
        self.assertAlmostEqual(float(backtest.iloc[0]["exit_price"]), 105.0)
        self.assertAlmostEqual(float(backtest.iloc[0]["net_return"]), 0.05)
        self.assertEqual(float(metrics.loc[metrics["metric"] == "trades", "value"].iloc[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
