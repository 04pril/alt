from __future__ import annotations

from typing import Dict

import pandas as pd

from storage.repository import TradingRepository


class Evaluator:
    def __init__(self, repository: TradingRepository):
        self.repository = repository

    def prediction_metrics(self) -> Dict[str, pd.DataFrame]:
        frame = self.repository.prediction_report(limit=5000)
        if frame.empty:
            return {"by_date": pd.DataFrame(), "by_symbol": pd.DataFrame(), "by_model": pd.DataFrame()}
        frame["created_date"] = pd.to_datetime(frame["created_at"], errors="coerce").dt.date
        by_date = (
            frame.groupby("created_date", dropna=False)
            .agg(
                predictions=("prediction_id", "count"),
                direction_acc=("directional_accuracy", "mean"),
                mae_return=("abs_error_return", "mean"),
                trade_return=("paper_trade_return", "mean"),
            )
            .reset_index()
        )
        by_symbol = (
            frame.groupby("symbol", dropna=False)
            .agg(
                predictions=("prediction_id", "count"),
                direction_acc=("directional_accuracy", "mean"),
                mae_return=("abs_error_return", "mean"),
                trade_return=("paper_trade_return", "mean"),
            )
            .reset_index()
        )
        by_model = (
            frame.groupby("model_version", dropna=False)
            .agg(
                predictions=("prediction_id", "count"),
                direction_acc=("directional_accuracy", "mean"),
                mae_return=("abs_error_return", "mean"),
                trade_return=("paper_trade_return", "mean"),
            )
            .reset_index()
        )
        return {"by_date": by_date, "by_symbol": by_symbol, "by_model": by_model}

    def trade_metrics(self) -> Dict[str, float]:
        return self.repository.trade_performance_report()
