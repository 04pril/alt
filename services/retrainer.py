from __future__ import annotations

from typing import Dict, List

from config.settings import RuntimeSettings
from predictor import MODEL_NAME, MODEL_VERSION, FEATURE_VERSION, run_forecast
from storage.repository import TradingRepository, make_id, parse_utc_timestamp, utc_now_iso


class Retrainer:
    def __init__(self, settings: RuntimeSettings, repository: TradingRepository):
        self.settings = settings
        self.repository = repository

    def retraining_due(self) -> bool:
        cadence = str(self.settings.retraining.cadence or "weekly").lower().strip()
        if cadence == "disabled":
            return False
        recent = self.repository.recent_job_health(limit=100)
        recent = recent[recent["job_name"] == "retrain_check"] if not recent.empty else recent
        if recent.empty:
            return True
        last = recent.iloc[0]
        last_started = str(last.get("started_at") or "")
        if not last_started:
            return True
        last_started_at = utc_now_iso()
        last_dt = parse_utc_timestamp(last_started)
        now_dt = parse_utc_timestamp(last_started_at)
        if last_dt is None or now_dt is None:
            return True
        if cadence == "weekly":
            return last_dt.isocalendar()[:2] != now_dt.isocalendar()[:2]
        if cadence == "daily":
            return last_dt.date() != now_dt.date()
        if cadence == "monthly":
            return (last_dt.year, last_dt.month) != (now_dt.year, now_dt.month)
        return True

    def run(self) -> Dict[str, float | str]:
        metrics: List[Dict[str, float]] = []
        for symbol in self.settings.retraining.benchmark_symbols:
            asset_type = "코인" if symbol.endswith("-USD") else ("한국주식" if symbol.endswith(".KS") or symbol.endswith(".KQ") else "미국주식")
            try:
                result = run_forecast(
                    symbol=symbol,
                    years=3,
                    validation_mode=self.settings.strategy.validation_mode,
                    test_days=self.settings.strategy.test_bars,
                    validation_days=self.settings.strategy.validation_bars,
                    final_holdout_days=self.settings.strategy.final_holdout_bars,
                    purge_days=self.settings.strategy.purge_bars,
                    embargo_days=self.settings.strategy.embargo_bars,
                    min_signal_strength_pct=self.settings.strategy.min_signal_strength_pct,
                    trade_mode=self.settings.strategy.trade_mode,
                    target_mode=self.settings.strategy.target_mode,
                    allow_short=self.settings.strategy.allow_short,
                    target_daily_vol_pct=self.settings.strategy.target_daily_vol_pct,
                    max_position_size=self.settings.strategy.max_position_size,
                    stop_loss_atr_mult=self.settings.strategy.stop_loss_atr_mult,
                    take_profit_atr_mult=self.settings.strategy.take_profit_atr_mult,
                )
            except Exception:
                continue
            ensemble_row = result.final_holdout_metrics[result.final_holdout_metrics["model"] == "Ensemble"]
            if ensemble_row.empty:
                continue
            trade_row = result.final_holdout_trade_metrics.set_index("metric")["value"] if not result.final_holdout_trade_metrics.empty else {}
            metrics.append(
                {
                    "directional_accuracy_pct": float(ensemble_row["direction_acc_pct"].iloc[0]),
                    "mae_pct": float(ensemble_row["mape_pct"].iloc[0]),
                    "trade_return_pct": float(trade_row.get("net_cum_return_pct", 0.0)) if isinstance(trade_row, dict) else float(trade_row.get("net_cum_return_pct", 0.0)),
                    "max_drawdown_pct": abs(float(trade_row.get("max_drawdown_pct", 0.0))) if not isinstance(trade_row, dict) else abs(float(trade_row.get("max_drawdown_pct", 0.0))),
                }
            )
        if not metrics:
            summary = {"status": "skipped", "reason": "no_benchmark_metrics"}
        else:
            directional = sum(item["directional_accuracy_pct"] for item in metrics) / len(metrics)
            mae_pct = sum(item["mae_pct"] for item in metrics) / len(metrics)
            trade_return = sum(item["trade_return_pct"] for item in metrics) / len(metrics)
            max_dd = sum(item["max_drawdown_pct"] for item in metrics) / len(metrics)
            promote = (
                directional >= self.settings.retraining.promotion_min_directional_accuracy_pct
                and mae_pct <= self.settings.retraining.promotion_max_mae_pct
                and trade_return >= self.settings.retraining.promotion_min_trade_return_pct
                and max_dd <= self.settings.retraining.promotion_max_drawdown_pct
            )
            self.repository.record_model_version(
                model_version=MODEL_VERSION,
                model_name=MODEL_NAME,
                feature_version=FEATURE_VERSION,
                strategy_version=self.settings.strategy.strategy_version,
                metrics={
                    "directional_accuracy_pct": directional,
                    "mae_pct": mae_pct,
                    "trade_return_pct": trade_return,
                    "max_drawdown_pct": max_dd,
                },
                is_champion=False,
                notes="retrainer evaluation",
            )
            if promote:
                self.repository.promote_model(MODEL_VERSION)
            summary = {
                "status": "promoted" if promote else "evaluated",
                "directional_accuracy_pct": directional,
                "mae_pct": mae_pct,
                "trade_return_pct": trade_return,
                "max_drawdown_pct": max_dd,
            }
        self.repository.insert_retrain_run(
            {
                "retrain_run_id": make_id("retrain"),
                "created_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": str(summary["status"]),
                "champion_before": MODEL_VERSION,
                "challenger_version": MODEL_VERSION,
                "champion_after": MODEL_VERSION,
                "directional_accuracy_pct": summary.get("directional_accuracy_pct"),
                "mae_pct": summary.get("mae_pct"),
                "trade_return_pct": summary.get("trade_return_pct"),
                "max_drawdown_pct": summary.get("max_drawdown_pct"),
                "summary": summary,
                "error_message": "",
            }
        )
        return summary
