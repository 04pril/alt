from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.settings import load_settings
from services.market_data_service import MarketDataService
from services.outcome_resolver import OutcomeResolver
from storage.models import AccountSnapshotRecord, OrderRecord, PredictionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _repo() -> TradingRepository:
    settings = load_settings()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    repository.initialize_runtime_flags()
    return repository


def _settings():
    return load_settings()


def _parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _market_timezone(asset_type: str) -> str:
    settings = _settings()
    schedule = settings.asset_schedules.get(asset_type)
    return str(schedule.timezone) if schedule is not None else "UTC"


def _feature_hash(result: Any) -> str:
    existing = str(getattr(result, "feature_hash", "") or "")
    if existing:
        return existing
    future_frame = getattr(result, "future_frame", pd.DataFrame())
    if future_frame is None or future_frame.empty:
        return ""
    latest = future_frame.iloc[0].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    payload = {key: float(value) if isinstance(value, (int, float, np.floating)) else str(value) for key, value in latest.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def init_store() -> None:
    _repo()


def register_model_version(result: Any) -> None:
    repository = _repo()
    repository.record_model_version(
        model_version=str(getattr(result, "model_version", "")),
        model_name=str(getattr(result, "model_name", "")),
        feature_version=str(getattr(result, "feature_version", "")),
        strategy_version=str(_settings().strategy.strategy_version),
        metrics={
            "target_mode": getattr(result, "target_mode", ""),
            "trade_mode": getattr(result, "trade_mode", ""),
            "validation_mode": getattr(result, "validation_mode", ""),
        },
        is_champion=False,
        notes="prediction_store adapter observed model version",
    )


def save_prediction_snapshot(*, asset_type: str, korea_market: str, result: Any, notes: str = "") -> str:
    repository = _repo()
    register_model_version(result)
    future_frame = getattr(result, "future_frame", pd.DataFrame())
    if future_frame is None or future_frame.empty:
        raise ValueError("future_frame is empty")

    run_id = make_id("run")
    created_at = utc_now_iso()
    current_price = _parse_float(getattr(result, "latest_close", np.nan))
    threshold = _parse_float(getattr(result, "signal_threshold_pct", np.nan)) / 100.0
    target_type = "next_close_return" if str(getattr(result, "target_mode", "return")) == "return" else "next_close_price"
    feature_hash = _feature_hash(result)
    settings = _settings()
    timeframe = str(getattr(result, "timeframe", settings.asset_schedules.get(asset_type).timeframe if asset_type in settings.asset_schedules else "1d"))
    rows: List[PredictionRecord] = []
    for horizon, (target_at, row) in enumerate(future_frame.iterrows(), start=1):
        signal_value = _parse_float(row.get("planned_signal"), 0.0)
        signal = "LONG" if signal_value > 1e-12 else "SHORT" if signal_value < -1e-12 else "FLAT"
        rows.append(
            PredictionRecord(
                prediction_id=make_id("pred"),
                created_at=created_at,
                run_id=run_id,
                symbol=str(getattr(result, "symbol", "")),
                asset_type=asset_type,
                timeframe=timeframe,
                market_timezone=_market_timezone(asset_type),
                data_cutoff_at=pd.Timestamp(getattr(result, "data_cutoff_at", pd.Timestamp.utcnow())).isoformat(),
                target_at=pd.Timestamp(target_at).isoformat(),
                forecast_horizon_bars=horizon,
                target_type=target_type,
                current_price=current_price,
                predicted_price=_parse_float(row.get("ensemble_pred")),
                predicted_return=_parse_float(row.get("ensemble_pred_return_pct")) / 100.0,
                signal=signal,
                score=_parse_float(row.get("score"), _parse_float(signal_value)),
                confidence=_parse_float(row.get("position_size"), abs(signal_value)),
                threshold=threshold,
                expected_return=_parse_float(row.get("ensemble_pred_return_pct")) / 100.0,
                expected_risk=_parse_float(row.get("atr_14")) / max(current_price, 1e-9),
                position_size=_parse_float(row.get("position_size")),
                model_name=str(getattr(result, "model_name", "")),
                model_version=str(getattr(result, "model_version", "")),
                feature_version=str(getattr(result, "feature_version", "")),
                strategy_version=str(settings.strategy.strategy_version),
                validation_mode=str(getattr(result, "validation_mode", "")),
                feature_hash=feature_hash,
                scan_id=None,
                notes=json.dumps(
                    {
                        "notes": notes,
                        "korea_market": korea_market,
                        "trade_mode": getattr(result, "trade_mode", ""),
                        "stop_level": _parse_float(row.get("stop_level")),
                        "take_level": _parse_float(row.get("take_level")),
                        "entry_estimate": _parse_float(row.get("entry_estimate")),
                    },
                    ensure_ascii=False,
                ),
            )
        )
    repository.insert_predictions(rows)
    return run_id


def refresh_prediction_actuals(symbol: str | None = None) -> pd.DataFrame:
    repository = _repo()
    settings = _settings()
    resolver = OutcomeResolver(repository, MarketDataService(settings))
    resolver.resolve(limit=5000, symbol=symbol)
    return load_prediction_log(symbol=symbol)


def load_prediction_log(symbol: str | None = None, status: str | None = None, limit: int | None = None) -> pd.DataFrame:
    repository = _repo()
    frame = repository.prediction_report(limit=int(limit or 5000))
    if frame.empty:
        return frame
    frame["status"] = np.where(frame["resolved_at"].notna(), "resolved", "unresolved")
    if symbol:
        frame = frame[frame["symbol"].astype(str) == str(symbol)]
    if status in {"resolved", "unresolved"}:
        frame = frame[frame["status"] == status]
    return frame.sort_values(["created_at", "forecast_horizon_bars"], ascending=[False, True]).reset_index(drop=True)


def filter_prediction_history(frame: pd.DataFrame, symbol: str | None = None, status: str | None = None) -> pd.DataFrame:
    filtered = frame.copy()
    if filtered.empty:
        return filtered
    if symbol:
        filtered = filtered[filtered["symbol"].astype(str) == str(symbol)]
    if status in {"resolved", "unresolved"} and "status" in filtered.columns:
        filtered = filtered[filtered["status"] == status]
    return filtered.sort_values(["created_at", "forecast_horizon_bars"], ascending=[False, True]).reset_index(drop=True)


def summarize_prediction_accuracy(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {
            "saved_runs": 0.0,
            "saved_rows": 0.0,
            "matured_rows": 0.0,
            "mae_price": np.nan,
            "rmse_price": np.nan,
            "mae_return_pct": np.nan,
            "rmse_return_pct": np.nan,
            "mape_pct": np.nan,
            "direction_acc_pct": np.nan,
            "brier_score": np.nan,
            "avg_trade_return_pct": np.nan,
            "avg_trade_pnl": np.nan,
        }
    matured = frame[frame["status"] == "resolved"].copy() if "status" in frame.columns else frame.copy()
    rmse_price = float(np.sqrt(pd.to_numeric(matured["abs_error_price"], errors="coerce").pow(2).mean())) if not matured.empty else np.nan
    rmse_return = (
        float(np.sqrt(pd.to_numeric(matured["abs_error_return"], errors="coerce").pow(2).mean()) * 100.0)
        if not matured.empty
        else np.nan
    )
    return {
        "saved_runs": float(frame["run_id"].nunique()) if "run_id" in frame.columns else 0.0,
        "saved_rows": float(len(frame)),
        "matured_rows": float(len(matured)),
        "mae_price": float(pd.to_numeric(matured["abs_error_price"], errors="coerce").mean()) if not matured.empty else np.nan,
        "rmse_price": rmse_price,
        "mae_return_pct": float(pd.to_numeric(matured["abs_error_return"], errors="coerce").mean() * 100.0) if not matured.empty else np.nan,
        "rmse_return_pct": rmse_return,
        "mape_pct": float(pd.to_numeric(matured["ape_pct"], errors="coerce").mean()) if not matured.empty else np.nan,
        "direction_acc_pct": float(pd.to_numeric(matured["directional_accuracy"], errors="coerce").mean() * 100.0) if not matured.empty else np.nan,
        "brier_score": float(pd.to_numeric(matured["brier_score"], errors="coerce").mean()) if not matured.empty else np.nan,
        "avg_trade_return_pct": float(pd.to_numeric(matured["paper_trade_return"], errors="coerce").mean() * 100.0) if not matured.empty else np.nan,
        "avg_trade_pnl": float(pd.to_numeric(matured["paper_trade_pnl"], errors="coerce").mean()) if not matured.empty else np.nan,
    }


def latest_prediction_id(symbol: str, forecast_horizon: int = 1) -> str | None:
    frame = load_prediction_log(symbol=symbol)
    if frame.empty:
        return None
    frame = frame[frame["forecast_horizon_bars"] == int(forecast_horizon)]
    return str(frame.iloc[0]["prediction_id"]) if not frame.empty else None


def prediction_id_for_run(run_id: str, forecast_horizon: int = 1) -> str | None:
    frame = load_prediction_log()
    if frame.empty:
        return None
    frame = frame[(frame["run_id"].astype(str) == str(run_id)) & (frame["forecast_horizon_bars"] == int(forecast_horizon))]
    return str(frame.iloc[0]["prediction_id"]) if not frame.empty else None


def attach_order_to_prediction(prediction_id: str, order_id: str) -> None:
    if not prediction_id or not order_id:
        return
    repository = _repo()
    order = repository.get_order(order_id) or repository.find_order_by_broker_order_id(order_id)
    if not order:
        return
    raw_json = json.loads(str(order.get("raw_json") or "{}"))
    raw_json["prediction_id"] = prediction_id
    repository.update_order(str(order["order_id"]), status=str(order["status"]), raw_json=raw_json)


def append_order_log(record: Dict[str, Any]) -> str:
    repository = _repo()
    order_id = str(record.get("order_id") or record.get("broker_order_no") or record.get("order_no") or make_id("ord"))
    existing = repository.get_order(order_id)
    if existing:
        return order_id
    repository.insert_order(
        OrderRecord(
            order_id=order_id,
            created_at=str(record.get("requested_at") or utc_now_iso()),
            updated_at=str(record.get("requested_at") or utc_now_iso()),
            prediction_id=record.get("prediction_id"),
            scan_id=record.get("scan_id"),
            symbol=str(record.get("symbol") or record.get("symbol_code") or ""),
            asset_type=str(record.get("asset_type") or "한국주식"),
            timeframe=str(record.get("timeframe") or "1d"),
            side=str(record.get("side") or "buy"),
            order_type=str(record.get("order_type") or "market"),
            requested_qty=int(record.get("quantity", 0) or 0),
            filled_qty=int(record.get("filled_qty", 0) or 0),
            remaining_qty=int(record.get("remaining_qty", 0) or 0),
            requested_price=_parse_float(record.get("requested_price")),
            limit_price=_parse_float(record.get("limit_price")),
            status=str(record.get("status") or "submitted"),
            fees_estimate=_parse_float(record.get("fees_estimate"), 0.0),
            slippage_bps=_parse_float(record.get("slippage_bps"), 0.0),
            retry_count=int(record.get("retry_count", 0) or 0),
            strategy_version=str(record.get("strategy_version") or _settings().strategy.strategy_version),
            reason=str(record.get("reason") or ""),
            broker_order_id=str(record.get("broker_order_no") or record.get("order_no") or ""),
            error_message=str(record.get("message") or record.get("error_message") or ""),
            raw_json=json.dumps(record, ensure_ascii=False),
        )
    )
    return order_id


def append_fill_log(record: Dict[str, Any]) -> str:
    repository = _repo()
    order_id = str(record.get("order_id") or "")
    if not order_id:
        raise ValueError("order_id is required")
    fill_id = str(record.get("fill_id") or f"fill_{order_id}")
    from storage.models import FillRecord

    repository.insert_fill(
        FillRecord(
            fill_id=fill_id,
            created_at=str(record.get("filled_at") or utc_now_iso()),
            order_id=order_id,
            symbol=str(record.get("symbol") or ""),
            side=str(record.get("side") or ""),
            quantity=int(record.get("quantity", 0) or 0),
            fill_price=_parse_float(record.get("fill_price")),
            fees=_parse_float(record.get("fees"), 0.0),
            slippage_bps=_parse_float(record.get("slippage_bps"), 0.0),
            status=str(record.get("fill_status") or record.get("status") or "filled"),
            raw_json=json.dumps(record, ensure_ascii=False),
        )
    )
    return fill_id


def load_order_log(limit: int = 200) -> pd.DataFrame:
    repository = _repo()
    frame = repository.recent_orders(limit=limit)
    if frame.empty:
        return frame
    if "created_at" in frame.columns:
        frame["requested_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    if "broker_order_id" in frame.columns and "order_no" not in frame.columns:
        frame["order_no"] = frame["broker_order_id"]
    return frame


def load_model_registry() -> pd.DataFrame:
    repository = _repo()
    return repository.load_model_registry(limit=200)


def append_equity_snapshot(summary: Dict[str, Any], holdings: pd.DataFrame | None = None, source: str = "kis_paper") -> None:
    repository = _repo()
    cash = _parse_float(summary.get("cash"), 0.0)
    stock_eval = _parse_float(summary.get("stock_eval"), 0.0)
    total_eval = _parse_float(summary.get("total_eval"), cash + stock_eval)
    pnl = _parse_float(summary.get("pnl"), 0.0)
    repository.insert_account_snapshot(
        AccountSnapshotRecord(
            snapshot_id=make_id("snap"),
            created_at=utc_now_iso(),
            cash=cash,
            equity=total_eval,
            gross_exposure=abs(stock_eval),
            net_exposure=stock_eval,
            realized_pnl=0.0,
            unrealized_pnl=pnl,
            daily_pnl=0.0,
            drawdown_pct=0.0,
            open_positions=int(summary.get("holding_count", len(holdings.index) if holdings is not None else 0) or 0),
            open_orders=0,
            paused=int(repository.get_control_flag_bool("entry_paused") or repository.get_control_flag_bool("worker_paused")),
            source=source,
            raw_json=json.dumps({"summary": summary}, ensure_ascii=False),
        )
    )


def load_equity_curve(limit: int | None = None) -> pd.DataFrame:
    repository = _repo()
    frame = repository.load_account_snapshots(limit=int(limit or 2000))
    if frame.empty:
        return frame
    curve = frame.rename(columns={"created_at": "timestamp", "equity": "total_eval"}).copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
    curve["stock_eval"] = pd.to_numeric(curve["gross_exposure"], errors="coerce")
    curve["return_pct"] = pd.to_numeric(curve["equity"], errors="coerce").pct_change() * 100.0
    curve["holding_count"] = pd.to_numeric(curve["open_positions"], errors="coerce")
    return curve.sort_values("timestamp").reset_index(drop=True)


def compute_equity_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    if equity_curve.empty or "total_eval" not in equity_curve:
        return {
            "samples": 0.0,
            "start_equity": np.nan,
            "latest_equity": np.nan,
            "total_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "exposure_pct": np.nan,
            "max_consecutive_losses": np.nan,
        }
    series = pd.to_numeric(equity_curve["total_eval"], errors="coerce").dropna()
    if series.empty:
        return {
            "samples": 0.0,
            "start_equity": np.nan,
            "latest_equity": np.nan,
            "total_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "exposure_pct": np.nan,
            "max_consecutive_losses": np.nan,
        }
    drawdown = series / series.cummax() - 1.0
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0]
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252.0)) if len(returns) >= 2 and returns.std(ddof=1) > 0 else np.nan
    sortino = float((returns.mean() / downside.std(ddof=1)) * np.sqrt(252.0)) if len(downside) >= 2 and downside.std(ddof=1) > 0 else np.nan
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    negative_sum = abs(float(negative.sum())) if not negative.empty else 0.0
    max_consecutive_losses = 0
    current_losses = 0
    for flag in (returns < 0).tolist():
        if flag:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0
    max_drawdown_pct = float(drawdown.min() * 100.0) if not drawdown.empty else np.nan
    total_return_pct = float((series.iloc[-1] / series.iloc[0] - 1.0) * 100.0) if float(series.iloc[0]) > 0 else np.nan
    calmar = float(total_return_pct / abs(max_drawdown_pct)) if np.isfinite(total_return_pct) and np.isfinite(max_drawdown_pct) and max_drawdown_pct < 0 else np.nan
    exposure_pct = float((pd.to_numeric(equity_curve.get("holding_count"), errors="coerce").fillna(0) > 0).mean() * 100.0) if "holding_count" in equity_curve.columns else np.nan
    return {
        "samples": float(len(series)),
        "start_equity": float(series.iloc[0]),
        "latest_equity": float(series.iloc[-1]),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate_pct": float((returns > 0).mean() * 100.0) if not returns.empty else np.nan,
        "profit_factor": float(positive.sum() / negative_sum) if negative_sum > 0 else (np.inf if not positive.empty else np.nan),
        "exposure_pct": exposure_pct,
        "max_consecutive_losses": float(max_consecutive_losses),
    }
