from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from predictor import download_price_data


SEOUL_TZ = timezone(timedelta(hours=9))
DATA_DIR = Path(".prediction_tracking")
DB_PATH = DATA_DIR / "tracking.sqlite3"


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    korea_market TEXT,
    market_timezone TEXT NOT NULL,
    data_cutoff_at TEXT NOT NULL,
    target_date TEXT NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    horizon_label TEXT NOT NULL,
    target_type TEXT NOT NULL,
    trade_mode TEXT NOT NULL,
    current_price REAL,
    predicted_price REAL,
    predicted_return REAL,
    signal TEXT NOT NULL,
    confidence_score REAL,
    threshold REAL,
    position_size REAL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    validation_mode TEXT NOT NULL,
    feature_hash TEXT,
    notes TEXT,
    paper_order_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_target_date
ON predictions(symbol, target_date);

CREATE INDEX IF NOT EXISTS idx_predictions_run_id
ON predictions(run_id);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    target_date TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    actual_price REAL,
    actual_return REAL,
    outcome_source TEXT,
    notes TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id TEXT PRIMARY KEY,
    evaluated_at TEXT NOT NULL,
    error_return REAL,
    abs_error_return REAL,
    squared_error_return REAL,
    error_price REAL,
    abs_error_price REAL,
    squared_error_price REAL,
    ape_pct REAL,
    directional_accuracy REAL,
    sign_hit_rate REAL,
    brier_score REAL,
    paper_trade_pnl REAL,
    paper_trade_return REAL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_champion INTEGER NOT NULL DEFAULT 1,
    model_params_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    retrain_run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    champion_model_version TEXT,
    challenger_model_version TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    prediction_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    requested_price REAL,
    quote_price REAL,
    requested_at TEXT NOT NULL,
    broker_order_no TEXT,
    message TEXT,
    raw_json TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    fill_status TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price REAL,
    raw_json TEXT,
    FOREIGN KEY(order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS positions (
    position_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price REAL,
    market_price REAL,
    pnl REAL,
    return_pct REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    cash REAL,
    stock_eval REAL,
    total_eval REAL,
    pnl REAL,
    return_pct REAL,
    holding_count INTEGER,
    source TEXT,
    raw_json TEXT
);
"""


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA_SQL)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now(tz_name: str = "Asia/Seoul") -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def _iso(dt: datetime | pd.Timestamp | str) -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            return dt.isoformat()
        return dt.to_pydatetime().isoformat(timespec="seconds")
    return dt.isoformat(timespec="seconds")


def _parse_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _summary_value(summary_df: pd.DataFrame, key: str, default: float | str = "") -> float | str:
    if summary_df is None or summary_df.empty or "item" not in summary_df.columns:
        return default
    row = summary_df.loc[summary_df["item"] == key, "value"]
    if row.empty:
        return default
    value = row.iloc[0]
    if isinstance(default, float):
        try:
            value = float(value)
        except Exception:
            return default
        if np.isnan(value):
            return default
    return value


def _market_timezone(asset_type: str) -> str:
    return "Asia/Seoul"


def _cutoff_timestamp(asset_type: str, cutoff_date: pd.Timestamp) -> str:
    market_tz = _market_timezone(asset_type)
    tz = ZoneInfo(market_tz)
    day = pd.Timestamp(cutoff_date).date()
    if asset_type == "한국주식":
        dt = datetime(day.year, day.month, day.day, 15, 30, tzinfo=tz)
    elif asset_type == "미국주식":
        dt = datetime(day.year, day.month, day.day, 16, 0, tzinfo=tz)
    else:
        dt = datetime(day.year, day.month, day.day, 23, 59, tzinfo=tz)
    return dt.isoformat(timespec="seconds")


def _target_type(result: Any) -> str:
    target_mode = str(getattr(result, "target_mode", "return"))
    if target_mode == "price":
        return "next_close_price"
    return "next_close_return"


def _signal_label(signal_value: float) -> str:
    if signal_value > 1e-12:
        return "LONG"
    if signal_value < -1e-12:
        return "SHORT"
    return "FLAT"


def _confidence_score(planned_signal: float, position_size: float, max_position_size: float, predicted_return: float, threshold: float) -> float:
    size_part = abs(position_size) / max(max_position_size, 1e-9)
    if abs(threshold) > 1e-12:
        move_part = abs(predicted_return) / abs(threshold)
    else:
        move_part = 0.0 if abs(predicted_return) < 1e-12 else 1.0
    signal_part = abs(planned_signal) / max(max_position_size, 1e-9)
    return float(np.clip(max(size_part, move_part, signal_part), 0.0, 1.0))


def _prediction_id(symbol: str, generated_at: datetime, forecast_day: int) -> str:
    clean_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum())
    return f"pred_{generated_at.strftime('%Y%m%dT%H%M%S')}_{clean_symbol}_{forecast_day:02d}"


def init_store() -> None:
    with _connect():
        pass


def register_model_version(result: Any) -> None:
    init_store()
    payload = {
        "target_mode": getattr(result, "target_mode", ""),
        "trade_mode": getattr(result, "trade_mode", ""),
        "validation_mode": getattr(result, "validation_mode", ""),
        "weights": getattr(result, "weights", {}),
        "signal_threshold_pct": getattr(result, "signal_threshold_pct", np.nan),
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO model_registry (
                model_version, model_name, feature_version, created_at, is_champion, model_params_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(getattr(result, "model_version", "")),
                str(getattr(result, "model_name", "")),
                str(getattr(result, "feature_version", "")),
                _iso(_now("Asia/Seoul")),
                1,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                "Current predictor.py ensemble registry entry",
            ),
        )


def save_prediction_snapshot(*, asset_type: str, korea_market: str, result: Any, notes: str = "") -> str:
    init_store()
    register_model_version(result)
    future_frame = getattr(result, "future_frame", pd.DataFrame())
    if future_frame is None or future_frame.empty:
        raise ValueError("저장할 미래 예측 결과가 없습니다.")

    market_tz = _market_timezone(asset_type)
    generated_at = _now(market_tz)
    run_id = f"run_{getattr(result, 'symbol', 'UNKNOWN')}_{generated_at.strftime('%Y%m%dT%H%M%S')}"
    current_price = _parse_float(getattr(result, "latest_close", np.nan))
    threshold_pct = _parse_float(getattr(result, "signal_threshold_pct", np.nan))
    threshold = threshold_pct / 100.0 if np.isfinite(threshold_pct) else np.nan
    max_position_size = _parse_float(_summary_value(getattr(result, "validation_summary", pd.DataFrame()), "max_position_size", 1.0))
    data_cutoff_at = _cutoff_timestamp(asset_type, pd.Timestamp(getattr(result, "data_cutoff_at", pd.Timestamp.utcnow())))

    rows = []
    for forecast_day, (target_dt, row) in enumerate(future_frame.iterrows(), start=1):
        predicted_price = _parse_float(row.get("ensemble_pred"))
        predicted_return_pct = _parse_float(row.get("ensemble_pred_return_pct"))
        predicted_return = predicted_return_pct / 100.0 if np.isfinite(predicted_return_pct) else np.nan
        planned_signal = _parse_float(row.get("planned_signal"))
        position_size = _parse_float(row.get("position_size"))
        prediction_id = _prediction_id(str(getattr(result, "symbol", "")), generated_at, forecast_day)
        rows.append(
            (
                prediction_id,
                run_id,
                _iso(generated_at),
                str(getattr(result, "symbol", "")),
                asset_type,
                korea_market,
                market_tz,
                data_cutoff_at,
                pd.Timestamp(target_dt).date().isoformat(),
                forecast_day,
                f"{forecast_day}d",
                _target_type(result),
                str(getattr(result, "trade_mode", "")),
                current_price,
                predicted_price,
                predicted_return,
                _signal_label(planned_signal),
                _confidence_score(
                    planned_signal=planned_signal,
                    position_size=position_size,
                    max_position_size=max_position_size if np.isfinite(max_position_size) else 1.0,
                    predicted_return=predicted_return,
                    threshold=threshold,
                ),
                threshold,
                position_size,
                str(getattr(result, "model_name", "")),
                str(getattr(result, "model_version", "")),
                str(getattr(result, "feature_version", "")),
                str(getattr(result, "validation_mode", "")),
                str(getattr(result, "feature_hash", "")),
                notes,
                None,
            )
        )

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO predictions (
                prediction_id, run_id, generated_at, symbol, asset_type, korea_market, market_timezone,
                data_cutoff_at, target_date, forecast_horizon, horizon_label, target_type, trade_mode,
                current_price, predicted_price, predicted_return, signal, confidence_score, threshold,
                position_size, model_name, model_version, feature_version, validation_mode, feature_hash,
                notes, paper_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return run_id


def _resolve_symbol_actuals(conn: sqlite3.Connection, symbol: str, pending_rows: pd.DataFrame) -> None:
    valid_dates = pd.to_datetime(pending_rows["target_date"], errors="coerce").dropna()
    if valid_dates.empty:
        return
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    min_date = valid_dates.min()
    years = max(1, int(np.ceil((today - min_date).days / 365.0)) + 1)
    try:
        price_data = download_price_data(symbol=symbol, years=years)
    except Exception:
        return
    if price_data.empty or "Close" not in price_data.columns:
        return

    close_map = pd.Series(price_data["Close"].astype(float).values, index=pd.to_datetime(price_data.index).normalize())
    for row in pending_rows.to_dict("records"):
        target_date = pd.to_datetime(row["target_date"], errors="coerce")
        if pd.isna(target_date):
            continue
        actual_price = close_map.get(pd.Timestamp(target_date).normalize())
        if pd.isna(actual_price):
            continue

        current_price = _parse_float(row.get("current_price"))
        predicted_price = _parse_float(row.get("predicted_price"))
        predicted_return = _parse_float(row.get("predicted_return"))
        confidence = _parse_float(row.get("confidence_score"))
        signal = str(row.get("signal") or "FLAT")

        actual_return = (float(actual_price) / current_price - 1.0) if np.isfinite(current_price) and current_price != 0 else np.nan
        error_return = actual_return - predicted_return if np.isfinite(actual_return) and np.isfinite(predicted_return) else np.nan
        abs_error_return = abs(error_return) if np.isfinite(error_return) else np.nan
        squared_error_return = error_return ** 2 if np.isfinite(error_return) else np.nan
        error_price = float(actual_price) - predicted_price if np.isfinite(predicted_price) else np.nan
        abs_error_price = abs(error_price) if np.isfinite(error_price) else np.nan
        squared_error_price = error_price ** 2 if np.isfinite(error_price) else np.nan
        ape_pct = abs_error_price / abs(float(actual_price)) * 100.0 if np.isfinite(abs_error_price) and float(actual_price) != 0 else np.nan
        predicted_sign = np.sign(predicted_return) if np.isfinite(predicted_return) else 0.0
        actual_sign = np.sign(actual_return) if np.isfinite(actual_return) else 0.0
        directional_accuracy = float(predicted_sign == actual_sign) if np.isfinite(actual_return) and np.isfinite(predicted_return) else np.nan
        sign_hit_rate = directional_accuracy
        position_size = _parse_float(row.get("position_size"))
        if signal == "LONG":
            trade_direction = 1.0
        elif signal == "SHORT":
            trade_direction = -1.0
        else:
            trade_direction = 0.0
        paper_trade_return = (
            actual_return * trade_direction * abs(position_size)
            if np.isfinite(actual_return) and np.isfinite(position_size)
            else np.nan
        )
        paper_trade_pnl = paper_trade_return * current_price if np.isfinite(paper_trade_return) and np.isfinite(current_price) else np.nan
        if signal == "LONG":
            prob_up = 0.5 + 0.5 * max(confidence, 0.0)
        elif signal == "SHORT":
            prob_up = 0.5 - 0.5 * max(confidence, 0.0)
        else:
            prob_up = 0.5
        actual_up = 1.0 if np.isfinite(actual_return) and actual_return > 0 else 0.0
        brier_score = (prob_up - actual_up) ** 2

        conn.execute(
            """
            INSERT OR IGNORE INTO outcomes (
                prediction_id, symbol, target_date, resolved_at, actual_price, actual_return, outcome_source, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["prediction_id"],
                symbol,
                row["target_date"],
                _iso(_now("Asia/Seoul")),
                float(actual_price),
                actual_return,
                "market_data_close",
                "",
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO evaluations (
                prediction_id, evaluated_at, error_return, abs_error_return, squared_error_return,
                error_price, abs_error_price, squared_error_price, ape_pct, directional_accuracy,
                sign_hit_rate, brier_score, paper_trade_pnl, paper_trade_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["prediction_id"],
                _iso(_now("Asia/Seoul")),
                error_return,
                abs_error_return,
                squared_error_return,
                error_price,
                abs_error_price,
                squared_error_price,
                ape_pct,
                directional_accuracy,
                sign_hit_rate,
                brier_score,
                paper_trade_pnl,
                paper_trade_return,
            ),
        )


def refresh_prediction_actuals(symbol: str | None = None) -> pd.DataFrame:
    init_store()
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None).date().isoformat()
    query = """
        SELECT p.*
        FROM predictions p
        LEFT JOIN outcomes o ON o.prediction_id = p.prediction_id
        WHERE o.prediction_id IS NULL
          AND p.target_date <= ?
    """
    params: List[Any] = [today]
    if symbol:
        query += " AND p.symbol = ?"
        params.append(symbol)

    with _connect() as conn:
        pending = pd.read_sql_query(query, conn, params=params)
        if not pending.empty:
            for symbol_key, group in pending.groupby("symbol"):
                _resolve_symbol_actuals(conn=conn, symbol=str(symbol_key), pending_rows=group)
    return load_prediction_log(symbol=symbol)


def load_prediction_log(symbol: str | None = None, status: str | None = None, limit: int | None = None) -> pd.DataFrame:
    init_store()
    query = """
        SELECT
            p.*,
            CASE WHEN o.prediction_id IS NULL THEN 'unresolved' ELSE 'resolved' END AS status,
            o.resolved_at,
            o.actual_price,
            o.actual_return,
            e.evaluated_at,
            e.error_return,
            e.abs_error_return,
            e.squared_error_return,
            e.error_price,
            e.abs_error_price,
            e.squared_error_price,
            e.ape_pct,
            e.directional_accuracy,
            e.sign_hit_rate,
            e.brier_score,
            e.paper_trade_pnl,
            e.paper_trade_return
        FROM predictions p
        LEFT JOIN outcomes o ON o.prediction_id = p.prediction_id
        LEFT JOIN evaluations e ON e.prediction_id = p.prediction_id
    """
    clauses: List[str] = []
    params: List[Any] = []
    if symbol:
        clauses.append("p.symbol = ?")
        params.append(symbol)
    if status in {"unresolved", "resolved"}:
        clauses.append(("o.prediction_id IS NULL") if status == "unresolved" else ("o.prediction_id IS NOT NULL"))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY p.generated_at DESC, p.forecast_horizon ASC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _connect() as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return frame
    for col in ["generated_at", "data_cutoff_at", "target_date", "resolved_at", "evaluated_at"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    numeric_cols = [
        "forecast_horizon",
        "current_price",
        "predicted_price",
        "predicted_return",
        "confidence_score",
        "threshold",
        "position_size",
        "actual_price",
        "actual_return",
        "error_return",
        "abs_error_return",
        "squared_error_return",
        "error_price",
        "abs_error_price",
        "squared_error_price",
        "ape_pct",
        "directional_accuracy",
        "sign_hit_rate",
        "brier_score",
        "paper_trade_pnl",
        "paper_trade_return",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def filter_prediction_history(frame: pd.DataFrame, symbol: str | None = None, status: str | None = None) -> pd.DataFrame:
    filtered = frame.copy()
    if filtered.empty:
        return filtered
    if symbol:
        filtered = filtered[filtered["symbol"].astype(str) == str(symbol)]
    if status in {"unresolved", "resolved"} and "status" in filtered.columns:
        filtered = filtered[filtered["status"] == status]
    return filtered.sort_values(["generated_at", "forecast_horizon"], ascending=[False, True]).reset_index(drop=True)


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
    matured = frame[frame["status"] == "resolved"].copy() if "status" in frame.columns else frame.dropna(subset=["actual_price"]).copy()
    rmse_price = float(np.sqrt(matured["squared_error_price"].mean())) if not matured.empty and matured["squared_error_price"].notna().any() else np.nan
    rmse_return = float(np.sqrt(matured["squared_error_return"].mean()) * 100.0) if not matured.empty and matured["squared_error_return"].notna().any() else np.nan
    return {
        "saved_runs": float(frame["run_id"].nunique()) if "run_id" in frame.columns else 0.0,
        "saved_rows": float(len(frame)),
        "matured_rows": float(len(matured)),
        "mae_price": float(matured["abs_error_price"].mean()) if not matured.empty else np.nan,
        "rmse_price": rmse_price,
        "mae_return_pct": float(matured["abs_error_return"].mean() * 100.0) if not matured.empty else np.nan,
        "rmse_return_pct": rmse_return,
        "mape_pct": float(matured["ape_pct"].mean()) if not matured.empty else np.nan,
        "direction_acc_pct": float(matured["directional_accuracy"].mean() * 100.0) if not matured.empty else np.nan,
        "brier_score": float(matured["brier_score"].mean()) if not matured.empty else np.nan,
        "avg_trade_return_pct": float(matured["paper_trade_return"].mean() * 100.0) if not matured.empty else np.nan,
        "avg_trade_pnl": float(matured["paper_trade_pnl"].mean()) if not matured.empty else np.nan,
    }


def latest_prediction_id(symbol: str, forecast_horizon: int = 1) -> str | None:
    init_store()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT prediction_id
            FROM predictions
            WHERE symbol = ? AND forecast_horizon = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (symbol, int(forecast_horizon)),
        ).fetchone()
    return str(row["prediction_id"]) if row else None


def prediction_id_for_run(run_id: str, forecast_horizon: int = 1) -> str | None:
    init_store()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT prediction_id
            FROM predictions
            WHERE run_id = ? AND forecast_horizon = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (str(run_id), int(forecast_horizon)),
        ).fetchone()
    return str(row["prediction_id"]) if row else None


def attach_order_to_prediction(prediction_id: str, order_id: str) -> None:
    if not prediction_id or not order_id:
        return
    init_store()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE predictions
            SET paper_order_id = ?
            WHERE prediction_id = ?
            """,
            (str(order_id), str(prediction_id)),
        )


def append_order_log(record: Dict[str, Any]) -> str:
    init_store()
    order_id = str(record.get("order_id") or record.get("order_no") or record.get("broker_order_no") or "")
    if not order_id:
        base = f"{record.get('requested_at','')}_{record.get('symbol','')}_{record.get('side','')}_{record.get('quantity',0)}"
        order_id = f"order_{abs(hash(base))}"

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO orders (
                order_id, prediction_id, symbol, side, order_type, quantity, requested_price,
                quote_price, requested_at, broker_order_no, message, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                record.get("prediction_id"),
                record.get("symbol"),
                record.get("side"),
                record.get("order_type"),
                int(record.get("quantity", 0) or 0),
                _parse_float(record.get("requested_price")),
                _parse_float(record.get("quote_price")),
                str(record.get("requested_at") or _iso(_now("Asia/Seoul"))),
                str(record.get("order_no") or record.get("broker_order_no") or ""),
                str(record.get("message") or ""),
                json.dumps(record, ensure_ascii=False),
            ),
        )
        if record.get("prediction_id"):
            conn.execute(
                """
                UPDATE predictions
                SET paper_order_id = ?
                WHERE prediction_id = ?
                """,
                (order_id, str(record.get("prediction_id"))),
            )
    return order_id


def append_fill_log(record: Dict[str, Any]) -> str:
    init_store()
    fill_id = str(record.get("fill_id") or "")
    if not fill_id:
        base = f"{record.get('order_id','')}_{record.get('filled_at','')}_{record.get('quantity',0)}_{record.get('fill_status','')}"
        fill_id = f"fill_{abs(hash(base))}"

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO fills (
                fill_id, order_id, fill_status, filled_at, quantity, fill_price, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_id,
                str(record.get("order_id") or ""),
                str(record.get("fill_status") or "unknown"),
                str(record.get("filled_at") or _iso(_now("Asia/Seoul"))),
                int(record.get("quantity", 0) or 0),
                _parse_float(record.get("fill_price")),
                json.dumps(record, ensure_ascii=False),
            ),
        )
    return fill_id


def load_order_log(limit: int = 200) -> pd.DataFrame:
    init_store()
    with _connect() as conn:
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM orders
            ORDER BY requested_at DESC
            LIMIT ?
            """,
            conn,
            params=[int(limit)],
        )
    if frame.empty:
        return frame
    if "raw_json" in frame.columns:
        expanded_rows: List[Dict[str, Any]] = []
        for payload in frame["raw_json"].fillna("{}").astype(str).tolist():
            try:
                expanded_rows.append(json.loads(payload))
            except Exception:
                expanded_rows.append({})
        expanded = pd.DataFrame(expanded_rows)
        if not expanded.empty:
            for col in expanded.columns:
                if col not in frame.columns:
                    frame[col] = expanded[col]
                else:
                    frame[col] = frame[col].where(frame[col].notna(), expanded[col])
    if "order_no" not in frame.columns and "broker_order_no" in frame.columns:
        frame["order_no"] = frame["broker_order_no"]
    if "requested_at" in frame.columns:
        frame["requested_at"] = pd.to_datetime(frame["requested_at"], errors="coerce", utc=True)
    return frame


def load_model_registry() -> pd.DataFrame:
    init_store()
    with _connect() as conn:
        frame = pd.read_sql_query(
            """
            SELECT model_version, model_name, feature_version, created_at, is_champion, model_params_json, notes
            FROM model_registry
            ORDER BY created_at DESC, model_version DESC
            """,
            conn,
        )
    if frame.empty:
        return frame
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    frame["is_champion"] = pd.to_numeric(frame["is_champion"], errors="coerce").fillna(0).astype(int)
    return frame


def append_equity_snapshot(summary: Dict[str, Any], holdings: pd.DataFrame | None = None, source: str = "kis_paper") -> None:
    init_store()
    captured_at = _iso(_now("Asia/Seoul"))
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO account_snapshots (
                captured_at, cash, stock_eval, total_eval, pnl, return_pct, holding_count, source, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                captured_at,
                _parse_float(summary.get("cash")),
                _parse_float(summary.get("stock_eval")),
                _parse_float(summary.get("total_eval")),
                _parse_float(summary.get("pnl")),
                _parse_float(summary.get("return_pct")),
                int(summary.get("holding_count", 0) or 0),
                source,
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        if holdings is not None and not holdings.empty:
            for row in holdings.to_dict("records"):
                quantity_value = _parse_float(row.get("보유수량"))
                conn.execute(
                    """
                    INSERT INTO positions (
                        captured_at, symbol, quantity, avg_price, market_price, pnl, return_pct, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        captured_at,
                        str(row.get("symbol_code") or row.get("심볼") or ""),
                        int(quantity_value) if np.isfinite(quantity_value) else 0,
                        _parse_float(row.get("매입평균가")),
                        _parse_float(row.get("현재가")),
                        _parse_float(row.get("평가손익")),
                        _parse_float(row.get("수익률(%)")),
                        source,
                    ),
                )


def load_equity_curve(limit: int | None = None) -> pd.DataFrame:
    init_store()
    query = """
        SELECT captured_at AS timestamp, cash, stock_eval, total_eval, pnl, return_pct, holding_count
        FROM account_snapshots
        ORDER BY captured_at ASC
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _connect() as conn:
        frame = pd.read_sql_query(query, conn)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for col in ("cash", "stock_eval", "total_eval", "pnl", "return_pct"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp"]).reset_index(drop=True)


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
    start_equity = float(series.iloc[0])
    latest_equity = float(series.iloc[-1])
    total_return_pct = (latest_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else np.nan
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0]
    sharpe = np.nan
    sortino = np.nan
    calmar = np.nan
    win_rate_pct = np.nan
    profit_factor = np.nan
    max_consecutive_losses = np.nan
    exposure_pct = np.nan
    if not returns.empty:
        mean_ret = float(returns.mean())
        std_ret = float(returns.std(ddof=1))
        down_std = float(downside.std(ddof=1)) if len(downside) >= 2 else np.nan
        periods_per_year = 252.0
        if np.isfinite(std_ret) and std_ret > 0:
            sharpe = float((mean_ret / std_ret) * np.sqrt(periods_per_year))
        if np.isfinite(down_std) and down_std > 0:
            sortino = float((mean_ret / down_std) * np.sqrt(periods_per_year))
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        win_rate_pct = float((returns > 0).mean() * 100.0)
        negative_sum = abs(float(negative.sum())) if not negative.empty else 0.0
        profit_factor = float(positive.sum() / negative_sum) if negative_sum > 0 else (np.inf if not positive.empty else np.nan)
        loss_runs = (returns < 0).astype(int)
        current_losses = 0
        max_losses = 0
        for flag in loss_runs.tolist():
            if flag:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        max_consecutive_losses = float(max_losses)
    max_drawdown_pct = float(drawdown.min() * 100.0) if not drawdown.empty else np.nan
    if np.isfinite(total_return_pct) and np.isfinite(max_drawdown_pct) and max_drawdown_pct < 0:
        calmar = float(total_return_pct / abs(max_drawdown_pct))
    if "holding_count" in equity_curve.columns:
        holding_count = pd.to_numeric(equity_curve["holding_count"], errors="coerce")
        valid_holdings = holding_count.dropna()
        if not valid_holdings.empty:
            exposure_pct = float((valid_holdings > 0).mean() * 100.0)
    return {
        "samples": float(len(series)),
        "start_equity": start_equity,
        "latest_equity": latest_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "exposure_pct": exposure_pct,
        "max_consecutive_losses": max_consecutive_losses,
    }
