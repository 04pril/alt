from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd

from storage.models import (
    AccountSnapshotRecord,
    CandidateScanRecord,
    EvaluationRecord,
    FillRecord,
    OrderRecord,
    OutcomeRecord,
    PositionRecord,
    PredictionRecord,
    SystemEventRecord,
)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    market_timezone TEXT NOT NULL,
    data_cutoff_at TEXT NOT NULL,
    target_at TEXT NOT NULL,
    forecast_horizon_bars INTEGER NOT NULL,
    target_type TEXT NOT NULL,
    current_price REAL,
    predicted_price REAL,
    predicted_return REAL,
    signal TEXT NOT NULL,
    score REAL,
    confidence REAL,
    threshold REAL,
    expected_return REAL,
    expected_risk REAL,
    position_size REAL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    validation_mode TEXT NOT NULL,
    feature_hash TEXT,
    status TEXT NOT NULL,
    scan_id TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_target ON predictions(symbol, target_at);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    actual_price REAL,
    actual_return REAL,
    outcome_source TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    error_price REAL,
    abs_error_price REAL,
    squared_error_price REAL,
    error_return REAL,
    abs_error_return REAL,
    squared_error_return REAL,
    ape_pct REAL,
    directional_accuracy REAL,
    sign_hit_rate REAL,
    brier_score REAL,
    paper_trade_return REAL,
    paper_trade_pnl REAL
);

CREATE TABLE IF NOT EXISTS candidate_scans (
    scan_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    score REAL,
    rank INTEGER,
    status TEXT NOT NULL,
    reason TEXT,
    expected_return REAL,
    expected_risk REAL,
    confidence REAL,
    threshold REAL,
    volatility REAL,
    liquidity_score REAL,
    cost_bps REAL,
    recent_performance REAL,
    signal TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    cooldown_until TEXT,
    is_holding INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    prediction_id TEXT,
    scan_id TEXT,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    requested_qty INTEGER NOT NULL,
    filled_qty INTEGER NOT NULL,
    remaining_qty INTEGER NOT NULL,
    requested_price REAL,
    limit_price REAL,
    status TEXT NOT NULL,
    fees_estimate REAL,
    slippage_bps REAL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    strategy_version TEXT NOT NULL,
    reason TEXT,
    broker_order_id TEXT,
    error_message TEXT,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price REAL,
    fees REAL,
    slippage_bps REAL,
    status TEXT NOT NULL,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT,
    prediction_id TEXT,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL,
    mark_price REAL,
    stop_loss REAL,
    take_profit REAL,
    trailing_stop REAL,
    highest_price REAL,
    lowest_price REAL,
    unrealized_pnl REAL,
    realized_pnl REAL,
    expected_risk REAL,
    exposure_value REAL,
    max_holding_until TEXT,
    strategy_version TEXT NOT NULL,
    cooldown_until TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    cash REAL,
    equity REAL,
    gross_exposure REAL,
    net_exposure REAL,
    realized_pnl REAL,
    unrealized_pnl REAL,
    daily_pnl REAL,
    drawdown_pct REAL,
    open_positions INTEGER,
    open_orders INTEGER,
    paused INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    promoted_at TEXT,
    is_champion INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    retrain_run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    champion_before TEXT,
    challenger_version TEXT,
    champion_after TEXT,
    directional_accuracy_pct REAL,
    mae_pct REAL,
    trade_return_pct REAL,
    max_drawdown_pct REAL,
    summary_json TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS job_runs (
    job_run_id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    run_key TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    lock_owner TEXT,
    error_message TEXT,
    metrics_json TEXT,
    UNIQUE(job_name, run_key)
);

CREATE TABLE IF NOT EXISTS system_events (
    event_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS control_flags (
    flag_name TEXT PRIMARY KEY,
    flag_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class TradingRepository:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA_SQL)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect():
            pass

    def set_control_flag(self, flag_name: str, flag_value: str, notes: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO control_flags(flag_name, flag_value, updated_at, notes)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(flag_name) DO UPDATE SET
                    flag_value=excluded.flag_value,
                    updated_at=excluded.updated_at,
                    notes=excluded.notes
                """,
                (flag_name, flag_value, utc_now_iso(), notes),
            )

    def get_control_flag(self, flag_name: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT flag_value FROM control_flags WHERE flag_name = ?", (flag_name,)).fetchone()
        return str(row["flag_value"]) if row else default

    def insert_system_event(self, record: SystemEventRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO system_events(event_id, created_at, level, component, event_type, message, details_json)
                VALUES(:event_id, :created_at, :level, :component, :event_type, :message, :details_json)
                """,
                asdict(record),
            )

    def log_event(self, level: str, component: str, event_type: str, message: str, details: Dict[str, Any] | None = None) -> None:
        self.insert_system_event(
            SystemEventRecord(
                event_id=make_id("evt"),
                created_at=utc_now_iso(),
                level=level,
                component=component,
                event_type=event_type,
                message=message,
                details_json=json.dumps(details or {}, ensure_ascii=False),
            )
        )

    def begin_job_run(self, job_name: str, run_key: str, scheduled_at: str, lock_owner: str) -> str | None:
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT job_run_id, status FROM job_runs WHERE job_name = ? AND run_key = ?",
                (job_name, run_key),
            ).fetchone()
            if existing and str(existing["status"]) in {"running", "completed"}:
                return None
            job_run_id = str(existing["job_run_id"]) if existing else make_id("job")
            conn.execute(
                """
                INSERT INTO job_runs(job_run_id, job_name, run_key, scheduled_at, started_at, status, retry_count, lock_owner, error_message, metrics_json)
                VALUES(?, ?, ?, ?, ?, 'running', 0, ?, '', '{}')
                ON CONFLICT(job_name, run_key) DO UPDATE SET
                    started_at=excluded.started_at,
                    status='running',
                    lock_owner=excluded.lock_owner,
                    error_message=''
                """,
                (job_run_id, job_name, run_key, scheduled_at, utc_now_iso(), lock_owner),
            )
        return job_run_id

    def finish_job_run(self, job_run_id: str, status: str, error_message: str = "", metrics: Dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE job_runs
                SET finished_at = ?, status = ?, error_message = ?, metrics_json = ?
                WHERE job_run_id = ?
                """,
                (utc_now_iso(), status, error_message, json.dumps(metrics or {}, ensure_ascii=False), job_run_id),
            )

    def record_model_version(
        self,
        model_version: str,
        model_name: str,
        feature_version: str,
        strategy_version: str,
        metrics: Dict[str, Any] | None = None,
        is_champion: bool = False,
        notes: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_registry(model_version, model_name, feature_version, strategy_version, created_at, promoted_at, is_champion, metrics_json, notes)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    metrics_json=excluded.metrics_json,
                    notes=excluded.notes
                """,
                (
                    model_version,
                    model_name,
                    feature_version,
                    strategy_version,
                    utc_now_iso(),
                    utc_now_iso() if is_champion else None,
                    int(is_champion),
                    json.dumps(metrics or {}, ensure_ascii=False),
                    notes,
                ),
            )

    def promote_model(self, model_version: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE model_registry SET is_champion = 0")
            conn.execute(
                "UPDATE model_registry SET is_champion = 1, promoted_at = ? WHERE model_version = ?",
                (utc_now_iso(), model_version),
            )

    def insert_retrain_run(self, payload: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO retrain_runs(
                    retrain_run_id, created_at, finished_at, status, champion_before, challenger_version,
                    champion_after, directional_accuracy_pct, mae_pct, trade_return_pct, max_drawdown_pct, summary_json, error_message
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["retrain_run_id"],
                    payload["created_at"],
                    payload.get("finished_at"),
                    payload["status"],
                    payload.get("champion_before"),
                    payload.get("challenger_version"),
                    payload.get("champion_after"),
                    payload.get("directional_accuracy_pct"),
                    payload.get("mae_pct"),
                    payload.get("trade_return_pct"),
                    payload.get("max_drawdown_pct"),
                    json.dumps(payload.get("summary", {}), ensure_ascii=False),
                    payload.get("error_message", ""),
                ),
            )

    def insert_predictions(self, records: Iterable[PredictionRecord]) -> None:
        rows = [asdict(record) for record in records]
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO predictions (
                    prediction_id, created_at, run_id, symbol, asset_type, timeframe, market_timezone,
                    data_cutoff_at, target_at, forecast_horizon_bars, target_type, current_price, predicted_price,
                    predicted_return, signal, score, confidence, threshold, expected_return, expected_risk,
                    position_size, model_name, model_version, feature_version, strategy_version, validation_mode,
                    feature_hash, status, scan_id, notes
                ) VALUES (
                    :prediction_id, :created_at, :run_id, :symbol, :asset_type, :timeframe, :market_timezone,
                    :data_cutoff_at, :target_at, :forecast_horizon_bars, :target_type, :current_price, :predicted_price,
                    :predicted_return, :signal, :score, :confidence, :threshold, :expected_return, :expected_risk,
                    :position_size, :model_name, :model_version, :feature_version, :strategy_version, :validation_mode,
                    :feature_hash, :status, :scan_id, :notes
                )
                """,
                rows,
            )

    def unresolved_predictions(self, limit: int = 500) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT p.*
                FROM predictions p
                LEFT JOIN outcomes o ON o.prediction_id = p.prediction_id
                WHERE o.prediction_id IS NULL
                ORDER BY p.target_at ASC
                LIMIT ?
                """,
                conn,
                params=[int(limit)],
            )

    def insert_outcome(self, record: OutcomeRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO outcomes(
                    prediction_id, created_at, resolved_at, symbol, asset_type, timeframe, actual_price, actual_return, outcome_source, notes
                ) VALUES(
                    :prediction_id, :created_at, :resolved_at, :symbol, :asset_type, :timeframe, :actual_price, :actual_return, :outcome_source, :notes
                )
                """,
                asdict(record),
            )
            conn.execute("UPDATE predictions SET status = 'resolved' WHERE prediction_id = ?", (record.prediction_id,))

    def insert_evaluation(self, record: EvaluationRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations(
                    prediction_id, created_at, error_price, abs_error_price, squared_error_price, error_return,
                    abs_error_return, squared_error_return, ape_pct, directional_accuracy, sign_hit_rate, brier_score,
                    paper_trade_return, paper_trade_pnl
                ) VALUES(
                    :prediction_id, :created_at, :error_price, :abs_error_price, :squared_error_price, :error_return,
                    :abs_error_return, :squared_error_return, :ape_pct, :directional_accuracy, :sign_hit_rate, :brier_score,
                    :paper_trade_return, :paper_trade_pnl
                )
                """,
                asdict(record),
            )

    def recent_prediction_performance(self, symbol: str, lookback_days: int = 30) -> Dict[str, float]:
        with self.connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT e.directional_accuracy, e.paper_trade_return, e.abs_error_return
                FROM evaluations e
                JOIN predictions p ON p.prediction_id = e.prediction_id
                WHERE p.symbol = ?
                  AND datetime(e.created_at) >= datetime('now', ?)
                """,
                conn,
                params=[symbol, f"-{int(lookback_days)} day"],
            )
        if frame.empty:
            return {"directional_accuracy": 0.5, "paper_trade_return": 0.0, "abs_error_return": 0.0}
        return {
            "directional_accuracy": float(pd.to_numeric(frame["directional_accuracy"], errors="coerce").mean()),
            "paper_trade_return": float(pd.to_numeric(frame["paper_trade_return"], errors="coerce").mean()),
            "abs_error_return": float(pd.to_numeric(frame["abs_error_return"], errors="coerce").mean()),
        }

    def insert_candidate_scans(self, records: Iterable[CandidateScanRecord]) -> None:
        rows = [asdict(record) for record in records]
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candidate_scans(
                    scan_id, created_at, symbol, asset_type, timeframe, score, rank, status, reason,
                    expected_return, expected_risk, confidence, threshold, volatility, liquidity_score,
                    cost_bps, recent_performance, signal, model_version, feature_version, strategy_version,
                    cooldown_until, is_holding, raw_json
                ) VALUES(
                    :scan_id, :created_at, :symbol, :asset_type, :timeframe, :score, :rank, :status, :reason,
                    :expected_return, :expected_risk, :confidence, :threshold, :volatility, :liquidity_score,
                    :cost_bps, :recent_performance, :signal, :model_version, :feature_version, :strategy_version,
                    :cooldown_until, :is_holding, :raw_json
                )
                """,
                rows,
            )

    def latest_candidates(self, asset_type: str | None = None, timeframe: str | None = None, limit: int = 50) -> pd.DataFrame:
        clauses: List[str] = []
        params: List[Any] = []
        if asset_type:
            clauses.append("asset_type = ?")
            params.append(asset_type)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        query = "SELECT * FROM candidate_scans"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, rank ASC LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def insert_order(self, record: OrderRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders(
                    order_id, created_at, updated_at, prediction_id, scan_id, symbol, asset_type, timeframe, side, order_type,
                    requested_qty, filled_qty, remaining_qty, requested_price, limit_price, status, fees_estimate, slippage_bps,
                    retry_count, strategy_version, reason, broker_order_id, error_message, raw_json
                ) VALUES(
                    :order_id, :created_at, :updated_at, :prediction_id, :scan_id, :symbol, :asset_type, :timeframe, :side, :order_type,
                    :requested_qty, :filled_qty, :remaining_qty, :requested_price, :limit_price, :status, :fees_estimate, :slippage_bps,
                    :retry_count, :strategy_version, :reason, :broker_order_id, :error_message, :raw_json
                )
                """,
                asdict(record),
            )

    def update_order(
        self,
        order_id: str,
        *,
        status: str,
        filled_qty: int | None = None,
        remaining_qty: int | None = None,
        error_message: str = "",
        raw_json: Dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute("SELECT filled_qty, remaining_qty, raw_json FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if not row:
                return
            payload = json.loads(str(row["raw_json"] or "{}"))
            if raw_json:
                payload.update(raw_json)
            conn.execute(
                """
                UPDATE orders
                SET updated_at = ?, status = ?, filled_qty = ?, remaining_qty = ?, error_message = ?, raw_json = ?
                WHERE order_id = ?
                """,
                (
                    utc_now_iso(),
                    status,
                    int(filled_qty if filled_qty is not None else row["filled_qty"]),
                    int(remaining_qty if remaining_qty is not None else row["remaining_qty"]),
                    error_message,
                    json.dumps(payload, ensure_ascii=False),
                    order_id,
                ),
            )

    def open_orders(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM orders WHERE status IN ('new', 'partially_filled') ORDER BY created_at ASC",
                conn,
            )

    def insert_fill(self, record: FillRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO fills(fill_id, created_at, order_id, symbol, side, quantity, fill_price, fees, slippage_bps, status, raw_json)
                VALUES(:fill_id, :created_at, :order_id, :symbol, :side, :quantity, :fill_price, :fees, :slippage_bps, :status, :raw_json)
                """,
                asdict(record),
            )

    def upsert_position(self, record: PositionRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO positions(
                    position_id, created_at, updated_at, closed_at, prediction_id, symbol, asset_type, timeframe, side, status,
                    quantity, entry_price, mark_price, stop_loss, take_profit, trailing_stop, highest_price, lowest_price,
                    unrealized_pnl, realized_pnl, expected_risk, exposure_value, max_holding_until, strategy_version, cooldown_until, notes
                ) VALUES(
                    :position_id, :created_at, :updated_at, :closed_at, :prediction_id, :symbol, :asset_type, :timeframe, :side, :status,
                    :quantity, :entry_price, :mark_price, :stop_loss, :take_profit, :trailing_stop, :highest_price, :lowest_price,
                    :unrealized_pnl, :realized_pnl, :expected_risk, :exposure_value, :max_holding_until, :strategy_version, :cooldown_until, :notes
                )
                ON CONFLICT(position_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    closed_at=excluded.closed_at,
                    status=excluded.status,
                    quantity=excluded.quantity,
                    mark_price=excluded.mark_price,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    trailing_stop=excluded.trailing_stop,
                    highest_price=excluded.highest_price,
                    lowest_price=excluded.lowest_price,
                    unrealized_pnl=excluded.unrealized_pnl,
                    realized_pnl=excluded.realized_pnl,
                    expected_risk=excluded.expected_risk,
                    exposure_value=excluded.exposure_value,
                    max_holding_until=excluded.max_holding_until,
                    cooldown_until=excluded.cooldown_until,
                    notes=excluded.notes
                """,
                asdict(record),
            )

    def open_positions(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM positions WHERE status = 'open' ORDER BY created_at ASC",
                conn,
            )

    def latest_position_by_symbol(self, symbol: str, timeframe: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM positions
                WHERE symbol = ? AND timeframe = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                conn,
                params=[symbol, timeframe],
            )

    def latest_cooldown_until(self, symbol: str, timeframe: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT cooldown_until
                FROM positions
                WHERE symbol = ? AND timeframe = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (symbol, timeframe),
            ).fetchone()
        if not row or row["cooldown_until"] is None:
            return None
        return str(row["cooldown_until"])

    def insert_account_snapshot(self, record: AccountSnapshotRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO account_snapshots(
                    snapshot_id, created_at, cash, equity, gross_exposure, net_exposure, realized_pnl, unrealized_pnl,
                    daily_pnl, drawdown_pct, open_positions, open_orders, paused, source, raw_json
                ) VALUES(
                    :snapshot_id, :created_at, :cash, :equity, :gross_exposure, :net_exposure, :realized_pnl, :unrealized_pnl,
                    :daily_pnl, :drawdown_pct, :open_positions, :open_orders, :paused, :source, :raw_json
                )
                """,
                asdict(record),
            )

    def latest_account_snapshot(self) -> Dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def load_account_snapshots(self, limit: int = 500) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT ?",
                conn,
                params=[int(limit)],
            )

    def count_daily_entries(self, created_date: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM orders
                WHERE substr(created_at, 1, 10) = ?
                  AND side IN ('buy', 'sell')
                  AND status IN ('new', 'partially_filled', 'filled')
                """,
                (created_date,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def recent_closed_realized_pnl(self, created_date: str) -> float:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(realized_pnl), 0.0) AS pnl
                FROM positions
                WHERE status = 'closed'
                  AND closed_at IS NOT NULL
                  AND substr(closed_at, 1, 10) = ?
                """,
                (created_date,),
            ).fetchone()
        return float(row["pnl"]) if row else 0.0

    def dashboard_counts(self) -> Dict[str, Any]:
        with self.connect() as conn:
            unresolved = conn.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE status = 'unresolved'").fetchone()
            open_positions = conn.execute("SELECT COUNT(*) AS cnt FROM positions WHERE status = 'open'").fetchone()
            open_orders = conn.execute("SELECT COUNT(*) AS cnt FROM orders WHERE status IN ('new', 'partially_filled')").fetchone()
            latest_account = conn.execute("SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1").fetchone()
        return {
            "unresolved_predictions": int(unresolved["cnt"]) if unresolved else 0,
            "open_positions": int(open_positions["cnt"]) if open_positions else 0,
            "open_orders": int(open_orders["cnt"]) if open_orders else 0,
            "latest_account": dict(latest_account) if latest_account else {},
        }

    def recent_job_health(self, limit: int = 20) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM job_runs
                ORDER BY COALESCE(finished_at, started_at, scheduled_at) DESC
                LIMIT ?
                """,
                conn,
                params=[int(limit)],
            )

    def latest_job_heartbeat(self) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    job_run_id,
                    job_name,
                    status,
                    scheduled_at,
                    started_at,
                    finished_at,
                    COALESCE(finished_at, started_at, scheduled_at) AS heartbeat_at
                FROM job_runs
                ORDER BY COALESCE(finished_at, started_at, scheduled_at) DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else {}

    def recent_system_events(self, level: str | None = None, limit: int = 50) -> pd.DataFrame:
        query = "SELECT * FROM system_events"
        params: List[Any] = []
        if level:
            query += " WHERE level = ?"
            params.append(level)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def prediction_report(self, limit: int = 500) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    p.*,
                    o.resolved_at,
                    o.actual_price,
                    o.actual_return,
                    e.error_price,
                    e.abs_error_price,
                    e.error_return,
                    e.abs_error_return,
                    e.ape_pct,
                    e.directional_accuracy,
                    e.paper_trade_return,
                    e.paper_trade_pnl
                FROM predictions p
                LEFT JOIN outcomes o ON o.prediction_id = p.prediction_id
                LEFT JOIN evaluations e ON e.prediction_id = p.prediction_id
                ORDER BY p.created_at DESC
                LIMIT ?
                """,
                conn,
                params=[int(limit)],
            )

    def prediction_by_scan(self, scan_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM predictions
                WHERE scan_id = ?
                ORDER BY created_at DESC, forecast_horizon_bars ASC
                """,
                conn,
                params=[scan_id],
            )

    def trade_performance_report(self) -> Dict[str, float]:
        equity = self.load_account_snapshots(limit=1000)
        if equity.empty:
            return {"samples": 0.0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "today_pnl": 0.0}
        equity = equity.sort_values("created_at")
        values = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if values.empty:
            return {"samples": 0.0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "today_pnl": 0.0}
        drawdown = values / values.cummax() - 1.0
        latest = float(values.iloc[-1])
        start = float(values.iloc[0])
        today = str(pd.Timestamp.utcnow().date())
        daily_pnl = self.recent_closed_realized_pnl(today)
        return {
            "samples": float(len(values)),
            "total_return_pct": (latest / start - 1.0) * 100.0 if start > 0 else np.nan,
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "today_pnl": float(daily_pnl),
        }
