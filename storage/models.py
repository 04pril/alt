from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    created_at: str
    run_id: str
    symbol: str
    asset_type: str
    timeframe: str
    market_timezone: str
    data_cutoff_at: str
    target_at: str
    forecast_horizon_bars: int
    target_type: str
    current_price: float
    predicted_price: float
    predicted_return: float
    signal: str
    score: float
    confidence: float
    threshold: float
    expected_return: float
    expected_risk: float
    position_size: float
    model_name: str
    model_version: str
    feature_version: str
    strategy_version: str
    validation_mode: str
    feature_hash: str
    status: str = 'unresolved'
    scan_id: str | None = None
    notes: str = ''
    execution_account_id: str = ''


@dataclass(frozen=True)
class OutcomeRecord:
    prediction_id: str
    created_at: str
    resolved_at: str
    symbol: str
    asset_type: str
    timeframe: str
    actual_price: float
    actual_return: float
    outcome_source: str
    notes: str = ''


@dataclass(frozen=True)
class EvaluationRecord:
    prediction_id: str
    created_at: str
    error_price: float
    abs_error_price: float
    squared_error_price: float
    error_return: float
    abs_error_return: float
    squared_error_return: float
    ape_pct: float
    directional_accuracy: float
    sign_hit_rate: float
    brier_score: float
    paper_trade_return: float
    paper_trade_pnl: float


@dataclass(frozen=True)
class CandidateScanRecord:
    scan_id: str
    created_at: str
    symbol: str
    asset_type: str
    timeframe: str
    score: float
    rank: int
    status: str
    reason: str
    expected_return: float
    expected_risk: float
    confidence: float
    threshold: float
    volatility: float
    liquidity_score: float
    cost_bps: float
    recent_performance: float
    signal: str
    model_version: str
    feature_version: str
    strategy_version: str
    cooldown_until: str | None = None
    is_holding: int = 0
    raw_json: str = '{}'
    execution_account_id: str = ''


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    created_at: str
    updated_at: str
    prediction_id: str | None
    scan_id: str | None
    symbol: str
    asset_type: str
    timeframe: str
    side: str
    order_type: str
    requested_qty: int
    filled_qty: int
    remaining_qty: int
    requested_price: float
    limit_price: float
    status: str
    fees_estimate: float
    slippage_bps: float
    retry_count: int
    strategy_version: str
    reason: str = ''
    broker_order_id: str = ''
    error_message: str = ''
    raw_json: str = '{}'
    account_id: str = ''


@dataclass(frozen=True)
class FillRecord:
    fill_id: str
    created_at: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    fill_price: float
    fees: float
    slippage_bps: float
    status: str
    raw_json: str = '{}'
    account_id: str = ''


@dataclass(frozen=True)
class PositionRecord:
    position_id: str
    created_at: str
    updated_at: str
    closed_at: str | None
    prediction_id: str | None
    symbol: str
    asset_type: str
    timeframe: str
    side: str
    status: str
    quantity: int
    entry_price: float
    mark_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    highest_price: float
    lowest_price: float
    unrealized_pnl: float
    realized_pnl: float
    expected_risk: float
    exposure_value: float
    max_holding_until: str
    strategy_version: str
    cooldown_until: str | None = None
    notes: str = ''
    account_id: str = ''
    strategy_family: str = ''
    session_mode: str = ''
    price_policy: str = ''


@dataclass(frozen=True)
class AccountSnapshotRecord:
    snapshot_id: str
    created_at: str
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    drawdown_pct: float
    open_positions: int
    open_orders: int
    paused: int
    source: str
    raw_json: str = '{}'
    account_id: str = ''


@dataclass(frozen=True)
class BrokerAccountRecord:
    account_id: str
    broker_mode: str
    asset_scope: str
    currency: str
    display_name: str
    is_active: int
    metadata_json: str = '{}'


@dataclass(frozen=True)
class LiveMarketQuoteRecord:
    symbol_code: str
    symbol: str
    asset_type: str
    currency: str
    source: str
    current_price: float
    previous_close: float
    change_pct: float
    ask_price: float
    bid_price: float
    volume: float
    updated_at: str
    raw_json: str = '{}'


@dataclass(frozen=True)
class JobRunRecord:
    job_run_id: str
    job_name: str
    run_key: str
    scheduled_at: str
    started_at: str
    finished_at: str
    status: str
    retry_count: int
    lock_owner: str
    error_message: str = ''
    metrics_json: str = '{}'


@dataclass(frozen=True)
class JobRunLease:
    job_run_id: str
    acquired: bool
    status: str
    retry_count: int = 0


@dataclass(frozen=True)
class SystemEventRecord:
    event_id: str
    created_at: str
    level: str
    component: str
    event_type: str
    message: str
    details_json: str = '{}'
    account_id: str = ''


@dataclass(frozen=True)
class DashboardSnapshot:
    summary: Dict[str, Any] = field(default_factory=dict)
