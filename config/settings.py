from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SETTINGS_PATH = Path("config/runtime_settings.json")


@dataclass
class StorageSettings:
    db_path: str = ".runtime/paper_trading.sqlite3"


@dataclass
class AssetScheduleConfig:
    asset_type: str
    timeframe: str
    scan_interval_minutes: int
    entry_interval_minutes: int
    exit_interval_minutes: int
    outcome_interval_minutes: int
    session_mode: str
    timezone: str
    market_open: str
    market_close: str
    holiday_country: str
    lookback_bars: int
    forecast_horizon_bars: int
    pre_close_buffer_minutes: int = 15
    min_history_bars: int = 320


@dataclass
class UniverseSettings:
    watchlist: List[str] = field(default_factory=list)
    top_universe: List[str] = field(default_factory=list)


@dataclass
class StrategySettings:
    strategy_version: str = "paper_strategy_v1"
    validation_mode: str = "holdout"
    target_mode: str = "return"
    trade_mode: str = "close_to_close"
    retrain_every_bars: int = 5
    test_bars: int = 60
    validation_bars: int = 40
    final_holdout_bars: int = 40
    purge_bars: int = 2
    embargo_bars: int = 1
    round_trip_cost_bps: float = 8.0
    min_signal_strength_pct: float = 0.4
    min_expected_return_pct: float = 0.8
    min_confidence: float = 0.55
    max_expected_risk_pct: float = 4.0
    max_cost_bps: float = 25.0
    score_decay_exit_threshold: float = 0.2
    trailing_stop_atr_mult: float = 1.0
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    max_holding_bars: int = 5
    time_stop_bars: int = 3
    allow_short: bool = False
    allow_short_asset_types: List[str] = field(default_factory=list)
    target_daily_vol_pct: float = 1.0
    max_position_size: float = 1.0
    scan_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "expected_return": 0.40,
            "confidence": 0.25,
            "cost": -0.10,
            "volatility": -0.10,
            "recent_performance": 0.15,
        }
    )


@dataclass
class KRStrategyConfig:
    strategy_id: str
    strategy_family: str
    display_name: str
    enabled: bool
    experimental: bool
    execution_account_id: str
    session_mode: str
    order_division: str
    session_price_policy: str
    auction_interval_minutes: int
    timeframe: str
    feature_profile: str
    validation_mode: str
    target_mode: str
    trade_mode: str
    primary_target: str
    secondary_target: str
    analysis_target: str
    decision_horizon_bars: int
    forecast_horizon_bars: int
    retrain_every_bars: int
    test_bars: int
    validation_bars: int
    final_holdout_bars: int
    purge_bars: int
    embargo_bars: int
    round_trip_cost_bps: float
    min_signal_strength_pct: float
    min_expected_return_pct: float
    min_confidence: float
    max_expected_risk_pct: float
    max_cost_bps: float
    score_decay_exit_threshold: float
    trailing_stop_atr_mult: float
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    max_holding_bars: int
    time_stop_bars: int
    allow_short: bool
    target_daily_vol_pct: float
    max_position_size: float
    scan_score_weights: Dict[str, float]
    scan_interval_minutes: int
    entry_interval_minutes: int
    exit_interval_minutes: int
    outcome_interval_minutes: int
    lookback_bars: int
    min_history_bars: int
    liquidity_min_score: float
    liquidity_min_median_value: float
    bar_close_only: bool
    block_opening_bar: bool
    entry_window_start: str
    entry_window_end: str
    flatten_window_start: str
    flatten_window_end: str
    max_daily_new_entries: int
    cooldown_bars_after_exit: int


@dataclass
class RiskSettings:
    starting_cash: float = 30_000_000.0
    max_open_positions: int = 8
    max_daily_new_entries: int = 4
    symbol_max_weight: float = 0.18
    asset_type_max_weight: Dict[str, float] = field(
        default_factory=lambda: {"코인": 0.35, "미국주식": 0.45, "한국주식": 0.45}
    )
    per_trade_risk_budget_pct: float = 0.012
    total_risk_budget_pct: float = 0.10
    daily_loss_limit_pct: float = 0.03
    max_drawdown_limit_pct: float = 0.15
    max_same_direction_correlation: float = 0.82
    correlation_window_bars: int = 60
    cooldown_bars_after_exit: int = 2


@dataclass
class BrokerSettings:
    fee_bps: float = 5.0
    base_slippage_bps: float = 3.0
    volatility_slippage_mult: float = 0.25
    max_volume_participation: float = 0.05
    allow_partial_fills: bool = True
    default_order_type: str = "market"
    websocket_reconnect_interval_seconds: int = 30
    stale_submitted_order_timeout_minutes: int = 20


@dataclass
class SchedulerSettings:
    loop_sleep_seconds: int = 30
    retry_backoff_seconds: int = 30
    max_retry_count: int = 3
    job_lease_seconds: int = 180
    exit_management_interval_minutes: int = 15
    outcome_resolution_interval_minutes: int = 60
    broker_market_status_interval_minutes: int = 5
    broker_order_sync_interval_minutes: int = 5
    broker_position_sync_interval_minutes: int = 5
    broker_account_sync_interval_minutes: int = 15
    lock_owner: str = "paper-worker"


@dataclass
class RetrainingSettings:
    cadence: str = "weekly"
    benchmark_symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "AAPL", "005930.KS"])
    promotion_min_directional_accuracy_pct: float = 52.0
    promotion_max_mae_pct: float = 5.0
    promotion_min_trade_return_pct: float = 0.0
    promotion_max_drawdown_pct: float = 15.0


@dataclass
class RuntimeSettings:
    profile_name: str = "baseline"
    profile_source: str = "embedded_defaults"
    storage: StorageSettings = field(default_factory=StorageSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    retraining: RetrainingSettings = field(default_factory=RetrainingSettings)
    kr_default_strategy_id: str = "kr_intraday_1h_v1"
    kr_strategies: Dict[str, KRStrategyConfig] = field(
        default_factory=lambda: {
            "kr_daily_preclose_v1": KRStrategyConfig(
                strategy_id="kr_daily_preclose_v1",
                strategy_family="kr_daily",
                display_name="KR Daily Pre-close v1",
                enabled=False,
                experimental=False,
                execution_account_id="kis_kr_paper",
                session_mode="pre_close",
                order_division="00",
                session_price_policy="limit",
                auction_interval_minutes=0,
                timeframe="1d",
                feature_profile="default",
                validation_mode="holdout",
                target_mode="return",
                trade_mode="close_to_close",
                primary_target="next_close_return",
                secondary_target="session_close_return",
                analysis_target="session_close_return",
                decision_horizon_bars=1,
                forecast_horizon_bars=1,
                retrain_every_bars=5,
                test_bars=60,
                validation_bars=40,
                final_holdout_bars=40,
                purge_bars=2,
                embargo_bars=1,
                round_trip_cost_bps=8.0,
                min_signal_strength_pct=0.40,
                min_expected_return_pct=0.80,
                min_confidence=0.55,
                max_expected_risk_pct=4.0,
                max_cost_bps=25.0,
                score_decay_exit_threshold=0.20,
                trailing_stop_atr_mult=1.0,
                stop_loss_atr_mult=1.5,
                take_profit_atr_mult=3.0,
                max_holding_bars=1,
                time_stop_bars=1,
                allow_short=False,
                target_daily_vol_pct=1.0,
                max_position_size=1.0,
                scan_score_weights={
                    "expected_return": 0.40,
                    "confidence": 0.25,
                    "cost": -0.10,
                    "volatility": -0.10,
                    "recent_performance": 0.15,
                },
                scan_interval_minutes=60,
                entry_interval_minutes=15,
                exit_interval_minutes=15,
                outcome_interval_minutes=60,
                lookback_bars=520,
                min_history_bars=320,
                liquidity_min_score=0.30,
                liquidity_min_median_value=0.0,
                bar_close_only=False,
                block_opening_bar=False,
                entry_window_start="15:10",
                entry_window_end="15:30",
                flatten_window_start="15:20",
                flatten_window_end="15:30",
                max_daily_new_entries=4,
                cooldown_bars_after_exit=2,
            ),
            "kr_intraday_1h_v1": KRStrategyConfig(
                strategy_id="kr_intraday_1h_v1",
                strategy_family="kr_intraday",
                display_name="KR Intraday 1h v1",
                enabled=True,
                experimental=False,
                execution_account_id="kis_kr_paper",
                session_mode="regular",
                order_division="01",
                session_price_policy="market_best_effort",
                auction_interval_minutes=0,
                timeframe="1h",
                feature_profile="kr_intraday",
                validation_mode="holdout",
                target_mode="return",
                trade_mode="close_to_close",
                primary_target="next_1_bar_return",
                secondary_target="next_2_bar_return",
                analysis_target="session_close_return",
                decision_horizon_bars=1,
                forecast_horizon_bars=2,
                retrain_every_bars=8,
                test_bars=80,
                validation_bars=60,
                final_holdout_bars=60,
                purge_bars=2,
                embargo_bars=1,
                round_trip_cost_bps=8.0,
                min_signal_strength_pct=0.16,
                min_expected_return_pct=0.22,
                min_confidence=0.54,
                max_expected_risk_pct=2.8,
                max_cost_bps=18.0,
                score_decay_exit_threshold=0.18,
                trailing_stop_atr_mult=0.9,
                stop_loss_atr_mult=1.0,
                take_profit_atr_mult=1.4,
                max_holding_bars=3,
                time_stop_bars=3,
                allow_short=False,
                target_daily_vol_pct=0.9,
                max_position_size=0.8,
                scan_score_weights={
                    "expected_return": 0.44,
                    "confidence": 0.24,
                    "cost": -0.08,
                    "volatility": -0.08,
                    "recent_performance": 0.12,
                },
                scan_interval_minutes=60,
                entry_interval_minutes=60,
                exit_interval_minutes=15,
                outcome_interval_minutes=60,
                lookback_bars=420,
                min_history_bars=260,
                liquidity_min_score=0.42,
                liquidity_min_median_value=2_500_000_000.0,
                bar_close_only=True,
                block_opening_bar=False,
                entry_window_start="10:00",
                entry_window_end="14:30",
                flatten_window_start="15:10",
                flatten_window_end="15:20",
                max_daily_new_entries=6,
                cooldown_bars_after_exit=1,
            ),
            "kr_intraday_15m_v1": KRStrategyConfig(
                strategy_id="kr_intraday_15m_v1",
                strategy_family="kr_intraday_15m",
                display_name="KR Intraday 15m v1",
                enabled=False,
                experimental=True,
                execution_account_id="kis_kr_paper",
                session_mode="regular",
                order_division="01",
                session_price_policy="market_best_effort",
                auction_interval_minutes=0,
                timeframe="15m",
                feature_profile="kr_intraday",
                validation_mode="walk_forward",
                target_mode="return",
                trade_mode="close_to_close",
                primary_target="next_2_bar_return",
                secondary_target="next_4_bar_return",
                analysis_target="session_close_return",
                decision_horizon_bars=2,
                forecast_horizon_bars=4,
                retrain_every_bars=12,
                test_bars=120,
                validation_bars=80,
                final_holdout_bars=80,
                purge_bars=2,
                embargo_bars=1,
                round_trip_cost_bps=8.0,
                min_signal_strength_pct=0.12,
                min_expected_return_pct=0.18,
                min_confidence=0.53,
                max_expected_risk_pct=2.2,
                max_cost_bps=16.0,
                score_decay_exit_threshold=0.16,
                trailing_stop_atr_mult=0.7,
                stop_loss_atr_mult=0.8,
                take_profit_atr_mult=1.2,
                max_holding_bars=4,
                time_stop_bars=4,
                allow_short=False,
                target_daily_vol_pct=0.8,
                max_position_size=0.65,
                scan_score_weights={
                    "expected_return": 0.46,
                    "confidence": 0.24,
                    "cost": -0.08,
                    "volatility": -0.10,
                    "recent_performance": 0.12,
                },
                scan_interval_minutes=15,
                entry_interval_minutes=15,
                exit_interval_minutes=15,
                outcome_interval_minutes=15,
                lookback_bars=480,
                min_history_bars=320,
                liquidity_min_score=0.55,
                liquidity_min_median_value=4_000_000_000.0,
                bar_close_only=True,
                block_opening_bar=True,
                entry_window_start="09:15",
                entry_window_end="14:45",
                flatten_window_start="15:15",
                flatten_window_end="15:20",
                max_daily_new_entries=8,
                cooldown_bars_after_exit=2,
            ),
            "kr_intraday_15m_v1_after_close_close": KRStrategyConfig(
                strategy_id="kr_intraday_15m_v1_after_close_close",
                strategy_family="kr_intraday_15m",
                display_name="KR 15m After-close Close v1",
                enabled=False,
                experimental=True,
                execution_account_id="kis_kr_paper",
                session_mode="after_close_close_price",
                order_division="06",
                session_price_policy="close_price",
                auction_interval_minutes=0,
                timeframe="15m",
                feature_profile="kr_intraday",
                validation_mode="walk_forward",
                target_mode="return",
                trade_mode="close_to_close",
                primary_target="next_2_bar_return",
                secondary_target="next_4_bar_return",
                analysis_target="session_close_return",
                decision_horizon_bars=2,
                forecast_horizon_bars=4,
                retrain_every_bars=12,
                test_bars=120,
                validation_bars=80,
                final_holdout_bars=80,
                purge_bars=2,
                embargo_bars=1,
                round_trip_cost_bps=8.0,
                min_signal_strength_pct=0.12,
                min_expected_return_pct=0.18,
                min_confidence=0.53,
                max_expected_risk_pct=2.2,
                max_cost_bps=16.0,
                score_decay_exit_threshold=0.16,
                trailing_stop_atr_mult=0.7,
                stop_loss_atr_mult=0.8,
                take_profit_atr_mult=1.2,
                max_holding_bars=4,
                time_stop_bars=4,
                allow_short=False,
                target_daily_vol_pct=0.8,
                max_position_size=0.65,
                scan_score_weights={
                    "expected_return": 0.46,
                    "confidence": 0.24,
                    "cost": -0.08,
                    "volatility": -0.10,
                    "recent_performance": 0.12,
                },
                scan_interval_minutes=5,
                entry_interval_minutes=5,
                exit_interval_minutes=5,
                outcome_interval_minutes=15,
                lookback_bars=480,
                min_history_bars=320,
                liquidity_min_score=0.55,
                liquidity_min_median_value=4_000_000_000.0,
                bar_close_only=True,
                block_opening_bar=False,
                entry_window_start="15:40",
                entry_window_end="16:00",
                flatten_window_start="",
                flatten_window_end="",
                max_daily_new_entries=4,
                cooldown_bars_after_exit=2,
            ),
            "kr_intraday_15m_v1_after_close_single": KRStrategyConfig(
                strategy_id="kr_intraday_15m_v1_after_close_single",
                strategy_family="kr_intraday_15m",
                display_name="KR 15m After-close Single v1",
                enabled=False,
                experimental=True,
                execution_account_id="kis_kr_paper",
                session_mode="after_close_single_price",
                order_division="07",
                session_price_policy="auction_expected_price",
                auction_interval_minutes=10,
                timeframe="15m",
                feature_profile="kr_intraday",
                validation_mode="walk_forward",
                target_mode="return",
                trade_mode="close_to_close",
                primary_target="next_2_bar_return",
                secondary_target="next_4_bar_return",
                analysis_target="session_close_return",
                decision_horizon_bars=2,
                forecast_horizon_bars=4,
                retrain_every_bars=12,
                test_bars=120,
                validation_bars=80,
                final_holdout_bars=80,
                purge_bars=2,
                embargo_bars=1,
                round_trip_cost_bps=10.0,
                min_signal_strength_pct=0.14,
                min_expected_return_pct=0.22,
                min_confidence=0.56,
                max_expected_risk_pct=2.6,
                max_cost_bps=18.0,
                score_decay_exit_threshold=0.16,
                trailing_stop_atr_mult=0.7,
                stop_loss_atr_mult=0.8,
                take_profit_atr_mult=1.2,
                max_holding_bars=4,
                time_stop_bars=4,
                allow_short=False,
                target_daily_vol_pct=0.75,
                max_position_size=0.55,
                scan_score_weights={
                    "expected_return": 0.48,
                    "confidence": 0.24,
                    "cost": -0.10,
                    "volatility": -0.12,
                    "recent_performance": 0.10,
                },
                scan_interval_minutes=10,
                entry_interval_minutes=10,
                exit_interval_minutes=10,
                outcome_interval_minutes=15,
                lookback_bars=480,
                min_history_bars=320,
                liquidity_min_score=0.60,
                liquidity_min_median_value=5_000_000_000.0,
                bar_close_only=True,
                block_opening_bar=False,
                entry_window_start="16:00",
                entry_window_end="18:00",
                flatten_window_start="",
                flatten_window_end="",
                max_daily_new_entries=4,
                cooldown_bars_after_exit=2,
            ),
        }
    )
    asset_schedules: Dict[str, AssetScheduleConfig] = field(
        default_factory=lambda: {
            # Assumption: crypto uses 1h bars and can trade on each bar close 24/7.
            "코인": AssetScheduleConfig(
                asset_type="코인",
                timeframe="1h",
                scan_interval_minutes=60,
                entry_interval_minutes=60,
                exit_interval_minutes=15,
                outcome_interval_minutes=60,
                session_mode="always",
                timezone="UTC",
                market_open="00:00",
                market_close="23:59",
                holiday_country="",
                lookback_bars=720,
                forecast_horizon_bars=1,
                min_history_bars=400,
            ),
            # Assumption: US equities generate signals in the pre-close window and fill with a paper MOC price proxy.
            "미국주식": AssetScheduleConfig(
                asset_type="미국주식",
                timeframe="1d",
                scan_interval_minutes=60,
                entry_interval_minutes=15,
                exit_interval_minutes=15,
                outcome_interval_minutes=60,
                session_mode="market_hours",
                timezone="America/New_York",
                market_open="09:30",
                market_close="16:00",
                holiday_country="US",
                lookback_bars=520,
                forecast_horizon_bars=1,
                pre_close_buffer_minutes=20,
                min_history_bars=320,
            ),
            # Assumption: KR equities also trade using a pre-close paper MOC proxy to avoid target/open-close mismatch.
            "한국주식": AssetScheduleConfig(
                asset_type="한국주식",
                timeframe="1d",
                scan_interval_minutes=60,
                entry_interval_minutes=15,
                exit_interval_minutes=15,
                outcome_interval_minutes=60,
                session_mode="market_hours",
                timezone="Asia/Seoul",
                market_open="09:00",
                market_close="15:30",
                holiday_country="KR",
                lookback_bars=520,
                forecast_horizon_bars=1,
                pre_close_buffer_minutes=20,
                min_history_bars=320,
            ),
        }
    )
    universes: Dict[str, UniverseSettings] = field(
        default_factory=lambda: {
            "코인": UniverseSettings(
                watchlist=["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
                top_universe=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"],
            ),
            "미국주식": UniverseSettings(
                watchlist=["AAPL", "MSFT", "NVDA", "AMZN"],
                top_universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "AVGO"],
            ),
            "한국주식": UniverseSettings(
                watchlist=["005930.KS", "000660.KS", "035420.KS", "005380.KS"],
                top_universe=["005930.KS", "000660.KS", "035420.KS", "005380.KS", "068270.KS", "247540.KS", "373220.KS"],
            ),
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, overrides: Dict[str, Any]) -> Any:
    for key, value in overrides.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            setattr(instance, key, _merge_dataclass(current, value))
        elif isinstance(current, dict) and isinstance(value, dict):
            if current and all(hasattr(v, "__dataclass_fields__") for v in current.values()):
                merged: Dict[str, Any] = {}
                for sub_key, sub_value in value.items():
                    base_value = current.get(sub_key)
                    if base_value is None:
                        continue
                    merged[sub_key] = _merge_dataclass(base_value, sub_value) if isinstance(sub_value, dict) else sub_value
                current.update(merged)
            else:
                current.update(value)
        else:
            setattr(instance, key, value)
    return instance


def load_settings(path: str | Path | None = None) -> RuntimeSettings:
    settings = RuntimeSettings()
    source_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if source_path.exists():
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        settings = _merge_dataclass(settings, raw)
        settings.profile_source = source_path.as_posix()
    else:
        settings.profile_source = "embedded_defaults"
    return settings


def write_example_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(RuntimeSettings().to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
