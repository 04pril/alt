from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


logger = logging.getLogger(__name__)
DEFAULT_SETTINGS_PATH = Path("config/runtime_settings.json")


class SettingsMergeError(ValueError):
    pass


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
    watchlist: list[str] = field(default_factory=list)
    top_universe: list[str] = field(default_factory=list)


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
    kis_config_path: str = "kis_devlp.yaml"
    asset_broker_mode: Dict[str, str] = field(
        default_factory=lambda: {"코인": "sim", "미국주식": "sim", "한국주식": "kis_paper"}
    )
    broker_short_support: Dict[str, bool] = field(
        default_factory=lambda: {"sim": True, "kis_paper": False}
    )


@dataclass
class SchedulerSettings:
    loop_sleep_seconds: int = 30
    retry_backoff_seconds: int = 30
    max_retry_count: int = 3
    lock_owner: str = "paper-worker"
    job_lease_timeout_seconds: int = 180


@dataclass
class RetrainingSettings:
    cadence: str = "weekly"
    benchmark_symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "AAPL", "005930.KS"])
    promotion_min_directional_accuracy_pct: float = 52.0
    promotion_max_mae_pct: float = 5.0
    promotion_min_trade_return_pct: float = 0.0
    promotion_max_drawdown_pct: float = 15.0


@dataclass
class RuntimeSettings:
    storage: StorageSettings = field(default_factory=StorageSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    retraining: RetrainingSettings = field(default_factory=RetrainingSettings)
    asset_schedules: Dict[str, AssetScheduleConfig] = field(
        default_factory=lambda: {
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

    def broker_mode_for(self, asset_type: str) -> str:
        return str(self.broker.asset_broker_mode.get(asset_type, "sim"))

    def asset_types_for_broker_mode(self, broker_mode: str) -> list[str]:
        return [asset_type for asset_type, mode in self.broker.asset_broker_mode.items() if str(mode) == str(broker_mode)]

    def broker_supports_short(self, broker_mode: str) -> bool:
        return bool(self.broker.broker_short_support.get(broker_mode, False))

    def short_allowed_for(self, asset_type: str) -> bool:
        return bool(self.strategy.allow_short and self.broker_supports_short(self.broker_mode_for(asset_type)))


def _merge_error(path: str, message: str, *, strict: bool, warn_unknown: bool) -> None:
    full_message = f"settings override error at '{path}': {message}" if path else f"settings override error: {message}"
    if strict:
        raise SettingsMergeError(full_message)
    if warn_unknown:
        logger.warning(full_message)


def _merge_dataclass(instance: Any, overrides: Mapping[str, Any], *, path: str = "", strict: bool = False, warn_unknown: bool = True) -> Any:
    valid_fields = getattr(instance, "__dataclass_fields__", {})
    for key, value in overrides.items():
        current_path = f"{path}.{key}" if path else str(key)
        if key not in valid_fields:
            _merge_error(current_path, "unknown key", strict=strict, warn_unknown=warn_unknown)
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, Mapping):
                _merge_error(current_path, f"expected object, got {type(value).__name__}", strict=strict, warn_unknown=warn_unknown)
                continue
            setattr(
                instance,
                key,
                _merge_dataclass(current, value, path=current_path, strict=strict, warn_unknown=warn_unknown),
            )
            continue
        if isinstance(current, dict) and isinstance(value, Mapping):
            if current and all(hasattr(v, "__dataclass_fields__") for v in current.values()):
                merged = dict(current)
                for sub_key, sub_value in value.items():
                    sub_path = f"{current_path}.{sub_key}"
                    base_value = merged.get(sub_key)
                    if base_value is None:
                        _merge_error(sub_path, "unknown key", strict=strict, warn_unknown=warn_unknown)
                        continue
                    if isinstance(sub_value, Mapping):
                        merged[sub_key] = _merge_dataclass(
                            base_value,
                            sub_value,
                            path=sub_path,
                            strict=strict,
                            warn_unknown=warn_unknown,
                        )
                    else:
                        merged[sub_key] = sub_value
                setattr(instance, key, merged)
            else:
                merged = dict(current)
                known_subkeys = set(current.keys())
                for sub_key, sub_value in value.items():
                    sub_path = f"{current_path}.{sub_key}"
                    if known_subkeys and sub_key not in known_subkeys:
                        _merge_error(sub_path, "unknown key", strict=strict, warn_unknown=warn_unknown)
                        if strict:
                            continue
                    merged[sub_key] = sub_value
                setattr(instance, key, merged)
            continue
        setattr(instance, key, value)
    return instance


def load_settings(path: str | Path | None = None, *, strict: bool = False, warn_unknown: bool = True) -> RuntimeSettings:
    settings = RuntimeSettings()
    source_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if source_path.exists():
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        settings = _merge_dataclass(settings, raw, strict=strict, warn_unknown=warn_unknown)
    return settings


def write_example_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(RuntimeSettings().to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
