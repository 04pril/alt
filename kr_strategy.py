from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import AssetScheduleConfig, KRStrategyConfig, RuntimeSettings
from runtime_accounts import ACCOUNT_KIS_KR_PAPER


KR_DAILY_PRE_CLOSE_V1 = "kr_daily_preclose_v1"
KR_INTRADAY_1H_V1 = "kr_intraday_1h_v1"
KR_INTRADAY_15M_V1 = "kr_intraday_15m_v1"


def all_kr_strategies(settings: RuntimeSettings) -> dict[str, KRStrategyConfig]:
    return {str(key): value for key, value in (settings.kr_strategies or {}).items()}


def enabled_kr_strategies(settings: RuntimeSettings) -> list[KRStrategyConfig]:
    ordered = list(all_kr_strategies(settings).values())
    return [strategy for strategy in ordered if bool(strategy.enabled)]


def kr_strategy_ids(settings: RuntimeSettings, *, enabled_only: bool = False) -> list[str]:
    source = enabled_kr_strategies(settings) if enabled_only else all_kr_strategies(settings).values()
    return [str(strategy.strategy_id) for strategy in source]


def get_kr_strategy(settings: RuntimeSettings, strategy_id: str) -> KRStrategyConfig | None:
    return all_kr_strategies(settings).get(str(strategy_id))


def is_kr_strategy(strategy_id: str, settings: RuntimeSettings) -> bool:
    return get_kr_strategy(settings, strategy_id) is not None


def active_kr_strategy_ids(settings: RuntimeSettings) -> list[str]:
    return kr_strategy_ids(settings, enabled_only=True)


def default_kr_strategy(settings: RuntimeSettings) -> KRStrategyConfig | None:
    preferred = get_kr_strategy(settings, str(settings.kr_default_strategy_id or ""))
    if preferred and preferred.enabled:
        return preferred
    enabled = enabled_kr_strategies(settings)
    return enabled[0] if enabled else preferred


def strategy_schedule(settings: RuntimeSettings, strategy_id: str) -> AssetScheduleConfig:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        raise KeyError(f"unknown KR strategy: {strategy_id}")
    base = settings.asset_schedules["한국주식"]
    return replace(
        base,
        timeframe=str(strategy.timeframe),
        scan_interval_minutes=int(strategy.scan_interval_minutes),
        entry_interval_minutes=int(strategy.entry_interval_minutes),
        exit_interval_minutes=int(strategy.exit_interval_minutes),
        outcome_interval_minutes=int(strategy.outcome_interval_minutes),
        lookback_bars=int(strategy.lookback_bars),
        forecast_horizon_bars=int(strategy.forecast_horizon_bars),
        min_history_bars=int(strategy.min_history_bars),
    )


def strategy_label(strategy: KRStrategyConfig) -> str:
    suffix = " Experimental" if bool(strategy.experimental) else ""
    return f"{strategy.display_name}{suffix}".strip()


def strategy_requires_flatten(strategy: KRStrategyConfig) -> bool:
    return bool(strategy.strategy_family == "kr_intraday")


def strategy_is_intraday(strategy: KRStrategyConfig) -> bool:
    return str(strategy.timeframe) in {"15m", "1h"}


def _intraday_minutes(timeframe: str) -> int:
    normalized = str(timeframe or "").strip().lower()
    if normalized.endswith("m"):
        return max(int(normalized[:-1] or 0), 1)
    if normalized.endswith("h"):
        return max(int(normalized[:-1] or 0), 1) * 60
    return 24 * 60


def _local_dt(schedule: AssetScheduleConfig, when: datetime | None = None) -> datetime:
    if when is not None:
        return when.astimezone(ZoneInfo(schedule.timezone))
    return datetime.now(ZoneInfo(schedule.timezone))


def latest_completed_bar_close(schedule: AssetScheduleConfig, timeframe: str, when: datetime | None = None) -> pd.Timestamp | None:
    local_now = _local_dt(schedule, when)
    if local_now.weekday() >= 5:
        return None
    minutes = _intraday_minutes(timeframe)
    if minutes >= 24 * 60:
        return pd.Timestamp(local_now.replace(hour=0, minute=0, second=0, microsecond=0))
    session_open = datetime.combine(local_now.date(), time.fromisoformat(schedule.market_open), local_now.tzinfo)
    session_close = datetime.combine(local_now.date(), time.fromisoformat(schedule.market_close), local_now.tzinfo)
    if local_now < session_open + timedelta(minutes=minutes):
        return None
    capped_now = min(local_now, session_close)
    elapsed = int((capped_now - session_open).total_seconds() // 60)
    completed_minutes = (elapsed // minutes) * minutes
    if completed_minutes <= 0:
        return None
    return pd.Timestamp(session_open + timedelta(minutes=completed_minutes))


def entry_gate_reason(
    settings: RuntimeSettings,
    strategy_id: str,
    *,
    when: datetime | None = None,
    market_is_open: bool,
) -> str | None:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        return None
    schedule = strategy_schedule(settings, strategy_id)
    if not market_is_open:
        return "market_closed"
    if not strategy.enabled:
        return "strategy_disabled"
    if str(strategy.timeframe) == "1d":
        local_now = _local_dt(schedule, when)
        close_dt = datetime.combine(local_now.date(), time.fromisoformat(schedule.market_close), local_now.tzinfo)
        window_start = close_dt - timedelta(minutes=int(schedule.pre_close_buffer_minutes))
        return None if window_start <= local_now <= close_dt else "outside_preclose_window"

    completed_close = latest_completed_bar_close(schedule, strategy.timeframe, when=when)
    if completed_close is None:
        return "waiting_for_bar_close"
    close_time = completed_close.tz_convert(ZoneInfo(schedule.timezone)).time()
    opening_cutoff = time.fromisoformat(strategy.entry_window_start)
    if bool(strategy.block_opening_bar) and close_time <= opening_cutoff:
        return "opening_bar_blocked"
    if close_time < time.fromisoformat(strategy.entry_window_start) or close_time > time.fromisoformat(strategy.entry_window_end):
        return "outside_intraday_entry_window"
    return None


def flatten_due(settings: RuntimeSettings, strategy_id: str, when: datetime | None = None) -> bool:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None or not strategy_requires_flatten(strategy):
        return False
    schedule = strategy_schedule(settings, strategy_id)
    local_now = _local_dt(schedule, when)
    if local_now.weekday() >= 5:
        return False
    current_time = local_now.time()
    return time.fromisoformat(strategy.flatten_window_start) <= current_time <= time.fromisoformat(strategy.flatten_window_end)


def strategy_execution_account_id(settings: RuntimeSettings, strategy_id: str) -> str:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        return ACCOUNT_KIS_KR_PAPER
    return str(strategy.execution_account_id or ACCOUNT_KIS_KR_PAPER)


def strategy_conflict_ids(settings: RuntimeSettings, strategy_id: str) -> set[str]:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None or str(strategy.strategy_family) != "kr_intraday":
        return set()
    return {str(item.strategy_id) for item in all_kr_strategies(settings).values() if item.execution_account_id == strategy.execution_account_id}


def experimental_kr_strategy_ids(settings: RuntimeSettings) -> set[str]:
    return {str(strategy.strategy_id) for strategy in all_kr_strategies(settings).values() if bool(strategy.experimental)}


def enabled_strategy_labels(settings: RuntimeSettings) -> list[str]:
    return [strategy_label(strategy) for strategy in enabled_kr_strategies(settings)]


def iter_strategy_rows(settings: RuntimeSettings) -> Iterable[KRStrategyConfig]:
    return all_kr_strategies(settings).values()
