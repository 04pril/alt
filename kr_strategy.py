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
KR_INTRADAY_15M_V1_AUTO = "kr_intraday_15m_v1_auto"
KR_INTRADAY_15M_V1_AFTER_CLOSE_CLOSE = "kr_intraday_15m_v1_after_close_close"
KR_INTRADAY_15M_V1_AFTER_CLOSE_SINGLE = "kr_intraday_15m_v1_after_close_single"
KR_COMBO_1H_AHC_REGULAR_V1 = "kr_combo_1h_ahc_regular_v1"
KR_COMBO_1H_AHC_AFTERCLOSE_V1 = "kr_combo_1h_ahc_afterclose_v1"
US_INTRADAY_1H_V1 = "us_intraday_1h_v1"
US_AFTERHOURS_V1 = "us_afterhours_v1"
US_COMBO_1H_AHC_REGULAR_V1 = "us_combo_1h_ahc_regular_v1"
US_COMBO_1H_AHC_AFTERHOURS_V1 = "us_combo_1h_ahc_afterhours_v1"
KR_COMBO_15M_AHC_REGULAR_V2 = "kr_combo_15m_ahc_regular_v2"
KR_COMBO_15M_AHC_AFTERCLOSE_V2 = "kr_combo_15m_ahc_afterclose_v2"
US_COMBO_15M_AHC_REGULAR_V1 = "us_combo_15m_ahc_regular_v1"
US_COMBO_15M_AHC_AFTERHOURS_V1 = "us_combo_15m_ahc_afterhours_v1"


def strategy_asset_schedule_key(strategy: KRStrategyConfig | None) -> str:
    return str((getattr(strategy, "asset_schedule_key", None) if strategy is not None else None) or "한국주식")


def strategy_asset_type(settings: RuntimeSettings, strategy_id: str) -> str:
    return strategy_asset_schedule_key(get_kr_strategy(settings, strategy_id))


def _default_strategy_attr(asset_schedule_key: str) -> str:
    if str(asset_schedule_key) == "미국주식":
        return "us_default_strategy_id"
    return "kr_default_strategy_id"


def all_kr_strategies(settings: RuntimeSettings) -> dict[str, KRStrategyConfig]:
    return {str(key): value for key, value in (settings.kr_strategies or {}).items()}


def strategy_visible_in_ui(strategy: KRStrategyConfig | None) -> bool:
    if strategy is None:
        return False
    return bool(getattr(strategy, "visible_in_ui", True))


def visible_kr_strategies(settings: RuntimeSettings, *, asset_schedule_key: str | None = None) -> list[KRStrategyConfig]:
    ordered = list(all_kr_strategies(settings).values())
    return [
        strategy
        for strategy in ordered
        if strategy_visible_in_ui(strategy) and (asset_schedule_key is None or strategy_asset_schedule_key(strategy) == str(asset_schedule_key))
    ]


def enabled_kr_strategies(settings: RuntimeSettings, *, asset_schedule_key: str | None = None) -> list[KRStrategyConfig]:
    ordered = list(all_kr_strategies(settings).values())
    return [
        strategy
        for strategy in ordered
        if bool(strategy.enabled) and (asset_schedule_key is None or strategy_asset_schedule_key(strategy) == str(asset_schedule_key))
    ]


def kr_strategy_ids(
    settings: RuntimeSettings,
    *,
    enabled_only: bool = False,
    asset_schedule_key: str | None = None,
) -> list[str]:
    source = (
        enabled_kr_strategies(settings, asset_schedule_key=asset_schedule_key)
        if enabled_only
        else all_kr_strategies(settings).values()
    )
    return [str(strategy.strategy_id) for strategy in source]


def get_kr_strategy(settings: RuntimeSettings, strategy_id: str) -> KRStrategyConfig | None:
    return all_kr_strategies(settings).get(str(strategy_id))


def is_kr_strategy(strategy_id: str, settings: RuntimeSettings) -> bool:
    return get_kr_strategy(settings, strategy_id) is not None


def active_kr_strategy_ids(settings: RuntimeSettings) -> list[str]:
    return active_strategy_ids(settings, asset_schedule_key="한국주식")


def active_strategy_ids(settings: RuntimeSettings, *, asset_schedule_key: str | None = None) -> list[str]:
    return kr_strategy_ids(settings, enabled_only=True, asset_schedule_key=asset_schedule_key)


def default_strategy(settings: RuntimeSettings, asset_schedule_key: str = "한국주식") -> KRStrategyConfig | None:
    preferred = get_kr_strategy(settings, str(getattr(settings, _default_strategy_attr(asset_schedule_key), "") or ""))
    if preferred and preferred.enabled and strategy_asset_schedule_key(preferred) == str(asset_schedule_key):
        return preferred
    enabled = enabled_kr_strategies(settings, asset_schedule_key=asset_schedule_key)
    return enabled[0] if enabled else preferred


def default_kr_strategy(settings: RuntimeSettings) -> KRStrategyConfig | None:
    return default_strategy(settings, "한국주식")


def active_us_strategy_ids(settings: RuntimeSettings) -> list[str]:
    return active_strategy_ids(settings, asset_schedule_key="미국주식")


def default_us_strategy(settings: RuntimeSettings) -> KRStrategyConfig | None:
    return default_strategy(settings, "미국주식")


def strategy_schedule(settings: RuntimeSettings, strategy_id: str, *, when: datetime | None = None) -> AssetScheduleConfig:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        raise KeyError(f"unknown KR strategy: {strategy_id}")
    strategy = strategy_runtime_config(strategy, when=when)
    schedule_key = strategy_asset_schedule_key(strategy)
    base = settings.asset_schedules.get(schedule_key) or settings.asset_schedules["한국주식"]
    return replace(
        base,
        session_mode=str(strategy.session_mode),
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
    return str(strategy.display_name or "").strip()


def _kr_intraday_15m_auto_session(strategy: KRStrategyConfig, when: datetime | None = None) -> KRStrategyConfig:
    local_now = when.astimezone(ZoneInfo("Asia/Seoul")) if when is not None else datetime.now(ZoneInfo("Asia/Seoul"))
    common = {
        "flatten_window_start": "17:50",
        "flatten_window_end": "18:00",
    }
    if _time_in_window(local_now.time(), "16:00", "18:00"):
        return replace(
            strategy,
            session_mode="after_close_single_price",
            order_division="07",
            session_price_policy="auction_expected_price",
            auction_interval_minutes=10,
            scan_interval_minutes=10,
            entry_interval_minutes=10,
            exit_interval_minutes=10,
            entry_window_start="16:00",
            entry_window_end="18:00",
            **common,
        )
    if _time_in_window(local_now.time(), "15:40", "16:00"):
        return replace(
            strategy,
            session_mode="after_close_close_price",
            order_division="06",
            session_price_policy="close_price",
            auction_interval_minutes=0,
            scan_interval_minutes=5,
            entry_interval_minutes=5,
            exit_interval_minutes=5,
            entry_window_start="15:40",
            entry_window_end="16:00",
            **common,
        )
    return replace(
        strategy,
        session_mode="regular",
        order_division="01",
        session_price_policy="market_best_effort",
        auction_interval_minutes=0,
        scan_interval_minutes=15,
        entry_interval_minutes=15,
        exit_interval_minutes=15,
        entry_window_start="09:15",
        entry_window_end="14:45",
        **common,
    )


def strategy_runtime_config(strategy: KRStrategyConfig | None, when: datetime | None = None) -> KRStrategyConfig | None:
    if strategy is None:
        return None
    if str(strategy.strategy_id) == KR_INTRADAY_15M_V1_AUTO and strategy_asset_schedule_key(strategy) == "한국주식":
        return _kr_intraday_15m_auto_session(strategy, when=when)
    return strategy


def strategy_session_mode(strategy: KRStrategyConfig | None, when: datetime | None = None) -> str:
    effective = strategy_runtime_config(strategy, when=when)
    return str((effective.session_mode if effective is not None else "") or "regular")


def strategy_session_label(strategy: KRStrategyConfig | None, when: datetime | None = None) -> str:
    mode = strategy_session_mode(strategy, when=when)
    return {
        "regular": "regular",
        "pre_close": "pre_close",
        "after_close_close_price": "after_close_close",
        "after_close_single_price": "after_close_single",
    }.get(mode, mode)


def strategy_requires_flatten(strategy: KRStrategyConfig, when: datetime | None = None) -> bool:
    effective = strategy_runtime_config(strategy, when=when)
    return bool(
        effective is not None
        and strategy_is_intraday(effective)
        and str(effective.flatten_window_start or "").strip()
        and str(effective.flatten_window_end or "").strip()
    )


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


def _time_in_window(current: time, start_text: str, end_text: str) -> bool:
    if not str(start_text or "").strip() or not str(end_text or "").strip():
        return False
    return time.fromisoformat(start_text) <= current <= time.fromisoformat(end_text)


def _single_price_auction_open(local_now: datetime, strategy: KRStrategyConfig) -> bool:
    interval = max(int(strategy.auction_interval_minutes or 10), 1)
    if local_now.minute % interval != 0:
        return False
    return True


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
    strategy = strategy_runtime_config(strategy, when=when)
    schedule = strategy_schedule(settings, strategy_id, when=when)
    session_mode = strategy_session_mode(strategy, when=when)
    local_now = _local_dt(schedule, when)
    if local_now.weekday() >= 5:
        return "market_closed"
    if not strategy.enabled:
        if session_mode in {"after_close_close_price", "after_close_single_price"}:
            return "after_close_strategy_disabled"
        return "strategy_disabled"
    if str(strategy.timeframe) == "1d":
        if not market_is_open:
            return "market_closed"
        close_dt = datetime.combine(local_now.date(), time.fromisoformat(schedule.market_close), local_now.tzinfo)
        window_start = close_dt - timedelta(minutes=int(schedule.pre_close_buffer_minutes))
        return None if window_start <= local_now <= close_dt else "outside_preclose_window"

    if session_mode == "after_close_close_price":
        return None if _time_in_window(local_now.time(), str(strategy.entry_window_start), str(strategy.entry_window_end)) else "outside_after_close_close_session"

    if session_mode == "after_close_single_price":
        if not _time_in_window(local_now.time(), str(strategy.entry_window_start), str(strategy.entry_window_end)):
            return "outside_after_close_single_session"
        if not _single_price_auction_open(local_now, strategy):
            return "after_close_single_waiting_auction"
        return None

    if not market_is_open:
        return "market_closed"

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
    strategy = strategy_runtime_config(strategy, when=when)
    if strategy is None or not strategy_requires_flatten(strategy, when=when):
        return False
    schedule = strategy_schedule(settings, strategy_id, when=when)
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


def strategy_runtime_metadata(
    settings: RuntimeSettings,
    strategy_id: str,
    *,
    when: datetime | None = None,
) -> dict[str, str]:
    strategy = get_kr_strategy(settings, strategy_id)
    effective = strategy_runtime_config(strategy, when=when)
    if effective is None:
        return {
            "strategy_family": "",
            "session_mode": "",
            "price_policy": "",
            "account_id": strategy_execution_account_id(settings, strategy_id),
        }
    return {
        "strategy_family": str(effective.strategy_family or ""),
        "session_mode": str(effective.session_mode or ""),
        "price_policy": str(effective.session_price_policy or ""),
        "account_id": str(effective.execution_account_id or strategy_execution_account_id(settings, strategy_id)),
    }


def strategy_conflict_ids(settings: RuntimeSettings, strategy_id: str) -> set[str]:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        return set()
    return {str(item.strategy_id) for item in all_kr_strategies(settings).values() if item.execution_account_id == strategy.execution_account_id}


def strategy_session_is_open(
    settings: RuntimeSettings,
    strategy_id: str,
    *,
    when: datetime | None = None,
    market_is_open: bool,
) -> bool:
    strategy = get_kr_strategy(settings, strategy_id)
    if strategy is None:
        return bool(market_is_open)
    strategy = strategy_runtime_config(strategy, when=when)
    session_mode = strategy_session_mode(strategy, when=when)
    if session_mode in {"regular", "pre_close"}:
        return bool(market_is_open)
    schedule = strategy_schedule(settings, strategy_id, when=when)
    local_now = _local_dt(schedule, when)
    if local_now.weekday() >= 5:
        return False
    if session_mode == "after_close_close_price":
        return _time_in_window(local_now.time(), str(strategy.entry_window_start), str(strategy.entry_window_end))
    if session_mode == "after_close_single_price":
        return _time_in_window(local_now.time(), str(strategy.entry_window_start), str(strategy.entry_window_end))
    return bool(market_is_open)


def experimental_kr_strategy_ids(settings: RuntimeSettings) -> set[str]:
    return {str(strategy.strategy_id) for strategy in all_kr_strategies(settings).values() if bool(strategy.experimental)}


def enabled_strategy_labels(settings: RuntimeSettings) -> list[str]:
    return [strategy_label(strategy) for strategy in enabled_kr_strategies(settings)]


def iter_strategy_rows(settings: RuntimeSettings) -> Iterable[KRStrategyConfig]:
    return all_kr_strategies(settings).values()


def iter_visible_strategy_rows(settings: RuntimeSettings, *, asset_schedule_key: str | None = None) -> Iterable[KRStrategyConfig]:
    return visible_kr_strategies(settings, asset_schedule_key=asset_schedule_key)
