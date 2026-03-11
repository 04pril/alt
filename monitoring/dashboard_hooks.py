from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from kr_strategy import (
    active_kr_strategy_ids,
    active_strategy_ids,
    default_kr_strategy,
    default_strategy,
    get_kr_strategy,
    iter_strategy_rows,
    iter_visible_strategy_rows,
    strategy_asset_schedule_key,
    strategy_label,
    strategy_runtime_config,
    strategy_schedule,
    strategy_session_label,
)
from runtime_accounts import (
    ACCOUNT_KIS_KR_PAPER,
    ACCOUNT_SIM_CRYPTO,
    ACCOUNT_SIM_US_EQUITY,
)
from services.broker_router import resolve_broker_mode
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from storage.repository import TradingRepository

KST = ZoneInfo("Asia/Seoul")
BROKER_SYNC_JOBS = ["broker_account_sync", "broker_order_sync", "broker_position_sync", "broker_market_status"]
EXECUTION_EVENT_KEYS = [
    "candidate",
    "entry_allowed",
    "entry_rejected",
    "submit_requested",
    "submitted",
    "acknowledged",
    "pending_fill",
    "filled",
    "rejected",
    "cancelled",
    "noop",
]

_FRAME_CACHE_TTL_SECONDS = 30.0
_SHORT_FRAME_CACHE_TTL_SECONDS = 10.0
_FRAME_CACHE_LOCK = threading.Lock()
_FRAME_CACHE: Dict[tuple[str, str, str], tuple[float, Any]] = {}


def _parse_utc_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_kst_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or str(value).strip() == "":
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_convert(KST)


def _localize_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    localized = frame.copy()
    for column in localized.columns:
        if column.endswith("_at") or column in {"created_at", "updated_at", "resolved_at", "closed_at", "cooldown_until"}:
            try:
                parsed = pd.to_datetime(localized[column], errors="coerce", utc=True)
            except Exception:
                continue
            if parsed.notna().any():
                localized[column] = parsed.dt.tz_convert(KST)
    return localized


def _cache_key(repository: TradingRepository, name: str, *parts: object) -> tuple[str, str, str]:
    db_path = str(getattr(repository, "db_path", ""))
    suffix = "|".join(str(part or "") for part in parts)
    return (db_path, name, suffix)


def _cached_dataframe(
    repository: TradingRepository,
    name: str,
    *parts: object,
    ttl_seconds: float = _FRAME_CACHE_TTL_SECONDS,
    loader,
) -> pd.DataFrame:
    key = _cache_key(repository, name, *parts)
    now = time.time()
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(key)
        if cached is not None and cached[0] > now:
            value = cached[1]
            return value.copy(deep=True) if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    value = loader()
    stored = value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = (now + float(ttl_seconds), stored)
    return value.copy(deep=True) if isinstance(value, pd.DataFrame) else value


def _kis_enabled_for_monitor(settings: RuntimeSettings, repository: TradingRepository) -> bool:
    try:
        return bool(KISPaperBroker(settings, repository, PaperBroker(settings, repository)).is_enabled())
    except Exception:
        return False


def _runtime_profile_read_model(repository: TradingRepository, settings: RuntimeSettings) -> Dict[str, str]:
    profile_name = repository.get_control_flag("runtime_profile_name", str(settings.profile_name or "baseline"))
    profile_source = repository.get_control_flag(
        "runtime_profile_source",
        str(settings.profile_source or "embedded_defaults"),
    )
    normalized_name = str(profile_name or "baseline")
    profile_meta = {
        "baseline": {
            "mode": "baseline",
            "recommended_default": "false",
            "experimental": "false",
            "note": "production-equivalent baseline profile",
        },
        "balanced": {
            "mode": "recommended",
            "recommended_default": "true",
            "experimental": "false",
            "note": "recommended default profile for steady paper trading",
        },
        "active": {
            "mode": "experimental",
            "recommended_default": "false",
            "experimental": "true",
            "note": "aggressive experimental profile with US 15m cadence",
        },
    }.get(
        normalized_name,
        {
            "mode": "custom",
            "recommended_default": "false",
            "experimental": "false",
            "note": "custom runtime profile",
        },
    )
    default_kr = default_kr_strategy(settings)
    recommended_strategy = next((strategy for strategy in iter_strategy_rows(settings) if str(strategy.strategy_id) == "kr_intraday_1h_v1"), None)
    active_ids = active_kr_strategy_ids(settings)
    default_us = default_strategy(settings, "미국주식")
    recommended_us = next((strategy for strategy in iter_strategy_rows(settings) if str(strategy.strategy_id) == "us_intraday_1h_v1"), None)
    active_us_ids = active_strategy_ids(settings, asset_schedule_key="미국주식")
    active_labels = [
        strategy_label(strategy)
        for strategy in iter_strategy_rows(settings)
        if str(strategy.strategy_id) in set(active_ids)
    ]
    active_session_modes = [
        strategy_session_label(strategy)
        for strategy in iter_strategy_rows(settings)
        if str(strategy.strategy_id) in set(active_ids)
    ]
    active_us_labels = [
        strategy_label(strategy)
        for strategy in iter_strategy_rows(settings)
        if str(strategy.strategy_id) in set(active_us_ids)
    ]
    active_us_session_modes = [
        strategy_session_label(strategy)
        for strategy in iter_strategy_rows(settings)
        if str(strategy.strategy_id) in set(active_us_ids)
    ]
    return {
        "name": normalized_name,
        "source": str(profile_source or "embedded_defaults"),
        "mode": str(profile_meta["mode"]),
        "recommended_default": str(profile_meta["recommended_default"]),
        "experimental": str(profile_meta["experimental"]),
        "note": str(profile_meta["note"]),
        "kr_default_strategy_id": str((default_kr.strategy_id if default_kr is not None else settings.kr_default_strategy_id) or ""),
        "kr_default_strategy_label": str(strategy_label(default_kr) if default_kr is not None else settings.kr_default_strategy_id or ""),
        "kr_default_strategy_session_mode": str(strategy_session_label(default_kr)),
        "kr_recommended_strategy_id": str(recommended_strategy.strategy_id if recommended_strategy is not None else "kr_intraday_1h_v1"),
        "kr_recommended_strategy_label": str(strategy_label(recommended_strategy) if recommended_strategy is not None else "kr_intraday_1h_v1"),
        "kr_active_strategies": ",".join(active_ids),
        "kr_active_strategy_labels": ", ".join(active_labels),
        "kr_active_strategy_session_modes": ", ".join(active_session_modes),
        "us_default_strategy_id": str((default_us.strategy_id if default_us is not None else settings.us_default_strategy_id) or ""),
        "us_default_strategy_label": str(strategy_label(default_us) if default_us is not None else settings.us_default_strategy_id or ""),
        "us_default_strategy_session_mode": str(strategy_session_label(default_us)),
        "us_recommended_strategy_id": str(recommended_us.strategy_id if recommended_us is not None else "us_intraday_1h_v1"),
        "us_recommended_strategy_label": str(strategy_label(recommended_us) if recommended_us is not None else "us_intraday_1h_v1"),
        "us_active_strategies": ",".join(active_us_ids),
        "us_active_strategy_labels": ", ".join(active_us_labels),
        "us_active_strategy_session_modes": ", ".join(active_us_session_modes),
    }


def _account_display_order() -> list[str]:
    return [ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO]


def _sync_status_from_events_frame(events: pd.DataFrame, account_id: str, latest_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not events.empty:
        account_events = events.loc[events["account_id"].fillna("").astype(str) == str(account_id)].copy() if "account_id" in events.columns else pd.DataFrame()
        sync_events = account_events.loc[
            account_events["event_type"].astype(str).isin({"broker_account_sync", "broker_order_sync", "broker_position_sync", "broker_market_status"})
            | account_events["component"].astype(str).str.contains("kis_execution|execution_pipeline", na=False)
        ] if not account_events.empty else pd.DataFrame()
        if not sync_events.empty:
            row = sync_events.iloc[0]
            return {
                "last_sync_time": row.get("created_at") or "",
                "last_sync_status": "failed" if str(row.get("level") or "").upper() == "ERROR" else "completed",
            }
    if latest_snapshot:
        return {
            "last_sync_time": latest_snapshot.get("created_at") or "",
            "last_sync_status": "completed",
        }
    return {"last_sync_time": "", "last_sync_status": "never"}


def _trade_performance_from_snapshots(
    snapshots: pd.DataFrame,
    repository: TradingRepository,
    *,
    account_id: str | None = None,
) -> Dict[str, float]:
    frame = snapshots.copy()
    if account_id is not None and not frame.empty and "account_id" in frame.columns:
        frame = frame.loc[frame["account_id"].fillna("").astype(str) == str(account_id)].copy()
    if frame.empty or "equity" not in frame.columns:
        return {"samples": 0.0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "today_pnl": 0.0}
    created = pd.to_datetime(frame["created_at"], errors="coerce", utc=True) if "created_at" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    rowids = pd.to_numeric(frame["rowid"], errors="coerce") if "rowid" in frame.columns else pd.Series(0, index=frame.index)
    frame = frame.assign(_created_sort=created, _rowid_sort=rowids).sort_values(["_created_sort", "_rowid_sort"], ascending=[True, True], na_position="last")
    values = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    if values.empty:
        return {"samples": 0.0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "today_pnl": 0.0}
    drawdown = values / values.cummax() - 1.0
    latest = float(values.iloc[-1])
    start = float(values.iloc[0])
    today = str(pd.Timestamp.utcnow().date())
    daily_pnl = repository.recent_closed_realized_pnl(today, account_id=account_id)
    return {
        "samples": float(len(values)),
        "total_return_pct": (latest / start - 1.0) * 100.0 if start > 0 else np.nan,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "today_pnl": float(daily_pnl),
    }


def _build_accounts_overview(
    repository: TradingRepository,
    accounts: pd.DataFrame,
    *,
    latest_snapshots: pd.DataFrame,
    open_positions: pd.DataFrame,
    open_orders: pd.DataFrame,
    recent_events: pd.DataFrame,
    trade_performance_snapshots: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    if accounts.empty:
        return {}
    latest_snapshot_by_account: Dict[str, Dict[str, Any]] = {}
    if not latest_snapshots.empty and "account_id" in latest_snapshots.columns:
        created = pd.to_datetime(latest_snapshots["created_at"], errors="coerce", utc=True) if "created_at" in latest_snapshots.columns else pd.Series(pd.NaT, index=latest_snapshots.index)
        rowids = pd.to_numeric(latest_snapshots["rowid"], errors="coerce") if "rowid" in latest_snapshots.columns else pd.Series(0, index=latest_snapshots.index)
        ordered_snapshots = latest_snapshots.assign(_created_sort=created, _rowid_sort=rowids).sort_values(
            ["_created_sort", "_rowid_sort"],
            ascending=[False, False],
            na_position="last",
        )
        latest_rows = ordered_snapshots.drop_duplicates(subset=["account_id"], keep="first")
        latest_snapshot_by_account = {
            str(row.get("account_id") or ""): {key: value for key, value in row.items() if key not in {"_created_sort", "_rowid_sort"}}
            for _, row in latest_rows.iterrows()
        }
    overviews: Dict[str, Dict[str, Any]] = {}
    for account_id in _account_display_order():
        matched = accounts.loc[accounts["account_id"].astype(str) == account_id]
        if matched.empty:
            continue
        account_row = matched.iloc[0].to_dict()
        latest_snapshot = latest_snapshot_by_account.get(str(account_id), {})
        sync_state = _sync_status_from_events_frame(recent_events, account_id, latest_snapshot)
        account_positions = open_positions.loc[open_positions["account_id"].fillna("").astype(str) == str(account_id)].copy() if not open_positions.empty and "account_id" in open_positions.columns else pd.DataFrame()
        account_orders = open_orders.loc[open_orders["account_id"].fillna("").astype(str) == str(account_id)].copy() if not open_orders.empty and "account_id" in open_orders.columns else pd.DataFrame()
        pending_orders = int(
            len(account_orders.loc[account_orders["status"].astype(str).isin({"new", "submitted", "acknowledged", "pending_fill", "partially_filled"})])
        ) if not account_orders.empty else 0
        overviews[account_id] = {
            "account_id": account_id,
            "display_name": str(account_row.get("display_name") or account_id),
            "broker_mode": str(account_row.get("broker_mode") or ""),
            "asset_scope": str(account_row.get("asset_scope") or ""),
            "currency": str(account_row.get("currency") or ""),
            "cash": float(latest_snapshot.get("cash", 0.0) or 0.0),
            "equity": float(latest_snapshot.get("equity", 0.0) or 0.0),
            "gross_exposure": float(latest_snapshot.get("gross_exposure", 0.0) or 0.0),
            "net_exposure": float(latest_snapshot.get("net_exposure", 0.0) or 0.0),
            "realized_pnl": float(latest_snapshot.get("daily_pnl", 0.0) or 0.0),
            "unrealized_pnl": float(latest_snapshot.get("unrealized_pnl", 0.0) or 0.0),
            "drawdown_pct": float(latest_snapshot.get("drawdown_pct", 0.0) or 0.0),
            "open_positions": int(len(account_positions)),
            "pending_orders": pending_orders,
            "last_sync_time": sync_state["last_sync_time"],
            "last_sync_status": sync_state["last_sync_status"],
            "latest_snapshot": latest_snapshot,
            "trade_performance": _trade_performance_from_snapshots(trade_performance_snapshots, repository, account_id=account_id),
        }
    return overviews


def _build_total_portfolio_overview(accounts_overview: Dict[str, Dict[str, Any]], repository: TradingRepository) -> Dict[str, Any]:
    cash_by_currency: Dict[str, float] = {}
    equity_by_currency: Dict[str, float] = {}
    gross_exposure_by_currency: Dict[str, float] = {}
    realized_pnl_by_currency: Dict[str, float] = {}
    unrealized_pnl_by_currency: Dict[str, float] = {}
    for item in accounts_overview.values():
        currency = str((item or {}).get("currency") or "KRW").upper()
        cash_by_currency[currency] = float(cash_by_currency.get(currency, 0.0) + float((item or {}).get("cash", 0.0) or 0.0))
        equity_by_currency[currency] = float(equity_by_currency.get(currency, 0.0) + float((item or {}).get("equity", 0.0) or 0.0))
        gross_exposure_by_currency[currency] = float(
            gross_exposure_by_currency.get(currency, 0.0) + abs(float((item or {}).get("gross_exposure", 0.0) or 0.0))
        )
        realized_pnl_by_currency[currency] = float(
            realized_pnl_by_currency.get(currency, 0.0) + float((item or {}).get("realized_pnl", 0.0) or 0.0)
        )
        unrealized_pnl_by_currency[currency] = float(
            unrealized_pnl_by_currency.get(currency, 0.0) + float((item or {}).get("unrealized_pnl", 0.0) or 0.0)
        )
    currencies = [code for code, value in equity_by_currency.items() if abs(float(value)) > 1e-9]
    single_currency = currencies[0] if len(currencies) == 1 else ""
    cash = cash_by_currency.get(single_currency, float("nan")) if single_currency else float("nan")
    equity = equity_by_currency.get(single_currency, float("nan")) if single_currency else float("nan")
    open_positions = sum(int((item or {}).get("open_positions", 0) or 0) for item in accounts_overview.values())
    pending_orders = sum(int((item or {}).get("pending_orders", 0) or 0) for item in accounts_overview.values())
    drawdowns = [
        float((item or {}).get("drawdown_pct", 0.0) or 0.0)
        for item in accounts_overview.values()
        if item is not None
    ]
    latest_sync_time = max((str((item or {}).get("last_sync_time") or "") for item in accounts_overview.values()), default="")
    statuses = {str((item or {}).get("last_sync_status") or "never") for item in accounts_overview.values()}
    if "failed" in statuses:
        sync_status = "failed"
    elif "completed" in statuses:
        sync_status = "completed"
    else:
        sync_status = "never"
    return {
        "cash": float(cash),
        "equity": float(equity),
        "cash_by_currency": cash_by_currency,
        "equity_by_currency": equity_by_currency,
        "gross_exposure_by_currency": gross_exposure_by_currency,
        "realized_pnl_by_currency": realized_pnl_by_currency,
        "unrealized_pnl_by_currency": unrealized_pnl_by_currency,
        "display_currency": single_currency,
        "drawdown_pct": float(min(drawdowns) if drawdowns else 0.0),
        "open_positions": int(open_positions),
        "pending_orders": int(pending_orders),
        "last_sync_time": latest_sync_time,
        "last_sync_status": sync_status,
        "trade_performance": repository.trade_performance_report(),
        "warning": (
            "\uc804\uccb4 \ud569\uc0b0 \ubdf0\ub294 \ucc38\uace0\uc6a9\uc774\uba70 \uc8fc\ubb38 \uac00\ub2a5 \uc794\uace0 \uae30\uc900\uc774 \uc544\ub2d9\ub2c8\ub2e4."
            " \ub2ec\ub7ec/\uc6d0\ud654 \ud63c\uc6a9 \uacc4\uc88c\ub294 \ubd88\ub7ec\uc628 FX \uae30\uc900\uc73c\ub85c\ub9cc \ud45c\uc2dc\ud574\uc57c \ud569\ub2c8\ub2e4."
        ),
    }


def load_monitor_open_positions(settings: RuntimeSettings) -> pd.DataFrame:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    return _localize_timestamp_columns(repository.open_positions())


def load_monitor_recent_orders(settings: RuntimeSettings, limit: int = 30) -> pd.DataFrame:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    return _localize_timestamp_columns(repository.recent_orders(limit=limit))


def build_asset_overview(settings: RuntimeSettings, kis_enabled: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_ids_by_asset = {
        asset_type: set(active_strategy_ids(settings, asset_schedule_key=asset_type))
        for asset_type in settings.asset_schedules.keys()
    }
    for asset_type, schedule in settings.asset_schedules.items():
        universe = settings.universes.get(asset_type)
        watchlist = list(universe.watchlist) if universe else []
        top_universe = list(universe.top_universe) if universe else []
        representative_symbols = watchlist or top_universe
        representative_symbol = representative_symbols[0] if representative_symbols else ""
        active_ids = active_ids_by_asset.get(asset_type) or set()
        visible_strategies = list(iter_visible_strategy_rows(settings, asset_schedule_key=asset_type))
        active_visible = [strategy for strategy in visible_strategies if str(strategy.strategy_id) in active_ids]
        display_strategies = active_visible
        if not display_strategies and visible_strategies:
            if asset_type == "한국주식":
                preferred = default_kr_strategy(settings)
            elif asset_type == "미국주식":
                preferred = default_strategy(settings, asset_type)
            else:
                preferred = None
            if preferred is not None and strategy_asset_schedule_key(preferred) == asset_type and getattr(preferred, "visible_in_ui", True):
                display_strategies = [preferred]
            else:
                display_strategies = visible_strategies[:1]
        if display_strategies:
            for strategy in display_strategies:
                strategy_view = strategy_runtime_config(strategy)
                rows.append(
                    {
                        "자산유형": asset_type,
                        "전략": str(strategy_label(strategy)),
                        "전략ID": str(strategy.strategy_id),
                        "실험전략": "예" if bool(strategy.experimental) else "",
                        "타임프레임": str(strategy_view.timeframe if strategy_view is not None else strategy.timeframe),
                        "세션모드": strategy_session_label(strategy),
                        "시간대": schedule.timezone,
                        "스캔주기(분)": int(strategy_view.scan_interval_minutes if strategy_view is not None else strategy.scan_interval_minutes),
                        "진입주기(분)": int(strategy_view.entry_interval_minutes if strategy_view is not None else strategy.entry_interval_minutes),
                        "청산주기(분)": int(strategy_view.exit_interval_minutes if strategy_view is not None else strategy.exit_interval_minutes),
                        "실행브로커": resolve_broker_mode(symbol=representative_symbol, asset_type=asset_type, kis_enabled=kis_enabled),
                        "Watchlist 개수": len(watchlist),
                        "Top Universe 개수": len(top_universe),
                        "대표 종목": ", ".join((watchlist or top_universe)[:4]),
                    }
                )
            continue
        rows.append(
            {
                "자산유형": asset_type,
                "전략": "",
                "전략ID": "",
                "실험전략": "",
                "타임프레임": schedule.timeframe,
                "세션모드": schedule.session_mode,
                "시간대": schedule.timezone,
                "스캔주기(분)": schedule.scan_interval_minutes,
                "진입주기(분)": schedule.entry_interval_minutes,
                "청산주기(분)": schedule.exit_interval_minutes,
                "실행브로커": resolve_broker_mode(symbol=representative_symbol, asset_type=asset_type, kis_enabled=kis_enabled),
                "Watchlist 개수": len(watchlist),
                "Top Universe 개수": len(top_universe),
                "대표 종목": ", ".join((watchlist or top_universe)[:4]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["자산유형", "전략ID"]).reset_index(drop=True) if not frame.empty else frame


def compute_auto_trading_status(repository: TradingRepository, loop_sleep_seconds: int, now: datetime | None = None) -> Dict[str, Any]:
    now_utc = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    paused = repository.get_control_flag("trading_paused", "0") == "1"
    stale_after_seconds = max(int(loop_sleep_seconds) * 3, 180)
    heartbeat_at = _parse_utc_timestamp(repository.get_control_flag("worker_heartbeat_at", ""))
    heartbeat_source = "worker_heartbeat_at"
    if heartbeat_at is None:
        heartbeat = repository.latest_job_heartbeat()
        heartbeat_at = _parse_utc_timestamp(heartbeat.get("heartbeat_at"))
        heartbeat_source = str(heartbeat.get("job_name") or "job_runs")
    if heartbeat_at is None:
        return {"state": "stopped", "label": "Stopped", "heartbeat_at": "", "heartbeat_at_kst": "", "heartbeat_age_seconds": None, "reason": "worker heartbeat가 없습니다.", "source": "none"}
    heartbeat_age_seconds = max((now_utc - heartbeat_at).total_seconds(), 0.0)
    heartbeat_at_kst = heartbeat_at.astimezone(KST)
    if heartbeat_age_seconds > stale_after_seconds:
        return {"state": "stopped", "label": "Stopped", "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"), "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"), "heartbeat_age_seconds": heartbeat_age_seconds, "reason": f"마지막 heartbeat는 {heartbeat_at_kst.strftime('%Y-%m-%d %H:%M:%S')} 입니다.", "source": heartbeat_source}
    if paused:
        return {"state": "paused", "label": "Paused", "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"), "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"), "heartbeat_age_seconds": heartbeat_age_seconds, "reason": "신규 진입이 일시 중단된 상태입니다.", "source": heartbeat_source}
    return {"state": "running", "label": "Running", "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"), "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"), "heartbeat_age_seconds": heartbeat_age_seconds, "reason": "worker heartbeat가 정상입니다.", "source": heartbeat_source}


def _parse_details(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "details_json" not in frame.columns:
        return frame
    out = frame.copy()
    out["details"] = out["details_json"].fillna("{}").map(lambda value: json.loads(str(value or "{}")))
    return out


def _asset_type_from_account_id(account_id: object) -> str:
    mapping = {
        ACCOUNT_KIS_KR_PAPER: "한국주식",
        ACCOUNT_SIM_US_EQUITY: "미국주식",
        ACCOUNT_SIM_CRYPTO: "코인",
    }
    return mapping.get(str(account_id or "").strip(), "")


def _asset_timezone(settings: RuntimeSettings, asset_type: str) -> str:
    schedule = settings.asset_schedules.get(str(asset_type or "").strip())
    return str(getattr(schedule, "timezone", "") or "UTC")


def _is_timestamp_in_local_today(value: object, timezone_name: str) -> bool:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return False
    local_now = pd.Timestamp.now(tz=timezone_name)
    local_value = parsed.tz_convert(timezone_name)
    start = local_now.normalize()
    end = start + pd.Timedelta(days=1)
    return bool(start <= local_value < end)


def _local_today_bounds_utc_by_asset(settings: RuntimeSettings) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for asset_type in settings.asset_schedules.keys():
        timezone_name = _asset_timezone(settings, asset_type)
        local_now = pd.Timestamp.now(tz=timezone_name)
        local_start = local_now.normalize()
        local_end = local_start + pd.Timedelta(days=1)
        bounds[str(asset_type)] = (local_start.tz_convert("UTC"), local_end.tz_convert("UTC"))
    return bounds


def _filter_candidate_scans_to_local_today(frame: pd.DataFrame, settings: RuntimeSettings) -> pd.DataFrame:
    if frame.empty or "created_at" not in frame.columns or "asset_type" not in frame.columns:
        return frame
    created = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    if created.isna().all():
        return frame.iloc[0:0].copy()
    bounds_by_asset = _local_today_bounds_utc_by_asset(settings)
    masks: list[pd.Series] = []
    asset_series = frame["asset_type"].fillna("").astype(str)
    for asset_type, (start_utc, end_utc) in bounds_by_asset.items():
        asset_mask = asset_series == str(asset_type)
        if not bool(asset_mask.any()):
            continue
        masks.append(asset_mask & created.ge(start_utc) & created.lt(end_utc))
    if not masks:
        return frame.iloc[0:0].copy()
    mask = masks[0].copy()
    for extra_mask in masks[1:]:
        mask |= extra_mask
    return frame.loc[mask.fillna(False)].copy()


def _event_asset_type(settings: RuntimeSettings, row: pd.Series) -> str:
    details = row.get("details") or {}
    asset_type = str(details.get("asset_type") or "").strip()
    if asset_type:
        return asset_type
    strategy_id = str(details.get("strategy_version") or "").strip()
    if strategy_id:
        strategy = get_kr_strategy(settings, strategy_id)
        if strategy is not None:
            return strategy_asset_schedule_key(strategy)
    account_id = (
        row.get("account_id")
        or details.get("account_id")
        or details.get("execution_account_id")
        or ""
    )
    return _asset_type_from_account_id(account_id)


def _filter_events_to_local_today(frame: pd.DataFrame, settings: RuntimeSettings) -> pd.DataFrame:
    if frame.empty or "created_at" not in frame.columns:
        return frame
    created = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    if created.isna().all():
        return frame.iloc[0:0].copy()
    details_series = frame["details"] if "details" in frame.columns else pd.Series([{}] * len(frame), index=frame.index)
    strategy_asset_map = {
        str(strategy.strategy_id): str(strategy_asset_schedule_key(strategy))
        for strategy in iter_strategy_rows(settings)
    }
    account_series = frame["account_id"].fillna("").astype(str) if "account_id" in frame.columns else pd.Series("", index=frame.index, dtype="object")
    details_asset = details_series.map(lambda item: str((item or {}).get("asset_type") or "").strip())
    strategy_asset = details_series.map(
        lambda item: strategy_asset_map.get(str((item or {}).get("strategy_version") or "").strip(), "")
    )
    details_account = details_series.map(
        lambda item: str((item or {}).get("account_id") or (item or {}).get("execution_account_id") or "").strip()
    )
    resolved_account = account_series.where(account_series != "", details_account)
    asset_series = details_asset.where(details_asset != "", strategy_asset)
    asset_series = asset_series.where(asset_series != "", resolved_account.map(_asset_type_from_account_id))
    asset_series = asset_series.where(asset_series != "", "한국주식")
    bounds_by_asset = _local_today_bounds_utc_by_asset(settings)
    masks: list[pd.Series] = []
    for asset_type, (start_utc, end_utc) in bounds_by_asset.items():
        asset_mask = asset_series == str(asset_type)
        if not bool(asset_mask.any()):
            continue
        masks.append(asset_mask & created.ge(start_utc) & created.lt(end_utc))
    if not masks:
        return frame.iloc[0:0].copy()
    mask = masks[0].copy()
    for extra_mask in masks[1:]:
        mask |= extra_mask
    return frame.loc[mask.fillna(False)].copy()


def _execution_summary(events: pd.DataFrame) -> Dict[str, Any]:
    summary = {f"today_{key}_count": 0 for key in EXECUTION_EVENT_KEYS}
    if events.empty:
        summary["today_noop_breakdown"] = pd.DataFrame(columns=["reason", "count"])
        summary["today_entry_rejected_breakdown"] = pd.DataFrame(columns=["reason", "count"])
        return summary
    counts = events["event_type"].value_counts()
    for key in EXECUTION_EVENT_KEYS:
        summary[f"today_{key}_count"] = int(counts.get(key, 0))
    noop_reasons = (
        events.loc[events["event_type"].astype(str) == "noop", "details"]
        .map(lambda item: str((item or {}).get("reason") or "unknown"))
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )
    summary["today_noop_breakdown"] = noop_reasons
    entry_rejected_reasons = (
        events.loc[events["event_type"].astype(str) == "entry_rejected", "details"]
        .map(lambda item: str((item or {}).get("reason") or "unknown"))
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )
    summary["today_entry_rejected_breakdown"] = entry_rejected_reasons
    return summary


def _broker_sync_status(job_health: pd.DataFrame) -> pd.DataFrame:
    if job_health.empty:
        return pd.DataFrame(columns=["job_name", "status", "heartbeat_at"])
    rows = []
    for job_name in BROKER_SYNC_JOBS:
        job_rows = job_health.loc[job_health["job_name"].astype(str) == job_name]
        if job_rows.empty:
            continue
        row = job_rows.iloc[0]
        rows.append(
            {
                "job_name": job_name,
                "status": str(row["status"]),
                "heartbeat_at": row.get("finished_at") or row.get("started_at") or row.get("scheduled_at"),
                "error_message": str(row.get("error_message") or ""),
            }
        )
    return pd.DataFrame(rows)


def _strategy_event_mask(frame: pd.DataFrame, strategy_id: str) -> pd.Series:
    if frame.empty or "details" not in frame.columns:
        return pd.Series(dtype=bool)
    return frame["details"].map(lambda item: str((item or {}).get("strategy_version") or "") == str(strategy_id))


def _top_reason(frame: pd.DataFrame, event_type: str) -> str:
    if frame.empty:
        return ""
    series = (
        frame.loc[frame["event_type"].astype(str) == event_type, "details"]
        .map(lambda item: str((item or {}).get("reason") or ""))
        .loc[lambda values: values.astype(str).str.strip() != ""]
        .value_counts()
    )
    if series.empty:
        return ""
    return str(series.index[0])


def _reason_breakdown(frame: pd.DataFrame, event_type: str, *, limit: int = 3) -> str:
    if frame.empty:
        return ""
    series = (
        frame.loc[frame["event_type"].astype(str) == event_type, "details"]
        .map(lambda item: str((item or {}).get("reason") or ""))
        .loc[lambda values: values.astype(str).str.strip() != ""]
        .value_counts()
        .head(limit)
    )
    if series.empty:
        return ""
    return ", ".join(f"{idx}({int(value)})" for idx, value in series.items())


def _normalize_session_mode_label(value: object) -> str:
    mode = str(value or "").strip()
    return {
        "regular": "regular",
        "pre_close": "pre_close",
        "after_close_close_price": "after_close_close",
        "after_close_close": "after_close_close",
        "after_close_single_price": "after_close_single",
        "after_close_single": "after_close_single",
    }.get(mode, mode)


def _monitor_strategy_rows(
    repository: TradingRepository,
    settings: RuntimeSettings,
    candidate_scans: pd.DataFrame,
    today_events: pd.DataFrame,
) -> list[Any]:
    strategy_by_id = {str(strategy.strategy_id): strategy for strategy in iter_strategy_rows(settings)}
    visible_ids = [str(strategy.strategy_id) for strategy in iter_visible_strategy_rows(settings)]
    active_ids: set[str] = set()
    open_positions = repository.open_positions()
    if not open_positions.empty and "strategy_version" in open_positions.columns:
        active_ids.update(
            str(value)
            for value in open_positions["strategy_version"].dropna().astype(str).tolist()
            if str(value).strip()
        )
    open_orders = repository.open_orders()
    if not open_orders.empty and "strategy_version" in open_orders.columns:
        active_ids.update(
            str(value)
            for value in open_orders["strategy_version"].dropna().astype(str).tolist()
            if str(value).strip()
        )
    ordered_ids = [strategy_id for strategy_id in visible_ids if strategy_id in strategy_by_id]
    ordered_ids.extend(
        strategy_id
        for strategy_id in sorted(active_ids)
        if strategy_id in strategy_by_id and strategy_id not in ordered_ids
    )
    return [strategy_by_id[strategy_id] for strategy_id in ordered_ids]


def _kr_strategy_session_window(strategy) -> str:
    strategy_view = strategy_runtime_config(strategy)
    start = str((strategy_view.entry_window_start if strategy_view is not None else strategy.entry_window_start) or "").strip()
    end = str((strategy_view.entry_window_end if strategy_view is not None else strategy.entry_window_end) or "").strip()
    if start and end:
        return f"{start}~{end}"
    return "-"


def _kr_strategy_price_policy(strategy) -> str:
    strategy_view = strategy_runtime_config(strategy)
    mode = str((strategy_view.session_mode if strategy_view is not None else strategy.session_mode) or "regular")
    return {
        "regular": "시장가/호가 기반",
        "pre_close": "종가 근접 pre-close",
        "after_close_close_price": "당일 종가 고정",
        "after_close_single_price": "시간외 단일가 예상체결",
    }.get(mode, str((strategy_view.session_price_policy if strategy_view is not None else strategy.session_price_policy) or "-"))


def _kr_strategy_execution_cadence(strategy) -> str:
    strategy_view = strategy_runtime_config(strategy)
    mode = str((strategy_view.session_mode if strategy_view is not None else strategy.session_mode) or "regular")
    if mode == "after_close_single_price":
        return f"{int((strategy_view.auction_interval_minutes if strategy_view is not None else strategy.auction_interval_minutes) or 10)}분 단일가 경매"
    timeframe = str(strategy_view.timeframe if strategy_view is not None else strategy.timeframe)
    if timeframe == "1d":
        return "종가 기준"
    return f"{timeframe} 완성봉 기준"


def _kr_strategy_intended_use(strategy) -> str:
    strategy_id = str(strategy.strategy_id)
    if strategy_id == "kr_daily_preclose_v1":
        return "legacy"
    if strategy_id == "kr_intraday_1h_v1":
        return "recommended default"
    if strategy_id == "kr_intraday_15m_v1":
        return "regular intraday only"
    if strategy_id == "kr_intraday_15m_v1_auto":
        return "auto regular + after-close"
    if strategy_id == "kr_intraday_15m_v1_after_close_close":
        return "post-close fixed close"
    if strategy_id == "kr_intraday_15m_v1_after_close_single":
        return "10-minute single-price auction"
    return "custom"


def _build_kr_strategy_overview(
    repository: TradingRepository,
    settings: RuntimeSettings,
    candidate_scans: pd.DataFrame,
    today_events: pd.DataFrame,
    *,
    kis_enabled: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in _monitor_strategy_rows(repository, settings, candidate_scans, today_events):
        strategy_id = str(strategy.strategy_id)
        account_id = str(strategy.execution_account_id or "")
        asset_type = strategy_asset_schedule_key(strategy)
        strategy_view = strategy_runtime_config(strategy)
        universe = settings.universes.get(asset_type)
        candidate_symbols = list(getattr(universe, "watchlist", []) or []) + list(getattr(universe, "top_universe", []) or [])
        representative_symbol = str(candidate_symbols[0] if candidate_symbols else ("005930.KS" if asset_type == "한국주식" else "AAPL"))
        broker_mode = resolve_broker_mode(symbol=representative_symbol, asset_type=asset_type, kis_enabled=kis_enabled)
        candidate_frame = candidate_scans.loc[
            (candidate_scans["asset_type"].astype(str) == asset_type)
            & (candidate_scans["strategy_version"].astype(str) == strategy_id)
            & (candidate_scans["execution_account_id"].astype(str) == account_id)
        ].copy() if not candidate_scans.empty else pd.DataFrame()
        position_frame = repository.open_positions(account_id=account_id, strategy_version=strategy_id)
        order_frame = repository.open_orders(account_id=account_id, strategy_version=strategy_id)
        pending_orders = int(
            len(
                order_frame.loc[
                    order_frame["status"].astype(str).isin({"new", "submitted", "acknowledged", "pending_fill", "partially_filled"})
                ]
            )
        ) if not order_frame.empty else 0
        strategy_events = today_events.loc[_strategy_event_mask(today_events, strategy_id)].copy() if not today_events.empty else pd.DataFrame()
        event_counts = strategy_events["event_type"].astype(str).value_counts() if not strategy_events.empty else pd.Series(dtype=int)
        last_event_at = ""
        last_fill_at = ""
        if not strategy_events.empty:
            last_event_at = str(strategy_events.iloc[0].get("created_at") or "")
            fill_rows = strategy_events.loc[strategy_events["event_type"].astype(str) == "filled"]
            if not fill_rows.empty:
                last_fill_at = str(fill_rows.iloc[0].get("created_at") or "")
        pending_fill_count = int(
            len(
                order_frame.loc[
                    order_frame["status"].astype(str).isin({"pending_fill", "partially_filled"})
                ]
            )
        ) if not order_frame.empty else 0
        rows.append(
            {
                "strategy_id": strategy_id,
                "display_name": str(strategy.display_name),
                "label": str(strategy_label(strategy)),
                "strategy_family": str(strategy.strategy_family),
                "session_mode": str(strategy_session_label(strategy)),
                "timeframe": str(strategy_view.timeframe if strategy_view is not None else strategy.timeframe),
                "enabled": bool(strategy.enabled),
                "experimental": bool(strategy.experimental),
                "visible_in_ui": bool(getattr(strategy, "visible_in_ui", True)),
                "broker_mode": broker_mode,
                "execution_account_id": account_id,
                "session_window": _kr_strategy_session_window(strategy),
                "price_policy": _kr_strategy_price_policy(strategy),
                "execution_cadence": _kr_strategy_execution_cadence(strategy),
                "intended_use": _kr_strategy_intended_use(strategy),
                "primary_target": str(strategy.primary_target),
                "secondary_target": str(strategy.secondary_target),
                "analysis_target": str(strategy.analysis_target),
                "today_candidate_count": int(len(candidate_frame)),
                "today_entry_allowed_count": int(event_counts.get("entry_allowed", 0)),
                "today_entry_rejected_count": int(event_counts.get("entry_rejected", 0)),
                "today_submit_requested_count": int(event_counts.get("submit_requested", 0)),
                "today_submitted_count": int(event_counts.get("submitted", 0)),
                "today_acknowledged_count": int(event_counts.get("acknowledged", 0)),
                "today_pending_fill_count": max(int(event_counts.get("pending_fill", 0)), pending_fill_count),
                "today_filled_count": int(event_counts.get("filled", 0)),
                "today_rejected_count": int(event_counts.get("rejected", 0)),
                "today_cancelled_count": int(event_counts.get("cancelled", 0)),
                "today_noop_count": int(event_counts.get("noop", 0)),
                "today_noop_top_reason": _top_reason(strategy_events, "noop"),
                "today_noop_breakdown": _reason_breakdown(strategy_events, "noop"),
                "today_reject_top_reason": _top_reason(strategy_events, "entry_rejected"),
                "today_reject_breakdown": _reason_breakdown(strategy_events, "entry_rejected"),
                "open_positions": int(len(position_frame)),
                "pending_orders": pending_orders,
                "last_event_at": last_event_at,
                "last_fill_at": last_fill_at,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["enabled", "experimental", "timeframe", "strategy_id"], ascending=[False, True, True, True]).reset_index(drop=True) if not frame.empty else frame


def _build_kr_strategy_recent_events(settings: RuntimeSettings, today_events: pd.DataFrame, limit: int = 60) -> pd.DataFrame:
    if today_events.empty:
        return pd.DataFrame(columns=["created_at", "strategy_id", "account_id", "symbol", "event_type", "reason", "message"])
    rows: list[dict[str, Any]] = []
    strategy_map = {str(strategy.strategy_id): strategy for strategy in iter_strategy_rows(settings)}
    valid_ids = set(strategy_map)
    for _, row in today_events.iterrows():
        details = row.get("details") or {}
        strategy_id = str(details.get("strategy_version") or "").strip()
        if not strategy_id or strategy_id not in valid_ids:
            continue
        rows.append(
            {
                "created_at": row.get("created_at"),
                "strategy_id": strategy_id,
                "session_mode": _normalize_session_mode_label(
                    details.get("session_mode") or strategy_session_label(strategy_map.get(strategy_id))
                ),
                "account_id": row.get("account_id") or details.get("account_id") or details.get("execution_account_id") or "",
                "symbol": details.get("symbol") or "",
                "event_type": row.get("event_type") or "",
                "reason": details.get("reason") or "",
                "message": row.get("message") or "",
            }
        )
    frame = pd.DataFrame(rows)
    return frame.head(limit) if not frame.empty else frame


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    repository.expire_stale_job_runs()
    kis_enabled = _kis_enabled_for_monitor(settings, repository)
    runtime_profile = _runtime_profile_read_model(repository, settings)
    accounts = repository.load_broker_accounts(active_only=True)
    summary = repository.dashboard_counts()
    equity_curve = _localize_timestamp_columns(
        _cached_dataframe(repository, "account_snapshots", "500", loader=lambda: repository.load_account_snapshots(limit=500))
    )
    trade_performance_snapshots = _localize_timestamp_columns(
        _cached_dataframe(
            repository,
            "account_snapshots",
            "2000",
            "trade",
            loader=lambda: repository.load_account_snapshots(limit=2000),
        )
    )
    open_positions = _localize_timestamp_columns(repository.open_positions())
    open_orders = _localize_timestamp_columns(repository.open_orders())
    job_health = _localize_timestamp_columns(
        _cached_dataframe(
            repository,
            "job_health",
            "200",
            ttl_seconds=_SHORT_FRAME_CACHE_TTL_SECONDS,
            loader=lambda: repository.recent_job_health(limit=200),
        )
    )
    recent_events = _localize_timestamp_columns(
        _cached_dataframe(
            repository,
            "system_events",
            "200",
            ttl_seconds=_SHORT_FRAME_CACHE_TTL_SECONDS,
            loader=lambda: repository.recent_system_events(limit=200),
        )
    )
    recent_errors = _localize_timestamp_columns(
        _cached_dataframe(
            repository,
            "system_events",
            "ERROR",
            "50",
            ttl_seconds=_SHORT_FRAME_CACHE_TTL_SECONDS,
            loader=lambda: repository.recent_system_events(level="ERROR", limit=50),
        )
    )
    accounts_overview = _build_accounts_overview(
        repository,
        accounts,
        latest_snapshots=trade_performance_snapshots,
        open_positions=open_positions,
        open_orders=open_orders,
        recent_events=recent_events,
        trade_performance_snapshots=trade_performance_snapshots,
    )
    total_portfolio_overview = _build_total_portfolio_overview(accounts_overview, repository)
    equity_curves_by_account = {
        account_id: _localize_timestamp_columns(
            _cached_dataframe(
                repository,
                "account_snapshots",
                "200",
                account_id,
                loader=lambda account_id=account_id: repository.load_account_snapshots(limit=200, account_id=account_id),
            )
        ).sort_values("created_at")
        for account_id in accounts_overview.keys()
    }
    raw_today_events = _parse_details(
        _cached_dataframe(repository, "system_events", "5000", loader=lambda: repository.recent_system_events(limit=5000))
    )
    today_events = _localize_timestamp_columns(_filter_events_to_local_today(raw_today_events, settings))
    execution_summary = _execution_summary(today_events)
    raw_candidate_scans = _cached_dataframe(repository, "candidate_scans", "5000", loader=lambda: repository.latest_candidates(limit=5000))
    candidate_scans = _localize_timestamp_columns(_filter_candidate_scans_to_local_today(raw_candidate_scans, settings))
    kr_strategy_overview = _build_kr_strategy_overview(repository, settings, candidate_scans, today_events, kis_enabled=kis_enabled)
    kr_strategy_recent_events = _localize_timestamp_columns(_build_kr_strategy_recent_events(settings, today_events))
    broker_sync_status = _localize_timestamp_columns(_broker_sync_status(job_health))
    broker_error_mask = (
        (today_events["level"].astype(str) == "ERROR")
        & (
            today_events["component"].astype(str).str.contains("broker|kis_", na=False)
            | today_events["event_type"].astype(str).isin(BROKER_SYNC_JOBS)
        )
    )
    recent_broker_errors = _localize_timestamp_columns(today_events.loc[broker_error_mask].head(100))
    kis_open_orders = open_orders.loc[
        open_orders["raw_json"].fillna("{}").astype(str).map(lambda value: json.loads(str(value or "{}")).get("broker") == "kis_mock")
    ].copy() if not open_orders.empty else pd.DataFrame()
    pending_submitted_orders = int(
        len(
            kis_open_orders.loc[
                kis_open_orders["status"].astype(str).isin({"submitted", "acknowledged", "pending_fill", "partially_filled"})
            ]
        )
    ) if not kis_open_orders.empty else 0
    broker_rejects_today = int(
        len(
            today_events.loc[
                (today_events["component"].astype(str) == "kis_execution")
                & (today_events["event_type"].astype(str) == "rejected")
            ]
        )
    )
    kis_runtime = {
        "last_broker_account_sync": repository.get_control_flag("kis_last_account_sync_at", ""),
        "last_broker_order_sync": repository.get_control_flag("kis_last_order_sync_at", ""),
        "last_websocket_execution_event": repository.get_control_flag("kis_last_websocket_execution_at", ""),
        "last_websocket_quote_at": repository.get_control_flag("kis_last_websocket_quote_at", ""),
        "quote_stream_status": repository.get_control_flag("kis_quote_stream_status", ""),
        "quote_stream_symbols": repository.get_control_flag("kis_quote_stream_symbols", ""),
        "pending_submitted_orders": pending_submitted_orders,
        "broker_rejects_today": broker_rejects_today,
    }
    return {
        "summary": summary,
        "accounts_overview": accounts_overview,
        "total_portfolio_overview": total_portfolio_overview,
        "prediction_report": _localize_timestamp_columns(
            _cached_dataframe(repository, "prediction_report", "200", loader=lambda: repository.prediction_report(limit=200))
        ),
        "open_positions": open_positions,
        "recent_realized_trades": _localize_timestamp_columns(repository.recent_closed_positions(limit=100)),
        "open_orders": open_orders,
        "candidate_scans": candidate_scans,
        "asset_overview": build_asset_overview(settings, kis_enabled=kis_enabled),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
        "equity_curves_by_account": equity_curves_by_account,
        "job_health": job_health,
        "recent_errors": recent_errors,
        "recent_events": recent_events,
        "trade_performance": _trade_performance_from_snapshots(trade_performance_snapshots, repository),
        "auto_trading_status": compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds),
        "broker_sync_status": broker_sync_status,
        "broker_sync_errors": recent_broker_errors,
        "execution_summary": execution_summary,
        "today_execution_events": _localize_timestamp_columns(today_events),
        "kis_runtime": kis_runtime,
        "runtime_profile": runtime_profile,
        "kr_strategy_overview": kr_strategy_overview,
        "kr_strategy_recent_events": kr_strategy_recent_events,
    }
