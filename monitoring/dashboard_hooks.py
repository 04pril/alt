from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
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
    "filled",
    "rejected",
    "cancelled",
    "noop",
]


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
    return {
        "name": str(profile_name or "baseline"),
        "source": str(profile_source or "embedded_defaults"),
    }


def _account_display_order() -> list[str]:
    return [ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO]


def _sync_status_from_events(repository: TradingRepository, account_id: str, latest_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    events = repository.recent_system_events(limit=200, account_id=account_id)
    if not events.empty:
        sync_events = events.loc[
            events["event_type"].astype(str).isin({"broker_account_sync", "broker_order_sync", "broker_position_sync", "broker_market_status"})
            | events["component"].astype(str).str.contains("kis_execution|execution_pipeline", na=False)
        ]
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


def _build_accounts_overview(repository: TradingRepository) -> Dict[str, Dict[str, Any]]:
    accounts = repository.load_broker_accounts(active_only=True)
    if accounts.empty:
        return {}
    overviews: Dict[str, Dict[str, Any]] = {}
    for account_id in _account_display_order():
        matched = accounts.loc[accounts["account_id"].astype(str) == account_id]
        if matched.empty:
            continue
        account_row = matched.iloc[0].to_dict()
        latest_snapshot = repository.latest_account_snapshot(account_id=account_id) or {}
        sync_state = _sync_status_from_events(repository, account_id, latest_snapshot)
        open_positions = repository.open_positions(account_id=account_id)
        open_orders = repository.open_orders(account_id=account_id)
        pending_orders = int(
            len(open_orders.loc[open_orders["status"].astype(str).isin({"new", "submitted", "acknowledged", "pending_fill", "partially_filled"})])
        ) if not open_orders.empty else 0
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
            "open_positions": int(len(open_positions)),
            "pending_orders": pending_orders,
            "last_sync_time": sync_state["last_sync_time"],
            "last_sync_status": sync_state["last_sync_status"],
            "latest_snapshot": latest_snapshot,
            "trade_performance": repository.trade_performance_report(account_id=account_id),
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
    for asset_type, schedule in settings.asset_schedules.items():
        universe = settings.universes.get(asset_type)
        watchlist = list(universe.watchlist) if universe else []
        top_universe = list(universe.top_universe) if universe else []
        representative_symbols = watchlist or top_universe
        representative_symbol = representative_symbols[0] if representative_symbols else ""
        rows.append(
            {
                "자산유형": asset_type,
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
    return frame.sort_values("자산유형").reset_index(drop=True) if not frame.empty else frame


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


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    kis_enabled = _kis_enabled_for_monitor(settings, repository)
    runtime_profile = _runtime_profile_read_model(repository, settings)
    accounts_overview = _build_accounts_overview(repository)
    total_portfolio_overview = _build_total_portfolio_overview(accounts_overview, repository)
    summary = repository.dashboard_counts()
    equity_curve = _localize_timestamp_columns(repository.load_account_snapshots(limit=500))
    equity_curves_by_account = {
        account_id: _localize_timestamp_columns(repository.load_account_snapshots(limit=200, account_id=account_id)).sort_values("created_at")
        for account_id in accounts_overview.keys()
    }
    job_health = _localize_timestamp_columns(repository.recent_job_health(limit=200))
    recent_events = _localize_timestamp_columns(repository.recent_system_events(limit=200))
    today_events = _parse_details(repository.system_events_by_date(str(pd.Timestamp.utcnow().date()), limit=2000))
    execution_summary = _execution_summary(today_events)
    broker_sync_status = _localize_timestamp_columns(_broker_sync_status(job_health))
    broker_error_mask = (
        (today_events["level"].astype(str) == "ERROR")
        & (
            today_events["component"].astype(str).str.contains("broker|kis_", na=False)
            | today_events["event_type"].astype(str).isin(BROKER_SYNC_JOBS)
        )
    )
    recent_broker_errors = _localize_timestamp_columns(today_events.loc[broker_error_mask].head(100))
    open_orders = _localize_timestamp_columns(repository.open_orders())
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
        "pending_submitted_orders": pending_submitted_orders,
        "broker_rejects_today": broker_rejects_today,
    }
    return {
        "summary": summary,
        "accounts_overview": accounts_overview,
        "total_portfolio_overview": total_portfolio_overview,
        "prediction_report": _localize_timestamp_columns(repository.prediction_report(limit=200)),
        "open_positions": _localize_timestamp_columns(repository.open_positions()),
        "open_orders": open_orders,
        "candidate_scans": _localize_timestamp_columns(repository.latest_candidates(limit=100)),
        "asset_overview": build_asset_overview(settings, kis_enabled=kis_enabled),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
        "equity_curves_by_account": equity_curves_by_account,
        "job_health": job_health,
        "recent_errors": _localize_timestamp_columns(repository.recent_system_events(level="ERROR", limit=50)),
        "recent_events": recent_events,
        "trade_performance": repository.trade_performance_report(),
        "auto_trading_status": compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds),
        "broker_sync_status": broker_sync_status,
        "broker_sync_errors": recent_broker_errors,
        "execution_summary": execution_summary,
        "today_execution_events": _localize_timestamp_columns(today_events),
        "kis_runtime": kis_runtime,
        "runtime_profile": runtime_profile,
    }
