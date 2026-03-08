from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
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


def build_asset_overview(settings: RuntimeSettings, kis_enabled: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset_type, schedule in settings.asset_schedules.items():
        universe = settings.universes.get(asset_type)
        watchlist = list(universe.watchlist) if universe else []
        top_universe = list(universe.top_universe) if universe else []
        rows.append(
            {
                "자산유형": asset_type,
                "타임프레임": schedule.timeframe,
                "세션모드": schedule.session_mode,
                "시간대": schedule.timezone,
                "스캔주기(분)": schedule.scan_interval_minutes,
                "진입주기(분)": schedule.entry_interval_minutes,
                "청산주기(분)": schedule.exit_interval_minutes,
                "실행브로커": resolve_broker_mode(asset_type=asset_type, kis_enabled=kis_enabled),
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
    summary = repository.dashboard_counts()
    equity_curve = _localize_timestamp_columns(repository.load_account_snapshots(limit=500))
    job_health = _localize_timestamp_columns(repository.recent_job_health(limit=200))
    recent_events = _localize_timestamp_columns(repository.recent_system_events(limit=200))
    today_events = _parse_details(repository.system_events_by_date(str(pd.Timestamp.utcnow().date()), limit=2000))
    execution_summary = _execution_summary(today_events)
    broker_sync_status = _localize_timestamp_columns(_broker_sync_status(job_health))
    broker_error_mask = (
        (today_events["level"].astype(str) == "ERROR")
        & (
            today_events["component"].astype(str).str.contains("kis_|execution|broker", na=False)
            | today_events["message"].astype(str).str.contains("broker_|KIS|execution", case=False, na=False)
            | today_events["event_type"].astype(str).isin({"job_failed", "rejected", "cancelled", "expired", *BROKER_SYNC_JOBS})
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
        "prediction_report": _localize_timestamp_columns(repository.prediction_report(limit=200)),
        "open_positions": _localize_timestamp_columns(repository.open_positions()),
        "open_orders": open_orders,
        "candidate_scans": _localize_timestamp_columns(repository.latest_candidates(limit=100)),
        "asset_overview": build_asset_overview(settings, kis_enabled=kis_enabled),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
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
    }
