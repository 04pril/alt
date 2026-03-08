from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from storage.repository import TradingRepository

KST = ZoneInfo("Asia/Seoul")
BROKER_SYNC_JOBS = ["broker_account_sync", "broker_order_sync", "broker_position_sync"]


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


def _runtime_repository(settings: RuntimeSettings) -> TradingRepository:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    return repository


def set_trading_pause_state(settings: RuntimeSettings, paused: bool, notes: str = "set from streamlit monitor") -> None:
    repository = _runtime_repository(settings)
    repository.set_control_flag("trading_paused", "1" if paused else "0", notes)


def build_broker_sync_read_model(
    job_health: pd.DataFrame,
    recent_events: pd.DataFrame,
) -> tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    broker_sync_status: Dict[str, Dict[str, Any]] = {}
    for job_name in BROKER_SYNC_JOBS:
        if not job_health.empty and "job_name" in job_health.columns:
            matched = job_health.loc[job_health["job_name"].astype(str) == job_name]
            if matched.empty:
                broker_sync_status[job_name] = {"job_name": job_name, "status": "never", "finished_at": pd.NaT, "error_message": ""}
                continue
            row = matched.sort_values(["scheduled_at", "finished_at"], ascending=[False, False]).iloc[0].to_dict()
            broker_sync_status[job_name] = row
            continue
        broker_sync_status[job_name] = {"job_name": job_name, "status": "never", "finished_at": pd.NaT, "error_message": ""}
    broker_sync_errors = recent_events
    if not broker_sync_errors.empty:
        broker_sync_errors = broker_sync_errors[
            (
                broker_sync_errors["message"].astype(str).str.contains("broker_", case=False, na=False)
                | broker_sync_errors["event_type"].astype(str).str.contains("broker_", case=False, na=False)
            )
            & (broker_sync_errors["level"].astype(str).str.upper() == "ERROR")
        ].copy()
    return broker_sync_status, broker_sync_errors


def build_account_snapshot_read_model(summary: Dict[str, Any]) -> Dict[str, Any]:
    latest_account = dict(summary.get("latest_account") or {})
    if not latest_account:
        return {}
    created_at_kst = _to_kst_timestamp(latest_account.get("created_at"))
    latest_account["created_at_kst"] = (
        created_at_kst.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(created_at_kst) else ""
    )
    return latest_account


def build_broker_sync_status_frame(broker_sync_status: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job_name in BROKER_SYNC_JOBS:
        row = dict(broker_sync_status.get(job_name) or {})
        rows.append(
            {
                "job": job_name,
                "status": row.get("status", "never"),
                "finished_at": row.get("finished_at", pd.NaT),
                "retry_count": row.get("retry_count", 0),
                "error_message": row.get("error_message", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return _localize_timestamp_columns(pd.DataFrame(rows))


def build_recent_job_status_summary(
    job_health: pd.DataFrame,
    auto_trading_status: Dict[str, Any],
    *,
    loop_sleep_seconds: int,
    now: datetime | None = None,
) -> Dict[str, str]:
    now_utc = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    stale_after_seconds = max(int(loop_sleep_seconds) * 3, 180)

    latest_timestamp: datetime | None = None
    latest_job_name = ""
    latest_status = ""
    if not job_health.empty:
        latest_row = job_health.iloc[0]
        latest_job_name = str(latest_row.get("job_name") or "")
        latest_status = str(latest_row.get("status") or "").lower()
        for column in ("finished_at", "started_at", "scheduled_at"):
            parsed = _parse_utc_timestamp(latest_row.get(column))
            if parsed is not None:
                latest_timestamp = parsed
                break

    if latest_status == "failed":
        return {
            "label": "오류",
            "reason": latest_job_name or "recent job failed",
        }

    if latest_timestamp is None:
        state = str(auto_trading_status.get("state") or "").lower()
        if state == "running":
            return {"label": "정상", "reason": "worker heartbeat active"}
        return {"label": "지연", "reason": "recent job history unavailable"}

    if (now_utc - latest_timestamp).total_seconds() > stale_after_seconds:
        return {
            "label": "지연",
            "reason": latest_job_name or "recent jobs are stale",
        }

    return {
        "label": "정상",
        "reason": latest_job_name or "recent jobs healthy",
    }


def build_kis_monitor_read_model(
    repository: TradingRepository,
    settings: RuntimeSettings,
    *,
    now: datetime | None = None,
) -> Dict[str, Any]:
    now_utc = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    last_account_sync_at = repository.get_control_flag("kis_last_account_sync_at", "")
    last_account_sync_status = repository.get_control_flag("kis_last_account_sync_status", "")
    last_auth_error_at = repository.get_control_flag("kis_last_auth_error_at", "")
    last_auth_error = repository.get_control_flag("kis_last_auth_error", "")
    last_buying_power_success_at = repository.get_control_flag("kis_last_buying_power_success_at", "")
    last_buying_power_symbol = repository.get_control_flag("kis_last_buying_power_symbol", "")

    sync_dt = _parse_utc_timestamp(last_account_sync_at)
    auth_error_dt = _parse_utc_timestamp(last_auth_error_at)
    stale_after_seconds = max(int(settings.scheduler.broker_account_sync_interval_minutes) * 120, 900)

    if auth_error_dt is not None and (sync_dt is None or auth_error_dt >= sync_dt):
        connection_status = "인증오류"
        status_reason = str(last_auth_error or "KIS 인증 오류")
    elif sync_dt is None or (now_utc - sync_dt).total_seconds() > stale_after_seconds:
        connection_status = "sync지연"
        status_reason = "최근 account sync가 오래되었습니다."
    else:
        connection_status = "정상"
        status_reason = "최근 account sync가 정상입니다."

    return {
        "connection_status": connection_status,
        "status_reason": status_reason,
        "last_account_sync_at": last_account_sync_at,
        "last_account_sync_status": last_account_sync_status or ("ok" if sync_dt is not None else ""),
        "last_auth_error_at": last_auth_error_at,
        "last_auth_error": last_auth_error,
        "last_buying_power_success_at": last_buying_power_success_at,
        "last_buying_power_symbol": last_buying_power_symbol,
    }


def build_asset_overview(settings: RuntimeSettings) -> pd.DataFrame:
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
                "Watchlist 개수": len(watchlist),
                "Top Universe 개수": len(top_universe),
                "대표 심볼": ", ".join((watchlist or top_universe)[:4]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("자산유형").reset_index(drop=True)


def load_monitor_open_positions(settings: RuntimeSettings) -> pd.DataFrame:
    repository = _runtime_repository(settings)
    return _localize_timestamp_columns(repository.open_positions())


def load_monitor_recent_orders(settings: RuntimeSettings, limit: int = 30) -> pd.DataFrame:
    repository = _runtime_repository(settings)
    return _localize_timestamp_columns(repository.recent_orders(limit=limit))


def compute_auto_trading_status(
    repository: TradingRepository,
    loop_sleep_seconds: int,
    now: datetime | None = None,
) -> Dict[str, Any]:
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
        return {
            "state": "stopped",
            "label": "Stopped",
            "heartbeat_at": "",
            "heartbeat_at_kst": "",
            "heartbeat_age_seconds": None,
            "reason": "worker heartbeat가 없습니다.",
            "source": "none",
        }

    heartbeat_age_seconds = max((now_utc - heartbeat_at).total_seconds(), 0.0)
    heartbeat_at_kst = heartbeat_at.astimezone(KST)
    if heartbeat_age_seconds > stale_after_seconds:
        return {
            "state": "stopped",
            "label": "Stopped",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"),
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "worker heartbeat가 지연되었습니다.",
            "source": heartbeat_source,
        }

    if paused:
        return {
            "state": "paused",
            "label": "Paused",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"),
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "신규 진입이 일시 중단된 상태입니다.",
            "source": heartbeat_source,
        }

    return {
        "state": "running",
        "label": "Running",
        "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
        "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S"),
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "reason": "worker heartbeat가 정상입니다.",
        "source": heartbeat_source,
    }


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = _runtime_repository(settings)
    summary = repository.dashboard_counts()
    equity_curve = _localize_timestamp_columns(repository.load_account_snapshots(limit=500))
    job_health = _localize_timestamp_columns(repository.recent_job_health(limit=200))
    recent_events = _localize_timestamp_columns(repository.recent_system_events(limit=100))
    broker_sync_status, broker_sync_errors = build_broker_sync_read_model(job_health, recent_events)
    account_snapshot = build_account_snapshot_read_model(summary)
    broker_sync_frame = build_broker_sync_status_frame(broker_sync_status)
    auto_trading_status = compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds)
    recent_job_status_summary = build_recent_job_status_summary(
        job_health,
        auto_trading_status,
        loop_sleep_seconds=settings.scheduler.loop_sleep_seconds,
    )
    kis_monitor = build_kis_monitor_read_model(repository, settings)
    return {
        "summary": summary,
        "account_snapshot": account_snapshot,
        "prediction_report": _localize_timestamp_columns(repository.prediction_report(limit=200)),
        "open_positions": _localize_timestamp_columns(repository.open_positions()),
        "open_orders": _localize_timestamp_columns(repository.open_orders()),
        "candidate_scans": _localize_timestamp_columns(repository.latest_candidates(limit=100)),
        "asset_overview": build_asset_overview(settings),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
        "job_health": job_health,
        "recent_errors": _localize_timestamp_columns(repository.recent_system_events(level="ERROR", limit=50)),
        "recent_events": recent_events,
        "broker_sync_status": broker_sync_status,
        "broker_sync_frame": broker_sync_frame,
        "broker_sync_errors": broker_sync_errors,
        "trade_performance": repository.trade_performance_report(),
        "auto_trading_status": auto_trading_status,
        "recent_job_status_summary": recent_job_status_summary,
        "kis_monitor": kis_monitor,
    }
