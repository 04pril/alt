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
            broker_sync_errors["message"].astype(str).str.contains("broker_", case=False, na=False)
            | broker_sync_errors["event_type"].astype(str).str.contains("broker_", case=False, na=False)
        ].copy()
    return broker_sync_status, broker_sync_errors


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
            "reason": f"마지막 heartbeat는 {heartbeat_at_kst.strftime('%Y-%m-%d %H:%M:%S')} 입니다.",
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
    return {
        "summary": summary,
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
        "broker_sync_errors": broker_sync_errors,
        "trade_performance": repository.trade_performance_report(),
        "auto_trading_status": compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds),
    }
