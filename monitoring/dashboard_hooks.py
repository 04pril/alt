from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from storage.repository import TradingRepository, parse_utc_timestamp

KST = ZoneInfo("Asia/Seoul")


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
            parsed = pd.to_datetime(localized[column], errors="coerce", utc=True)
            if parsed.notna().any():
                localized[column] = parsed.dt.tz_convert(KST)
    return localized


def compute_auto_trading_status(repository: TradingRepository, loop_sleep_seconds: int, now: datetime | None = None) -> Dict[str, Any]:
    now_utc = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    stale_after_seconds = max(int(loop_sleep_seconds) * 3, 180)
    heartbeat_at = parse_utc_timestamp(repository.get_control_flag("worker_heartbeat_at", ""))
    heartbeat_source = "worker_heartbeat_at"
    if heartbeat_at is None:
        heartbeat = repository.latest_job_heartbeat()
        heartbeat_at = parse_utc_timestamp(heartbeat.get("heartbeat_at"))
        heartbeat_source = str(heartbeat.get("job_name") or "job_runs")

    if heartbeat_at is None:
        return {
            "state": "stopped",
            "label": "Stopped",
            "heartbeat_at": "",
            "heartbeat_at_kst": "",
            "heartbeat_age_seconds": None,
            "reason": "worker heartbeat missing",
            "source": "none",
        }

    heartbeat_age_seconds = max((now_utc - heartbeat_at).total_seconds(), 0.0)
    heartbeat_at_kst = heartbeat_at.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    if heartbeat_age_seconds > stale_after_seconds:
        return {
            "state": "stopped",
            "label": "Stopped",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": f"last heartbeat at {heartbeat_at_kst}",
            "source": heartbeat_source,
        }

    if repository.get_control_flag_bool("worker_paused", False):
        return {
            "state": "paused",
            "label": "Paused",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "worker_paused",
            "source": heartbeat_source,
        }

    if repository.get_control_flag_bool("exit_only_mode", False):
        return {
            "state": "paused",
            "label": "Paused",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "exit_only_mode",
            "source": heartbeat_source,
        }

    if repository.get_control_flag_bool("entry_paused", False):
        return {
            "state": "paused",
            "label": "Paused",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "entry_paused",
            "source": heartbeat_source,
        }

    return {
        "state": "running",
        "label": "Running",
        "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
        "heartbeat_at_kst": heartbeat_at_kst,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "reason": "worker heartbeat is healthy",
        "source": heartbeat_source,
    }


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    repository.initialize_runtime_flags()
    summary = repository.dashboard_counts()
    equity_curve = _localize_timestamp_columns(repository.load_account_snapshots(limit=500))
    return {
        "summary": summary,
        "prediction_report": _localize_timestamp_columns(repository.prediction_report(limit=200)),
        "open_positions": _localize_timestamp_columns(repository.open_positions()),
        "open_orders": _localize_timestamp_columns(repository.open_orders()),
        "candidate_scans": _localize_timestamp_columns(repository.latest_candidates(limit=100)),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
        "job_health": _localize_timestamp_columns(repository.recent_job_health(limit=50)),
        "recent_errors": _localize_timestamp_columns(repository.recent_system_events(level="ERROR", limit=50)),
        "recent_broker_errors": _localize_timestamp_columns(repository.recent_component_events(component="kis_broker", level="ERROR", limit=20)),
        "recent_events": _localize_timestamp_columns(repository.recent_system_events(limit=50)),
        "trade_performance": repository.trade_performance_report(),
        "auto_trading_status": compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds),
        "kis_sync_status": {
            "last_sync_at": repository.get_control_flag("broker_kis_last_sync_at", ""),
            "last_sync_status": repository.get_control_flag("broker_kis_last_sync_status", "never"),
            "last_sync_message": repository.get_control_flag("broker_kis_last_sync_message", ""),
        },
        "broker_modes": dict(settings.broker.asset_broker_mode),
        "control_flags": {
            "entry_paused": repository.get_control_flag_bool("entry_paused", False),
            "worker_paused": repository.get_control_flag_bool("worker_paused", False),
            "exit_only_mode": repository.get_control_flag_bool("exit_only_mode", False),
        },
    }
