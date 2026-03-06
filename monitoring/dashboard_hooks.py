from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from storage.repository import TradingRepository

KST = ZoneInfo("Asia/Seoul")


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
            "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": f"마지막 heartbeat는 {heartbeat_at_kst.strftime('%Y-%m-%d %H:%M:%S KST')} 입니다.",
            "source": heartbeat_source,
        }

    if paused:
        return {
            "state": "paused",
            "label": "Paused",
            "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
            "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "reason": "신규 진입이 일시 중단된 상태입니다.",
            "source": heartbeat_source,
        }

    return {
        "state": "running",
        "label": "Running",
        "heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z"),
        "heartbeat_at_kst": heartbeat_at_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "reason": "worker heartbeat가 정상입니다.",
        "source": heartbeat_source,
    }


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
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
        "recent_events": _localize_timestamp_columns(repository.recent_system_events(limit=50)),
        "trade_performance": repository.trade_performance_report(),
        "auto_trading_status": compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds),
    }
