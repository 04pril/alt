from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import load_settings
from jobs.tasks import (
    build_task_context,
    daily_report_job,
    entry_decision_job,
    exit_management_job,
    outcome_resolution_job,
    retrain_check_job,
    scan_job,
)
from storage.repository import utc_now_iso


def _bucket_key(dt: datetime, minutes: int) -> str:
    bucket = (dt.minute // max(minutes, 1)) * max(minutes, 1)
    rounded = dt.replace(minute=bucket, second=0, microsecond=0)
    return rounded.isoformat(timespec="minutes")


def _run_guarded(context, job_name: str, run_key: str, fn):
    job_run_id = context.repository.begin_job_run(
        job_name=job_name,
        run_key=run_key,
        scheduled_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        lock_owner=context.settings.scheduler.lock_owner,
    )
    if job_run_id is None:
        return None
    try:
        result = fn()
    except Exception as exc:  # pragma: no cover - exercised in runtime
        context.repository.finish_job_run(job_run_id, status="failed", error_message=str(exc), metrics={})
        context.repository.log_event("ERROR", "scheduler", "job_failed", f"{job_name} failed", {"error": str(exc)})
        return None
    context.repository.finish_job_run(job_run_id, status="completed", metrics=result if isinstance(result, dict) else {})
    return result


def run_once(settings_path: str | None = None) -> None:
    context = build_task_context(settings_path)
    settings = context.settings
    context.repository.set_control_flag("worker_heartbeat_at", utc_now_iso(), "scheduler loop heartbeat")
    for asset_type, schedule in settings.asset_schedules.items():
        now = datetime.now(ZoneInfo(schedule.timezone))
        _run_guarded(
            context,
            job_name=f"scan:{asset_type}",
            run_key=_bucket_key(now, schedule.scan_interval_minutes),
            fn=lambda asset_type=asset_type: scan_job(context, [asset_type]),
        )
        _run_guarded(
            context,
            job_name=f"entry:{asset_type}",
            run_key=_bucket_key(now, schedule.entry_interval_minutes),
            fn=lambda asset_type=asset_type: entry_decision_job(context, [asset_type]),
        )
    _run_guarded(
        context,
        job_name="exit_management",
        run_key=_bucket_key(datetime.utcnow(), 15),
        fn=lambda: exit_management_job(context),
    )
    _run_guarded(
        context,
        job_name="outcome_resolution",
        run_key=_bucket_key(datetime.utcnow(), 60),
        fn=lambda: outcome_resolution_job(context),
    )
    _run_guarded(
        context,
        job_name="daily_report",
        run_key=str(datetime.utcnow().date()),
        fn=lambda: daily_report_job(context),
    )
    _run_guarded(
        context,
        job_name="retrain_check",
        run_key=str(datetime.utcnow().date()),
        fn=lambda: retrain_check_job(context),
    )


def run_loop(settings_path: str | None = None) -> None:
    settings = load_settings(settings_path)
    while True:
        run_once(settings_path)
        time.sleep(max(settings.scheduler.loop_sleep_seconds, 5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="24/7 paper trading worker")
    parser.add_argument("--settings", default=None, help="JSON settings override path")
    parser.add_argument("--once", action="store_true", help="run one scheduler cycle and exit")
    parser.add_argument("--init-db", action="store_true", help="initialize database only and exit")
    args = parser.parse_args(argv)

    if args.init_db:
        context = build_task_context(args.settings)
        context.repository.log_event("INFO", "scheduler", "init_db", "database initialized", {})
        return 0
    if args.once:
        run_once(args.settings)
        return 0
    run_loop(args.settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
