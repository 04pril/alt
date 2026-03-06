from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
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
    lease = context.repository.begin_job_run(
        job_name=job_name,
        run_key=run_key,
        scheduled_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        lock_owner=context.settings.scheduler.lock_owner,
        lease_timeout_seconds=context.settings.scheduler.job_lease_timeout_seconds,
        max_retry_count=context.settings.scheduler.max_retry_count,
    )
    if lease is None:
        return None
    job_run_id = str(lease["job_run_id"])
    attempt = int(lease.get("retry_count", 0))
    while True:
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover - exercised in runtime
            if attempt >= context.settings.scheduler.max_retry_count:
                context.repository.finish_job_run(job_run_id, status="failed", error_message=str(exc), metrics={})
                context.repository.log_event("ERROR", "scheduler", "job_failed", f"{job_name} failed", {"error": str(exc), "attempt": attempt})
                return None
            attempt += 1
            context.repository.mark_job_retry(job_run_id, retry_count=attempt, error_message=str(exc))
            context.repository.log_event(
                "WARNING",
                "scheduler",
                "job_retry",
                f"{job_name} retry scheduled",
                {"error": str(exc), "attempt": attempt, "run_key": run_key},
            )
            time.sleep(max(context.settings.scheduler.retry_backoff_seconds, 1) * attempt)
            continue
        context.repository.finish_job_run(job_run_id, status="completed", metrics=result if isinstance(result, dict) else {})
        return result


def run_once(settings_path: str | None = None) -> None:
    context = build_task_context(settings_path)
    settings = context.settings
    context.repository.set_control_flag("worker_heartbeat_at", utc_now_iso(), "scheduler loop heartbeat")
    if context.repository.get_control_flag_bool("worker_paused", False):
        context.repository.log_event("INFO", "scheduler", "worker_paused", "worker loop skipped due to worker_paused", {})
        return
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
        run_key=_bucket_key(datetime.now(timezone.utc), 15),
        fn=lambda: exit_management_job(context),
    )
    _run_guarded(
        context,
        job_name="outcome_resolution",
        run_key=_bucket_key(datetime.now(timezone.utc), 60),
        fn=lambda: outcome_resolution_job(context),
    )
    _run_guarded(
        context,
        job_name="daily_report",
        run_key=str(datetime.now(timezone.utc).date()),
        fn=lambda: daily_report_job(context),
    )
    _run_guarded(
        context,
        job_name="retrain_check",
        run_key=str(datetime.now(timezone.utc).date()),
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
