from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.settings import load_settings
from jobs.tasks import (
    broker_account_sync_job,
    broker_market_status_job,
    broker_order_sync_job,
    broker_position_sync_job,
    build_task_context,
    daily_report_job,
    entry_decision_job,
    exit_management_job,
    outcome_resolution_job,
    retrain_check_job,
    scan_job,
)
from kr_strategy import active_strategy_ids, strategy_schedule
from services.kis_quote_stream import KISKRQuoteStream
from storage.repository import TradingRepository
from storage.repository import utc_now_iso


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bucket_key(dt: datetime, minutes: int) -> str:
    bucket = (dt.minute // max(minutes, 1)) * max(minutes, 1)
    rounded = dt.replace(minute=bucket, second=0, microsecond=0)
    return rounded.isoformat(timespec="minutes")


def _ordered_asset_schedule_items(settings) -> list[tuple[str, object]]:
    priority = {"한국주식": 0, "미국주식": 1, "코인": 2}
    items = list(settings.asset_schedules.items())
    return sorted(items, key=lambda item: (priority.get(str(item[0]), 99), str(item[0])))


def _run_guarded(context, job_name: str, run_key: str, fn):
    lease = context.repository.begin_job_run(
        job_name=job_name,
        run_key=run_key,
        scheduled_at=_utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        lock_owner=context.settings.scheduler.lock_owner,
        retry_backoff_seconds=context.settings.scheduler.retry_backoff_seconds,
        max_retry_count=context.settings.scheduler.max_retry_count,
        lease_seconds=context.settings.scheduler.job_lease_seconds,
    )
    if not lease.acquired:
        return None

    def touch(stage=None, details=None):
        context.repository.refresh_job_lease(lease.job_run_id, context.settings.scheduler.job_lease_seconds)
        context.repository.set_control_flag("worker_heartbeat_at", utc_now_iso(), f"{job_name}:{stage or 'tick'}")
        context.repository.set_control_flag("worker_heartbeat_job", job_name, "active job")

    previous_touch = getattr(context, "job_touch", None)
    context.job_touch = touch
    try:
        touch("job_start", {"job_name": job_name})
        result = fn()
    except Exception as exc:  # pragma: no cover
        context.repository.finish_job_run(
            lease.job_run_id,
            status="failed",
            error_message=str(exc),
            metrics={},
            retry_backoff_seconds=context.settings.scheduler.retry_backoff_seconds,
            max_retry_count=context.settings.scheduler.max_retry_count,
        )
        context.repository.log_event("ERROR", "scheduler", "job_failed", f"{job_name} failed", {"error": str(exc)})
        return None
    finally:
        context.job_touch = previous_touch or (lambda _stage=None, _details=None: None)

    context.repository.finish_job_run(
        lease.job_run_id,
        status="completed",
        metrics=result if isinstance(result, dict) else {},
        retry_backoff_seconds=context.settings.scheduler.retry_backoff_seconds,
        max_retry_count=context.settings.scheduler.max_retry_count,
    )
    return result


def run_once(settings_path: str | None = None) -> None:
    context = build_task_context(settings_path)
    expire_stale_job_runs = getattr(context.repository, "expire_stale_job_runs", None)
    if callable(expire_stale_job_runs):
        expire_stale_job_runs()
    settings = context.settings
    context.repository.set_control_flag("worker_heartbeat_at", utc_now_iso(), "scheduler loop heartbeat")
    _run_guarded(
        context,
        job_name="broker_market_status",
        run_key=_bucket_key(_utc_now(), settings.scheduler.broker_market_status_interval_minutes),
        fn=lambda: broker_market_status_job(context),
    )
    _run_guarded(
        context,
        job_name="broker_account_sync",
        run_key=_bucket_key(_utc_now(), settings.scheduler.broker_account_sync_interval_minutes),
        fn=lambda: broker_account_sync_job(context),
    )
    _run_guarded(
        context,
        job_name="exit_management",
        run_key=_bucket_key(_utc_now(), settings.scheduler.exit_management_interval_minutes),
        fn=lambda: exit_management_job(context),
    )
    for asset_type, schedule in _ordered_asset_schedule_items(settings):
        asset_strategy_ids = active_strategy_ids(settings, asset_schedule_key=asset_type)
        if asset_strategy_ids:
            for strategy_id in asset_strategy_ids:
                strategy_cfg = strategy_schedule(settings, strategy_id)
                now = datetime.now(ZoneInfo(strategy_cfg.timezone))
                _run_guarded(
                    context,
                    job_name=f"scan:{strategy_id}",
                    run_key=_bucket_key(now, strategy_cfg.scan_interval_minutes),
                    fn=lambda strategy_id=strategy_id: scan_job(context, strategy_ids=[strategy_id]),
                )
                _run_guarded(
                    context,
                    job_name=f"entry:{strategy_id}",
                    run_key=_bucket_key(now, strategy_cfg.entry_interval_minutes),
                    fn=lambda strategy_id=strategy_id: entry_decision_job(context, strategy_ids=[strategy_id]),
                )
            continue
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
        job_name="broker_position_sync",
        run_key=_bucket_key(_utc_now(), settings.scheduler.broker_position_sync_interval_minutes),
        fn=lambda: broker_position_sync_job(context),
    )
    _run_guarded(
        context,
        job_name="broker_order_sync",
        run_key=_bucket_key(_utc_now(), settings.scheduler.broker_order_sync_interval_minutes),
        fn=lambda: broker_order_sync_job(context),
    )
    _run_guarded(
        context,
        job_name="outcome_resolution",
        run_key=_bucket_key(_utc_now(), settings.scheduler.outcome_resolution_interval_minutes),
        fn=lambda: outcome_resolution_job(context),
    )
    _run_guarded(
        context,
        job_name="daily_report",
        run_key=str(_utc_now().date()),
        fn=lambda: daily_report_job(context),
    )
    _run_guarded(
        context,
        job_name="retrain_check",
        run_key=str(_utc_now().date()),
        fn=lambda: retrain_check_job(context),
    )


def run_loop(settings_path: str | None = None) -> None:
    settings = load_settings(settings_path)
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    quote_stream = KISKRQuoteStream(settings, repository)
    quote_stream.refresh_symbols()
    quote_stream.start()
    while True:
        quote_stream.refresh_symbols()
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
