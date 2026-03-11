from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config.settings import DEFAULT_SETTINGS_PATH


def restart_background_worker() -> Tuple[bool, str]:
    workspace = Path(__file__).resolve().parent
    python_executable = Path(sys.executable)

    if sys.platform.startswith("win"):
        command = f"""
$ErrorActionPreference = 'Stop'
$python = '{str(python_executable).replace("'", "''")}'
$cwd = '{str(workspace).replace("'", "''")}'
Get-CimInstance Win32_Process |
  Where-Object {{
    $_.CommandLine -and $_.CommandLine -like '*-m jobs.scheduler*'
  }} |
  ForEach-Object {{
    try {{
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
    }} catch {{
    }}
  }}
Start-Sleep -Milliseconds 800
Start-Process -FilePath $python -ArgumentList '-m','jobs.scheduler' -WorkingDirectory $cwd -WindowStyle Hidden | Out-Null
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        if completed.returncode == 0:
            return True, "worker 재시작 요청을 보냈습니다."
        stderr = (completed.stderr or completed.stdout or "").strip()
        return False, stderr or "worker 재시작에 실패했습니다."

    return False, "worker 재시작 버튼은 현재 Windows 환경에서만 지원합니다."


def stop_background_worker() -> Tuple[bool, str]:
    workspace = Path(__file__).resolve().parent

    if sys.platform.startswith("win"):
        command = """
$ErrorActionPreference = 'Stop'
$killed = 0
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -and $_.CommandLine -like '*-m jobs.scheduler*'
  } |
  ForEach-Object {
    try {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
      $script:killed += 1
    } catch {
    }
  }
Write-Output $killed
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        if completed.returncode == 0:
            killed_text = (completed.stdout or "").strip()
            killed = int(killed_text) if killed_text.isdigit() else 0
            if killed > 0:
                return True, f"worker {killed}개를 중지했습니다."
            return True, "중지할 worker가 없었습니다."
        stderr = (completed.stderr or completed.stdout or "").strip()
        return False, stderr or "worker 중지에 실패했습니다."

    return False, "worker 중지 버튼은 현재 Windows 환경에서만 지원합니다."


def run_manual_runtime_job(job_name: str) -> Tuple[bool, str]:
    from jobs.scheduler import _run_guarded
    from jobs.tasks import (
        broker_account_sync_job,
        broker_market_status_job,
        broker_order_sync_job,
        broker_position_sync_job,
        build_task_context,
    )

    job_map = {
        "broker_market_status": broker_market_status_job,
        "broker_position_sync": broker_position_sync_job,
        "broker_order_sync": broker_order_sync_job,
        "broker_account_sync": broker_account_sync_job,
    }
    fn = job_map.get(job_name)
    if fn is None:
        return False, f"지원하지 않는 job 입니다: {job_name}"
    context = build_task_context()
    run_key = f"manual:{pd.Timestamp.utcnow().isoformat()}"
    result = _run_guarded(context, job_name=job_name, run_key=run_key, fn=lambda: fn(context))
    if result is None:
        return False, f"{job_name} 실행에 실패했습니다."
    return True, f"{job_name} 실행 완료"


def run_manual_scan_job(asset_types: List[str] | None = None) -> Tuple[bool, str]:
    from jobs.scheduler import _run_guarded
    from jobs.tasks import build_task_context, scan_job
    from kr_strategy import active_strategy_ids

    context = build_task_context()
    targets = list(asset_types or context.settings.asset_schedules.keys())
    if not targets:
        return False, "실행할 스캔 자산이 없습니다."

    scanned_counts: Dict[str, int] = {}
    failures: List[str] = []
    for asset_type in targets:
        strategy_ids = active_strategy_ids(context.settings, asset_schedule_key=asset_type)
        if strategy_ids:
            for strategy_id in strategy_ids:
                run_key = f"manual-scan:{strategy_id}:{pd.Timestamp.utcnow().isoformat()}"
                result = _run_guarded(
                    context,
                    job_name=f"scan:{strategy_id}",
                    run_key=run_key,
                    fn=lambda strategy_id=strategy_id: scan_job(context, strategy_ids=[strategy_id]),
                )
                if result is None:
                    failures.append(strategy_id)
                    continue
                count = 0
                if isinstance(result, dict):
                    count = int(result.get(strategy_id, result.get(asset_type, 0)) or 0)
                scanned_counts[str(strategy_id)] = count
            continue

        run_key = f"manual-scan:{asset_type}:{pd.Timestamp.utcnow().isoformat()}"
        result = _run_guarded(
            context,
            job_name=f"scan:{asset_type}",
            run_key=run_key,
            fn=lambda asset_type=asset_type: scan_job(context, asset_types=[asset_type]),
        )
        if result is None:
            failures.append(asset_type)
            continue
        count = 0
        if isinstance(result, dict):
            count = int(result.get(asset_type, 0) or 0)
        scanned_counts[str(asset_type)] = count

    if failures and not scanned_counts:
        return False, " / ".join(failures) + " 스캔 실행 실패"
    summary = ", ".join(f"{key} {value}건" for key, value in scanned_counts.items()) if scanned_counts else "0건"
    if failures:
        return True, f"{summary} · 일부 실패: {' / '.join(failures)}"
    return True, f"즉시 스캔 완료: {summary}"


def set_default_strategy(strategy_id: str, current_settings) -> Tuple[bool, str]:
    from kr_strategy import get_kr_strategy, strategy_asset_schedule_key, strategy_label

    all_ids = list((current_settings.kr_strategies or {}).keys())
    if strategy_id not in all_ids:
        return False, f"알 수 없는 전략 ID: {strategy_id}"

    strategy = get_kr_strategy(current_settings, strategy_id)
    asset_type = strategy_asset_schedule_key(strategy)
    default_key = "us_default_strategy_id" if asset_type == "미국주식" else "kr_default_strategy_id"
    asset_label = "US" if asset_type == "미국주식" else "KR"

    path = Path(DEFAULT_SETTINGS_PATH)
    raw: dict = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    raw[default_key] = strategy_id
    if not isinstance(raw.get("kr_strategies"), dict):
        raw["kr_strategies"] = {}
    for known_strategy_id, known_strategy in (current_settings.kr_strategies or {}).items():
        if strategy_asset_schedule_key(known_strategy) != asset_type:
            continue
        if not isinstance(raw["kr_strategies"].get(known_strategy_id), dict):
            raw["kr_strategies"][known_strategy_id] = {}
        raw["kr_strategies"][known_strategy_id]["enabled"] = bool(str(known_strategy_id) == str(strategy_id))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = current_settings.kr_strategies.get(strategy_id)
    label = strategy_label(selected) if selected else strategy_id
    return True, f"{asset_label} 기본 전략을 '{label}'(으)로 변경했습니다. 다음 워커 루프에서 적용됩니다."
