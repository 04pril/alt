from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import write_example_settings
from jobs.tasks import build_task_context


def main() -> int:
    write_example_settings()
    context = build_task_context()
    context.repository.initialize_runtime_flags()
    context.repository.set_entry_paused(False, "initialized")
    context.repository.set_worker_paused(False, "initialized")
    context.repository.set_exit_only_mode(False, "initialized")
    context.repository.log_event("INFO", "init_runtime", "initialized", "runtime storage initialized", {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
