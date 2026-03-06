from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from config.settings import RuntimeSettings
from storage.repository import TradingRepository


def load_dashboard_data(settings: RuntimeSettings) -> Dict[str, Any]:
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    summary = repository.dashboard_counts()
    equity_curve = repository.load_account_snapshots(limit=500)
    return {
        "summary": summary,
        "prediction_report": repository.prediction_report(limit=200),
        "open_positions": repository.open_positions(),
        "open_orders": repository.open_orders(),
        "candidate_scans": repository.latest_candidates(limit=100),
        "equity_curve": equity_curve.sort_values("created_at") if not equity_curve.empty else pd.DataFrame(),
        "job_health": repository.recent_job_health(limit=50),
        "recent_errors": repository.recent_system_events(level="ERROR", limit=50),
        "recent_events": repository.recent_system_events(limit=50),
        "trade_performance": repository.trade_performance_report(),
    }
