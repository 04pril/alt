from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from config.settings import RuntimeSettings
from jobs.tasks import broker_account_sync_job, broker_market_status_job, broker_order_sync_job, broker_position_sync_job
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.repository import TradingRepository


class _StubBroker:
    def __init__(self) -> None:
        self.account_calls = 0
        self.order_calls = 0

    def sync_account(self, touch=None):
        self.account_calls += 1
        if callable(touch):
            touch("stub_account_sync", {"call": self.account_calls})
        return {"kis": {"enabled": True}, "sim": {"enabled": True}}

    def sync_orders(self, market_data_service, touch=None):
        self.order_calls += 1
        if callable(touch):
            touch("stub_sync", {"call": self.order_calls})
        return {"fills": 0, "acknowledged": 0, "rejected": 0, "cancelled": 0, "expired": 0}


class _StubPortfolioManager:
    def __init__(self) -> None:
        self.calls = 0

    def mark_to_market(self, market_data_service, touch=None):
        self.calls += 1
        if callable(touch):
            touch("mark", {"call": self.calls})


class _StubMarketDataService:
    def market_phase(self, asset_type: str) -> str:
        return "pre_close"


class ExecutionAssuranceJobsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.context = SimpleNamespace(
            settings=self.settings,
            repository=self.repo,
            paper_broker=_StubBroker(),
            portfolio_manager=_StubPortfolioManager(),
            market_data_service=_StubMarketDataService(),
            job_touch=lambda stage=None, details=None: None,
            touch_runtime=lambda stage=None, details=None: None,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_broker_account_sync_job_is_repeatable(self) -> None:
        first = broker_account_sync_job(self.context)
        second = broker_account_sync_job(self.context)
        self.assertIn("kis", first)
        self.assertIn("kis", second)
        self.assertEqual(self.context.paper_broker.account_calls, 2)

    def test_broker_order_sync_job_is_repeatable(self) -> None:
        first = broker_order_sync_job(self.context)
        second = broker_order_sync_job(self.context)
        self.assertEqual(first["fills"], 0)
        self.assertEqual(second["fills"], 0)
        self.assertEqual(self.context.paper_broker.order_calls, 2)

    def test_broker_position_sync_job_is_repeatable(self) -> None:
        first = broker_position_sync_job(self.context)
        second = broker_position_sync_job(self.context)
        self.assertEqual(first["open_positions"], 0)
        self.assertEqual(second["open_positions"], 0)
        self.assertEqual(self.context.portfolio_manager.calls, 2)

    def test_broker_market_status_job_records_flags(self) -> None:
        result = broker_market_status_job(self.context)
        self.assertTrue(result)
        for asset_type in self.settings.asset_schedules.keys():
            self.assertEqual(self.repo.get_control_flag(f"market_phase:{asset_type}", ""), "pre_close")

    def test_broker_market_status_job_records_account_scoped_events(self) -> None:
        broker_market_status_job(self.context)
        events = self.repo.system_events_by_date(str(pd.Timestamp.utcnow().date()), limit=50)
        scoped = events.loc[events["event_type"].astype(str) == "broker_market_status"].copy()
        self.assertEqual(
            set(scoped["account_id"].astype(str)),
            {ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO},
        )


if __name__ == "__main__":
    unittest.main()
