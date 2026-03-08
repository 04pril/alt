from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from config.settings import RuntimeSettings
from jobs.tasks import broker_account_sync_job, broker_order_sync_job, broker_position_sync_job
from services.broker_router import BrokerRouter
from services.market_data_service import MarketQuote
from services.paper_broker import PaperBroker
from services.portfolio_manager import PortfolioManager
from services.signal_engine import SignalDecision
from storage.models import PositionRecord
from storage.repository import TradingRepository


class _StubMarketDataService:
    def is_market_open(self, asset_type: str) -> bool:
        return True

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str) -> MarketQuote:
        return MarketQuote(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            price=101.0,
            high=102.0,
            low=99.0,
            open=100.0,
            volume=1000.0,
            timestamp=pd.Timestamp("2026-03-09T09:00:00Z"),
        )


class BrokerSyncJobsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.settings.broker.fee_bps = 0.0
        self.settings.broker.base_slippage_bps = 0.0
        self.repo = TradingRepository(self.settings.storage.db_path)
        self.repo.initialize()
        self.sim_broker = PaperBroker(self.settings, self.repo)
        self.router = BrokerRouter(self.sim_broker)
        self.router.ensure_account_initialized()
        self.market_data = _StubMarketDataService()
        self.portfolio_manager = PortfolioManager(self.settings, self.repo, self.router)
        self.context = SimpleNamespace(
            repository=self.repo,
            paper_broker=self.router,
            market_data_service=self.market_data,
            portfolio_manager=self.portfolio_manager,
            touch_runtime=lambda *_args, **_kwargs: None,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _signal(self) -> SignalDecision:
        return SignalDecision(
            symbol="BTC-USD",
            asset_type="코인",
            timeframe="1h",
            prediction_id="pred1",
            scan_id="scan1",
            score=1.0,
            signal="LONG",
            expected_return=0.02,
            expected_risk=0.01,
            confidence=0.9,
            threshold=0.003,
            position_size=0.5,
            current_price=100.0,
            predicted_price=102.0,
            predicted_return=0.02,
            stop_level=98.0,
            take_level=105.0,
            model_version="v1",
            feature_version="f1",
            strategy_version="s1",
            validation_mode="holdout",
            result=None,
        )

    def test_broker_account_sync_job_is_repeatable(self) -> None:
        first = broker_account_sync_job(self.context)
        second = broker_account_sync_job(self.context)

        self.assertEqual(first["sim_snapshot_written"], 1)
        self.assertEqual(second["sim_snapshot_written"], 1)
        self.assertIsNotNone(self.repo.latest_account_snapshot())

    def test_broker_order_sync_job_is_idempotent_after_fill(self) -> None:
        self.sim_broker.submit_entry_order(self._signal(), quantity=1)

        first = broker_order_sync_job(self.context)
        second = broker_order_sync_job(self.context)

        self.assertEqual(first["fills"], 1)
        self.assertEqual(second["fills"], 0)
        with self.repo.connect() as conn:
            fill_count = conn.execute("SELECT COUNT(*) AS cnt FROM fills").fetchone()["cnt"]
        self.assertEqual(int(fill_count), 1)

    def test_broker_position_sync_job_updates_in_place(self) -> None:
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos1",
                created_at="2026-03-09T09:00:00Z",
                updated_at="2026-03-09T09:00:00Z",
                closed_at=None,
                prediction_id="pred1",
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="LONG",
                status="open",
                quantity=1,
                entry_price=100.0,
                mark_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                trailing_stop=98.0,
                highest_price=100.0,
                lowest_price=100.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.02,
                exposure_value=100.0,
                max_holding_until="2026-03-10T00:00:00Z",
                strategy_version="s1",
            )
        )

        first = broker_position_sync_job(self.context)
        second = broker_position_sync_job(self.context)

        latest = self.repo.latest_position_by_symbol("BTC-USD", "1h")
        self.assertEqual(first["open_positions"], 1)
        self.assertEqual(second["open_positions"], 1)
        self.assertEqual(len(self.repo.open_positions()), 1)
        self.assertAlmostEqual(float(latest.iloc[0]["mark_price"]), 101.0)


if __name__ == "__main__":
    unittest.main()
