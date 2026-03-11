from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from jobs.scheduler import _run_guarded
from jobs.tasks import (
    TaskContext,
    broker_account_sync_job,
    broker_market_status_job,
    broker_order_sync_job,
    broker_position_sync_job,
    entry_decision_job,
)
from kr_strategy import default_kr_strategy, get_kr_strategy, strategy_schedule
from kis_paper import KISPaperSnapshot
from monitoring.dashboard_hooks import build_asset_overview, load_dashboard_data
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from services.broker_router import BrokerRouter
from services.kis_paper_broker import KISPaperBroker
from services.paper_broker import PaperBroker
from services.portfolio_manager import PortfolioManager
from services.risk_engine import RiskEngine
from storage.models import AccountSnapshotRecord, CandidateScanRecord, OrderRecord, PositionRecord, PredictionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class _FakeQuote:
    def __init__(self, price: float):
        self.price = float(price)


class _FakeMarketDataService:
    def __init__(self, *, market_open: bool = True, pre_close: bool = True, latest_price: float = 70_000.0):
        self.market_open = market_open
        self.pre_close = pre_close
        self.latest_price = float(latest_price)
        self.current_times: dict[str, datetime] = {}

    def current_time(self, asset_type: str) -> datetime:
        if asset_type in self.current_times:
            return self.current_times[asset_type]
        if asset_type == "한국주식":
            return datetime(2026, 3, 9, 14, 10, tzinfo=ZoneInfo("Asia/Seoul"))
        if asset_type == "미국주식":
            return datetime(2026, 3, 9, 15, 40, tzinfo=ZoneInfo("America/New_York"))
        return datetime(2026, 3, 9, 12, 0, tzinfo=ZoneInfo("UTC"))

    def is_market_open(self, asset_type: str, when=None) -> bool:
        return self.market_open

    def is_pre_close_window(self, asset_type: str, when=None) -> bool:
        return self.pre_close

    def market_phase(self, asset_type: str, when=None) -> str:
        if not self.market_open:
            return "closed"
        return "pre_close" if self.pre_close else "regular"

    def correlation_matrix(self, symbols, asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        return pd.DataFrame()

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str) -> _FakeQuote:
        return _FakeQuote(self.latest_price)


class _FakeKISClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_paper=True, hts_id="demo-user")
        self.cash = 30_000_000.0
        self.current_price = 70_000.0
        self.quote_available = True
        self.market_halted = False
        self.order_no = 0
        self.place_calls = 0
        self.place_orders: list[dict[str, object]] = []
        self.holdings = pd.DataFrame(columns=["symbol_code", "quantity", "avg_price", "current_price", "unrealized_pnl", "market_value"])
        self.buying_power = {"cash_buy_qty": 1_000, "max_buy_qty": 1_000, "cash_buy_amount": 100_000_000.0}
        self.sellable = {"sellable_qty": 10}
        self.daily_rows = pd.DataFrame()

    def get_account_snapshot(self) -> KISPaperSnapshot:
        market_value = float(pd.to_numeric(self.holdings.get("market_value"), errors="coerce").fillna(0.0).sum()) if not self.holdings.empty else 0.0
        unrealized = float(pd.to_numeric(self.holdings.get("unrealized_pnl"), errors="coerce").fillna(0.0).sum()) if not self.holdings.empty else 0.0
        return KISPaperSnapshot(
            summary={
                "cash": self.cash,
                "stock_eval": market_value,
                "total_eval": self.cash + market_value,
                "pnl": unrealized,
                "holding_count": int(len(self.holdings)),
            },
            holdings=self.holdings.copy(),
            raw_summary={},
        )

    def get_quote(self, symbol: str):
        price = self.current_price if self.quote_available else float("nan")
        return {"symbol_code": "005930", "current_price": price, "raw": {}}

    def get_orderbook(self, symbol: str):
        price = self.current_price if self.quote_available else float("nan")
        return {"expected_price": price, "best_ask": price, "best_bid": price - 100.0 if self.quote_available else float("nan")}

    def get_overtime_price(self, symbol: str):
        price = self.current_price if self.quote_available else float("nan")
        return {
            "symbol_code": "005930",
            "current_price": price,
            "close_price": price,
            "upper_limit": price * 1.1 if self.quote_available else float("nan"),
            "lower_limit": price * 0.9 if self.quote_available else float("nan"),
            "raw": {},
        }

    def get_overtime_asking_price(self, symbol: str):
        price = self.current_price if self.quote_available else float("nan")
        return {
            "expected_price": price,
            "best_ask": price,
            "best_bid": price - 100.0 if self.quote_available else float("nan"),
        }

    def get_market_status(self, symbol: str):
        return {"is_halted": self.market_halted, "phase_code": "halted" if self.market_halted else "open"}

    def get_buying_power(self, symbol: str, *, order_price: float, order_division: str = "01", include_cma: str = "N", include_overseas: str = "N"):
        return dict(self.buying_power)

    def get_sellable_quantity(self, symbol: str):
        return dict(self.sellable)

    def get_daily_order_fills(self, **kwargs):
        return self.daily_rows.copy()

    def get_websocket_approval_key(self) -> str:
        return "approval"

    def place_cash_order(
        self,
        symbol_or_code: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        price: float | None = None,
        order_division: str | None = None,
    ):
        self.place_calls += 1
        self.order_no += 1
        self.place_orders.append(
            {
                "symbol": symbol_or_code,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "order_division": order_division,
            }
        )
        return {
            "order_no": f"ODR{self.order_no}",
            "requested_at": utc_now_iso(),
            "message": "accepted",
        }


@dataclass
class _Fixture:
    tmp: tempfile.TemporaryDirectory
    settings: RuntimeSettings
    repository: TradingRepository
    market: _FakeMarketDataService
    client: _FakeKISClient
    sim_broker: PaperBroker
    kis_broker: KISPaperBroker
    router: BrokerRouter
    risk_engine: RiskEngine
    portfolio_manager: PortfolioManager
    context: TaskContext
    kr_asset_type: str
    us_asset_type: str
    crypto_asset_type: str

    def cleanup(self) -> None:
        self.tmp.cleanup()


class MergeReadinessSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = self._build_fixture()
        self.run_counter = 0

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def _build_fixture(self) -> _Fixture:
        tmp = tempfile.TemporaryDirectory()
        settings = RuntimeSettings()
        settings.storage.db_path = f"{tmp.name}/runtime.sqlite3"
        settings.broker.fee_bps = 0.0
        settings.broker.stale_submitted_order_timeout_minutes = 1
        repository = TradingRepository(settings.storage.db_path)
        repository.initialize()
        market = _FakeMarketDataService()
        client = _FakeKISClient()
        sim_broker = PaperBroker(settings, repository)
        sim_broker.ensure_account_initialized()
        kis_broker = KISPaperBroker(settings, repository, sim_broker, client_factory=lambda: client)
        router = BrokerRouter(sim_broker=sim_broker, kis_broker=kis_broker)
        risk_engine = RiskEngine(settings, repository)
        portfolio_manager = PortfolioManager(settings, repository, router)
        kr_asset_type = next(asset_type for asset_type, schedule in settings.asset_schedules.items() if schedule.timezone == "Asia/Seoul")
        us_asset_type = next(asset_type for asset_type, schedule in settings.asset_schedules.items() if schedule.timezone == "America/New_York")
        crypto_asset_type = next(asset_type for asset_type, schedule in settings.asset_schedules.items() if schedule.timeframe == "1h")
        context = TaskContext(
            settings=settings,
            repository=repository,
            market_data_service=market,
            signal_engine=SimpleNamespace(),
            risk_engine=risk_engine,
            paper_broker=router,
            portfolio_manager=portfolio_manager,
            universe_scanner=SimpleNamespace(),
            outcome_resolver=SimpleNamespace(),
            evaluator=SimpleNamespace(),
            retrainer=SimpleNamespace(),
        )
        return _Fixture(
            tmp=tmp,
            settings=settings,
            repository=repository,
            market=market,
            client=client,
            sim_broker=sim_broker,
            kis_broker=kis_broker,
            router=router,
            risk_engine=risk_engine,
            portfolio_manager=portfolio_manager,
            context=context,
            kr_asset_type=kr_asset_type,
            us_asset_type=us_asset_type,
            crypto_asset_type=crypto_asset_type,
        )

    def _run_job(self, job_name: str, fn):
        self.run_counter += 1
        run_key = f"2026-03-09T15:{self.run_counter:02d}"
        return _run_guarded(self.fixture.context, job_name=job_name, run_key=run_key, fn=fn)

    def _kr_strategy_id(self) -> str:
        strategy = default_kr_strategy(self.fixture.settings)
        self.assertIsNotNone(strategy)
        return str(strategy.strategy_id)

    def _kr_timeframe(self) -> str:
        return str(strategy_schedule(self.fixture.settings, self._kr_strategy_id()).timeframe)

    def _insert_candidate_prediction(
        self,
        *,
        symbol: str,
        asset_type: str,
        timeframe: str,
        scan_id: str,
        prediction_id: str | None = None,
        strategy_id: str | None = None,
    ) -> None:
        now_iso = utc_now_iso()
        prediction_id = prediction_id or make_id("pred")
        strategy_version = "s1"
        target_type = "next_close_return"
        forecast_horizon_bars = 1
        validation_mode = "holdout"
        notes_payload = {"stop_level": 68_000.0, "take_level": 73_000.0}
        if asset_type == self.fixture.kr_asset_type:
            strategy = get_kr_strategy(self.fixture.settings, str(strategy_id or "")) if strategy_id else default_kr_strategy(self.fixture.settings)
            self.assertIsNotNone(strategy)
            timeframe = str(strategy_schedule(self.fixture.settings, str(strategy.strategy_id)).timeframe)
            strategy_version = str(strategy.strategy_id)
            target_type = str(strategy.primary_target)
            forecast_horizon_bars = int(strategy.forecast_horizon_bars)
            validation_mode = str(strategy.validation_mode)
            notes_payload.update(
                {
                    "strategy_family": str(strategy.strategy_family),
                    "session_mode": str(strategy.session_mode),
                    "decision_horizon_bars": int(strategy.decision_horizon_bars),
                    "primary_target": str(strategy.primary_target),
                    "secondary_target": str(strategy.secondary_target),
                    "analysis_target": str(strategy.analysis_target),
                    "experimental": bool(strategy.experimental),
                }
            )
        notes = json.dumps(notes_payload, ensure_ascii=False)
        self.fixture.repository.insert_predictions(
            [
                PredictionRecord(
                    prediction_id=prediction_id,
                    created_at=now_iso,
                    run_id=make_id("run"),
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    market_timezone="Asia/Seoul",
                    data_cutoff_at=now_iso,
                    target_at=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat(),
                    forecast_horizon_bars=forecast_horizon_bars,
                    target_type=target_type,
                    current_price=70_000.0,
                    predicted_price=71_400.0,
                    predicted_return=0.02,
                    signal="LONG",
                    score=1.0,
                    confidence=0.9,
                    threshold=0.003,
                    expected_return=0.02,
                    expected_risk=0.01,
                    position_size=0.5,
                    model_name="demo",
                    model_version="v1",
                    feature_version="f1",
                    strategy_version=strategy_version,
                    validation_mode=validation_mode,
                    feature_hash="hash",
                    status="unresolved",
                    scan_id=scan_id,
                    notes=notes,
                    execution_account_id=ACCOUNT_KIS_KR_PAPER if asset_type == self.fixture.kr_asset_type else "",
                )
            ]
        )
        self.fixture.repository.insert_candidate_scans(
            [
                CandidateScanRecord(
                    scan_id=scan_id,
                    created_at=now_iso,
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    score=1.0,
                    rank=1,
                    status="candidate",
                    reason="ok",
                    expected_return=0.02,
                    expected_risk=0.01,
                    confidence=0.9,
                    threshold=0.003,
                    volatility=0.02,
                    liquidity_score=1.0,
                    cost_bps=5.0,
                    recent_performance=0.01,
                    signal="LONG",
                    model_version="v1",
                    feature_version="f1",
                    strategy_version=strategy_version,
                    is_holding=0,
                    raw_json="{}",
                    execution_account_id=ACCOUNT_KIS_KR_PAPER if asset_type == self.fixture.kr_asset_type else "",
                )
            ]
        )

    def _event_frame(self) -> pd.DataFrame:
        return self.fixture.repository.system_events_by_date(str(pd.Timestamp.utcnow().date()), limit=500)

    def _assert_event_reason(self, *, event_type: str, reason: str, account_id: str | None = None) -> None:
        events = self._event_frame()
        self.assertFalse(events.empty)
        matched = events.loc[events["event_type"].astype(str) == event_type]
        self.assertFalse(matched.empty)
        if account_id is not None:
            matched = matched.loc[matched["account_id"].fillna("").astype(str) == str(account_id)]
            self.assertFalse(matched.empty)
        reasons = matched["details_json"].fillna("{}").map(lambda payload: json.loads(str(payload or "{}")).get("reason"))
        self.assertIn(reason, set(reasons.tolist()))

    def _latest_event_details(self, *, event_type: str, account_id: str | None = None) -> dict:
        events = self._event_frame()
        matched = events.loc[events["event_type"].astype(str) == event_type].copy()
        self.assertFalse(matched.empty)
        if account_id is not None:
            matched = matched.loc[matched["account_id"].fillna("").astype(str) == str(account_id)]
            self.assertFalse(matched.empty)
        matched = matched.sort_values("created_at", ascending=False)
        return json.loads(str(matched.iloc[0]["details_json"] or "{}"))

    def _load_dashboard_data_for_kis(self):
        with patch("monitoring.dashboard_hooks._kis_enabled_for_monitor", return_value=True):
            return load_dashboard_data(self.fixture.settings)

    def test_kr_bootstrap_smoke_prefers_kis_account_snapshot(self) -> None:
        self.fixture.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 10,
                    "avg_price": 70_000.0,
                    "current_price": 72_000.0,
                    "unrealized_pnl": 20_000.0,
                    "market_value": 720_000.0,
                }
            ]
        )

        result = self._run_job("broker_account_sync", lambda: broker_account_sync_job(self.fixture.context))
        latest_kis = self.fixture.repository.latest_account_snapshot(account_id=ACCOUNT_KIS_KR_PAPER)
        state = self.fixture.risk_engine._latest_account_state(asset_type=self.fixture.kr_asset_type, symbol="005930.KS")

        self.assertIsNotNone(result)
        self.assertTrue(result["kis"]["enabled"])
        self.assertIsNotNone(latest_kis)
        self.assertEqual(str(latest_kis["source"]), "kis_account_sync")
        self.assertEqual(float(latest_kis["cash"]), 30_000_000.0)
        self.assertEqual(float(latest_kis["equity"]), 30_720_000.0)
        self.assertEqual(state["cash"], 30_000_000.0)
        self.assertEqual(state["equity"], 30_720_000.0)
        self.assertEqual(state["drawdown_pct"], float(latest_kis["drawdown_pct"]))
        self.assertEqual(self.fixture.repository.get_control_flag("worker_heartbeat_job", ""), "broker_account_sync")

    def test_kr_allowed_execution_smoke_submit_ack_fill_and_reconcile(self) -> None:
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-kr-allowed")
        self._run_job("broker_market_status", lambda: broker_market_status_job(self.fixture.context))
        self._run_job("broker_account_sync", lambda: broker_account_sync_job(self.fixture.context))

        entered = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        orders = self.fixture.repository.open_orders(
            statuses=("submitted", "acknowledged", "pending_fill", "partially_filled", "filled"),
            account_id=ACCOUNT_KIS_KR_PAPER,
        )
        self.assertEqual(entered[self._kr_strategy_id()], 1)
        self.assertEqual(len(orders), 1)
        order_id = str(orders.iloc[0]["order_id"])
        requested_qty = int(orders.iloc[0]["requested_qty"])
        self.assertEqual(str(orders.iloc[0]["status"]), "acknowledged")

        pending = self._run_job("broker_order_sync", lambda: broker_order_sync_job(self.fixture.context))
        self.assertEqual(pending["fills"], 0)
        self.assertEqual(self.fixture.repository.get_order(order_id)["status"], "pending_fill")

        self.fixture.client.daily_rows = pd.DataFrame([{"broker_order_id": "ODR1", "tot_ccld_qty": requested_qty, "avg_prvs": 70_000.0}])
        self.fixture.client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": requested_qty,
                    "avg_price": 70_000.0,
                    "current_price": 70_000.0,
                    "unrealized_pnl": 0.0,
                    "market_value": 70_000.0 * requested_qty,
                }
            ]
        )
        filled = self._run_job("broker_order_sync", lambda: broker_order_sync_job(self.fixture.context))
        self._run_job("broker_position_sync", lambda: broker_position_sync_job(self.fixture.context))
        self._run_job("broker_account_sync", lambda: broker_account_sync_job(self.fixture.context))

        with self.fixture.repository.connect() as conn:
            fill_count = int(conn.execute("SELECT COUNT(*) AS cnt FROM fills").fetchone()["cnt"])
        latest_snapshot = self.fixture.repository.latest_account_snapshot(account_id=ACCOUNT_KIS_KR_PAPER)
        dashboard_data = self._load_dashboard_data_for_kis()

        self.assertGreaterEqual(filled["fills"], 1)
        self.assertEqual(fill_count, 1)
        self.assertEqual(self.fixture.repository.get_order(order_id)["status"], "filled")
        self.assertFalse(self.fixture.repository.open_positions(account_id=ACCOUNT_KIS_KR_PAPER).empty)
        self.assertIsNotNone(latest_snapshot)
        self.assertEqual(str(latest_snapshot["source"]), "kis_account_sync")
        self.assertEqual(str(self.fixture.repository.get_order(order_id)["account_id"]), ACCOUNT_KIS_KR_PAPER)
        self.assertGreaterEqual(dashboard_data["execution_summary"]["today_submit_requested_count"], 1)
        self.assertGreaterEqual(dashboard_data["execution_summary"]["today_submitted_count"], 1)
        self.assertGreaterEqual(dashboard_data["execution_summary"]["today_acknowledged_count"], 1)
        self.assertGreaterEqual(dashboard_data["execution_summary"]["today_filled_count"], 1)

    def test_crypto_candidate_event_includes_expected_return_metrics(self) -> None:
        self._insert_candidate_prediction(
            symbol="BTC-USD",
            asset_type=self.fixture.crypto_asset_type,
            timeframe="1h",
            scan_id="scan-crypto-metrics",
        )

        entry_decision_job(self.fixture.context, [self.fixture.crypto_asset_type])

        details = self._latest_event_details(event_type="candidate", account_id=ACCOUNT_SIM_CRYPTO)
        self.assertAlmostEqual(float(details["expected_return"]), 0.02)
        self.assertAlmostEqual(float(details["confidence"]), 0.9)
        self.assertAlmostEqual(float(details["threshold"]), 0.003)

    def test_kr_15m_after_close_close_smoke_submit_requested_and_session_split(self) -> None:
        strategy_id = "kr_intraday_15m_v1_after_close_close"
        self.fixture.settings.kr_strategies[strategy_id].enabled = True
        self.fixture.market.market_open = False
        self.fixture.market.current_times[self.fixture.kr_asset_type] = datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul"))
        self._insert_candidate_prediction(
            symbol="005930.KS",
            asset_type=self.fixture.kr_asset_type,
            timeframe="15m",
            scan_id="scan-kr-after-close-close",
            strategy_id=strategy_id,
        )

        entered = entry_decision_job(self.fixture.context, strategy_ids=[strategy_id])
        orders = self.fixture.repository.open_orders(
            statuses=("submitted", "acknowledged", "pending_fill", "partially_filled", "filled"),
            account_id=ACCOUNT_KIS_KR_PAPER,
        )
        self.assertEqual(entered[strategy_id], 1)
        self.assertEqual(len(orders), 1)
        self.assertEqual(str(orders.iloc[0]["strategy_version"]), strategy_id)
        raw_payload = json.loads(str(orders.iloc[0]["raw_json"] or "{}"))
        self.assertEqual(str(orders.iloc[0]["order_type"]), "after_close_close")
        self.assertEqual(str(raw_payload.get("order_division") or ""), "06")
        self.assertEqual(str(raw_payload.get("session_mode") or ""), "after_close_close_price")
        self.assertEqual(self.fixture.client.place_orders[-1]["order_division"], "06")
        self.assertEqual(self.fixture.client.place_orders[-1]["order_type"], "after_close_close")

    def test_kr_15m_after_close_single_smoke_submit_requested_on_auction_boundary(self) -> None:
        strategy_id = "kr_intraday_15m_v1_after_close_single"
        self.fixture.settings.kr_strategies[strategy_id].enabled = True
        self.fixture.market.market_open = False
        self.fixture.market.current_times[self.fixture.kr_asset_type] = datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul"))
        self._insert_candidate_prediction(
            symbol="000660.KS",
            asset_type=self.fixture.kr_asset_type,
            timeframe="15m",
            scan_id="scan-kr-after-close-single",
            strategy_id=strategy_id,
        )

        entered = entry_decision_job(self.fixture.context, strategy_ids=[strategy_id])
        orders = self.fixture.repository.open_orders(
            statuses=("submitted", "acknowledged", "pending_fill", "partially_filled", "filled"),
            account_id=ACCOUNT_KIS_KR_PAPER,
        )
        matched = orders.loc[orders["strategy_version"].astype(str) == strategy_id]
        self.assertEqual(entered[strategy_id], 1)
        self.assertEqual(len(matched), 1)
        raw_payload = json.loads(str(matched.iloc[0]["raw_json"] or "{}"))
        self.assertEqual(str(matched.iloc[0]["order_type"]), "after_close_single")
        self.assertEqual(str(raw_payload.get("order_division") or ""), "07")
        self.assertEqual(str(raw_payload.get("session_mode") or ""), "after_close_single_price")
        self.assertEqual(self.fixture.client.place_orders[-1]["order_division"], "07")
        self.assertEqual(self.fixture.client.place_orders[-1]["order_type"], "after_close_single")

    def test_kr_noop_and_rejection_reasons_are_explicit(self) -> None:
        self.fixture.market.market_open = False
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-market-closed")
        entered_closed = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_closed[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="noop", reason="market_closed", account_id=ACCOUNT_KIS_KR_PAPER)

        self.fixture.market.market_open = True
        self.fixture.market.current_times[self.fixture.kr_asset_type] = datetime(2026, 3, 9, 15, 5, tzinfo=ZoneInfo("Asia/Seoul"))
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-not-preclose")
        entered_preclose = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_preclose[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="noop", reason="outside_intraday_entry_window", account_id=ACCOUNT_KIS_KR_PAPER)

        self.fixture.market.current_times[self.fixture.kr_asset_type] = datetime(2026, 3, 9, 14, 10, tzinfo=ZoneInfo("Asia/Seoul"))
        self.fixture.client.buying_power = {"cash_buy_qty": 0, "max_buy_qty": 0, "cash_buy_amount": 0.0}
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-buying-power")
        entered_power = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_power[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="entry_rejected", reason="insufficient_buying_power")

        self.fixture.client.buying_power = {"cash_buy_qty": 1_000, "max_buy_qty": 1_000, "cash_buy_amount": 100_000_000.0}
        self.fixture.repository.insert_order(
            OrderRecord(
                order_id="ord_pending",
                created_at=utc_now_iso(),
                updated_at=utc_now_iso(),
                prediction_id="pred-old",
                scan_id="scan-old",
                symbol="005930.KS",
                asset_type=self.fixture.kr_asset_type,
                timeframe=self._kr_timeframe(),
                side="buy",
                order_type="market",
                requested_qty=1,
                filled_qty=0,
                remaining_qty=1,
                requested_price=70_000.0,
                limit_price=0.0,
                status="submitted",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version=self._kr_strategy_id(),
                reason="entry",
                raw_json='{"broker":"kis_mock"}',
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-duplicate")
        entered_dup = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_dup[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="entry_rejected", reason="kr_strategy_conflict_pending")

        self.fixture.repository.upsert_position(
            PositionRecord(
                position_id="pos_cooldown",
                created_at=utc_now_iso(),
                updated_at=utc_now_iso(),
                closed_at=utc_now_iso(),
                prediction_id="pred-old",
                symbol="000660.KS",
                asset_type=self.fixture.kr_asset_type,
                timeframe=self._kr_timeframe(),
                side="LONG",
                status="closed",
                quantity=0,
                entry_price=70_000.0,
                mark_price=70_000.0,
                stop_loss=68_000.0,
                take_profit=73_000.0,
                trailing_stop=69_000.0,
                highest_price=70_000.0,
                lowest_price=70_000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.01,
                exposure_value=0.0,
                max_holding_until=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat(),
                strategy_version=self._kr_strategy_id(),
                cooldown_until=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat(),
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )
        self._insert_candidate_prediction(symbol="000660.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-cooldown")
        entered_cooldown = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_cooldown[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="entry_rejected", reason="cooldown_active")

        self.fixture.client.quote_available = False
        self._insert_candidate_prediction(symbol="068270.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-no-quote")
        entered_no_quote = entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self.assertEqual(entered_no_quote[self._kr_strategy_id()], 0)
        self._assert_event_reason(event_type="entry_rejected", reason="no_quote")

    def test_us_crypto_isolation_smoke(self) -> None:
        self.fixture.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_us",
                created_at="2099-03-09T05:00:00Z",
                cash=12_000_000.0,
                equity=12_500_000.0,
                gross_exposure=500_000.0,
                net_exposure=500_000.0,
                realized_pnl=0.0,
                unrealized_pnl=50_000.0,
                daily_pnl=0.0,
                drawdown_pct=-1.0,
                open_positions=1,
                open_orders=0,
                paused=0,
                source="paper_broker",
                raw_json="{}",
                account_id=ACCOUNT_SIM_US_EQUITY,
            )
        )
        self.fixture.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_crypto",
                created_at="2099-03-09T05:00:30Z",
                cash=8_000_000.0,
                equity=8_300_000.0,
                gross_exposure=300_000.0,
                net_exposure=300_000.0,
                realized_pnl=0.0,
                unrealized_pnl=30_000.0,
                daily_pnl=0.0,
                drawdown_pct=-1.5,
                open_positions=1,
                open_orders=0,
                paused=0,
                source="paper_broker",
                raw_json="{}",
                account_id=ACCOUNT_SIM_CRYPTO,
            )
        )
        self.fixture.repository.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_kis",
                created_at="2099-03-09T05:01:00Z",
                cash=30_000_000.0,
                equity=31_000_000.0,
                gross_exposure=1_000_000.0,
                net_exposure=1_000_000.0,
                realized_pnl=0.0,
                unrealized_pnl=100_000.0,
                daily_pnl=0.0,
                drawdown_pct=-0.3,
                open_positions=1,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
                raw_json="{}",
                account_id=ACCOUNT_KIS_KR_PAPER,
            )
        )

        us_state = self.fixture.risk_engine._latest_account_state(asset_type=self.fixture.us_asset_type, symbol="AAPL")
        crypto_state = self.fixture.risk_engine._latest_account_state(asset_type=self.fixture.crypto_asset_type, symbol="BTC-USD")

        self.assertEqual(self.fixture.router.broker_mode_for_asset(self.fixture.kr_asset_type, symbol="005930.KS"), "kis_mock")
        self.assertEqual(self.fixture.router.broker_mode_for_asset(self.fixture.us_asset_type, symbol="AAPL"), "sim")
        self.assertEqual(self.fixture.router.broker_mode_for_asset(self.fixture.crypto_asset_type, symbol="BTC-USD"), "sim")
        self.assertEqual(self.fixture.router.broker_mode_for_asset(self.fixture.us_asset_type, symbol="005930.KS"), "sim")
        self.assertEqual(us_state["cash"], 12_000_000.0)
        self.assertEqual(us_state["equity"], 12_500_000.0)
        self.assertEqual(us_state["account_id"], ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(crypto_state["cash"], 8_000_000.0)
        self.assertEqual(crypto_state["equity"], 8_300_000.0)
        self.assertEqual(crypto_state["account_id"], ACCOUNT_SIM_CRYPTO)

    def test_monitoring_smoke_exposes_sync_and_execution_summary(self) -> None:
        self._insert_candidate_prediction(symbol="005930.KS", asset_type=self.fixture.kr_asset_type, timeframe="1d", scan_id="scan-monitor")
        self._run_job("broker_market_status", lambda: broker_market_status_job(self.fixture.context))
        self._run_job("broker_account_sync", lambda: broker_account_sync_job(self.fixture.context))
        entry_decision_job(self.fixture.context, [self.fixture.kr_asset_type])
        self._run_job("broker_order_sync", lambda: broker_order_sync_job(self.fixture.context))
        self._run_job("broker_position_sync", lambda: broker_position_sync_job(self.fixture.context))

        data = self._load_dashboard_data_for_kis()
        overview = build_asset_overview(self.fixture.settings, kis_enabled=True)
        broker_modes = dict(zip(overview["자산유형"], overview["실행브로커"]))

        self.assertIn("broker_sync_status", data)
        self.assertIn("execution_summary", data)
        self.assertIn("kis_runtime", data)
        self.assertIn("runtime_profile", data)
        self.assertIn("accounts_overview", data)
        self.assertIn("total_portfolio_overview", data)
        self.assertIn("kr_strategy_overview", data)
        self.assertIn("kr_strategy_recent_events", data)
        self.assertFalse(data["broker_sync_status"].empty)
        self.assertTrue(data["broker_sync_errors"].empty)
        self.assertEqual(set(data["broker_sync_status"]["job_name"]), {"broker_account_sync", "broker_order_sync", "broker_position_sync", "broker_market_status"})
        self.assertEqual(broker_modes[self.fixture.kr_asset_type], "kis_mock")
        self.assertEqual(broker_modes[self.fixture.us_asset_type], "sim")
        self.assertEqual(broker_modes[self.fixture.crypto_asset_type], "sim")
        self.assertIn(ACCOUNT_KIS_KR_PAPER, data["accounts_overview"])
        self.assertIn(ACCOUNT_SIM_US_EQUITY, data["accounts_overview"])
        self.assertIn(ACCOUNT_SIM_CRYPTO, data["accounts_overview"])
        self.assertIn("warning", data["total_portfolio_overview"])
        self.assertTrue({"last_broker_account_sync", "last_broker_order_sync", "last_websocket_execution_event", "pending_submitted_orders", "broker_rejects_today"} <= set(data["kis_runtime"].keys()))
        self.assertGreaterEqual(data["execution_summary"]["today_candidate_count"], 1)
        self.assertIn("kr_intraday_1h_v1", set(data["kr_strategy_overview"]["strategy_id"].astype(str)))
        self.assertIn("session_mode", set(data["kr_strategy_overview"].columns))

    def test_account_scoped_sync_events_exist_for_all_execution_accounts(self) -> None:
        self._run_job("broker_market_status", lambda: broker_market_status_job(self.fixture.context))
        self._run_job("broker_account_sync", lambda: broker_account_sync_job(self.fixture.context))
        self._run_job("broker_position_sync", lambda: broker_position_sync_job(self.fixture.context))
        self._run_job("broker_order_sync", lambda: broker_order_sync_job(self.fixture.context))

        for account_id in (ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO):
            self.assertIsNotNone(self.fixture.repository.latest_system_event("broker_market_status", account_id=account_id))
            self.assertIsNotNone(self.fixture.repository.latest_system_event("broker_account_sync", account_id=account_id))
            self.assertIsNotNone(self.fixture.repository.latest_system_event("broker_position_sync", account_id=account_id))
            self.assertIsNotNone(self.fixture.repository.latest_system_event("broker_order_sync", account_id=account_id))


if __name__ == "__main__":
    unittest.main()
