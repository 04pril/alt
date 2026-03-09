from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings, load_settings
from jobs.tasks import broker_account_sync_job, broker_order_sync_job, broker_position_sync_job, entry_decision_job
from monitoring.dashboard_hooks import load_dashboard_data
from services.broker_router import BrokerRouter, resolve_broker_mode
from services.kis_paper_broker import KISPaperBroker
from services.market_data_service import MarketQuote
from services.paper_broker import PaperBroker
from services.portfolio_manager import PortfolioManager
from services.risk_engine import RiskEngine
from storage.models import CandidateScanRecord, OrderRecord, PositionRecord, PredictionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


PROFILE_PATHS = {
    "baseline": Path("config/runtime_settings.baseline.json"),
    "balanced": Path("config/runtime_settings.balanced.json"),
    "active": Path("config/runtime_settings.active.json"),
}

EXECUTION_EVENT_KEYS = [
    "candidate",
    "entry_allowed",
    "entry_rejected",
    "submit_requested",
    "submitted",
    "acknowledged",
    "filled",
    "rejected",
    "cancelled",
    "noop",
]


@dataclass(frozen=True)
class SeedCandidate:
    symbol: str
    asset_type: str
    score: float
    signal_strength_pct: float
    expected_return_pct: float
    confidence: float
    expected_risk_pct: float
    current_price: float


class _FakeMarketDataService:
    def __init__(
        self,
        settings: RuntimeSettings,
        prices: Dict[str, float],
        *,
        required_preclose_minutes: Dict[str, int] | None = None,
        market_open: Dict[str, bool] | None = None,
        correlations: Dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.settings = settings
        self.prices = dict(prices)
        self.required_preclose_minutes = dict(required_preclose_minutes or {})
        self.market_open = dict(market_open or {})
        self.correlations = dict(correlations or {})

    def is_market_open(self, asset_type: str, when=None) -> bool:
        return bool(self.market_open.get(asset_type, True))

    def is_pre_close_window(self, asset_type: str, when=None) -> bool:
        if not self.is_market_open(asset_type, when):
            return False
        required = self.required_preclose_minutes.get(asset_type)
        if required is None:
            return True
        return int(self.settings.asset_schedules[asset_type].pre_close_buffer_minutes) >= int(required)

    def market_phase(self, asset_type: str, when=None) -> str:
        if not self.is_market_open(asset_type, when):
            return "closed"
        return "pre_close" if self.is_pre_close_window(asset_type, when) else "open"

    def correlation_matrix(self, symbols: Iterable[str], asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        ordered = list(dict.fromkeys(str(symbol) for symbol in symbols))
        if not ordered:
            return pd.DataFrame()
        frame = pd.DataFrame(np.eye(len(ordered)), index=ordered, columns=ordered, dtype=float)
        for (left, right), corr in self.correlations.items():
            if left in frame.index and right in frame.columns:
                frame.loc[left, right] = float(corr)
                frame.loc[right, left] = float(corr)
        return frame

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str) -> MarketQuote:
        price = float(self.prices.get(symbol, 100.0))
        return MarketQuote(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            price=price,
            high=price,
            low=price,
            open=price,
            volume=1_000_000.0,
            timestamp=pd.Timestamp.now(tz="UTC"),
        )


class _FakeKISClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_paper=True, hts_id="demo-user")
        self.order_no = 0
        self.place_calls = 0
        self.daily_rows = pd.DataFrame()
        self.holdings = pd.DataFrame(columns=["symbol_code", "quantity", "avg_price", "market_value", "unrealized_pnl"])
        self.buying_power_overrides: Dict[str, Dict[str, Any]] = {}
        self.quote_overrides: Dict[str, Dict[str, Any]] = {}
        self.cash = 30_000_000.0

    def get_account_snapshot(self):
        market_value = (
            float(pd.to_numeric(self.holdings.get("market_value"), errors="coerce").fillna(0.0).sum())
            if not self.holdings.empty
            else 0.0
        )
        unrealized = (
            float(pd.to_numeric(self.holdings.get("unrealized_pnl"), errors="coerce").fillna(0.0).sum())
            if not self.holdings.empty
            else 0.0
        )
        from kis_paper import KISPaperSnapshot

        return KISPaperSnapshot(
            summary={
                "cash": float(self.cash),
                "stock_eval": float(market_value),
                "total_eval": float(self.cash + market_value),
                "pnl": float(unrealized),
                "holding_count": int(len(self.holdings)),
            },
            holdings=self.holdings.copy(),
            raw_summary={},
        )

    def get_quote(self, symbol: str):
        if symbol in self.quote_overrides:
            return dict(self.quote_overrides[symbol])
        current_price = 70_000.0 if symbol.endswith((".KS", ".KQ")) or symbol.isdigit() else 200.0
        return {"symbol_code": symbol.split(".")[0], "current_price": current_price, "raw": {}}

    def get_orderbook(self, symbol: str):
        quote = self.get_quote(symbol)
        current_price = float(quote.get("current_price", np.nan))
        return {
            "expected_price": current_price,
            "best_ask": current_price,
            "best_bid": current_price * 0.999 if np.isfinite(current_price) else np.nan,
        }

    def get_market_status(self, symbol: str):
        return {"is_halted": False, "phase_code": "open"}

    def get_buying_power(
        self,
        symbol: str,
        *,
        order_price: float,
        order_division: str = "01",
        include_cma: str = "N",
        include_overseas: str = "N",
    ):
        override = self.buying_power_overrides.get(symbol)
        if override is not None:
            return dict(override)
        qty = int(self.cash // max(float(order_price), 1.0))
        return {"cash_buy_qty": qty, "max_buy_qty": qty, "cash_buy_amount": float(self.cash)}

    def get_sellable_quantity(self, symbol: str):
        return {"sellable_qty": 99_999}

    def get_daily_order_fills(self, **kwargs):
        return self.daily_rows.copy()

    def get_websocket_approval_key(self) -> str:
        return "approval"

    def place_cash_order(self, symbol_or_code: str, side: str, quantity: int, order_type: str = "market", price: float | None = None):
        self.place_calls += 1
        self.order_no += 1
        return {
            "order_no": f"ODR{self.order_no}",
            "requested_at": "2026-03-09T15:20:00+09:00",
            "message": "accepted",
        }

    def set_daily_fills_from_repository(self, repository: TradingRepository) -> None:
        orders = repository.recent_orders(limit=500)
        if orders.empty:
            self.daily_rows = pd.DataFrame()
            return
        rows = []
        for _, order in orders.iterrows():
            try:
                payload = json.loads(str(order.get("raw_json") or "{}"))
            except Exception:
                payload = {}
            if str(payload.get("broker") or "") != "kis_mock":
                continue
            broker_order_id = str(order.get("broker_order_id") or "")
            if not broker_order_id:
                continue
            rows.append(
                {
                    "broker_order_id": broker_order_id,
                    "tot_ccld_qty": int(order["requested_qty"]),
                    "avg_prvs": float(order["requested_price"]),
                }
            )
        self.daily_rows = pd.DataFrame(rows)


def _asset_types(settings: RuntimeSettings) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for asset_type, schedule in settings.asset_schedules.items():
        if schedule.session_mode == "market_hours" and schedule.timezone == "Asia/Seoul":
            result["kr"] = asset_type
        elif schedule.session_mode == "market_hours" and schedule.timezone == "America/New_York":
            result["us"] = asset_type
        elif schedule.session_mode == "always":
            result["crypto"] = asset_type
    return result


def _profile_settings(profile_path: str | Path) -> RuntimeSettings:
    settings = load_settings(profile_path)
    settings.broker.fee_bps = 0.0
    settings.broker.base_slippage_bps = 0.0
    settings.broker.max_volume_participation = 1.0
    settings.broker.allow_partial_fills = True
    return settings


def _seed_candidate(repository: TradingRepository, settings: RuntimeSettings, spec: SeedCandidate) -> None:
    created_at = utc_now_iso()
    scan_id = make_id("scan")
    prediction_id = make_id("pred")
    timeframe = settings.asset_schedules[spec.asset_type].timeframe
    is_candidate = float(spec.signal_strength_pct) >= float(settings.strategy.min_signal_strength_pct)
    threshold = float(spec.signal_strength_pct) / 100.0
    repository.insert_candidate_scans(
        [
            CandidateScanRecord(
                scan_id=scan_id,
                created_at=created_at,
                symbol=spec.symbol,
                asset_type=spec.asset_type,
                timeframe=timeframe,
                score=float(spec.score),
                rank=1,
                status="candidate" if is_candidate else "flat",
                reason="signal_ready" if is_candidate else "flat_signal",
                expected_return=float(spec.expected_return_pct) / 100.0,
                expected_risk=float(spec.expected_risk_pct) / 100.0,
                confidence=float(spec.confidence),
                threshold=threshold,
                volatility=0.02,
                liquidity_score=0.9,
                cost_bps=float(settings.strategy.round_trip_cost_bps),
                recent_performance=0.0,
                signal="LONG" if is_candidate else "FLAT",
                model_version="smoke-model-v1",
                feature_version="smoke-features-v1",
                strategy_version=str(settings.strategy.strategy_version),
                cooldown_until=None,
                is_holding=0,
                raw_json=json.dumps({"signal_strength_pct": float(spec.signal_strength_pct)}, ensure_ascii=False),
            )
        ]
    )
    if not is_candidate:
        return
    repository.insert_predictions(
        [
            PredictionRecord(
                prediction_id=prediction_id,
                created_at=created_at,
                run_id=make_id("run"),
                symbol=spec.symbol,
                asset_type=spec.asset_type,
                timeframe=timeframe,
                market_timezone="Asia/Seoul" if spec.asset_type == "한국주식" else ("America/New_York" if spec.asset_type == "미국주식" else "UTC"),
                data_cutoff_at=created_at,
                target_at=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat(),
                forecast_horizon_bars=1,
                target_type="next_close_return",
                current_price=float(spec.current_price),
                predicted_price=float(spec.current_price) * (1.0 + float(spec.expected_return_pct) / 100.0),
                predicted_return=float(spec.expected_return_pct) / 100.0,
                signal="LONG",
                score=float(spec.score),
                confidence=float(spec.confidence),
                threshold=threshold,
                expected_return=float(spec.expected_return_pct) / 100.0,
                expected_risk=float(spec.expected_risk_pct) / 100.0,
                position_size=float(spec.confidence),
                model_name="smoke-model",
                model_version="smoke-model-v1",
                feature_version="smoke-features-v1",
                strategy_version=str(settings.strategy.strategy_version),
                validation_mode=str(settings.strategy.validation_mode),
                feature_hash="smoke-hash",
                scan_id=scan_id,
                notes=json.dumps(
                    {
                        "timeframe": timeframe,
                        "stop_level": float(spec.current_price) * 0.98,
                        "take_level": float(spec.current_price) * 1.04,
                        "atr_14": float(spec.current_price) * (float(spec.expected_risk_pct) / 100.0),
                        "planned_signal": 1.0,
                    },
                    ensure_ascii=False,
                ),
            )
        ]
    )


def _seed_prior_entries(repository: TradingRepository, settings: RuntimeSettings, asset_type: str, symbol: str, count: int) -> None:
    timeframe = settings.asset_schedules[asset_type].timeframe
    created_at = utc_now_iso()
    for index in range(int(count)):
        repository.insert_order(
            OrderRecord(
                order_id=make_id(f"ord{index}"),
                created_at=created_at,
                updated_at=created_at,
                prediction_id=f"pred-prior-{index}",
                scan_id=f"scan-prior-{index}",
                symbol=symbol,
                asset_type=asset_type,
                timeframe=timeframe,
                side="buy",
                order_type="market",
                requested_qty=1,
                filled_qty=1,
                remaining_qty=0,
                requested_price=100.0,
                limit_price=np.nan,
                status="filled",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version=str(settings.strategy.strategy_version),
                reason="entry",
                raw_json=json.dumps({"broker": "sim"}, ensure_ascii=False),
            )
        )


def _seed_pending_entry(repository: TradingRepository, settings: RuntimeSettings, asset_type: str, symbol: str) -> None:
    timeframe = settings.asset_schedules[asset_type].timeframe
    created_at = utc_now_iso()
    repository.insert_order(
        OrderRecord(
            order_id=make_id("ord_pending"),
            created_at=created_at,
            updated_at=created_at,
            prediction_id="pred-pending",
            scan_id="scan-pending",
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            side="buy",
            order_type="market",
            requested_qty=1,
            filled_qty=0,
            remaining_qty=1,
            requested_price=70_000.0,
            limit_price=np.nan,
            status="submitted",
            fees_estimate=0.0,
            slippage_bps=0.0,
            retry_count=0,
            strategy_version=str(settings.strategy.strategy_version),
            reason="entry",
            raw_json=json.dumps({"broker": "kis_mock" if symbol.endswith((".KS", ".KQ")) or symbol.isdigit() else "sim"}, ensure_ascii=False),
        )
    )


def _seed_cooldown(repository: TradingRepository, settings: RuntimeSettings, asset_type: str, symbol: str) -> None:
    timeframe = settings.asset_schedules[asset_type].timeframe
    now_iso = utc_now_iso()
    repository.upsert_position(
        PositionRecord(
            position_id=make_id("pos"),
            created_at=now_iso,
            updated_at=now_iso,
            closed_at=now_iso,
            prediction_id="pred-cooldown",
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            side="LONG",
            status="closed",
            quantity=0,
            entry_price=100.0,
            mark_price=100.0,
            stop_loss=np.nan,
            take_profit=np.nan,
            trailing_stop=np.nan,
            highest_price=100.0,
            lowest_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            expected_risk=0.01,
            exposure_value=0.0,
            max_holding_until=now_iso,
            strategy_version=str(settings.strategy.strategy_version),
            cooldown_until=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).isoformat(),
            notes="cooldown",
        )
    )


def _seed_open_position(repository: TradingRepository, settings: RuntimeSettings, asset_type: str, symbol: str) -> None:
    timeframe = settings.asset_schedules[asset_type].timeframe
    now_iso = utc_now_iso()
    repository.upsert_position(
        PositionRecord(
            position_id=make_id("pos"),
            created_at=now_iso,
            updated_at=now_iso,
            closed_at=None,
            prediction_id="pred-open",
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            side="LONG",
            status="open",
            quantity=10,
            entry_price=100.0,
            mark_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            trailing_stop=95.0,
            highest_price=100.0,
            lowest_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            expected_risk=0.01,
            exposure_value=1_000.0,
            max_holding_until=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=5)).isoformat(),
            strategy_version=str(settings.strategy.strategy_version),
            cooldown_until=None,
            notes="open-for-correlation",
        )
    )


def _build_context(
    settings: RuntimeSettings,
    *,
    prices: Dict[str, float],
    required_preclose_minutes: Dict[str, int] | None = None,
    market_open: Dict[str, bool] | None = None,
    correlations: Dict[tuple[str, str], float] | None = None,
    bootstrap_kis_holding: bool = False,
):
    temp_dir = tempfile.TemporaryDirectory()
    settings.storage.db_path = f"{temp_dir.name}/runtime.sqlite3"
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    repository.set_control_flag("runtime_profile_name", str(settings.profile_name), "profile_smoke")
    repository.set_control_flag("runtime_profile_source", str(settings.profile_source), "profile_smoke")
    market_data_service = _FakeMarketDataService(
        settings,
        prices,
        required_preclose_minutes=required_preclose_minutes,
        market_open=market_open,
        correlations=correlations,
    )
    sim_broker = PaperBroker(settings, repository)
    sim_broker.ensure_account_initialized()
    client = _FakeKISClient()
    if bootstrap_kis_holding:
        client.holdings = pd.DataFrame(
            [
                {
                    "symbol_code": "005930",
                    "quantity": 10,
                    "avg_price": 70_000.0,
                    "market_value": 720_000.0,
                    "unrealized_pnl": 20_000.0,
                }
            ]
        )
    kis_broker = KISPaperBroker(settings, repository, sim_broker, client_factory=lambda: client)
    router = BrokerRouter(sim_broker=sim_broker, kis_broker=kis_broker)
    risk_engine = RiskEngine(settings, repository)
    portfolio_manager = PortfolioManager(settings, repository, router)
    context = SimpleNamespace(
        settings=settings,
        repository=repository,
        market_data_service=market_data_service,
        signal_engine=None,
        risk_engine=risk_engine,
        paper_broker=router,
        portfolio_manager=portfolio_manager,
        universe_scanner=None,
        outcome_resolver=None,
        evaluator=None,
        retrainer=None,
        job_touch=lambda stage=None, details=None: None,
        touch_runtime=lambda stage=None, details=None: None,
    )
    return temp_dir, context, repository, client


def _event_frame(repository: TradingRepository) -> pd.DataFrame:
    events = repository.system_events_by_date(str(pd.Timestamp.utcnow().date()), limit=5000)
    if events.empty:
        events["details"] = []
        return events
    parsed = events.copy()
    parsed["details"] = parsed["details_json"].fillna("{}").map(lambda value: json.loads(str(value or "{}")))
    return parsed


def _asset_counts(events: pd.DataFrame, repository: TradingRepository) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {key: 0 for key in EXECUTION_EVENT_KEYS})
    if events.empty:
        return {}
    orders = repository.recent_orders(limit=500)
    by_order = {str(row["order_id"]): str(row["asset_type"]) for _, row in orders.iterrows()} if not orders.empty else {}
    by_symbol = {str(row["symbol"]): str(row["asset_type"]) for _, row in orders.iterrows()} if not orders.empty else {}
    for _, row in events.iterrows():
        event_type = str(row.get("event_type") or "")
        if event_type not in EXECUTION_EVENT_KEYS:
            continue
        details = row.get("details") or {}
        asset_type = str(details.get("asset_type") or "")
        if not asset_type:
            order_id = str(details.get("order_id") or "")
            symbol = str(details.get("symbol") or "")
            asset_type = by_order.get(order_id) or by_symbol.get(symbol) or ""
        if asset_type:
            counts[asset_type][event_type] += 1
    return dict(counts)


def _reason_breakdown(events: pd.DataFrame, event_type: str) -> Dict[str, int]:
    if events.empty:
        return {}
    series = (
        events.loc[events["event_type"].astype(str) == event_type, "details"]
        .map(lambda item: str((item or {}).get("reason") or "unknown"))
        .value_counts()
    )
    return {str(index): int(value) for index, value in series.items()}


def _fills_count(repository: TradingRepository) -> int:
    with repository.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM fills").fetchone()
    return int(row["cnt"]) if row else 0


def _broker_modes(settings: RuntimeSettings) -> Dict[str, str]:
    modes: Dict[str, str] = {}
    for asset_type, schedule in settings.asset_schedules.items():
        universe = settings.universes.get(asset_type)
        representative_symbols = list((universe.watchlist if universe else []) or (universe.top_universe if universe else []))
        representative_symbol = representative_symbols[0] if representative_symbols else ""
        modes[asset_type] = resolve_broker_mode(symbol=representative_symbol, asset_type=asset_type, kis_enabled=True)
    return modes


def _summarize(repository: TradingRepository, settings: RuntimeSettings) -> Dict[str, Any]:
    events = _event_frame(repository)
    counts = {f"today_{key}_count": 0 for key in EXECUTION_EVENT_KEYS}
    if not events.empty:
        raw_counts = events["event_type"].astype(str).value_counts()
        for key in EXECUTION_EVENT_KEYS:
            counts[f"today_{key}_count"] = int(raw_counts.get(key, 0))
    counts["today_noop_reason_breakdown"] = _reason_breakdown(events, "noop")
    counts["today_entry_rejected_reason_breakdown"] = _reason_breakdown(events, "entry_rejected")
    counts["asset_counts"] = _asset_counts(events, repository)
    counts["fills"] = _fills_count(repository)
    counts["open_positions"] = int(len(repository.open_positions()))
    counts["trade_performance"] = repository.trade_performance_report()
    counts["monitoring"] = load_dashboard_data(settings)
    counts["broker_modes"] = _broker_modes(settings)
    return counts


def _run_bootstrap(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {"005930.KS": 72_000.0, "AAPL": 200.0, "BTC-USD": 60_000.0}
    temp_dir, context, repository, _client = _build_context(settings, prices=prices, bootstrap_kis_holding=True)
    try:
        context.paper_broker.sim_broker.snapshot_account(cash_override=1_000_000.0)
        sync = broker_account_sync_job(context)
        latest = repository.latest_account_snapshot(source="kis_account_sync") or {}
        return {
            "sync": sync,
            "latest_snapshot_source": str(latest.get("source") or ""),
            "kr_state": context.risk_engine._latest_account_state(asset_type=asset_types["kr"], symbol="005930.KS"),
            "us_state": context.risk_engine._latest_account_state(asset_type=asset_types["us"], symbol="AAPL"),
            "crypto_state": context.risk_engine._latest_account_state(asset_type=asset_types["crypto"], symbol="BTC-USD"),
        }
    finally:
        temp_dir.cleanup()


def _run_preclose(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {"005930.KS": 70_000.0, "AAPL": 200.0}
    temp_dir, context, repository, client = _build_context(
        settings,
        prices=prices,
        required_preclose_minutes={asset_types["kr"]: 35, asset_types["us"]: 35},
    )
    try:
        _seed_candidate(repository, settings, SeedCandidate("005930.KS", asset_types["kr"], 96.0, 0.70, 1.10, 0.65, 1.2, 70_000.0))
        _seed_candidate(repository, settings, SeedCandidate("AAPL", asset_types["us"], 91.0, 0.70, 1.05, 0.64, 1.0, 200.0))
        entry_decision_job(context, [asset_types["kr"], asset_types["us"]])
        client.set_daily_fills_from_repository(repository)
        broker_order_sync_job(context)
        broker_position_sync_job(context)
        return _summarize(repository, settings)
    finally:
        temp_dir.cleanup()


def _run_thresholds(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {
        "005930.KS": 70_000.0,
        "000660.KS": 150_000.0,
        "AAPL": 200.0,
        "MSFT": 300.0,
        "NVDA": 900.0,
    }
    temp_dir, context, repository, client = _build_context(settings, prices=prices)
    try:
        _seed_candidate(repository, settings, SeedCandidate("005930.KS", asset_types["kr"], 98.0, 0.70, 1.10, 0.65, 1.2, 70_000.0))
        _seed_candidate(repository, settings, SeedCandidate("000660.KS", asset_types["kr"], 79.0, 0.45, 0.58, 0.60, 1.1, 150_000.0))
        _seed_candidate(repository, settings, SeedCandidate("AAPL", asset_types["us"], 95.0, 0.45, 0.95, 0.48, 1.0, 200.0))
        _seed_candidate(repository, settings, SeedCandidate("MSFT", asset_types["us"], 92.0, 0.70, 1.00, 0.64, 0.9, 300.0))
        _seed_candidate(repository, settings, SeedCandidate("NVDA", asset_types["us"], 71.0, 0.24, 0.45, 0.43, 1.0, 900.0))
        entry_decision_job(context, [asset_types["kr"], asset_types["us"]])
        client.set_daily_fills_from_repository(repository)
        broker_order_sync_job(context)
        broker_position_sync_job(context)
        return _summarize(repository, settings)
    finally:
        temp_dir.cleanup()


def _run_max_daily(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {"META": 500.0}
    temp_dir, context, repository, _client = _build_context(settings, prices=prices)
    try:
        _seed_candidate(repository, settings, SeedCandidate("META", asset_types["us"], 93.0, 0.70, 1.00, 0.66, 0.8, 500.0))
        _seed_prior_entries(repository, settings, asset_types["us"], "META", count=4)
        entry_decision_job(context, [asset_types["us"]])
        broker_order_sync_job(context)
        broker_position_sync_job(context)
        return _summarize(repository, settings)
    finally:
        temp_dir.cleanup()


def _run_correlation(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {"AAPL": 200.0, "QQQ": 450.0}
    temp_dir, context, repository, _client = _build_context(
        settings,
        prices=prices,
        correlations={("AAPL", "QQQ"): 0.86},
    )
    try:
        _seed_candidate(repository, settings, SeedCandidate("AAPL", asset_types["us"], 94.0, 0.70, 0.96, 0.63, 1.0, 200.0))
        _seed_open_position(repository, settings, asset_types["us"], "QQQ")
        entry_decision_job(context, [asset_types["us"]])
        broker_order_sync_job(context)
        broker_position_sync_job(context)
        return _summarize(repository, settings)
    finally:
        temp_dir.cleanup()


def _run_reason_logging(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    prices = {
        "068270.KS": 120_000.0,
        "035420.KS": 210_000.0,
        "AMZN": 180.0,
    }
    temp_dir, context, repository, client = _build_context(settings, prices=prices)
    try:
        _seed_candidate(repository, settings, SeedCandidate("068270.KS", asset_types["kr"], 90.0, 0.75, 0.62, 0.61, 1.0, 120_000.0))
        _seed_candidate(repository, settings, SeedCandidate("035420.KS", asset_types["kr"], 91.0, 0.88, 0.64, 0.63, 1.0, 210_000.0))
        _seed_candidate(repository, settings, SeedCandidate("AMZN", asset_types["us"], 87.0, 0.90, 0.66, 0.62, 1.0, 180.0))
        _seed_pending_entry(repository, settings, asset_types["kr"], "035420.KS")
        _seed_cooldown(repository, settings, asset_types["us"], "AMZN")
        client.quote_overrides["068270.KS"] = {"symbol_code": "068270", "current_price": np.nan, "raw": {}}
        entry_decision_job(context, [asset_types["kr"], asset_types["us"]])
        return _summarize(repository, settings)
    finally:
        temp_dir.cleanup()


def _merge_summaries(parts: list[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {f"today_{key}_count": 0 for key in EXECUTION_EVENT_KEYS}
    noop_counter: Counter[str] = Counter()
    rejected_counter: Counter[str] = Counter()
    asset_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    monitoring = {}
    broker_modes = {}
    fills = 0
    open_positions = 0
    worst_drawdown = 0.0
    total_today_pnl = 0.0
    for part in parts:
        for key in EXECUTION_EVENT_KEYS:
            merged[f"today_{key}_count"] += int(part.get(f"today_{key}_count", 0))
        noop_counter.update(part.get("today_noop_reason_breakdown", {}))
        rejected_counter.update(part.get("today_entry_rejected_reason_breakdown", {}))
        for asset_type, counts in part.get("asset_counts", {}).items():
            asset_counts[asset_type].update(counts)
        fills += int(part.get("fills", 0))
        open_positions += int(part.get("open_positions", 0))
        trade_performance = part.get("trade_performance", {})
        worst_drawdown = min(float(trade_performance.get("max_drawdown_pct", 0.0) or 0.0), worst_drawdown)
        total_today_pnl += float(trade_performance.get("today_pnl", 0.0) or 0.0)
        monitoring = part.get("monitoring", monitoring)
        broker_modes.update(part.get("broker_modes", {}))
    merged["today_noop_reason_breakdown"] = dict(noop_counter)
    merged["today_entry_rejected_reason_breakdown"] = dict(rejected_counter)
    merged["asset_counts"] = {asset_type: dict(counter) for asset_type, counter in asset_counts.items()}
    merged["fills"] = fills
    merged["open_positions"] = open_positions
    merged["trade_performance"] = {
        "max_drawdown_pct": worst_drawdown,
        "today_pnl": total_today_pnl,
    }
    merged["monitoring"] = monitoring
    merged["broker_modes"] = broker_modes
    return merged


def run_profile_smoke(profile_path: str | Path) -> Dict[str, Any]:
    settings = _profile_settings(profile_path)
    asset_types = _asset_types(settings)
    merged = _merge_summaries(
        [
            _run_preclose(profile_path),
            _run_thresholds(profile_path),
            _run_max_daily(profile_path),
            _run_correlation(profile_path),
            _run_reason_logging(profile_path),
        ]
    )
    merged["profile_name"] = str(settings.profile_name)
    merged["profile_source"] = str(settings.profile_source)
    merged["gates"] = {
        "min_signal_strength_pct": float(settings.strategy.min_signal_strength_pct),
        "min_expected_return_pct": float(settings.strategy.min_expected_return_pct),
        "min_confidence": float(settings.strategy.min_confidence),
        "max_daily_new_entries": int(settings.risk.max_daily_new_entries),
        "cooldown_bars_after_exit": int(settings.risk.cooldown_bars_after_exit),
        "max_same_direction_correlation": float(settings.risk.max_same_direction_correlation),
        "kr_pre_close_buffer_minutes": int(settings.asset_schedules[asset_types["kr"]].pre_close_buffer_minutes),
        "us_pre_close_buffer_minutes": int(settings.asset_schedules[asset_types["us"]].pre_close_buffer_minutes),
    }
    merged["bootstrap"] = _run_bootstrap(profile_path)
    return merged


def compare_profiles(profile_paths: Dict[str, str | Path] | None = None) -> list[Dict[str, Any]]:
    paths = profile_paths or PROFILE_PATHS
    return [run_profile_smoke(paths[name]) for name in ("baseline", "balanced", "active")]
