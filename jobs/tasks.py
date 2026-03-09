from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable

import pandas as pd

from config.settings import RuntimeSettings, load_settings
from services.broker_router import BrokerRouter
from services.evaluator import Evaluator
from services.kis_paper_broker import KISPaperBroker
from services.market_data_service import MarketDataService
from services.outcome_resolver import OutcomeResolver
from services.paper_broker import PaperBroker
from services.portfolio_manager import PortfolioManager
from services.retrainer import Retrainer
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision, SignalEngine
from services.universe_scanner import UniverseScanner
from storage.repository import TradingRepository


TouchFn = Callable[[str | None, Dict[str, Any] | None], None]


@dataclass
class TaskContext:
    settings: RuntimeSettings
    repository: TradingRepository
    market_data_service: MarketDataService
    signal_engine: SignalEngine
    risk_engine: RiskEngine
    paper_broker: BrokerRouter
    portfolio_manager: PortfolioManager
    universe_scanner: UniverseScanner
    outcome_resolver: OutcomeResolver
    evaluator: Evaluator
    retrainer: Retrainer
    job_touch: TouchFn = field(default=lambda _stage=None, _details=None: None)

    def touch_runtime(self, stage: str | None = None, details: Dict[str, Any] | None = None) -> None:
        self.job_touch(stage, details or {})


def build_task_context(settings_path: str | None = None) -> TaskContext:
    settings = load_settings(settings_path)
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    market_data_service = MarketDataService(settings)
    signal_engine = SignalEngine(settings, repository)
    sim_broker = PaperBroker(settings, repository)
    kis_broker = KISPaperBroker(settings, repository, sim_broker)
    paper_broker = BrokerRouter(sim_broker=sim_broker, kis_broker=kis_broker)
    risk_engine = RiskEngine(settings, repository)
    portfolio_manager = PortfolioManager(settings, repository, paper_broker)
    universe_scanner = UniverseScanner(settings, repository, market_data_service, signal_engine)
    outcome_resolver = OutcomeResolver(repository, market_data_service)
    evaluator = Evaluator(repository)
    retrainer = Retrainer(settings, repository)
    paper_broker.ensure_account_initialized()
    repository.set_control_flag("runtime_profile_name", str(settings.profile_name or "baseline"), "task_context")
    repository.set_control_flag(
        "runtime_profile_source",
        str(settings.profile_source or "embedded_defaults"),
        "task_context",
    )
    return TaskContext(
        settings=settings,
        repository=repository,
        market_data_service=market_data_service,
        signal_engine=signal_engine,
        risk_engine=risk_engine,
        paper_broker=paper_broker,
        portfolio_manager=portfolio_manager,
        universe_scanner=universe_scanner,
        outcome_resolver=outcome_resolver,
        evaluator=evaluator,
        retrainer=retrainer,
    )


def _rebuild_signal(context: TaskContext, candidate_row: pd.Series) -> SignalDecision | None:
    predictions = context.repository.prediction_by_scan(str(candidate_row["scan_id"]))
    if predictions.empty:
        return None
    pred = predictions.iloc[0]
    notes = json.loads(str(pred.get("notes") or "{}"))
    return SignalDecision(
        symbol=str(pred["symbol"]),
        asset_type=str(pred["asset_type"]),
        timeframe=str(pred["timeframe"]),
        prediction_id=str(pred["prediction_id"]),
        scan_id=str(pred.get("scan_id") or ""),
        score=float(pred["score"]),
        signal=str(pred["signal"]),
        expected_return=float(pred["expected_return"]),
        expected_risk=float(pred["expected_risk"]),
        confidence=float(pred["confidence"]),
        threshold=float(pred["threshold"]),
        position_size=float(pred["position_size"]),
        current_price=float(pred["current_price"]),
        predicted_price=float(pred["predicted_price"]),
        predicted_return=float(pred["predicted_return"]),
        stop_level=float(notes.get("stop_level", float("nan"))),
        take_level=float(notes.get("take_level", float("nan"))),
        model_version=str(pred["model_version"]),
        feature_version=str(pred["feature_version"]),
        strategy_version=str(pred["strategy_version"]),
        validation_mode=str(pred["validation_mode"]),
        result=None,
    )


def _execution_event(context: TaskContext, event_type: str, message: str, details: Dict[str, Any], *, level: str = "INFO") -> None:
    context.repository.log_event(level, "execution_pipeline", event_type, message, details)


def _record_noop(context: TaskContext, asset_type: str, reason: str, details: Dict[str, Any] | None = None) -> None:
    _execution_event(context, "noop", f"{asset_type} entry noop", {"asset_type": asset_type, "reason": reason, **(details or {})})


def scan_job(context: TaskContext, asset_types: Iterable[str] | None = None) -> Dict[str, int]:
    asset_list = list(asset_types or context.settings.asset_schedules.keys())
    results: Dict[str, int] = {}
    for asset_type in asset_list:
        context.touch_runtime("scan_asset", {"asset_type": asset_type})
        rows = context.universe_scanner.scan_asset(asset_type, touch=context.touch_runtime)
        results[asset_type] = len(rows)
        context.repository.log_event("INFO", "scan_job", "scan_complete", f"{asset_type} scanned", {"count": len(rows)})
    return results


def entry_decision_job(context: TaskContext, asset_types: Iterable[str] | None = None) -> Dict[str, int]:
    asset_list = list(asset_types or context.settings.asset_schedules.keys())
    entered: Dict[str, int] = {}
    for asset_type in asset_list:
        schedule = context.settings.asset_schedules[asset_type]
        market_is_open = context.market_data_service.is_market_open(asset_type)
        if not market_is_open:
            entered[asset_type] = 0
            _record_noop(context, asset_type, "market_closed", {"market_phase": context.market_data_service.market_phase(asset_type)})
            continue
        if schedule.timeframe == "1d" and not context.market_data_service.is_pre_close_window(asset_type):
            entered[asset_type] = 0
            _record_noop(context, asset_type, "outside_preclose_window", {"market_phase": context.market_data_service.market_phase(asset_type)})
            continue

        latest = context.repository.latest_candidates(asset_type=asset_type, timeframe=schedule.timeframe, limit=100)
        if latest.empty:
            entered[asset_type] = 0
            _record_noop(context, asset_type, "no_candidate")
            continue

        latest = latest.sort_values(["created_at", "rank"], ascending=[False, True]).drop_duplicates(subset=["symbol", "timeframe"], keep="first")
        latest = latest[(latest["status"] == "candidate") & (latest["is_holding"] == 0)]
        if latest.empty:
            entered[asset_type] = 0
            _record_noop(context, asset_type, "no_candidate", {"filtered": True})
            continue

        count = 0
        detailed_no_trade_reason_logged = False
        open_positions = context.repository.open_positions()
        corr_symbols = list(open_positions["symbol"].astype(str).tolist()) if not open_positions.empty else []
        for _, candidate in latest.iterrows():
            context.touch_runtime("entry_candidate", {"asset_type": asset_type, "symbol": str(candidate["symbol"])})
            _execution_event(
                context,
                "candidate",
                "candidate considered for execution",
                {
                    "asset_type": asset_type,
                    "symbol": str(candidate["symbol"]),
                    "scan_id": str(candidate["scan_id"]),
                    "rank": int(candidate["rank"]),
                    "score": float(candidate["score"]),
                },
            )
            signal = _rebuild_signal(context, candidate)
            if signal is None:
                detailed_no_trade_reason_logged = True
                _execution_event(context, "entry_rejected", "candidate missing prediction", {"symbol": str(candidate["symbol"]), "reason": "missing_prediction", "scan_id": str(candidate["scan_id"])})
                continue
            correlation_matrix = (
                context.market_data_service.correlation_matrix(
                    symbols=list(set(corr_symbols + [signal.symbol])),
                    asset_type=asset_type,
                    timeframe=schedule.timeframe,
                    lookback_bars=context.settings.risk.correlation_window_bars,
                )
                if corr_symbols
                else pd.DataFrame()
            )
            risk_decision = context.risk_engine.evaluate_entry(signal, correlation_matrix=correlation_matrix, market_is_open=market_is_open)
            if not risk_decision.allowed:
                detailed_no_trade_reason_logged = True
                _execution_event(
                    context,
                    "entry_rejected",
                    "risk gate rejected candidate",
                    {"symbol": signal.symbol, "asset_type": asset_type, "reason": risk_decision.reason, "scan_id": signal.scan_id},
                )
                continue

            _execution_event(
                context,
                "entry_allowed",
                "entry passed risk gate",
                {"symbol": signal.symbol, "asset_type": asset_type, "quantity": int(risk_decision.quantity), "scan_id": signal.scan_id},
            )
            submit = context.paper_broker.submit_entry_order_result(
                signal=signal,
                quantity=risk_decision.quantity,
                scan_id=signal.scan_id,
                market_data_service=context.market_data_service,
            )
            if not submit.get("submitted", False):
                detailed_no_trade_reason_logged = True
                _execution_event(
                    context,
                    "entry_rejected",
                    "broker pre-submit rejected candidate",
                    {"symbol": signal.symbol, "asset_type": asset_type, "reason": str(submit.get("reason") or "broker_rejected"), "scan_id": signal.scan_id},
                )
                continue
            count += 1
        entered[asset_type] = count
        if count == 0:
            if not detailed_no_trade_reason_logged:
                _record_noop(context, asset_type, "no_submit")
        else:
            context.repository.log_event("INFO", "entry_job", "entry_orders_created", f"{asset_type} entries", {"count": count})
    return entered


def exit_management_job(context: TaskContext) -> Dict[str, int]:
    context.touch_runtime("exit_mark_to_market", {})
    context.portfolio_manager.mark_to_market(context.market_data_service, touch=context.touch_runtime)
    exit_orders = context.portfolio_manager.evaluate_exit_orders(context.market_data_service, touch=context.touch_runtime)
    context.repository.log_event("INFO", "exit_job", "exit_cycle", "exit management cycle completed", {"exit_orders": exit_orders})
    return {"exit_orders": exit_orders}


def broker_market_status_job(context: TaskContext) -> Dict[str, Any]:
    phases: Dict[str, str] = {}
    for asset_type in context.settings.asset_schedules.keys():
        context.touch_runtime("broker_market_status", {"asset_type": asset_type})
        phase = context.market_data_service.market_phase(asset_type)
        phases[asset_type] = phase
        context.repository.set_control_flag(f"market_phase:{asset_type}", phase, "broker_market_status")
    context.repository.log_event("INFO", "broker_market_status_job", "broker_market_status", "market status snapshot recorded", phases)
    return phases


def broker_order_sync_job(context: TaskContext) -> Dict[str, int]:
    result = context.paper_broker.sync_orders(context.market_data_service, touch=context.touch_runtime)
    context.repository.log_event("INFO", "broker_order_sync_job", "broker_order_sync", "broker order sync completed", result)
    return result


def broker_position_sync_job(context: TaskContext) -> Dict[str, int]:
    context.touch_runtime("broker_position_sync", {})
    context.portfolio_manager.mark_to_market(context.market_data_service, touch=context.touch_runtime)
    positions = context.repository.open_positions()
    result = {"open_positions": int(len(positions))}
    context.repository.log_event("INFO", "broker_position_sync_job", "broker_position_sync", "broker position sync completed", result)
    return result


def broker_account_sync_job(context: TaskContext) -> Dict[str, Any]:
    context.touch_runtime("broker_account_sync", {})
    result = context.paper_broker.sync_account(touch=context.touch_runtime)
    context.repository.log_event("INFO", "broker_account_sync_job", "broker_account_sync", "broker account sync completed", result)
    return result


def outcome_resolution_job(context: TaskContext) -> Dict[str, int]:
    resolved = context.outcome_resolver.resolve()
    context.repository.log_event("INFO", "outcome_job", "outcome_resolved", "resolved predictions", {"count": resolved})
    return {"resolved": resolved}


def daily_report_job(context: TaskContext) -> Dict[str, float]:
    pred_metrics = context.evaluator.prediction_metrics()
    trade_metrics = context.evaluator.trade_metrics()
    context.repository.log_event(
        "INFO",
        "daily_report_job",
        "daily_report",
        "daily report generated",
        {
            "trade_metrics": trade_metrics,
            "prediction_rows": int(len(pred_metrics["by_date"])) if not pred_metrics["by_date"].empty else 0,
        },
    )
    return trade_metrics


def retrain_check_job(context: TaskContext) -> Dict[str, float | str]:
    if not context.retrainer.retraining_due():
        summary = {"status": "skip_not_due"}
        context.repository.log_event("INFO", "retrain_job", "skip", "retraining not due", summary)
        return summary
    summary = context.retrainer.run()
    context.repository.log_event("INFO", "retrain_job", "retrain", "retraining check complete", summary)
    return summary
