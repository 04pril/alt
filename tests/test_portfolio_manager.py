from __future__ import annotations

import tempfile
import unittest

from config.settings import RuntimeSettings
from services.portfolio_manager import PortfolioManager
from storage.models import CandidateScanRecord, LiveMarketQuoteRecord, PositionRecord
from storage.repository import TradingRepository, utc_now_iso


class _StubBroker:
    def __init__(self) -> None:
        self.exit_calls: list[dict[str, str]] = []

    def snapshot_account(self) -> None:
        return None

    def submit_exit_order_result(self, position, reason: str, *, market_data_service=None):
        self.exit_calls.append(
            {
                "symbol": str(position["symbol"]),
                "reason": str(reason),
            }
        )
        return {"submitted": True, "reason": "ok"}


class _FailIfFetchedMarketData:
    def latest_quote(self, symbol: str, asset_type: str, timeframe: str):
        raise AssertionError(f"unexpected bar quote fetch for {symbol} {asset_type} {timeframe}")


class PortfolioManagerLiveQuoteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = RuntimeSettings()
        self.settings.storage.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.repository = TradingRepository(self.settings.storage.db_path)
        self.repository.initialize()
        self.broker = _StubBroker()
        self.manager = PortfolioManager(self.settings, self.repository, self.broker)
        now_iso = utc_now_iso()
        self.repository.upsert_position(
            PositionRecord(
                position_id="pos_373220",
                created_at=now_iso,
                updated_at=now_iso,
                closed_at=None,
                prediction_id="pred_373220",
                symbol="373220.KS",
                asset_type="한국주식",
                timeframe="1h",
                side="LONG",
                status="open",
                quantity=14,
                entry_price=365666.0,
                mark_price=365000.0,
                stop_loss=359785.71428571426,
                take_profit=377100.0,
                trailing_stop=360015.9887115609,
                highest_price=366500.0,
                lowest_price=360500.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.019657454262358896,
                exposure_value=365000.0 * 14,
                max_holding_until="2026-03-12T00:00:00Z",
                strategy_version="kr_intraday_1h_v1",
                cooldown_until=None,
                notes="opened_by_fill",
                account_id="kis_kr_paper",
            )
        )
        self.repository.upsert_live_market_quote(
            LiveMarketQuoteRecord(
                symbol_code="373220",
                symbol="373220",
                asset_type="한국주식",
                currency="KRW",
                source="kis_quote_websocket",
                current_price=378000.0,
                previous_close=367000.0,
                change_pct=3.0,
                ask_price=378000.0,
                bid_price=377500.0,
                volume=163294.0,
                updated_at=utc_now_iso(),
                raw_json="{}",
            )
        )
        self.market_data_service = _FailIfFetchedMarketData()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mark_to_market_prefers_live_kr_quote_snapshot(self) -> None:
        self.manager.mark_to_market(self.market_data_service)
        updated = self.repository.latest_position_by_symbol("373220.KS", "1h", account_id="kis_kr_paper").iloc[0]
        self.assertEqual(float(updated["mark_price"]), 378000.0)
        self.assertEqual(float(updated["highest_price"]), 378000.0)

    def test_evaluate_exit_orders_uses_live_kr_bid_for_take_profit(self) -> None:
        exit_orders = self.manager.evaluate_exit_orders(self.market_data_service)
        self.assertEqual(exit_orders, 1)
        self.assertEqual(self.broker.exit_calls, [{"symbol": "373220.KS", "reason": "take_profit"}])

    def test_evaluate_exit_orders_matches_latest_candidate_by_strategy_and_account(self) -> None:
        now_iso = utc_now_iso()
        existing = self.repository.latest_position_by_symbol("373220.KS", "1h", account_id="kis_kr_paper").iloc[0]
        self.repository.upsert_position(
            PositionRecord(
                **{
                    **existing.to_dict(),
                    "updated_at": now_iso,
                    "stop_loss": 300000.0,
                    "take_profit": 450000.0,
                    "trailing_stop": 300000.0,
                }
            )
        )
        self.repository.upsert_position(
            PositionRecord(
                position_id="pos_strategy_scope",
                created_at=now_iso,
                updated_at=now_iso,
                closed_at=None,
                prediction_id="pred_scope",
                symbol="005930.KS",
                asset_type="한국주식",
                timeframe="15m",
                side="LONG",
                status="open",
                quantity=1,
                entry_price=190000.0,
                mark_price=190000.0,
                stop_loss=180000.0,
                take_profit=210000.0,
                trailing_stop=180000.0,
                highest_price=191000.0,
                lowest_price=189000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                expected_risk=0.01,
                exposure_value=190000.0,
                max_holding_until="2026-03-12T00:00:00Z",
                strategy_version="kr_intraday_15m_v1",
                cooldown_until=None,
                notes="opened_by_fill",
                account_id="kis_kr_paper",
            )
        )
        self.repository.upsert_live_market_quote(
            LiveMarketQuoteRecord(
                symbol_code="005930",
                symbol="005930",
                asset_type="한국주식",
                currency="KRW",
                source="kis_quote_websocket",
                current_price=190000.0,
                previous_close=189000.0,
                change_pct=0.5,
                ask_price=190100.0,
                bid_price=190000.0,
                volume=1000.0,
                updated_at=utc_now_iso(),
                raw_json="{}",
            )
        )
        self.repository.insert_candidate_scans(
            [
                CandidateScanRecord(
                    scan_id="scan_matching_strategy",
                    created_at="2026-03-11T06:00:00Z",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe="15m",
                    score=0.0,
                    rank=1,
                    status="candidate",
                    reason="signal_ready",
                    expected_return=0.01,
                    expected_risk=0.01,
                    confidence=0.9,
                    threshold=0.003,
                    volatility=0.01,
                    liquidity_score=1.0,
                    cost_bps=5.0,
                    recent_performance=0.0,
                    signal="SHORT",
                    model_version="m",
                    feature_version="f",
                    strategy_version="kr_intraday_15m_v1",
                    raw_json="{}",
                    execution_account_id="kis_kr_paper",
                ),
                CandidateScanRecord(
                    scan_id="scan_other_strategy",
                    created_at="2026-03-11T06:01:00Z",
                    symbol="005930.KS",
                    asset_type="한국주식",
                    timeframe="15m",
                    score=10.0,
                    rank=1,
                    status="candidate",
                    reason="signal_ready",
                    expected_return=0.02,
                    expected_risk=0.01,
                    confidence=0.9,
                    threshold=0.003,
                    volatility=0.01,
                    liquidity_score=1.0,
                    cost_bps=5.0,
                    recent_performance=0.0,
                    signal="LONG",
                    model_version="m",
                    feature_version="f",
                    strategy_version="kr_intraday_1h_v1",
                    raw_json="{}",
                    execution_account_id="kis_kr_paper",
                ),
            ]
        )

        exit_orders = self.manager.evaluate_exit_orders(self.market_data_service)

        self.assertEqual(exit_orders, 1)
        self.assertIn({"symbol": "005930.KS", "reason": "opposite_signal"}, self.broker.exit_calls)


if __name__ == "__main__":
    unittest.main()
