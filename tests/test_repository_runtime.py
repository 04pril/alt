from __future__ import annotations

import tempfile
import threading
import time
import unittest

from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.models import AccountSnapshotRecord, OrderRecord, PositionRecord
from storage.repository import TradingRepository, make_id, utc_now_iso


class RepositoryRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmp.name}/runtime.sqlite3"
        self.repo = TradingRepository(self.db_path)
        self.repo.initialize()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_count_daily_entries_only_counts_entry_orders(self) -> None:
        created_at = "2026-03-08T09:00:00Z"
        rows = [
            OrderRecord(
                order_id="ord_entry_new",
                created_at=created_at,
                updated_at=created_at,
                prediction_id=None,
                scan_id=None,
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="buy",
                order_type="market",
                requested_qty=1,
                filled_qty=0,
                remaining_qty=1,
                requested_price=100.0,
                limit_price=0.0,
                status="new",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="s1",
                reason="entry",
            ),
            OrderRecord(
                order_id="ord_entry_partial",
                created_at=created_at,
                updated_at=created_at,
                prediction_id=None,
                scan_id=None,
                symbol="ETH-USD",
                asset_type="코인",
                timeframe="1h",
                side="buy",
                order_type="market",
                requested_qty=2,
                filled_qty=1,
                remaining_qty=1,
                requested_price=200.0,
                limit_price=0.0,
                status="partially_filled",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="s1",
                reason="entry",
            ),
            OrderRecord(
                order_id="ord_exit_stop",
                created_at=created_at,
                updated_at=created_at,
                prediction_id=None,
                scan_id=None,
                symbol="AAPL",
                asset_type="미국주식",
                timeframe="1d",
                side="sell",
                order_type="market",
                requested_qty=1,
                filled_qty=1,
                remaining_qty=0,
                requested_price=190.0,
                limit_price=0.0,
                status="filled",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="s1",
                reason="stop_loss",
            ),
            OrderRecord(
                order_id="ord_entry_cancelled",
                created_at=created_at,
                updated_at=created_at,
                prediction_id=None,
                scan_id=None,
                symbol="NVDA",
                asset_type="미국주식",
                timeframe="1d",
                side="buy",
                order_type="market",
                requested_qty=1,
                filled_qty=0,
                remaining_qty=1,
                requested_price=900.0,
                limit_price=0.0,
                status="cancelled",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="s1",
                reason="entry",
            ),
        ]
        for row in rows:
            self.repo.insert_order(row)

        self.assertEqual(self.repo.count_daily_entries("2026-03-08"), 2)

    def test_latest_account_snapshot_uses_rowid_tiebreaker(self) -> None:
        created_at = "2026-03-08T10:00:00Z"
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_old",
                created_at=created_at,
                cash=100.0,
                equity=100.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="test",
            )
        )
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_new",
                created_at=created_at,
                cash=150.0,
                equity=150.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="test",
            )
        )

        latest = self.repo.latest_account_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["snapshot_id"], "snap_new")
        self.assertEqual(float(latest["equity"]), 150.0)

    def test_max_account_equity_honors_source_scope(self) -> None:
        rows = [
            AccountSnapshotRecord(
                snapshot_id="snap_sim_1",
                created_at="2026-03-08T09:00:00Z",
                cash=100.0,
                equity=100.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="paper_broker",
            ),
            AccountSnapshotRecord(
                snapshot_id="snap_sim_2",
                created_at="2026-03-08T09:01:00Z",
                cash=120.0,
                equity=120.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="paper_broker",
            ),
            AccountSnapshotRecord(
                snapshot_id="snap_kis_1",
                created_at="2026-03-08T09:02:00Z",
                cash=500.0,
                equity=500.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=0,
                open_orders=0,
                paused=0,
                source="kis_account_sync",
            ),
        ]
        for row in rows:
            self.repo.insert_account_snapshot(row)

        self.assertEqual(self.repo.max_account_equity(source="kis_account_sync"), 500.0)
        self.assertEqual(self.repo.max_account_equity(exclude_sources=("kis_account_sync",)), 120.0)

    def test_latest_position_and_cooldown_use_rowid_tiebreaker(self) -> None:
        updated_at = "2026-03-08T11:00:00Z"
        base = {
            "created_at": "2026-03-08T09:00:00Z",
            "updated_at": updated_at,
            "prediction_id": None,
            "symbol": "BTC-USD",
            "asset_type": "코인",
            "timeframe": "1h",
            "side": "LONG",
            "entry_price": 100.0,
            "mark_price": 100.0,
            "stop_loss": 90.0,
            "take_profit": 120.0,
            "trailing_stop": 95.0,
            "highest_price": 100.0,
            "lowest_price": 100.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "expected_risk": 0.02,
            "exposure_value": 100.0,
            "max_holding_until": "2026-03-09T00:00:00Z",
            "strategy_version": "s1",
        }
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos_closed",
                closed_at=updated_at,
                status="closed",
                quantity=0,
                cooldown_until="2026-03-08T12:00:00Z",
                notes="closed",
                **base,
            )
        )
        self.repo.upsert_position(
            PositionRecord(
                position_id="pos_open",
                closed_at=None,
                status="open",
                quantity=2,
                cooldown_until="2026-03-08T13:00:00Z",
                notes="reopened",
                **base,
            )
        )

        latest = self.repo.latest_position_by_symbol("BTC-USD", "1h")
        self.assertFalse(latest.empty)
        self.assertEqual(latest.iloc[0]["position_id"], "pos_open")
        self.assertEqual(self.repo.latest_cooldown_until("BTC-USD", "1h"), "2026-03-08T13:00:00Z")

    def test_account_backfill_populates_blank_execution_and_account_ids(self) -> None:
        self.repo.insert_order(
            OrderRecord(
                order_id="ord_backfill",
                created_at="2026-03-08T09:00:00Z",
                updated_at="2026-03-08T09:00:00Z",
                prediction_id="pred_backfill",
                scan_id="scan_backfill",
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                side="buy",
                order_type="market",
                requested_qty=1,
                filled_qty=0,
                remaining_qty=1,
                requested_price=100.0,
                limit_price=0.0,
                status="new",
                fees_estimate=0.0,
                slippage_bps=0.0,
                retry_count=0,
                strategy_version="s1",
                reason="entry",
                raw_json="{}",
            )
        )
        self.repo.insert_account_snapshot(
            AccountSnapshotRecord(
                snapshot_id="snap_backfill",
                created_at="2026-03-08T09:00:00Z",
                cash=100.0,
                equity=120.0,
                gross_exposure=20.0,
                net_exposure=20.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                open_positions=1,
                open_orders=1,
                paused=0,
                source="kis_account_sync",
                raw_json="{}",
            )
        )
        with self.repo.connect() as conn:
            conn.execute("UPDATE predictions SET execution_account_id = ''")
            conn.execute("UPDATE candidate_scans SET execution_account_id = ''")
            conn.execute("UPDATE orders SET account_id = '', raw_json = '{}' WHERE order_id = 'ord_backfill'")
            conn.execute("INSERT INTO fills(fill_id, created_at, order_id, symbol, side, quantity, fill_price, fees, slippage_bps, status, raw_json, account_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("fill_backfill", "2026-03-08T09:01:00Z", "ord_backfill", "BTC-USD", "buy", 1, 100.0, 0.0, 0.0, "filled", "{}", ""))
            conn.execute("INSERT INTO positions(position_id, created_at, updated_at, closed_at, prediction_id, symbol, asset_type, timeframe, side, status, quantity, entry_price, mark_price, stop_loss, take_profit, trailing_stop, highest_price, lowest_price, unrealized_pnl, realized_pnl, expected_risk, exposure_value, max_holding_until, strategy_version, cooldown_until, notes, account_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("pos_backfill", "2026-03-08T09:00:00Z", "2026-03-08T09:00:00Z", None, "pred_backfill", "AAPL", "미국주식", "1d", "LONG", "open", 1, 100.0, 100.0, None, None, None, None, None, 0.0, 0.0, 0.01, 100.0, None, "s1", None, "", ""))
            conn.execute("UPDATE account_snapshots SET account_id = '', raw_json = '{}' WHERE snapshot_id = 'snap_backfill'")
            conn.execute("INSERT INTO predictions(prediction_id, created_at, run_id, symbol, asset_type, timeframe, market_timezone, data_cutoff_at, target_at, forecast_horizon_bars, target_type, current_price, predicted_price, predicted_return, signal, score, confidence, threshold, expected_return, expected_risk, position_size, model_name, model_version, feature_version, strategy_version, validation_mode, feature_hash, status, scan_id, notes, execution_account_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("pred_backfill", "2026-03-08T09:00:00Z", "run1", "AAPL", "미국주식", "1d", "America/New_York", "2026-03-08T09:00:00Z", "2026-03-09T09:00:00Z", 1, "next_close_return", 100.0, 102.0, 0.02, "LONG", 1.0, 0.9, 0.003, 0.02, 0.01, 0.5, "demo", "v1", "f1", "s1", "holdout", "hash", "unresolved", "scan_backfill", "{}", ""))
            conn.execute("INSERT INTO candidate_scans(scan_id, created_at, symbol, asset_type, timeframe, score, rank, status, reason, expected_return, expected_risk, confidence, threshold, volatility, liquidity_score, cost_bps, recent_performance, signal, model_version, feature_version, strategy_version, cooldown_until, is_holding, raw_json, execution_account_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("scan_backfill", "2026-03-08T09:00:00Z", "005930.KS", "한국주식", "1d", 1.0, 1, "candidate", "ok", 0.02, 0.01, 0.9, 0.003, 0.01, 1.0, 5.0, 0.0, "LONG", "v1", "f1", "s1", None, 0, "{}", ""))
            conn.execute("INSERT INTO system_events(event_id, created_at, level, component, event_type, message, details_json, account_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?)", ("evt_backfill", "2026-03-08T09:02:00Z", "INFO", "execution_pipeline", "candidate", "candidate", '{\"order_id\":\"ord_backfill\",\"symbol\":\"BTC-USD\"}', ""))
            self.repo._migrate_schema(conn)

        self.assertEqual(self.repo.get_order("ord_backfill")["account_id"], ACCOUNT_SIM_CRYPTO)
        self.assertEqual(self.repo.latest_position_by_symbol("AAPL", "1d").iloc[0]["account_id"], ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(self.repo.latest_account_snapshot(account_id=ACCOUNT_KIS_KR_PAPER)["snapshot_id"], "snap_backfill")
        self.assertEqual(self.repo.prediction_by_scan("scan_backfill").iloc[0]["execution_account_id"], ACCOUNT_SIM_US_EQUITY)
        self.assertEqual(self.repo.latest_candidates(limit=5).loc[0, "execution_account_id"], ACCOUNT_KIS_KR_PAPER)
        self.assertEqual(self.repo.latest_system_event("candidate")["account_id"], ACCOUNT_SIM_CRYPTO)

    def test_begin_job_run_returns_canonical_id_under_concurrency(self) -> None:
        repo_a = TradingRepository(self.db_path)
        repo_b = TradingRepository(self.db_path)
        barrier = threading.Barrier(2)
        results: list[tuple[str, bool]] = []
        errors: list[BaseException] = []

        def worker(repo: TradingRepository, owner: str) -> None:
            try:
                barrier.wait()
                lease = repo.begin_job_run(
                    job_name="scan:crypto",
                    run_key="2026-03-08T10:00",
                    scheduled_at="2026-03-08T10:00:00Z",
                    lock_owner=owner,
                    retry_backoff_seconds=1,
                    max_retry_count=2,
                    lease_seconds=30,
                )
                results.append((lease.job_run_id, lease.acquired))
            except BaseException as exc:  # pragma: no cover - defensive test harness
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=(repo_a, "worker-a"))
        t2 = threading.Thread(target=worker, args=(repo_b, "worker-b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        canonical_ids = {item[0] for item in results}
        self.assertEqual(len(canonical_ids), 1)
        self.assertEqual(sum(1 for _, acquired in results if acquired), 1)

        canonical_id = results[0][0]
        self.repo.finish_job_run(canonical_id, status="completed", metrics={"ok": True})
        row = self.repo.get_job_run(canonical_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "completed")

    def test_begin_job_run_retry_waits_for_backoff_and_preserves_retry_count(self) -> None:
        lease = self.repo.begin_job_run(
            job_name="outcome_resolution",
            run_key="2026-03-08T10:00",
            scheduled_at="2026-03-08T10:00:00Z",
            lock_owner="worker",
            retry_backoff_seconds=1,
            max_retry_count=2,
            lease_seconds=30,
        )
        self.repo.finish_job_run(
            lease.job_run_id,
            status="failed",
            error_message="boom",
            retry_backoff_seconds=1,
            max_retry_count=2,
        )

        blocked = self.repo.begin_job_run(
            job_name="outcome_resolution",
            run_key="2026-03-08T10:00",
            scheduled_at="2026-03-08T10:00:00Z",
            lock_owner="worker",
            retry_backoff_seconds=1,
            max_retry_count=2,
            lease_seconds=30,
        )
        self.assertFalse(blocked.acquired)
        time.sleep(1.1)

        retried = self.repo.begin_job_run(
            job_name="outcome_resolution",
            run_key="2026-03-08T10:00",
            scheduled_at="2026-03-08T10:00:00Z",
            lock_owner="worker",
            retry_backoff_seconds=1,
            max_retry_count=2,
            lease_seconds=30,
        )
        self.assertTrue(retried.acquired)
        self.assertEqual(retried.job_run_id, lease.job_run_id)
        self.assertEqual(retried.retry_count, 1)


if __name__ == "__main__":
    unittest.main()
