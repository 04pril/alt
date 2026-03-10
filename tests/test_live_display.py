from __future__ import annotations

import unittest

import pandas as pd

from monitoring.live_display import build_live_accounts_overview, build_live_total_portfolio_overview
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_US_EQUITY


class LiveDisplayTest(unittest.TestCase):
    def test_build_live_accounts_overview_reprices_open_positions(self) -> None:
        accounts_overview = {
            ACCOUNT_KIS_KR_PAPER: {
                "account_id": ACCOUNT_KIS_KR_PAPER,
                "currency": "KRW",
                "cash": 29810000.0,
                "equity": 29999900.0,
                "gross_exposure": 189900.0,
                "net_exposure": 189900.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_positions": 1,
                "pending_orders": 0,
                "latest_snapshot": {
                    "cash": 29810000.0,
                    "equity": 29999900.0,
                    "gross_exposure": 189900.0,
                    "net_exposure": 189900.0,
                    "daily_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                },
                "trade_performance": {},
            }
        }
        open_positions = pd.DataFrame(
            [
                {
                    "account_id": ACCOUNT_KIS_KR_PAPER,
                    "symbol": "005930",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "quantity": 1,
                    "entry_price": 189900.0,
                    "mark_price": 189900.0,
                    "unrealized_pnl": 0.0,
                    "exposure_value": 189900.0,
                }
            ]
        )
        quote_snapshots = {
            "005930": {
                "current_price": 190000.0,
                "currency": "KRW",
            }
        }

        live_accounts = build_live_accounts_overview(accounts_overview, open_positions, quote_snapshots)
        payload = live_accounts[ACCOUNT_KIS_KR_PAPER]

        self.assertEqual(payload["live_quote_count"], 1)
        self.assertAlmostEqual(float(payload["gross_exposure"]), 190000.0)
        self.assertAlmostEqual(float(payload["unrealized_pnl"]), 100.0)
        self.assertAlmostEqual(float(payload["equity"]), 30000000.0)

    def test_build_live_total_portfolio_overview_keeps_currency_buckets(self) -> None:
        total = build_live_total_portfolio_overview(
            {
                ACCOUNT_KIS_KR_PAPER: {
                    "currency": "KRW",
                    "cash": 30000000.0,
                    "equity": 30100000.0,
                    "gross_exposure": 100000.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 100000.0,
                    "open_positions": 1,
                    "pending_orders": 0,
                    "drawdown_pct": -1.0,
                    "last_sync_time": "2026-03-10T01:00:00Z",
                    "last_sync_status": "completed",
                },
                ACCOUNT_SIM_US_EQUITY: {
                    "currency": "USD",
                    "cash": 1000.0,
                    "equity": 1100.0,
                    "gross_exposure": 100.0,
                    "realized_pnl": 10.0,
                    "unrealized_pnl": 20.0,
                    "open_positions": 1,
                    "pending_orders": 1,
                    "drawdown_pct": -2.0,
                    "last_sync_time": "2026-03-10T01:01:00Z",
                    "last_sync_status": "completed",
                },
            }
        )

        self.assertAlmostEqual(float(total["equity_by_currency"]["KRW"]), 30100000.0)
        self.assertAlmostEqual(float(total["equity_by_currency"]["USD"]), 1100.0)
        self.assertEqual(int(total["open_positions"]), 2)
        self.assertEqual(int(total["pending_orders"]), 1)

    def test_build_live_accounts_overview_uses_snapshot_equity_anchor_for_kis_buying_power(self) -> None:
        accounts_overview = {
            ACCOUNT_KIS_KR_PAPER: {
                "account_id": ACCOUNT_KIS_KR_PAPER,
                "currency": "KRW",
                "cash": 30000000.0,
                "equity": 30016760.0,
                "gross_exposure": 5325900.0,
                "net_exposure": 5325900.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 17500.0,
                "open_positions": 2,
                "pending_orders": 0,
                "latest_snapshot": {
                    "cash": 30000000.0,
                    "equity": 30016760.0,
                    "gross_exposure": 5325900.0,
                    "net_exposure": 5325900.0,
                    "daily_pnl": 0.0,
                    "unrealized_pnl": 17500.0,
                },
                "trade_performance": {},
            }
        }
        open_positions = pd.DataFrame(
            [
                {
                    "account_id": ACCOUNT_KIS_KR_PAPER,
                    "symbol": "005930",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "quantity": 24,
                    "entry_price": 189500.0,
                    "mark_price": 187300.0,
                    "unrealized_pnl": -61600.0,
                    "exposure_value": 4495200.0,
                },
                {
                    "account_id": ACCOUNT_KIS_KR_PAPER,
                    "symbol": "373220",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "quantity": 1,
                    "entry_price": 831200.0,
                    "mark_price": 830700.0,
                    "unrealized_pnl": -500.0,
                    "exposure_value": 830700.0,
                },
            ]
        )
        quote_snapshots = {
            "005930": {"current_price": 187300.0, "currency": "KRW"},
            "373220": {"current_price": 830700.0, "currency": "KRW"},
        }

        live_accounts = build_live_accounts_overview(accounts_overview, open_positions, quote_snapshots)
        payload = live_accounts[ACCOUNT_KIS_KR_PAPER]

        self.assertAlmostEqual(float(payload["gross_exposure"]), 5325900.0)
        self.assertAlmostEqual(float(payload["equity"]), 30016760.0)
        self.assertLess(float(payload["equity"]), float(payload["cash"]) + float(payload["gross_exposure"]))
