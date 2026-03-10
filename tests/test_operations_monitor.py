from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app import (
    build_live_open_positions_view,
    build_monitor_table_view,
    build_recent_order_activity_view,
    build_recent_realized_trades_view,
)


class OperationsMonitorTest(unittest.TestCase):
    @patch("app.load_krx_symbol_name_map", return_value={"005930": "삼성전자", "005930.KS": "삼성전자"})
    def test_build_live_open_positions_view_formats_long_and_short_pnl(self, _mock_names) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "005930.KS",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "quantity": 10,
                    "entry_price": 70000.0,
                    "mark_price": 70000.0,
                    "unrealized_pnl": 0.0,
                    "updated_at": "2026-03-09T01:05:00Z",
                },
                {
                    "symbol": "BTC-USD",
                    "asset_type": "코인",
                    "side": "SHORT",
                    "quantity": 2,
                    "entry_price": 100.0,
                    "mark_price": 100.0,
                    "unrealized_pnl": 0.0,
                    "updated_at": "2026-03-09T01:06:00Z",
                },
            ]
        )
        with patch(
            "app.fetch_quote_snapshots",
            return_value={
                "005930.KS": {"currency": "KRW", "current_price": 71000.0, "change_pct": 1.25},
                "BTC-USD": {"currency": "USD", "current_price": 90.0, "change_pct": -2.5},
            },
        ):
            view = build_live_open_positions_view(positions, refresh_token=1)

        self.assertEqual(view.loc[0, "종목"], "삼성전자 (005930)")
        self.assertEqual(view.loc[0, "현재가"], "71,000 KRW")
        self.assertEqual(view.loc[0, "평가손익"], "10,000 KRW")
        self.assertEqual(view.loc[0, "수익률"], "+1.43%")
        self.assertEqual(view.loc[1, "종목"], "BTC-USD")
        self.assertEqual(view.loc[1, "현재가"], "90.00 USD")
        self.assertEqual(view.loc[1, "평가손익"], "20.00 USD")
        self.assertEqual(view.loc[1, "수익률"], "+10.00%")

    def test_build_live_open_positions_view_returns_empty_when_no_positions(self) -> None:
        view = build_live_open_positions_view(pd.DataFrame(), refresh_token=1)
        self.assertTrue(view.empty)

    @patch("app.load_krx_symbol_name_map", return_value={"005930": "삼성전자", "005930.KS": "삼성전자"})
    def test_build_recent_order_activity_view_formats_buy_and_sell_history(self, _mock_names) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "005930.KS",
                    "asset_type": "한국주식",
                    "side": "buy",
                    "requested_qty": 3,
                    "filled_qty": 3,
                    "requested_price": 70000.0,
                    "status": "filled",
                    "reason": "entry",
                    "updated_at": "2026-03-09T01:05:00Z",
                },
                {
                    "symbol": "BTC-USD",
                    "asset_type": "코인",
                    "side": "sell",
                    "requested_qty": 1,
                    "filled_qty": 0,
                    "requested_price": 90000.0,
                    "status": "pending_fill",
                    "reason": "take_profit",
                    "updated_at": "2026-03-09T01:06:00Z",
                },
            ]
        )

        view = build_recent_order_activity_view(orders)

        self.assertEqual(view.loc[0, "종목"], "삼성전자 (005930)")
        self.assertEqual(view.loc[0, "주문"], "매수")
        self.assertEqual(view.loc[0, "상태"], "체결완료")
        self.assertEqual(view.loc[0, "사유"], "신규진입")
        self.assertEqual(view.loc[0, "주문가"], "70,000 KRW")
        self.assertEqual(view.loc[1, "종목"], "BTC-USD")
        self.assertEqual(view.loc[1, "주문"], "매도")
        self.assertEqual(view.loc[1, "상태"], "체결대기")
        self.assertEqual(view.loc[1, "사유"], "익절")
        self.assertEqual(view.loc[1, "주문가"], "90,000.00 USD")

    def test_build_recent_order_activity_view_returns_empty_when_no_orders(self) -> None:
        view = build_recent_order_activity_view(pd.DataFrame())
        self.assertTrue(view.empty)

    @patch("app.load_krx_symbol_name_map", return_value={"005930": "삼성전자", "005930.KS": "삼성전자"})
    def test_build_recent_realized_trades_view_formats_kr_display_name(self, _mock_names) -> None:
        trades = pd.DataFrame(
            [
                {
                    "closed_at": "2026-03-09T01:15:00Z",
                    "created_at": "2026-03-09T01:00:00Z",
                    "account_id": "kis_kr_paper",
                    "symbol": "005930",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "entry_price": 70000.0,
                    "mark_price": 71000.0,
                    "realized_pnl": 1000.0,
                    "notes": "closed_by_take_profit",
                }
            ]
        )

        view = build_recent_realized_trades_view(trades)

        self.assertEqual(view.loc[0, "종목"], "삼성전자 (005930)")
        self.assertEqual(view.loc[0, "실현손익"], "1,000 KRW")

    @patch("app.load_krx_symbol_name_map", return_value={"373220": "LG에너지솔루션", "373220.KS": "LG에너지솔루션"})
    def test_build_monitor_table_view_formats_kr_symbol_column(self, _mock_names) -> None:
        frame = pd.DataFrame(
            [
                {"symbol": "373220.KS", "asset_type": "한국주식", "status": "filled"},
                {"symbol": "AAPL", "asset_type": "미국주식", "status": "filled"},
            ]
        )

        view = build_monitor_table_view(frame)

        self.assertEqual(view.loc[0, "종목"], "LG에너지솔루션 (373220)")
        self.assertEqual(view.loc[1, "종목"], "AAPL")


if __name__ == "__main__":
    unittest.main()
