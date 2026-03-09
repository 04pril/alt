from __future__ import annotations

import unittest

import pandas as pd

from beta_monitor_clone import (
    _account_card_compact,
    _build_entry_result_rows,
    _candidate_tabs_html,
    _equity_svg,
    _ensure_template_base_href,
    _money_display_pair,
    _replace_template_script,
)


class BetaMonitorCloneTest(unittest.TestCase):
    def test_ensure_template_base_href_inserts_root_base_tag(self) -> None:
        template = "<html><head><title>beta</title></head><body></body></html>"

        markup = _ensure_template_base_href(template, "/")

        self.assertIn('<base href="/">', markup)
        self.assertEqual(markup.count("<base "), 1)

    def test_replace_template_script_swaps_original_template_script(self) -> None:
        template = """<html><body><div>content</div><script>window.template = true;</script></body></html>"""

        markup = _replace_template_script(template, "<script>window.injected = true;</script>")

        self.assertIn("window.injected = true;", markup)
        self.assertNotIn("window.template = true;", markup)
        self.assertEqual(markup.count("<script"), 1)

    def test_build_entry_result_rows_filters_non_symbol_noise(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:04:55Z",
                    "component": "scan_job",
                    "event_type": "scan_complete",
                    "message": "scan done",
                    "details": {"count": 12},
                },
                {
                    "created_at": "2026-03-09T14:03:40Z",
                    "component": "execution_pipeline",
                    "event_type": "noop",
                    "message": "noop",
                    "details": {"reason": "outside_preclose_window"},
                },
                {
                    "created_at": "2026-03-09T14:02:40Z",
                    "component": "execution_pipeline",
                    "event_type": "entry_rejected",
                    "message": "rejected",
                    "details": {"symbol": "ETH-USD", "reason": "insufficient_buying_power"},
                },
                {
                    "created_at": "2026-03-09T14:01:40Z",
                    "component": "kis_execution",
                    "event_type": "filled",
                    "message": "filled",
                    "details": {"symbol": "005930.KS", "expected_return": 0.0125, "confidence": 0.61},
                },
            ]
        )

        rows = _build_entry_result_rows(events, limit=4)

        self.assertEqual(len(rows), 2)
        self.assertIn("ETH-USD", rows[0])
        self.assertIn("진입 거절", rows[0])
        self.assertIn("주문 가능 금액 부족", rows[0])
        self.assertIn("005930.KS", rows[1])
        self.assertIn("체결 완료", rows[1])
        self.assertNotIn("scan_complete", "".join(rows))
        self.assertNotIn("execution_pipeline", "".join(rows))

    def test_equity_svg_uses_responsive_viewbox_markup(self) -> None:
        curve = pd.DataFrame(
            [
                {"created_at": "2026-03-09T14:00:00Z", "equity": 1000000.0},
                {"created_at": "2026-03-09T14:05:00Z", "equity": 1012500.0},
                {"created_at": "2026-03-09T14:10:00Z", "equity": 1007500.0},
            ]
        )

        markup = _equity_svg(curve, theme_mode="dark")

        self.assertIn('preserveAspectRatio="xMidYMid meet"', markup)
        self.assertIn("최근 평가 자산", markup)

    def test_candidate_tabs_split_markets_and_show_kr_name(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:10:00Z",
                    "symbol": "005930.KS",
                    "asset_type": "kr_stock",
                    "signal": "LONG",
                    "expected_return": 0.012,
                    "confidence": 0.71,
                    "score": 3.4,
                },
                {
                    "created_at": "2026-03-09T14:09:00Z",
                    "symbol": "NVDA",
                    "asset_type": "us_stock",
                    "signal": "LONG",
                    "expected_return": 0.018,
                    "confidence": 0.64,
                    "score": 2.8,
                },
                {
                    "created_at": "2026-03-09T14:08:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "signal": "SHORT",
                    "expected_return": -0.006,
                    "confidence": 0.58,
                    "score": 1.7,
                },
            ]
        )

        markup = _candidate_tabs_html(
            candidates,
            kr_asset_types={"kr_stock"},
            kr_symbol_names={"005930.KS": "삼성전자"},
            current_tab="us",
            jobs="all",
        )

        self.assertIn("국내주식", markup)
        self.assertIn("해외주식", markup)
        self.assertIn("코인", markup)
        self.assertIn("삼성전자", markup)
        self.assertIn("005930", markup)
        self.assertIn("NVDA", markup)
        self.assertIn("BTC-USD", markup)
        self.assertIn('type="button"', markup)
        self.assertIn('data-cand-tab="us"', markup)
        self.assertNotIn("beta_cand_tab=us", markup)

    def test_money_display_pair_converts_usd_to_krw_when_fx_available(self) -> None:
        primary, secondary = _money_display_pair(
            100.0,
            "USD",
            {"KRW=X": {"current_price": 1450.0}},
        )

        self.assertEqual(primary, "145,000.00 KRW")
        self.assertEqual(secondary, "100.00 USD")

    def test_account_card_compact_uses_unrealized_pnl_in_current_pnl(self) -> None:
        markup = _account_card_compact(
            {
                "equity": 21000000.0,
                "cash": 9000000.0,
                "daily_pnl": 0.0,
                "unrealized_pnl": -125000.0,
                "gross_exposure": 11000000.0,
                "created_at": "2026-03-09T14:05:00Z",
            },
            {},
            {"today_pnl": 0.0},
            "SIM",
            "미국주식 · 코인",
            "sim",
        )

        self.assertIn("현재 손익", markup)
        self.assertIn("-12만", markup)

    def test_account_card_compact_converts_usd_account_to_krw_primary_display(self) -> None:
        markup = _account_card_compact(
            {
                "equity": 1000.0,
                "cash": 800.0,
                "daily_pnl": 10.0,
                "unrealized_pnl": 5.0,
                "gross_exposure": 200.0,
                "created_at": "2026-03-09T14:05:00Z",
            },
            {},
            {"today_pnl": 0.0},
            "SIM",
            "미국주식 계좌",
            "sim",
            currency="USD",
            quote_snapshots={"KRW=X": {"current_price": 1450.0}},
        )

        self.assertIn("145만", markup)
        self.assertIn("1,000.00 USD", markup)


if __name__ == "__main__":
    unittest.main()
