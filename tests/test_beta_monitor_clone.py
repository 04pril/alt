from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import beta_monitor_clone as clone
from beta_monitor_clone import (
    _account_card_compact,
    _action_button,
    _build_kr_symbol_name_map,
    _build_total_equity_curve,
    _build_realized_trade_table_frame,
    _build_entry_result_rows,
    _candidate_tabs_html,
    _equity_svg,
    _ensure_template_base_href,
    _fmt_time,
    _frame_for_account,
    _job_label,
    _money_display_pair,
    _positions_card,
    _replace_template_script,
    _theme_button,
    build_beta_live_payload,
    render_beta_live_payload_host,
)
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY


class BetaMonitorCloneTest(unittest.TestCase):
    def test_fmt_time_localizes_utc_to_kst(self) -> None:
        self.assertEqual(_fmt_time("2026-03-10T06:59:52.164629Z"), "2026-03-10 15:59:52")

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
        self.assertIn("005930", rows[1])
        self.assertIn("체결 완료", rows[1])
        self.assertNotIn("scan_complete", "".join(rows))
        self.assertNotIn("execution_pipeline", "".join(rows))

    def test_build_entry_result_rows_formats_kr_symbol_name(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:01:40Z",
                    "component": "kis_execution",
                    "event_type": "filled",
                    "message": "filled",
                    "details": {"symbol": "005930.KS", "expected_return": 0.0125, "confidence": 0.61},
                }
            ]
        )

        rows = _build_entry_result_rows(
            events,
            kr_symbol_names={"005930": "삼성전자", "005930.KS": "삼성전자"},
            limit=4,
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("삼성전자", rows[0])
        self.assertIn("005930", rows[0])

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
        self.assertIn('class="equity-area"', markup)
        self.assertIn('class="equity-line"', markup)
        self.assertIn("최근 평가 자산", markup)

    def test_build_total_equity_curve_converts_usd_series_to_krw(self) -> None:
        total_curve, note, currency = _build_total_equity_curve(
            {
                ACCOUNT_KIS_KR_PAPER: pd.DataFrame(
                    [
                        {"created_at": "2026-03-09T14:00:00Z", "equity": 30000000.0},
                        {"created_at": "2026-03-09T14:05:00Z", "equity": 30100000.0},
                    ]
                ),
                ACCOUNT_SIM_US_EQUITY: pd.DataFrame(
                    [
                        {"created_at": "2026-03-09T14:00:00Z", "equity": 1000.0},
                        {"created_at": "2026-03-09T14:05:00Z", "equity": 1010.0},
                    ]
                ),
            },
            {"KRW=X": {"current_price": 1450.0}},
        )

        self.assertFalse(total_curve.empty)
        self.assertEqual(currency, "KRW")
        self.assertIn("KRW 환산", note)
        self.assertAlmostEqual(float(total_curve.iloc[-1]["equity"]), 30100000.0 + 1010.0 * 1450.0)

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

    def test_candidate_tabs_hide_older_candidate_when_latest_decision_is_rejected(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:11:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "status": "rejected",
                    "signal": "FLAT",
                    "expected_return": float("nan"),
                    "confidence": 0.0,
                    "score": -999.0,
                },
                {
                    "created_at": "2026-03-09T14:10:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "status": "candidate",
                    "signal": "LONG",
                    "expected_return": 0.014,
                    "confidence": 0.62,
                    "score": 2.1,
                },
            ]
        )

        markup = _candidate_tabs_html(
            candidates,
            kr_asset_types={"한국주식"},
            kr_symbol_names={},
            current_tab="crypto",
            jobs="all",
        )

        self.assertNotIn("+1.40%", markup)
        self.assertNotIn(">N/A<", markup)
        self.assertIn("최근 스캔 실패 1건", markup)

    def test_candidate_tabs_keep_latest_candidate_for_same_symbol(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:09:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "status": "candidate",
                    "signal": "SHORT",
                    "expected_return": -0.009,
                    "confidence": 0.57,
                    "score": 1.5,
                },
                {
                    "created_at": "2026-03-09T14:10:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "status": "candidate",
                    "signal": "LONG",
                    "expected_return": 0.014,
                    "confidence": 0.62,
                    "score": 2.1,
                },
            ]
        )

        markup = _candidate_tabs_html(
            candidates,
            kr_asset_types={"한국주식"},
            kr_symbol_names={},
            current_tab="crypto",
            jobs="all",
        )

        self.assertIn("+1.40%", markup)
        self.assertNotIn("-0.90%", markup)

    def test_candidate_tabs_show_kr_name_for_bare_code(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:10:00Z",
                    "symbol": "005930",
                    "asset_type": "한국주식",
                    "signal": "LONG",
                    "expected_return": 0.012,
                    "confidence": 0.71,
                    "score": 3.4,
                }
            ]
        )

        markup = _candidate_tabs_html(
            candidates,
            kr_asset_types={"한국주식"},
            kr_symbol_names={"005930": "삼성전자"},
            current_tab="kr",
            jobs="all",
        )

        self.assertIn("삼성전자", markup)
        self.assertIn("005930", markup)

    def test_candidate_tabs_hide_error_rows_without_expected_return(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T14:10:00Z",
                    "symbol": "BTC-USD",
                    "asset_type": "코인",
                    "status": "error",
                    "signal": "FLAT",
                    "expected_return": float("nan"),
                    "confidence": 0.0,
                    "score": -999.0,
                    "reason": "시세 데이터가 비어 있습니다: BTC-USD 1h",
                }
            ]
        )

        markup = _candidate_tabs_html(
            candidates,
            kr_asset_types={"한국주식"},
            kr_symbol_names={},
            current_tab="crypto",
            jobs="all",
        )

        self.assertNotIn(">N/A<", markup)
        self.assertIn("최근 스캔 실패 1건", markup)
        self.assertIn("시세 데이터가 비어 있습니다", markup)

    def test_build_kr_symbol_name_map_includes_manual_kr_etf_aliases(self) -> None:
        symbol_map = _build_kr_symbol_name_map(
            {"삼성전자": {"market": "유가", "code": "005930"}}
        )

        self.assertEqual(symbol_map.get("005930"), "삼성전자")
        self.assertEqual(symbol_map.get("411060"), "ACE KRX금현물")
        self.assertEqual(symbol_map.get("411060.KS"), "ACE KRX금현물")

    def test_action_button_uses_button_markup_with_beta_href(self) -> None:
        markup = _action_button(
            "계좌 동기화",
            "sync_account",
            "btn-mini",
            "sync",
            token="token1",
            candidate_tab="crypto",
            jobs="all",
            signals="all",
            theme_mode="dark",
        )

        self.assertIn('type="button"', markup)
        self.assertIn('data-beta-href="/beta?beta_anchor=sync&amp;beta_action=sync_account&amp;beta_token=token1&amp;beta_cand_tab=crypto&amp;beta_jobs=all&amp;beta_signals=all&amp;beta_theme=dark"', markup)
        self.assertIn('data-action="sync_account"', markup)
        self.assertNotIn('target="_top"', markup)

    def test_theme_button_uses_button_markup_with_toggle_href(self) -> None:
        markup = _theme_button(
            "light",
            "beta-overview",
            token="theme1",
            candidate_tab="us",
            jobs=None,
            signals="all",
        )

        self.assertIn('type="button"', markup)
        self.assertIn('data-beta-href="/beta?beta_anchor=beta-overview&amp;beta_action=toggle_theme&amp;beta_token=theme1&amp;beta_cand_tab=us&amp;beta_signals=all&amp;beta_theme=dark"', markup)
        self.assertIn('data-theme-toggle="1"', markup)
        self.assertIn('id="theme-label"', markup)
        self.assertNotIn('target="_top"', markup)

    def test_job_label_maps_scan_prefix(self) -> None:
        self.assertEqual(_job_label("scan:코인"), "시그널 스캔 · 코인")
        self.assertEqual(_job_label("signal_scan"), "시그널 스캔")

    def test_frame_for_account_falls_back_to_execution_account_id(self) -> None:
        frame = pd.DataFrame(
            [
                {"execution_account_id": ACCOUNT_SIM_US_EQUITY, "symbol": "AAPL"},
                {"execution_account_id": ACCOUNT_SIM_CRYPTO, "symbol": "BTC-USD"},
            ]
        )

        filtered = _frame_for_account(frame, ACCOUNT_SIM_CRYPTO)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(str(filtered.iloc[0]["symbol"]), "BTC-USD")

    def test_money_display_pair_converts_usd_to_krw_when_fx_available(self) -> None:
        primary, secondary = _money_display_pair(
            100.0,
            "USD",
            {"KRW=X": {"current_price": 1450.0}},
        )

        self.assertEqual(primary, "₩145,000")
        self.assertEqual(secondary, "$100.00")

    def test_build_realized_trade_table_frame_formats_trade_history(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "created_at": "2026-03-09T13:00:00Z",
                    "closed_at": "2026-03-09T14:15:00Z",
                    "account_id": ACCOUNT_SIM_US_EQUITY,
                    "symbol": "AAPL",
                    "side": "LONG",
                    "entry_price": 100.0,
                    "mark_price": 105.0,
                    "realized_pnl": 5.0,
                    "notes": "closed_by_take_profit",
                }
            ]
        )

        view = _build_realized_trade_table_frame(
            trades,
            {"KRW=X": {"current_price": 1450.0}},
            kr_symbol_names={"005930": "삼성전자", "005930.KS": "삼성전자"},
        )

        self.assertEqual(list(view.columns), ["종료시각", "계좌", "종목", "방향", "보유기간", "진입가", "청산가", "실현손익", "청산사유"])
        self.assertEqual(view.iloc[0]["계좌"], "미장")
        self.assertEqual(view.iloc[0]["방향"], "롱")
        self.assertIn("₩145,000 / $100.00", view.iloc[0]["진입가"])
        self.assertEqual(view.iloc[0]["청산사유"], "익절")

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
            "미장",
            "sim",
            currency="USD",
            quote_snapshots={"KRW=X": {"current_price": 1450.0}},
        )

        self.assertIn("145만", markup)
        self.assertIn("$1,000.00", markup)

    def test_positions_card_uses_krw_quote_for_bare_kr_code(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "005930",
                    "asset_type": "한국주식",
                    "side": "LONG",
                    "quantity": 1,
                    "entry_price": 189900.0,
                    "mark_price": 189900.0,
                    "unrealized_pnl": -94.95,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                }
            ]
        )

        markup = _positions_card(
            frame,
            "국장 포지션",
            "국장",
            "broker-kis",
            30000000.0,
            {"005930.KS": {"currency": "KRW", "current_price": 190000.0}},
            {"한국주식"},
            kr_symbol_names={"005930": "삼성전자", "005930.KS": "삼성전자"},
        )

        self.assertIn("삼성전자", markup)
        self.assertIn("190,000 KRW", markup)
        self.assertIn("현재가 190,000 KRW", markup)
        self.assertIn("100 KRW", markup)
        self.assertIn("- / -", markup)
        self.assertNotIn("USD", markup)

    def test_render_beta_overview_component_uses_light_theme_and_compact_detail_stack(self) -> None:
        template = """<!DOCTYPE html><html lang="ko" data-theme="dark"><head><style></style></head><body>
<nav class="top-nav"></nav>
<div class="status-strip"></div>
<div class="main">
  <div class="account-row"></div>
  <div class="stat-bar"></div>
  <div class="content-grid"></div>
  <div class="bottom-grid"></div>
</div><!-- /main -->
<script>window.template = true;</script>
</body></html>"""
        data = {
            "summary": {"latest_account": {}},
            "auto_trading_status": {"state": "running", "label": "가동 중"},
            "execution_summary": {"today_noop_breakdown": pd.DataFrame()},
            "broker_sync_status": pd.DataFrame(),
            "broker_sync_errors": pd.DataFrame(),
            "kis_runtime": {},
            "runtime_profile": {"name": "active"},
            "job_health": pd.DataFrame(),
            "recent_errors": pd.DataFrame(),
            "recent_events": pd.DataFrame(),
            "open_positions": pd.DataFrame(),
            "open_orders": pd.DataFrame(),
            "candidate_scans": pd.DataFrame(),
            "prediction_report": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "equity_curves_by_account": {
                ACCOUNT_KIS_KR_PAPER: pd.DataFrame([{"created_at": "2026-03-09T14:00:00Z", "equity": 30000000.0}]),
                ACCOUNT_SIM_US_EQUITY: pd.DataFrame([{"created_at": "2026-03-09T14:00:00Z", "equity": 1000.0}]),
            },
            "today_execution_events": pd.DataFrame(),
            "recent_realized_trades": pd.DataFrame(),
            "asset_overview": pd.DataFrame(),
        }
        accounts_overview = {
            ACCOUNT_KIS_KR_PAPER: {"currency": "KRW", "latest_snapshot": {"equity": 30000000.0, "cash": 30000000.0}},
            ACCOUNT_SIM_US_EQUITY: {"currency": "USD", "latest_snapshot": {"equity": 21000.0, "cash": 21000.0}},
            ACCOUNT_SIM_CRYPTO: {"currency": "USD", "latest_snapshot": {"equity": 21000.0, "cash": 21000.0}},
        }
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "beta_template.html"
            template_path.write_text(template, encoding="utf-8")
            original_path = clone.TEMPLATE_PATH
            original_html = clone.components.html
            try:
                clone.TEMPLATE_PATH = template_path
                clone.components.html = lambda html, height, scrolling: captured.update(
                    {"html": html, "height": height, "scrolling": scrolling}
                )
                clone.render_beta_overview_component(
                    data=data,
                    theme_mode="light",
                    initial_anchor="beta-overview",
                    feedback=None,
                    accounts_overview=accounts_overview,
                    total_portfolio_overview={},
                    quote_snapshots={"KRW=X": {"current_price": 1450.0}},
                    kr_asset_types={"한국주식"},
                    recent_orders=pd.DataFrame(),
                    current_candidate_tab="us",
                    jobs_expanded=False,
                )
            finally:
                clone.TEMPLATE_PATH = original_path
                clone.components.html = original_html

        markup = str(captured.get("html") or "")
        self.assertIn('data-theme="light"', markup)
        self.assertIn('data-theme-toggle="1"', markup)
        self.assertIn("alt-beta-theme", markup)
        self.assertIn("applyTheme(readStoredTheme())", markup)
        self.assertIn("beta_theme=light", markup)
        self.assertIn("window.parent.scrollTo", markup)
        self.assertIn("window.parent.addEventListener(\"scroll\", parentScrollHandler", markup)
        self.assertIn('id="beta-live-status-strip"', markup)
        self.assertIn('id="beta-live-account-row"', markup)
        self.assertIn('id="beta-live-positions"', markup)
        self.assertIn("startLivePayloadPolling()", markup)
        self.assertIn("applyLivePayload(readLivePayload())", markup)
        self.assertNotIn("const autoRefreshMs = 5000;", markup)
        self.assertNotIn("beta_live_tick", markup)
        self.assertNotIn("window.parent.location.replace", markup)
        self.assertIn("detail-events-grid", markup)
        self.assertIn('id="errors"', markup)
        self.assertIn("최근 거래 손익", markup)
        self.assertIn("KRW 환산 합산", markup)
        self.assertIn("KR 전략", markup)
        self.assertIn('data-beta-href="/beta?beta_anchor=beta-overview', markup)
        self.assertNotIn("window.template = true;", markup)
        self.assertIs(captured.get("scrolling"), False)

    def test_render_beta_overview_component_renders_kr_strategy_summary(self) -> None:
        template = """<!DOCTYPE html><html lang="ko" data-theme="dark"><head><style></style></head><body>
<nav class="top-nav"></nav>
<div class="status-strip"></div>
<div class="main">
  <div class="account-row"></div>
  <div class="stat-bar"></div>
  <div class="content-grid"></div>
  <div class="bottom-grid"></div>
</div><!-- /main -->
<script>window.template = true;</script>
</body></html>"""
        data = {
            "summary": {"latest_account": {}},
            "auto_trading_status": {"state": "running", "label": "가동 중"},
            "execution_summary": {"today_noop_breakdown": pd.DataFrame()},
            "broker_sync_status": pd.DataFrame(),
            "broker_sync_errors": pd.DataFrame(),
            "kis_runtime": {},
            "runtime_profile": {
                "name": "balanced",
                "kr_default_strategy_label": "KR 1시간봉 v1",
                "kr_default_strategy_session_mode": "regular",
                "kr_recommended_strategy_label": "KR 1시간봉 v1",
                "kr_active_strategy_labels": "KR 1시간봉 v1",
            },
            "job_health": pd.DataFrame(),
            "recent_errors": pd.DataFrame(),
            "recent_events": pd.DataFrame(),
            "open_positions": pd.DataFrame(),
            "open_orders": pd.DataFrame(),
            "candidate_scans": pd.DataFrame(),
            "prediction_report": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "equity_curves_by_account": {},
            "today_execution_events": pd.DataFrame(),
            "recent_realized_trades": pd.DataFrame(),
            "asset_overview": pd.DataFrame(),
            "kr_strategy_overview": pd.DataFrame(
                [
                    {
                        "strategy_id": "kr_intraday_1h_v1",
                        "label": "KR 1시간봉 v1",
                        "session_mode": "regular",
                        "timeframe": "1h",
                        "enabled": True,
                        "experimental": False,
                        "broker_mode": "kis_mock",
                        "execution_account_id": ACCOUNT_KIS_KR_PAPER,
                        "execution_cadence": "1h 완성봉 기준",
                        "today_candidate_count": 3,
                        "today_submitted_count": 1,
                        "today_filled_count": 1,
                    },
                    {
                        "strategy_id": "kr_intraday_15m_v1",
                        "label": "KR 15분봉 정규장 v1",
                        "session_mode": "regular",
                        "timeframe": "15m",
                        "enabled": False,
                        "experimental": True,
                        "broker_mode": "kis_mock",
                        "execution_account_id": ACCOUNT_KIS_KR_PAPER,
                        "execution_cadence": "15m 완성봉 기준",
                        "today_candidate_count": 0,
                        "today_submitted_count": 0,
                        "today_filled_count": 0,
                    },
                    {
                        "strategy_id": "kr_intraday_15m_v1_after_close_single",
                        "label": "KR 15m After-close Single v1",
                        "session_mode": "after_close_single",
                        "timeframe": "15m",
                        "enabled": False,
                        "experimental": True,
                        "broker_mode": "kis_mock",
                        "execution_account_id": ACCOUNT_KIS_KR_PAPER,
                        "execution_cadence": "10분 단일가 경매",
                        "today_candidate_count": 0,
                        "today_submitted_count": 0,
                        "today_filled_count": 0,
                    },
                ]
            ),
            "kr_strategy_recent_events": pd.DataFrame(
                [
                    {
                        "created_at": "2026-03-09T14:00:00Z",
                        "strategy_id": "kr_intraday_1h_v1",
                        "symbol": "005930.KS",
                        "event_type": "filled",
                        "reason": "",
                        "message": "filled",
                    }
                ]
            ),
        }
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "beta_template.html"
            template_path.write_text(template, encoding="utf-8")
            original_path = clone.TEMPLATE_PATH
            original_html = clone.components.html
            try:
                clone.TEMPLATE_PATH = template_path
                clone.components.html = lambda html, height, scrolling: captured.update(
                    {"html": html, "height": height, "scrolling": scrolling}
                )
                clone.render_beta_overview_component(
                    data=data,
                    theme_mode="dark",
                    initial_anchor="beta-overview",
                    feedback=None,
                    accounts_overview={},
                    total_portfolio_overview={},
                    quote_snapshots={},
                    kr_asset_types={"한국주식"},
                    recent_orders=pd.DataFrame(),
                )
            finally:
                clone.TEMPLATE_PATH = original_path
                clone.components.html = original_html

        markup = str(captured.get("html") or "")
        self.assertIn("KR 1시간봉 v1", markup)
        self.assertIn("KR 15분봉 정규장 v1", markup)
        self.assertIn("기본 추천 KR 1시간봉 v1", markup)
        self.assertIn("세션 regular", markup)
        self.assertIn("10분 단일가 경매", markup)
        self.assertIn("auction experimental", markup)
        self.assertIn("experimental", markup)
        self.assertIs(captured.get("scrolling"), False)

    def test_render_beta_overview_component_wires_us_strategy_buttons(self) -> None:
        template = """<!DOCTYPE html><html lang="ko" data-theme="dark"><head><style></style></head><body>
<nav class="top-nav"></nav>
<div class="status-strip"></div>
<div class="main">
  <div class="account-row"></div>
  <div class="stat-bar"></div>
  <div class="content-grid"></div>
  <div class="bottom-grid"></div>
</div><!-- /main -->
<script>window.template = true;</script>
</body></html>"""
        data = {
            "summary": {"latest_account": {}},
            "auto_trading_status": {"state": "running", "label": "가동 중"},
            "execution_summary": {"today_noop_breakdown": pd.DataFrame()},
            "broker_sync_status": pd.DataFrame(),
            "broker_sync_errors": pd.DataFrame(),
            "kis_runtime": {},
            "runtime_profile": {
                "name": "active",
                "kr_default_strategy_label": "KR 15분봉 정규장 v1",
                "kr_default_strategy_id": "kr_intraday_15m_v1",
                "kr_default_strategy_session_mode": "regular",
                "kr_recommended_strategy_label": "KR 1시간봉 v1",
                "kr_active_strategy_labels": "KR 15분봉 정규장 v1",
                "us_default_strategy_label": "US Combo 15m AHC Regular v1",
                "us_default_strategy_id": "us_combo_15m_ahc_regular_v1",
            },
            "job_health": pd.DataFrame(),
            "recent_errors": pd.DataFrame(),
            "recent_events": pd.DataFrame(),
            "open_positions": pd.DataFrame(),
            "open_orders": pd.DataFrame(),
            "candidate_scans": pd.DataFrame(),
            "prediction_report": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "equity_curves_by_account": {},
            "today_execution_events": pd.DataFrame(),
            "recent_realized_trades": pd.DataFrame(),
            "asset_overview": pd.DataFrame(),
            "kr_strategy_overview": pd.DataFrame(
                [
                    {
                        "strategy_id": "us_combo_15m_ahc_regular_v1",
                        "label": "US Combo 15m AHC Regular v1",
                        "session_mode": "regular",
                        "timeframe": "15m",
                        "enabled": True,
                        "experimental": True,
                        "broker_mode": "sim",
                        "execution_account_id": ACCOUNT_SIM_US_EQUITY,
                        "execution_cadence": "15m 완성봉 기준",
                        "today_candidate_count": 2,
                        "today_submitted_count": 1,
                        "today_filled_count": 0,
                    },
                    {
                        "strategy_id": "us_combo_15m_ahc_afterhours_v1",
                        "label": "US Combo 15m AHC Afterhours v1",
                        "session_mode": "after_close_close",
                        "timeframe": "15m",
                        "enabled": True,
                        "experimental": True,
                        "broker_mode": "sim",
                        "execution_account_id": ACCOUNT_SIM_US_EQUITY,
                        "execution_cadence": "15m 완성봉 기준",
                        "today_candidate_count": 1,
                        "today_submitted_count": 0,
                        "today_filled_count": 0,
                    },
                ]
            ),
            "kr_strategy_recent_events": pd.DataFrame(),
        }
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "beta_template.html"
            template_path.write_text(template, encoding="utf-8")
            original_path = clone.TEMPLATE_PATH
            original_html = clone.components.html
            try:
                clone.TEMPLATE_PATH = template_path
                clone.components.html = lambda html, height, scrolling: captured.update(
                    {"html": html, "height": height, "scrolling": scrolling}
                )
                clone.render_beta_overview_component(
                    data=data,
                    theme_mode="dark",
                    initial_anchor="beta-overview",
                    feedback=None,
                    accounts_overview={},
                    total_portfolio_overview={},
                    quote_snapshots={},
                    kr_asset_types={"한국주식"},
                    recent_orders=pd.DataFrame(),
                )
            finally:
                clone.TEMPLATE_PATH = original_path
                clone.components.html = original_html

        markup = str(captured.get("html") or "")
        self.assertIn("US 전략", markup)
        self.assertIn("US Combo 15m AHC Regular v1", markup)
        self.assertIn("US Combo 15m AHC Afterhours v1", markup)
        self.assertIn('set_strategy:us_combo_15m_ahc_afterhours_v1', markup)
        self.assertIn("기본</span>", markup)

    def test_render_beta_overview_component_shows_signal_toggle_when_more_than_four_results(self) -> None:
        template = """<!DOCTYPE html><html lang="ko" data-theme="dark"><head><style></style></head><body>
<nav class="top-nav"></nav>
<div class="status-strip"></div>
<div class="main">
  <div class="account-row"></div>
  <div class="stat-bar"></div>
  <div class="content-grid"></div>
  <div class="bottom-grid"></div>
</div><!-- /main -->
<script>window.template = true;</script>
</body></html>"""
        events = pd.DataFrame(
            [
                {
                    "created_at": f"2026-03-09T14:0{i}:00Z",
                    "component": "execution_pipeline",
                    "event_type": "entry_rejected",
                    "message": "rejected",
                    "details": {"symbol": f"ASSET{i}", "reason": "insufficient_buying_power"},
                }
                for i in range(5)
            ]
        )
        data = {
            "summary": {"latest_account": {}},
            "auto_trading_status": {"state": "running", "label": "가동 중"},
            "execution_summary": {"today_noop_breakdown": pd.DataFrame(), "today_entry_rejected_count": 5},
            "broker_sync_status": pd.DataFrame(),
            "broker_sync_errors": pd.DataFrame(),
            "kis_runtime": {},
            "runtime_profile": {"name": "active"},
            "job_health": pd.DataFrame(),
            "recent_errors": pd.DataFrame(),
            "recent_events": pd.DataFrame(),
            "open_positions": pd.DataFrame(),
            "open_orders": pd.DataFrame(),
            "candidate_scans": pd.DataFrame(),
            "prediction_report": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "equity_curves_by_account": {},
            "today_execution_events": events,
            "recent_realized_trades": pd.DataFrame(),
            "asset_overview": pd.DataFrame(),
        }
        accounts_overview = {}
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "beta_template.html"
            template_path.write_text(template, encoding="utf-8")
            original_path = clone.TEMPLATE_PATH
            original_html = clone.components.html
            try:
                clone.TEMPLATE_PATH = template_path
                clone.components.html = lambda html, height, scrolling: captured.update(
                    {"html": html, "height": height, "scrolling": scrolling}
                )
                clone.render_beta_overview_component(
                    data=data,
                    theme_mode="light",
                    initial_anchor="events",
                    feedback=None,
                    accounts_overview=accounts_overview,
                    total_portfolio_overview={},
                    quote_snapshots={},
                    kr_asset_types={"한국주식"},
                    recent_orders=pd.DataFrame(),
                    current_candidate_tab="us",
                    jobs_expanded=False,
                    signals_expanded=False,
                )
            finally:
                clone.TEMPLATE_PATH = original_path
                clone.components.html = original_html

        markup = str(captured.get("html") or "")
        self.assertIn("표시 4 / 전체 5건", markup)
        self.assertIn("상세보기", markup)
        self.assertIn('data-toggle-target="signal-list"', markup)

    def test_build_beta_live_payload_and_host_include_live_sections(self) -> None:
        payload = build_beta_live_payload(
            data={
                "summary": {"latest_account": {}},
                "auto_trading_status": {"state": "running", "label": "가동 중"},
                "broker_sync_errors": pd.DataFrame(),
                "kis_runtime": {"last_broker_account_sync": "2026-03-10T05:00:00Z"},
                "job_health": pd.DataFrame(),
                "open_positions": pd.DataFrame(
                    [
                        {
                            "account_id": ACCOUNT_KIS_KR_PAPER,
                            "symbol": "005930",
                            "asset_type": "한국주식",
                            "side": "LONG",
                            "quantity": 1,
                            "entry_price": 187000.0,
                            "mark_price": 187300.0,
                            "unrealized_pnl": 300.0,
                        }
                    ]
                ),
            },
            accounts_overview={
                ACCOUNT_KIS_KR_PAPER: {
                    "currency": "KRW",
                    "latest_snapshot": {"equity": 30000000.0, "cash": 29813000.0},
                }
            },
            quote_snapshots={"005930": {"currency": "KRW", "current_price": 187300.0}},
            kr_asset_types={"한국주식"},
            krx_name_map={"삼성전자": {"code": "005930", "market": "KOSPI"}},
        )

        self.assertIn('id="beta-live-status-strip"', payload["status_strip_html"])
        self.assertIn('id="beta-live-account-row"', payload["account_row_html"])
        self.assertIn('id="beta-live-positions"', payload["positions_html"])

        host_markup = render_beta_live_payload_host(payload)
        self.assertIn("alt-beta-live-payload", host_markup)
        self.assertIn("localStorage.setItem", host_markup)
        self.assertIn('\\"version\\"', host_markup)


if __name__ == "__main__":
    unittest.main()
