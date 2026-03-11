from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import RuntimeSettings
from kr_strategy import active_kr_strategy_ids, active_us_strategy_ids, default_kr_strategy, default_us_strategy, entry_gate_reason, flatten_due, iter_visible_strategy_rows, strategy_conflict_ids, strategy_schedule


class KRStrategyTest(unittest.TestCase):
    def test_default_kr_strategy_is_1h_and_15m_is_disabled(self) -> None:
        settings = RuntimeSettings()

        default_strategy = default_kr_strategy(settings)

        self.assertIsNotNone(default_strategy)
        self.assertEqual(str(default_strategy.strategy_id), "kr_intraday_1h_v1")
        self.assertTrue(bool(settings.kr_strategies["kr_intraday_1h_v1"].enabled))
        self.assertFalse(bool(settings.kr_strategies["kr_intraday_15m_v1"].enabled))
        self.assertFalse(bool(settings.kr_strategies["kr_intraday_15m_v1_auto"].enabled))
        self.assertTrue(bool(settings.kr_strategies["kr_intraday_15m_v1"].experimental))
        self.assertFalse(bool(settings.kr_strategies["kr_intraday_15m_v1_after_close_close"].enabled))
        self.assertFalse(bool(settings.kr_strategies["kr_intraday_15m_v1_after_close_single"].enabled))

    def test_active_kr_strategy_ids_ignore_non_kr_schedule_entries(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_1h_v1"].enabled = False
        settings.kr_strategies["kr_intraday_15m_v1"].enabled = True
        settings.kr_strategies["us_combo_15m_ahc_regular_v1"].enabled = True
        settings.kr_strategies["us_combo_15m_ahc_afterhours_v1"].enabled = True
        settings.kr_strategies["kr_combo_15m_ahc_afterclose_v2"].enabled = True

        self.assertEqual(
            active_kr_strategy_ids(settings),
            ["kr_intraday_15m_v1", "kr_combo_15m_ahc_afterclose_v2"],
        )

    def test_default_kr_strategy_ignores_enabled_us_schedule_entries(self) -> None:
        settings = RuntimeSettings()
        settings.kr_default_strategy_id = "us_combo_15m_ahc_regular_v1"
        settings.kr_strategies["kr_intraday_1h_v1"].enabled = False
        settings.kr_strategies["kr_intraday_15m_v1"].enabled = True
        settings.kr_strategies["us_combo_15m_ahc_regular_v1"].enabled = True

        default_strategy = default_kr_strategy(settings)

        self.assertIsNotNone(default_strategy)
        self.assertEqual(str(default_strategy.strategy_id), "kr_intraday_15m_v1")

    def test_active_us_strategy_ids_only_include_us_schedule_entries(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["us_intraday_1h_v1"].enabled = True
        settings.kr_strategies["us_combo_15m_ahc_regular_v1"].enabled = True
        settings.kr_strategies["kr_intraday_1h_v1"].enabled = True

        self.assertEqual(
            active_us_strategy_ids(settings),
            ["us_intraday_1h_v1", "us_combo_15m_ahc_regular_v1"],
        )

    def test_default_us_strategy_prefers_enabled_us_entry(self) -> None:
        settings = RuntimeSettings()
        settings.us_default_strategy_id = "us_intraday_1h_v1"
        settings.kr_strategies["us_intraday_1h_v1"].enabled = False
        settings.kr_strategies["us_combo_15m_ahc_regular_v1"].enabled = True

        default_strategy = default_us_strategy(settings)

        self.assertIsNotNone(default_strategy)
        self.assertEqual(str(default_strategy.strategy_id), "us_combo_15m_ahc_regular_v1")

    def test_15m_entry_gate_uses_completed_bar_and_blocks_opening_bar(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1"].enabled = True

        waiting = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 9, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )
        opening_bar = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 9, 20, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )
        allowed = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 9, 31, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )

        self.assertEqual(waiting, "waiting_for_bar_close")
        self.assertEqual(opening_bar, "opening_bar_blocked")
        self.assertIsNone(allowed)

    def test_regular_15m_entry_window_and_flatten_policy(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1"].enabled = True

        outside_window = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 15, 5, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )
        flatten_now = flatten_due(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 15, 16, tzinfo=ZoneInfo("Asia/Seoul")),
        )
        flatten_late = flatten_due(
            settings,
            "kr_intraday_15m_v1",
            when=datetime(2026, 3, 9, 15, 21, tzinfo=ZoneInfo("Asia/Seoul")),
        )

        self.assertEqual(outside_window, "outside_intraday_entry_window")
        self.assertTrue(flatten_now)
        self.assertFalse(flatten_late)

    def test_after_close_close_session_gate(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1_after_close_close"].enabled = True

        before = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_close",
            when=datetime(2026, 3, 9, 15, 35, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        inside = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_close",
            when=datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        flatten = flatten_due(
            settings,
            "kr_intraday_15m_v1_after_close_close",
            when=datetime(2026, 3, 9, 15, 50, tzinfo=ZoneInfo("Asia/Seoul")),
        )

        self.assertEqual(before, "outside_after_close_close_session")
        self.assertIsNone(inside)
        self.assertFalse(flatten)

    def test_auto_15m_switches_to_after_close_close_profile(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1_auto"].enabled = True

        allowed = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_auto",
            when=datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        schedule = strategy_schedule(
            settings,
            "kr_intraday_15m_v1_auto",
            when=datetime(2026, 3, 9, 15, 45, tzinfo=ZoneInfo("Asia/Seoul")),
        )

        self.assertIsNone(allowed)
        self.assertEqual(schedule.session_mode, "after_close_close_price")
        self.assertEqual(schedule.scan_interval_minutes, 5)

    def test_auto_15m_switches_to_after_close_single_profile_and_late_flatten(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1_auto"].enabled = True

        allowed = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_auto",
            when=datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        schedule = strategy_schedule(
            settings,
            "kr_intraday_15m_v1_auto",
            when=datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul")),
        )
        flatten_now = flatten_due(
            settings,
            "kr_intraday_15m_v1_auto",
            when=datetime(2026, 3, 9, 17, 55, tzinfo=ZoneInfo("Asia/Seoul")),
        )

        self.assertIsNone(allowed)
        self.assertEqual(schedule.session_mode, "after_close_single_price")
        self.assertEqual(schedule.scan_interval_minutes, 10)
        self.assertTrue(flatten_now)

    def test_after_close_single_session_gate_and_auction_alignment(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_intraday_15m_v1_after_close_single"].enabled = True

        before = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_single",
            when=datetime(2026, 3, 9, 15, 59, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        waiting = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_single",
            when=datetime(2026, 3, 9, 16, 5, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        inside = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_single",
            when=datetime(2026, 3, 9, 16, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )
        after = entry_gate_reason(
            settings,
            "kr_intraday_15m_v1_after_close_single",
            when=datetime(2026, 3, 9, 18, 1, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=False,
        )

        self.assertEqual(before, "outside_after_close_single_session")
        self.assertEqual(waiting, "after_close_single_waiting_auction")
        self.assertIsNone(inside)
        self.assertEqual(after, "outside_after_close_single_session")

    def test_daily_legacy_strategy_keeps_preclose_gate(self) -> None:
        settings = RuntimeSettings()
        settings.kr_strategies["kr_daily_preclose_v1"].enabled = True
        settings.kr_strategies["kr_intraday_1h_v1"].enabled = False

        blocked = entry_gate_reason(
            settings,
            "kr_daily_preclose_v1",
            when=datetime(2026, 3, 9, 14, 0, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )
        allowed = entry_gate_reason(
            settings,
            "kr_daily_preclose_v1",
            when=datetime(2026, 3, 9, 15, 20, tzinfo=ZoneInfo("Asia/Seoul")),
            market_is_open=True,
        )

        self.assertEqual(blocked, "outside_preclose_window")
        self.assertIsNone(allowed)

    def test_strategy_schedule_reflects_intraday_timeframe(self) -> None:
        settings = RuntimeSettings()

        hourly = strategy_schedule(settings, "kr_intraday_1h_v1")
        fifteen = strategy_schedule(settings, "kr_intraday_15m_v1")
        after_close = strategy_schedule(settings, "kr_intraday_15m_v1_after_close_close")

        self.assertEqual(hourly.timeframe, "1h")
        self.assertEqual(hourly.scan_interval_minutes, 60)
        self.assertEqual(fifteen.timeframe, "15m")
        self.assertEqual(fifteen.scan_interval_minutes, 15)
        self.assertEqual(fifteen.forecast_horizon_bars, 4)
        self.assertEqual(after_close.session_mode, "after_close_close_price")
        self.assertEqual(after_close.scan_interval_minutes, 5)

    def test_kr_strategy_conflict_ids_cover_regular_and_after_hours(self) -> None:
        settings = RuntimeSettings()

        conflicts = strategy_conflict_ids(settings, "kr_intraday_15m_v1_after_close_close")

        self.assertTrue(
            {
                "kr_daily_preclose_v1",
                "kr_intraday_1h_v1",
                "kr_intraday_15m_v1",
                "kr_intraday_15m_v1_auto",
                "kr_intraday_15m_v1_after_close_close",
                "kr_intraday_15m_v1_after_close_single",
            }
            <= conflicts
        )

    def test_visible_kr_strategy_rows_hide_legacy_after_close_profiles(self) -> None:
        settings = RuntimeSettings()

        visible_ids = [
            str(strategy.strategy_id)
            for strategy in iter_visible_strategy_rows(settings, asset_schedule_key="한국주식")
        ]

        self.assertEqual(
            visible_ids,
            [
                "kr_daily_preclose_v1",
                "kr_intraday_1h_v1",
                "kr_intraday_15m_v1",
                "kr_intraday_15m_v1_auto",
            ],
        )


if __name__ == "__main__":
    unittest.main()
