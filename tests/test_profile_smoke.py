from __future__ import annotations

import unittest

from services.profile_smoke import compare_profiles


class ProfileSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        results = compare_profiles()
        cls.by_name = {result["profile_name"]: result for result in results}

    def test_profiles_are_reported(self) -> None:
        self.assertEqual(set(self.by_name.keys()), {"baseline", "balanced", "active"})

    def test_balanced_increases_submit_and_fill_counts(self) -> None:
        baseline = self.by_name["baseline"]
        balanced = self.by_name["balanced"]
        self.assertGreater(balanced["today_submit_requested_count"], baseline["today_submit_requested_count"])
        self.assertGreater(balanced["today_submitted_count"], baseline["today_submitted_count"])
        self.assertGreater(balanced["today_filled_count"], baseline["today_filled_count"])

    def test_active_is_at_least_as_active_as_balanced(self) -> None:
        balanced = self.by_name["balanced"]
        active = self.by_name["active"]
        self.assertGreaterEqual(active["today_submit_requested_count"], balanced["today_submit_requested_count"])
        self.assertGreaterEqual(active["today_submitted_count"], balanced["today_submitted_count"])
        self.assertGreaterEqual(active["today_filled_count"], balanced["today_filled_count"])

    def test_gate_related_reasons_drop_in_balanced_profile(self) -> None:
        baseline = self.by_name["baseline"]
        balanced = self.by_name["balanced"]
        baseline_noop = baseline["today_noop_reason_breakdown"]
        baseline_rejected = baseline["today_entry_rejected_reason_breakdown"]
        balanced_noop = balanced["today_noop_reason_breakdown"]
        balanced_rejected = balanced["today_entry_rejected_reason_breakdown"]

        self.assertGreater(baseline_noop.get("outside_preclose_window", 0), 0)
        self.assertEqual(balanced_noop.get("outside_preclose_window", 0), 0)
        self.assertGreater(baseline_rejected.get("expected_return_too_low", 0), 0)
        self.assertEqual(balanced_rejected.get("expected_return_too_low", 0), 0)
        self.assertGreater(baseline_rejected.get("confidence_too_low", 0), 0)
        self.assertEqual(balanced_rejected.get("confidence_too_low", 0), 0)

    def test_broker_isolation_is_maintained(self) -> None:
        for profile in self.by_name.values():
            self.assertEqual(profile["broker_modes"]["한국주식"], "kis_mock")
            self.assertEqual(profile["broker_modes"]["미국주식"], "sim")
            self.assertEqual(profile["broker_modes"]["코인"], "sim")

    def test_kr_bootstrap_uses_kis_snapshot_and_us_crypto_use_sim_snapshot(self) -> None:
        bootstrap = self.by_name["balanced"]["bootstrap"]
        self.assertEqual(bootstrap["latest_snapshot_source"], "kis_account_sync")
        self.assertGreater(float(bootstrap["kr_state"]["equity"]), float(bootstrap["us_state"]["equity"]))
        self.assertEqual(float(bootstrap["us_state"]["equity"]), float(bootstrap["crypto_state"]["equity"]))
        self.assertEqual(float(bootstrap["kr_state"]["cash"]), 30_000_000.0)

    def test_monitoring_reports_loaded_profile_name(self) -> None:
        for profile_name, result in self.by_name.items():
            monitoring = result["monitoring"]
            self.assertEqual(monitoring["runtime_profile"]["name"], profile_name)
            self.assertTrue(str(monitoring["runtime_profile"]["source"]).endswith(f"{profile_name}.json"))
        self.assertEqual(self.by_name["balanced"]["monitoring"]["runtime_profile"]["mode"], "recommended")
        self.assertEqual(self.by_name["balanced"]["monitoring"]["runtime_profile"]["recommended_default"], "true")
        self.assertEqual(self.by_name["active"]["monitoring"]["runtime_profile"]["mode"], "experimental")
        self.assertEqual(self.by_name["active"]["monitoring"]["runtime_profile"]["experimental"], "true")
        self.assertEqual(self.by_name["balanced"]["monitoring"]["runtime_profile"]["kr_default_strategy_id"], "kr_intraday_1h_v1")
        self.assertIn("kr_intraday_1h_v1", self.by_name["balanced"]["monitoring"]["runtime_profile"]["kr_active_strategies"])

    def test_kr_default_strategy_remains_1h_and_15m_is_opt_in(self) -> None:
        for result in self.by_name.values():
            self.assertEqual(result["kr_strategy"]["default_strategy_id"], "kr_intraday_1h_v1")
            self.assertIn("kr_intraday_1h_v1", result["kr_strategy"]["active_strategy_ids"])
            self.assertNotIn("kr_intraday_15m_v1", result["kr_strategy"]["active_strategy_ids"])
            self.assertFalse(bool(result["kr_strategy"]["experimental_15m_enabled"]))
            self.assertFalse(bool(result["kr_strategy"]["experimental_15m_after_close_close_enabled"]))
            self.assertFalse(bool(result["kr_strategy"]["experimental_15m_after_close_single_enabled"]))
            self.assertEqual(
                result["kr_strategy"]["experimental_15m_family_ids"],
                [
                    "kr_intraday_15m_v1",
                    "kr_intraday_15m_v1_after_close_close",
                    "kr_intraday_15m_v1_after_close_single",
                ],
            )
            self.assertEqual(
                result["kr_strategy"]["experimental_15m_session_modes"],
                {
                    "kr_intraday_15m_v1": "regular",
                    "kr_intraday_15m_v1_after_close_close": "after_close_close_price",
                    "kr_intraday_15m_v1_after_close_single": "after_close_single_price",
                },
            )


if __name__ == "__main__":
    unittest.main()
