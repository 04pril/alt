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


if __name__ == "__main__":
    unittest.main()
