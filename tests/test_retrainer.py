from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from config.settings import RuntimeSettings
from services.retrainer import Retrainer


class RetrainerCadenceTest(unittest.TestCase):
    def _retrainer(self, cadence: str, started_at: str | None) -> Retrainer:
        settings = RuntimeSettings()
        settings.retraining.cadence = cadence
        repository = MagicMock()
        if started_at is None:
            repository.recent_job_health.return_value = pd.DataFrame(columns=["job_name", "started_at"])
        else:
            repository.recent_job_health.return_value = pd.DataFrame(
                [{"job_name": "retrain_check", "started_at": started_at}]
            )
        return Retrainer(settings, repository)

    @patch("services.retrainer.utc_now_iso", return_value="2026-03-08T12:00:00Z")
    def test_retraining_due_disabled_daily_weekly_monthly(self, _mock_now) -> None:
        cases = [
            ("disabled", "2026-03-08T01:00:00Z", False),
            ("daily", "2026-03-08T01:00:00Z", False),
            ("daily", "2026-03-07T23:00:00Z", True),
            ("weekly", "2026-03-03T01:00:00Z", False),
            ("weekly", "2026-02-28T23:00:00Z", True),
            ("monthly", "2026-03-01T01:00:00Z", False),
            ("monthly", "2026-02-28T23:00:00Z", True),
        ]
        for cadence, started_at, expected in cases:
            with self.subTest(cadence=cadence, started_at=started_at):
                retrainer = self._retrainer(cadence, started_at)
                self.assertEqual(retrainer.retraining_due(), expected)

    @patch("services.retrainer.utc_now_iso", return_value="2026-03-08T12:00:00Z")
    def test_retraining_due_true_when_no_history(self, _mock_now) -> None:
        retrainer = self._retrainer("weekly", None)
        self.assertTrue(retrainer.retraining_due())


if __name__ == "__main__":
    unittest.main()
