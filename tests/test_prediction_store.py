from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from config.settings import RuntimeSettings
from prediction_store import append_fill_log, append_order_log, load_order_log
from storage.repository import TradingRepository


class PredictionStoreCompatibilityTest(unittest.TestCase):
    def test_legacy_append_order_log_uses_repository_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.initialize_runtime_flags()
            record = {
                "order_id": "ord1",
                "requested_at": "2026-03-06T00:00:00Z",
                "symbol": "BTC-USD",
                "asset_type": next(asset for asset, schedule in settings.asset_schedules.items() if schedule.session_mode == "always"),
                "timeframe": "1h",
                "side": "buy",
                "order_type": "market",
                "quantity": 1,
                "filled_qty": 1,
                "remaining_qty": 0,
                "requested_price": 100.0,
                "status": "filled",
                "reason": "entry",
            }
            with patch("prediction_store.load_settings", return_value=settings):
                append_order_log(record)
                append_order_log(record)
                append_fill_log(
                    {
                        "fill_id": "fill_ord1",
                        "order_id": "ord1",
                        "filled_at": "2026-03-06T00:00:01Z",
                        "symbol": "BTC-USD",
                        "side": "buy",
                        "quantity": 1,
                        "fill_price": 100.0,
                        "fill_status": "filled",
                    }
                )
                append_fill_log(
                    {
                        "fill_id": "fill_ord1",
                        "order_id": "ord1",
                        "filled_at": "2026-03-06T00:00:01Z",
                        "symbol": "BTC-USD",
                        "side": "buy",
                        "quantity": 1,
                        "fill_price": 100.0,
                        "fill_status": "filled",
                    }
                )
                orders = load_order_log()
            self.assertEqual(len(orders), 1)
            self.assertEqual(len(repo.fills_for_order("ord1")), 1)


if __name__ == "__main__":
    unittest.main()
