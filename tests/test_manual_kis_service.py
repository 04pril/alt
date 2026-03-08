from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from config.settings import RuntimeSettings
from services import manual_kis_service


class _ReadOnlyRepository:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[str, int]] = []

    def recent_orders(self, limit: int = 200) -> pd.DataFrame:
        self.calls.append(("recent_orders", int(limit)))
        return self.frame.copy()

    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected repository access: {name}")


class ManualKISServiceTest(unittest.TestCase):
    def test_submit_manual_kis_order_uses_runtime_builder_and_attaches_once(self) -> None:
        fake_repo = Mock()
        fake_repo.get_order.return_value = {"broker_order_id": "KIS-1", "status": "acknowledged", "error_message": ""}
        fake_router = Mock()
        fake_router.submit_manual_kis_order.return_value = "ord-1"
        runtime = manual_kis_service.ManualKISRuntime(
            settings=RuntimeSettings(),
            repository=fake_repo,
            router=fake_router,
        )

        with patch("services.manual_kis_service._build_manual_runtime", return_value=runtime), patch(
            "services.manual_kis_service._attach_prediction_order_link"
        ) as mock_attach:
            result = manual_kis_service.submit_manual_kis_order(
                symbol="005930.KS",
                side="buy",
                quantity=1,
                order_type="market",
                requested_price=70000.0,
                prediction_id="pred-1",
            )

        fake_router.submit_manual_kis_order.assert_called_once()
        fake_repo.get_order.assert_called_once_with("ord-1")
        mock_attach.assert_called_once_with("pred-1", "ord-1")
        self.assertEqual(result["order_id"], "ord-1")
        self.assertEqual(result["broker_order_id"], "KIS-1")
        self.assertEqual(result["status"], "acknowledged")

    def test_load_manual_order_history_uses_repository_read_path_only(self) -> None:
        frame = pd.DataFrame(
            [
                {"order_id": "ord-1", "raw_json": '{"broker":"kis_mock","stage":"submitted"}'},
                {"order_id": "ord-2", "raw_json": '{"broker":"sim","stage":"ignored"}'},
            ]
        )
        repo = _ReadOnlyRepository(frame)
        settings = RuntimeSettings()

        with patch("services.manual_kis_service.load_settings", return_value=settings), patch(
            "services.manual_kis_service._open_repository", return_value=repo
        ):
            result = manual_kis_service.load_manual_order_history(limit=25)

        self.assertEqual(repo.calls, [("recent_orders", 25)])
        self.assertEqual(result["order_id"].tolist(), ["ord-1"])
        self.assertIn("stage", result.columns)
