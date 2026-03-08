from __future__ import annotations

import unittest
from unittest.mock import patch

import prediction_memory


class PredictionMemoryWrapperTest(unittest.TestCase):
    @patch("prediction_store.save_prediction_snapshot", return_value="run-1")
    def test_save_prediction_snapshot_forwards_once(self, mock_save) -> None:
        result = prediction_memory.save_prediction_snapshot(asset_type="코인", korea_market="", result=object())
        self.assertEqual(result, "run-1")
        mock_save.assert_called_once()

    @patch("prediction_store.attach_order_to_prediction")
    def test_attach_order_to_prediction_forwards_once(self, mock_attach) -> None:
        prediction_memory.attach_order_to_prediction("pred-1", "ord-1")
        mock_attach.assert_called_once_with("pred-1", "ord-1")


if __name__ == "__main__":
    unittest.main()
