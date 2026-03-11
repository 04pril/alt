from __future__ import annotations

from datetime import datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import RuntimeSettings
from services.market_data_service import MarketDataService


class MarketDataServiceProviderTest(TestCase):
    def setUp(self) -> None:
        self.settings = RuntimeSettings()
        self.service = MarketDataService(self.settings)

    def test_trim_incomplete_intraday_bars_keeps_previous_session_before_first_bar_close(self) -> None:
        tz = ZoneInfo("Asia/Seoul")
        frame = pd.DataFrame(
            {
                "Open": [190000.0, 191000.0],
                "High": [191000.0, 191500.0],
                "Low": [189500.0, 190500.0],
                "Close": [190500.0, 191200.0],
                "Volume": [1200.0, 800.0],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2026, 3, 10, 15, 0, tzinfo=tz),
                    datetime(2026, 3, 11, 9, 0, tzinfo=tz),
                ]
            ),
        )

        self.service.now = lambda asset_type: datetime(2026, 3, 11, 9, 5, tzinfo=tz)
        with patch("services.market_data_service.latest_completed_bar_close", return_value=None):
            trimmed = self.service._trim_incomplete_intraday_bars(frame, asset_type="한국주식", timeframe="15m")

        self.assertEqual(len(trimmed), 1)
        self.assertEqual(pd.Timestamp(trimmed.index[-1]), pd.Timestamp(datetime(2026, 3, 10, 15, 0, tzinfo=tz)))

    def test_get_bars_uses_kis_fallback_when_kr_yfinance_is_empty(self) -> None:
        expected = pd.DataFrame(
            {
                "Open": [1.0] * 20,
                "High": [2.0] * 20,
                "Low": [0.5] * 20,
                "Close": [1.5] * 20,
                "Volume": [100.0] * 20,
            },
            index=pd.date_range("2026-03-10 09:00:00", periods=20, freq="15min", tz="Asia/Seoul"),
        )

        with patch("services.market_data_service.yf.download", return_value=pd.DataFrame()) as yf_mock:
            with patch.object(MarketDataService, "_get_kr_intraday_bars_from_kis", return_value=expected) as kis_mock:
                with patch.object(MarketDataService, "_get_kr_intraday_bars_from_naver", return_value=pd.DataFrame()) as naver_mock:
                    frame = self.service.get_bars("005930.KS", "한국주식", "15m", 10)

        yf_mock.assert_called_once()
        kis_mock.assert_called_once()
        naver_mock.assert_not_called()
        self.assertEqual(len(frame), 15)
        self.assertAlmostEqual(float(frame.iloc[-1]["Close"]), 1.5)

    def test_get_bars_uses_cryptocompare_for_crypto_hourly(self) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "Response": "Success",
            "Data": {
                "Data": [
                    {"time": 1773180000, "open": 98.0, "high": 102.0, "low": 97.0, "close": 101.0, "volumefrom": 500.0},
                    {"time": 1773183600, "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volumefrom": 650.0},
                    {"time": 1773187200, "open": 102.0, "high": 104.0, "low": 101.0, "close": 103.0, "volumefrom": 700.0},
                ]
            },
        }

        with patch.object(self.service._http, "get", return_value=response) as get_mock:
            with patch("services.market_data_service.yf.download", side_effect=AssertionError("yfinance fallback should not run")):
                frame = self.service.get_bars("BTC-USD", "코인", "1h", 2)

        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(float(frame.iloc[-1]["Close"]), 103.0)
        self.assertIn("histohour", get_mock.call_args.args[0])


if __name__ == "__main__":
    import unittest

    unittest.main()
