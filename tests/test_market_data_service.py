from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config.settings import RuntimeSettings
from services.market_data_service import MarketDataService


class _FakeKISClient:
    def __init__(self) -> None:
        self.quote_calls = 0
        self.history_calls = 0

    def get_quote(self, symbol_or_code: str):
        self.quote_calls += 1
        return {
            "current_price": 101.0,
            "high_price": 103.0,
            "low_price": 99.0,
            "open_price": 100.0,
            "volume": 1000.0,
        }

    def get_daily_history(self, symbol_or_code: str, years: int = 5):
        self.history_calls += 1
        index = pd.date_range("2025-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {
                "Open": np.linspace(100, 109, len(index)),
                "High": np.linspace(101, 110, len(index)),
                "Low": np.linspace(99, 108, len(index)),
                "Close": np.linspace(100, 109, len(index)),
                "Volume": np.full(len(index), 1000.0),
            },
            index=index,
        )


class MarketDataServiceTest(unittest.TestCase):
    def test_latest_quote_uses_kis_for_korean_execution(self) -> None:
        settings = RuntimeSettings()
        fake_client = _FakeKISClient()
        service = MarketDataService(settings, kis_client_factory=lambda: fake_client)
        korean_asset_type = next(asset for asset, mode in settings.broker.asset_broker_mode.items() if mode == "kis_paper")
        quote = service.latest_quote("005930.KS", asset_type=korean_asset_type, timeframe="1d", purpose="execution")
        self.assertEqual(quote.price, 101.0)
        self.assertEqual(fake_client.quote_calls, 1)

    def test_latest_quote_uses_bar_source_for_non_execution_path(self) -> None:
        settings = RuntimeSettings()
        fake_client = _FakeKISClient()
        service = MarketDataService(settings, kis_client_factory=lambda: fake_client)
        service.get_bars = lambda **kwargs: pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.5],
                "Volume": [5000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-06")]),
        )
        korean_asset_type = next(asset for asset, mode in settings.broker.asset_broker_mode.items() if mode == "kis_paper")
        quote = service.latest_quote("005930.KS", asset_type=korean_asset_type, timeframe="1d", purpose="analysis")
        self.assertEqual(quote.price, 101.5)
        self.assertEqual(fake_client.quote_calls, 0)

    def test_get_bars_uses_kis_history_for_korean_daily_mode(self) -> None:
        settings = RuntimeSettings()
        fake_client = _FakeKISClient()
        service = MarketDataService(settings, kis_client_factory=lambda: fake_client)
        korean_asset_type = next(asset for asset, mode in settings.broker.asset_broker_mode.items() if mode == "kis_paper")
        frame = service.get_bars("005930.KS", asset_type=korean_asset_type, timeframe="1d", lookback_bars=5)
        self.assertFalse(frame.empty)
        self.assertEqual(fake_client.history_calls, 1)


if __name__ == "__main__":
    unittest.main()
