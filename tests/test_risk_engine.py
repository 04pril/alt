from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from config.settings import RuntimeSettings
from services.risk_engine import RiskEngine
from services.signal_engine import SignalDecision
from storage.repository import TradingRepository


class RiskEngineTest(unittest.TestCase):
    def test_paused_flag_blocks_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = RuntimeSettings()
            settings.storage.db_path = f"{tmp}/runtime.sqlite3"
            repo = TradingRepository(settings.storage.db_path)
            repo.initialize()
            repo.set_control_flag("trading_paused", "1", "test")
            engine = RiskEngine(settings, repo)
            signal = SignalDecision(
                symbol="BTC-USD",
                asset_type="코인",
                timeframe="1h",
                prediction_id="pred1",
                scan_id="scan1",
                score=1.0,
                signal="LONG",
                expected_return=0.02,
                expected_risk=0.01,
                confidence=0.9,
                threshold=0.003,
                position_size=0.5,
                current_price=100.0,
                predicted_price=102.0,
                predicted_return=0.02,
                stop_level=98.0,
                take_level=105.0,
                model_version="v1",
                feature_version="f1",
                strategy_version="s1",
                validation_mode="holdout",
                result=None,
            )
            decision = engine.evaluate_entry(signal, correlation_matrix=pd.DataFrame(), market_is_open=True)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "paused")


if __name__ == "__main__":
    unittest.main()
