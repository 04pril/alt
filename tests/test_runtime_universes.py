from __future__ import annotations

from pathlib import Path
import unittest

from config.settings import load_settings


class RuntimeUniverseSettingsTest(unittest.TestCase):
    def test_runtime_universes_are_expanded_to_100(self) -> None:
        settings = load_settings(Path("config/runtime_settings.json"))

        for asset_type in ("한국주식", "미국주식", "코인"):
            universe = settings.universes[asset_type]
            self.assertEqual(len(universe.top_universe), 100, asset_type)
            self.assertTrue(set(universe.watchlist).issubset(set(universe.top_universe)), asset_type)
            self.assertGreaterEqual(len(universe.watchlist), 10, asset_type)


if __name__ == "__main__":
    unittest.main()
