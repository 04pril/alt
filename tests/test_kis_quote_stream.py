from __future__ import annotations

import unittest

import pandas as pd

from config.settings import RuntimeSettings, UniverseSettings
from services.kis_quote_stream import (
    KR_QUOTE_FIELDS,
    MAX_KR_QUOTE_SYMBOLS,
    MAX_KR_QUOTE_SYMBOLS_PER_CONNECTION,
    KISKRQuoteStream,
    _chunk_symbol_codes,
    _is_transient_stream_error,
    _stream_connect_kwargs,
    parse_kr_quote_message,
)


class _StubRepository:
    def open_positions(self, *, account_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({"symbol": [f"{index:06d}.KS" for index in range(1, 41)]})

    def open_orders(self, *, account_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({"symbol": [f"{index:06d}.KS" for index in range(41, 81)]})

    def latest_candidates(
        self,
        asset_type: str | None = None,
        execution_account_id: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        return pd.DataFrame({"symbol": [f"{index:06d}.KS" for index in range(81, 101)]})

    def set_control_flag(self, flag_name: str, flag_value: str, notes: str = "") -> None:
        return None


class KISQuoteStreamTest(unittest.TestCase):
    def test_stream_connect_kwargs_disable_protocol_keepalive_ping(self) -> None:
        kwargs = _stream_connect_kwargs()

        self.assertIsNone(kwargs["ping_interval"])
        self.assertIsNone(kwargs["ping_timeout"])
        self.assertEqual(kwargs["open_timeout"], 20)

    def test_parse_kr_quote_message_extracts_contract_fields(self) -> None:
        payload = [""] * len(KR_QUOTE_FIELDS)
        payload[0] = "005930"
        payload[1] = "102208"
        payload[2] = "190000"
        payload[4] = "500"
        payload[5] = "0.26"
        payload[10] = "190100"
        payload[11] = "189900"
        payload[13] = "1234567"
        message = "0|H0STCNT0|001|" + "^".join(payload)

        rows = parse_kr_quote_message(message, received_at="2026-03-10T01:22:08Z")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].symbol_code, "005930")
        self.assertEqual(rows[0].currency, "KRW")
        self.assertAlmostEqual(rows[0].current_price, 190000.0)
        self.assertAlmostEqual(rows[0].previous_close, 189500.0)
        self.assertAlmostEqual(rows[0].change_pct, 0.26)
        self.assertAlmostEqual(rows[0].ask_price, 190100.0)
        self.assertAlmostEqual(rows[0].bid_price, 189900.0)
        self.assertAlmostEqual(rows[0].volume, 1234567.0)

    def test_candidate_symbols_expand_to_runtime_cap_unique_codes(self) -> None:
        settings = RuntimeSettings()
        settings.universes["한국주식"] = UniverseSettings(
            watchlist=[f"{index:06d}.KS" for index in range(101, 111)],
            top_universe=[f"{index:06d}.KS" for index in range(111, 161)],
        )
        stream = KISKRQuoteStream(settings, _StubRepository(), client_factory=lambda: None)

        symbols = stream._candidate_symbols()

        self.assertEqual(len(symbols), MAX_KR_QUOTE_SYMBOLS)
        self.assertEqual(symbols[0], "000001")
        self.assertEqual(symbols[-1], f"{MAX_KR_QUOTE_SYMBOLS:06d}")
        self.assertEqual(len(set(symbols)), MAX_KR_QUOTE_SYMBOLS)

    def test_chunk_symbol_codes_shards_runtime_cap_across_connections(self) -> None:
        batches = _chunk_symbol_codes([f"{index:06d}" for index in range(1, MAX_KR_QUOTE_SYMBOLS + 1)])

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), MAX_KR_QUOTE_SYMBOLS_PER_CONNECTION)
        self.assertEqual(batches[0][0], "000001")
        self.assertEqual(batches[0][-1], f"{MAX_KR_QUOTE_SYMBOLS:06d}")

    def test_refresh_symbols_enforces_runtime_cap_for_explicit_symbol_lists(self) -> None:
        settings = RuntimeSettings()
        repository = _StubRepository()
        stream = KISKRQuoteStream(settings, repository, client_factory=lambda: None)

        changed = stream.refresh_symbols([f"{index:06d}" for index in range(1, 60)])

        self.assertTrue(changed)
        self.assertEqual(len(stream._desired_snapshot()), MAX_KR_QUOTE_SYMBOLS)
        self.assertEqual(stream._desired_snapshot()[-1], f"{MAX_KR_QUOTE_SYMBOLS:06d}")

    def test_transient_stream_error_classification(self) -> None:
        self.assertTrue(_is_transient_stream_error(RuntimeError("no close frame received or sent")))
        self.assertTrue(_is_transient_stream_error(RuntimeError("Connection closed unexpectedly")))
        self.assertFalse(_is_transient_stream_error(RuntimeError("MAX SUBSCRIBE OVER")))
