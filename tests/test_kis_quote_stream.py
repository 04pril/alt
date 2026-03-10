from __future__ import annotations

import unittest

from services.kis_quote_stream import KR_QUOTE_FIELDS, _stream_connect_kwargs, parse_kr_quote_message


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
