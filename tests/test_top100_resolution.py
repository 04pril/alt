from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import app
from top100_universe import KR_MANUAL_SYMBOLS, KR_NAME_ALIASES, KR_TOP100_NAMES


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding: str | None = None

    def raise_for_status(self) -> None:
        return None


class Top100ResolutionTest(unittest.TestCase):
    def tearDown(self) -> None:
        app.load_krx_name_map.clear()
        app.load_krx_symbol_name_map.clear()
        app.resolve_single_top100_entries.clear()

    def test_top100_universe_preserves_expected_korean_literals(self) -> None:
        self.assertEqual(
            KR_TOP100_NAMES[:6],
            ["삼성전자", "SK하이닉스", "한화시스템", "한미반도체", "현대차", "LIG넥스원"],
        )
        self.assertEqual(KR_MANUAL_SYMBOLS["KODEX 200"], "069500.KS")
        self.assertEqual(KR_NAME_ALIASES["현대차"], "현대자동차")

    @patch("app.requests.get")
    @patch("app.pd.read_html")
    def test_load_krx_name_map_falls_back_when_default_parser_is_unavailable(
        self,
        mock_read_html,
        mock_get,
    ) -> None:
        attempts: list[object] = []

        def fake_read_html(*args, **kwargs):
            attempts.append(kwargs.get("flavor"))
            if kwargs.get("flavor") is None:
                raise ImportError("Missing optional dependency 'lxml'")
            return [
                pd.DataFrame(
                    {
                        "회사명": ["삼성전자", "한화시스템"],
                        "시장구분": ["KOSPI", "KOSPI"],
                        "종목코드": ["5930", "272210"],
                    }
                )
            ]

        mock_get.return_value = _FakeResponse("<table></table>")
        mock_read_html.side_effect = fake_read_html

        name_map = app.load_krx_name_map()

        self.assertEqual(attempts, [None, ["bs4"]])
        self.assertEqual(name_map["삼성전자"]["code"], "005930")
        self.assertEqual(name_map[app.normalize_kr_name_key("한화시스템")]["code"], "272210")

    @patch("app.load_krx_name_map")
    def test_resolve_kr_name_to_symbol_uses_loaded_name_map(self, mock_load_krx_name_map) -> None:
        mock_load_krx_name_map.return_value = {
            "삼성전자": {"market": "KOSPI", "code": "005930"},
            app.normalize_kr_name_key("삼성전자"): {"market": "KOSPI", "code": "005930"},
            "현대자동차": {"market": "KOSPI", "code": "005380"},
            app.normalize_kr_name_key("현대자동차"): {"market": "KOSPI", "code": "005380"},
            "삼천당제약": {"market": "코스닥", "code": "000250"},
            app.normalize_kr_name_key("삼천당제약"): {"market": "코스닥", "code": "000250"},
        }

        self.assertEqual(app.resolve_kr_name_to_symbol("삼성전자"), "005930.KS")
        self.assertEqual(app.resolve_kr_name_to_symbol("현대차"), "005380.KS")
        self.assertEqual(app.resolve_kr_name_to_symbol("삼천당제약"), "000250.KQ")

    def test_bare_kr_code_uses_kr_quote_and_currency_defaults(self) -> None:
        self.assertTrue(app.is_korean_stock_symbol("005930"))
        self.assertEqual(app.default_currency_from_symbol("005930"), "KRW")


if __name__ == "__main__":
    unittest.main()
