from __future__ import annotations

import unittest
from types import SimpleNamespace

from kis_paper import KISPaperClient


class _StubKISPaperClient(KISPaperClient):
    def __init__(self, *, is_paper: bool) -> None:
        self.config = SimpleNamespace(
            is_paper=is_paper,
            account_no="12345678",
            product_code="01",
        )
        self.request_args = None

    def _request(self, method, path, tr_id, params=None, body=None):
        self.request_args = {
            "method": method,
            "path": path,
            "tr_id": tr_id,
            "params": params or {},
            "body": body,
        }
        return {"output": {"ord_psbl_qty": "7", "hldg_qty": "9"}}, {}


class KISPaperClientTest(unittest.TestCase):
    def test_get_sellable_quantity_uses_paper_tr_id_for_paper_account(self) -> None:
        client = _StubKISPaperClient(is_paper=True)

        result = client.get_sellable_quantity("373220.KS")

        self.assertEqual(client.request_args["tr_id"], "VTTC8408R")
        self.assertEqual(result["sellable_qty"], 7)
        self.assertEqual(result["held_qty"], 9)

    def test_get_sellable_quantity_keeps_live_tr_id_for_live_account(self) -> None:
        client = _StubKISPaperClient(is_paper=False)

        client.get_sellable_quantity("373220.KS")

        self.assertEqual(client.request_args["tr_id"], "TTTC8408R")


if __name__ == "__main__":
    unittest.main()
