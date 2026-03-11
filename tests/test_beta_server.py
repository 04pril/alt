from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlsplit

from beta_server import _beta_page_html, _redirect_params_after_action


class BetaServerTest(unittest.TestCase):
    def test_beta_page_html_embeds_live_payload_bridge(self) -> None:
        html = _beta_page_html({})
        self.assertIn("/api/live-payload", html)
        self.assertIn("alt-beta-live-payload", html)
        self.assertIn("beta-live-status-strip", html)

    def test_redirect_params_after_action_cleans_action_fields(self) -> None:
        target = _redirect_params_after_action(
            {
                "beta_action": ["restart_worker"],
                "beta_token": ["123"],
                "beta_anchor": ["sync"],
                "beta_theme": ["dark"],
            },
            True,
            "ok",
        )
        parsed = urlsplit(target)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/beta")
        self.assertNotIn("beta_action", query)
        self.assertNotIn("beta_token", query)
        self.assertEqual(query.get("beta_anchor"), ["sync"])
        self.assertEqual(query.get("beta_theme"), ["dark"])
        self.assertEqual(query.get("beta_ok"), ["1"])
        self.assertEqual(query.get("beta_msg"), ["ok"])


if __name__ == "__main__":
    unittest.main()
