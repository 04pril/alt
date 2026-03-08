from __future__ import annotations

import unittest
from unittest.mock import patch

from ui.floating_nav import (
    FloatingNavItem,
    _FLOATING_NAV_CSS,
    _FLOATING_NAV_JS,
    _theme_toggle_svg,
    _toolbar_status_label,
    build_nav_items,
    resolve_current_page_key,
    theme_tokens,
)


class FloatingNavTest(unittest.TestCase):
    def test_build_nav_items_preserves_order(self) -> None:
        items = [
            FloatingNavItem(key="analysis", label="종목 분석", icon="A"),
            FloatingNavItem(key="paper", label="모의매매", icon="P"),
        ]
        payload = build_nav_items(items)
        self.assertEqual(payload[0]["key"], "analysis")
        self.assertEqual(payload[1]["label"], "모의매매")

    def test_resolve_current_page_key_uses_identity(self) -> None:
        page_a = object()
        page_b = object()
        page_map = {"monitor": page_a, "analysis": page_b}
        self.assertEqual(resolve_current_page_key(page_map, page_b, default_key="monitor"), "analysis")
        self.assertEqual(resolve_current_page_key(page_map, object(), default_key="monitor"), "monitor")

    def test_toolbar_status_label_shortens_toolbar_copy(self) -> None:
        self.assertEqual(_toolbar_status_label({"state": "running"}), "Run")
        self.assertEqual(_toolbar_status_label({"state": "paused"}), "Pause")
        self.assertEqual(_toolbar_status_label({"state": "stopped"}), "Stop")

    def test_theme_toggle_svg_returns_svg_markup(self) -> None:
        dark_svg = _theme_toggle_svg("dark")
        light_svg = _theme_toggle_svg("light")
        self.assertIn("<svg", dark_svg)
        self.assertIn("<svg", light_svg)
        self.assertNotEqual(dark_svg, light_svg)

    @patch("ui.floating_nav.st.get_option")
    def test_theme_tokens_follow_streamlit_light_theme(self, mock_get_option) -> None:
        options = {
            "theme.primaryColor": "#2563eb",
            "theme.backgroundColor": "#ffffff",
            "theme.secondaryBackgroundColor": "#f8fafc",
            "theme.textColor": "#111827",
            "theme.base": "light",
        }
        mock_get_option.side_effect = lambda key: options.get(key)
        tokens = theme_tokens("light")
        self.assertEqual(tokens["base"], "light")
        self.assertIn("255, 255, 255", tokens["background"])
        self.assertIn("17, 24, 39", tokens["text"])

    @patch("ui.floating_nav.st.get_option")
    def test_theme_tokens_follow_streamlit_dark_theme(self, mock_get_option) -> None:
        options = {
            "theme.primaryColor": "#60a5fa",
            "theme.backgroundColor": "#ffffff",
            "theme.secondaryBackgroundColor": "#f8fafc",
            "theme.textColor": "#111827",
            "theme.base": "dark",
        }
        mock_get_option.side_effect = lambda key: options.get(key)
        tokens = theme_tokens("dark")
        self.assertEqual(tokens["base"], "dark")
        self.assertEqual(tokens["background"], "rgba(12, 18, 32, 0.76)")
        self.assertEqual(tokens["text"], "rgba(226, 232, 240, 0.86)")

    def test_mobile_nav_has_sidebar_suppression_hooks(self) -> None:
        self.assertIn(".alt-floating-nav.is-suppressed", _FLOATING_NAV_CSS)
        self.assertIn("const isSidebarOpen =", _FLOATING_NAV_JS)
        self.assertIn("bindSidebarObserver", _FLOATING_NAV_JS)
        self.assertIn("const updateTopOffset =", _FLOATING_NAV_JS)
        self.assertIn("data-testid=\"stHeader\"", _FLOATING_NAV_JS)


if __name__ == "__main__":
    unittest.main()
