from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd

from beta_actions import (
    restart_background_worker,
    run_manual_runtime_job,
    run_manual_scan_job,
    set_default_strategy,
    stop_background_worker,
)
from beta_monitor_clone import (
    BETA_LIVE_PAYLOAD_STORAGE_KEY,
    build_beta_live_payload,
    build_beta_overview_html,
    render_beta_live_payload_host,
)
from beta_quotes import fetch_quote_snapshots, load_krx_name_map
from config.settings import load_settings
from monitoring.dashboard_hooks import compute_auto_trading_status, load_dashboard_data, load_monitor_recent_orders
from monitoring.live_display import build_live_accounts_overview, build_live_total_portfolio_overview
from runtime_accounts import ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.repository import TradingRepository

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, ...], tuple[float, Any]] = {}
_LIVE_PAYLOAD_TTL_SECONDS = 3.0
_PAGE_CONTEXT_TTL_SECONDS = 2.0


def _cached(key: tuple[str, ...], ttl_seconds: float, loader):
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    value = loader()
    with _CACHE_LOCK:
        _CACHE[key] = (now + float(ttl_seconds), value)
    return value


def _query_value(query: dict[str, list[str]], name: str, default: str = "") -> str:
    values = query.get(name)
    if not values:
        return default
    return str(values[0] or default)


def _build_feedback(query: dict[str, list[str]]) -> Dict[str, Any] | None:
    message = _query_value(query, "beta_msg", "")
    if not message:
        return None
    return {"ok": _query_value(query, "beta_ok", "1") == "1", "message": message}


def _load_beta_live_context(settings) -> Dict[str, Any]:
    def _loader() -> Dict[str, Any]:
        data = load_dashboard_data(settings)
        open_positions = data.get("open_positions", pd.DataFrame())
        accounts_overview = build_live_accounts_overview(
            data.get("accounts_overview", {}),
            open_positions,
            {},
        )
        quote_symbols: list[str] = []
        if not open_positions.empty and "symbol" in open_positions.columns:
            quote_symbols.extend(open_positions["symbol"].dropna().astype(str).tolist())
        if any(
            str((accounts_overview.get(account_id) or {}).get("currency") or "").upper() == "USD"
            for account_id in (ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO)
        ):
            quote_symbols.append("KRW=X")
        quote_snapshots: Dict[str, Dict[str, Any]] = {}
        if quote_symbols:
            symbols = tuple(dict.fromkeys(quote_symbols))
            if symbols:
                quote_snapshots = fetch_quote_snapshots(
                    symbols=symbols,
                    db_path=settings.storage.db_path,
                    prefer_overtime_kr=True,
                )
        live_accounts_overview = build_live_accounts_overview(
            data.get("accounts_overview", {}),
            open_positions,
            quote_snapshots,
        )
        kr_asset_types = {
            asset_type
            for asset_type, schedule in settings.asset_schedules.items()
            if str(getattr(schedule, "timezone", "")) == "Asia/Seoul"
        }
        return {
            "data": data,
            "open_positions": open_positions,
            "accounts_overview": live_accounts_overview,
            "total_portfolio_overview": build_live_total_portfolio_overview(live_accounts_overview),
            "quote_snapshots": quote_snapshots,
            "kr_asset_types": kr_asset_types,
        }

    return _cached(("beta_live_context", str(settings.storage.db_path)), _PAGE_CONTEXT_TTL_SECONDS, _loader)


def _build_live_payload(settings) -> Dict[str, Any]:
    def _loader() -> Dict[str, Any]:
        context = _load_beta_live_context(settings)
        return build_beta_live_payload(
            data=context["data"],
            accounts_overview=context["accounts_overview"],
            quote_snapshots=context["quote_snapshots"],
            kr_asset_types=context["kr_asset_types"],
            krx_name_map=load_krx_name_map(),
        )

    return _cached(("beta_live_payload", str(settings.storage.db_path)), _LIVE_PAYLOAD_TTL_SECONDS, _loader)


def _inject_live_payload_bridge(html: str, *, initial_payload: Dict[str, Any], live_payload_url: str) -> str:
    storage_key = json.dumps(BETA_LIVE_PAYLOAD_STORAGE_KEY)
    bridge_script = f"""
<script>
(() => {{
  const livePayloadUrl = {json.dumps(live_payload_url)};
  const storageKey = {storage_key};
  let pending = false;
  const writePayload = (payload) => {{
    try {{
      window.localStorage.setItem(storageKey, JSON.stringify(payload));
    }} catch (error) {{
    }}
  }};
  const refresh = async () => {{
    if (pending || document.visibilityState === "hidden") return;
    pending = true;
    try {{
      const response = await fetch(livePayloadUrl, {{ cache: "no-store" }});
      if (!response.ok) return;
      const payload = await response.json();
      writePayload(payload);
    }} catch (error) {{
    }} finally {{
      pending = false;
    }}
  }};
  writePayload({json.dumps(initial_payload, ensure_ascii=False)});
  refresh();
  setInterval(refresh, 5000);
}})();
</script>
"""
    return html.replace("</body>", render_beta_live_payload_host(initial_payload) + bridge_script + "</body>", 1)


def _beta_page_html(query: dict[str, list[str]]) -> str:
    settings = load_settings()
    context = _load_beta_live_context(settings)
    feedback = _build_feedback(query)
    theme_mode = _query_value(query, "beta_theme", "dark")
    initial_anchor = _query_value(query, "beta_anchor", "")
    current_candidate_tab = _query_value(query, "beta_cand_tab", "")
    jobs_expanded = _query_value(query, "beta_jobs", "") == "all"
    signals_expanded = _query_value(query, "beta_signals", "") == "all"
    trades_expanded = _query_value(query, "beta_trades", "") == "all"
    recent_orders = load_monitor_recent_orders(settings, limit=40)
    krx_name_map = load_krx_name_map()
    page_html = build_beta_overview_html(
        data=context["data"],
        theme_mode=theme_mode if theme_mode in {"light", "dark"} else "dark",
        initial_anchor=initial_anchor,
        feedback=feedback,
        accounts_overview=context["accounts_overview"],
        total_portfolio_overview=context["total_portfolio_overview"],
        quote_snapshots=context["quote_snapshots"],
        kr_asset_types=context["kr_asset_types"],
        recent_orders=recent_orders,
        krx_name_map=krx_name_map,
        current_candidate_tab=current_candidate_tab,
        jobs_expanded=jobs_expanded,
        signals_expanded=signals_expanded,
        trades_expanded=trades_expanded,
    )
    initial_payload = build_beta_live_payload(
        data=context["data"],
        accounts_overview=context["accounts_overview"],
        quote_snapshots=context["quote_snapshots"],
        kr_asset_types=context["kr_asset_types"],
        krx_name_map=krx_name_map,
    )
    return _inject_live_payload_bridge(page_html, initial_payload=initial_payload, live_payload_url="/api/live-payload")


def _handle_beta_action(action: str, query: dict[str, list[str]]) -> tuple[bool, str]:
    settings = load_settings()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    if action == "restart_worker":
        return restart_background_worker()
    if action == "pause_entries":
        repository.set_control_flag("trading_paused", "1", "set from beta server")
        return True, "자동 진입을 일시중지했습니다."
    if action == "resume_entries":
        repository.set_control_flag("trading_paused", "0", "set from beta server")
        return True, "자동 진입을 재개했습니다."
    if action == "halt_all":
        repository.set_control_flag("trading_paused", "1", "set from beta server")
        ok, stop_message = stop_background_worker()
        return ok, f"전체 정지 요청 완료. {stop_message}" if ok else stop_message
    if action == "sync_market":
        return run_manual_runtime_job("broker_market_status")
    if action == "sync_account":
        return run_manual_runtime_job("broker_account_sync")
    if action == "sync_order":
        return run_manual_runtime_job("broker_order_sync")
    if action == "sync_position":
        return run_manual_runtime_job("broker_position_sync")
    if action == "scan_now":
        return run_manual_scan_job()
    if action == "order_check":
        return run_manual_runtime_job("broker_order_sync")
    if action == "clear_broker_errors":
        cleared = repository.clear_broker_error_events()
        return True, f"브로커 오류 이벤트 {cleared}건을 정리했습니다."
    if action.startswith("set_strategy:"):
        target_strategy_id = action.split(":", 1)[1].strip()
        return set_default_strategy(target_strategy_id, settings)
    if action == "toggle_theme":
        return True, ""
    return False, f"알 수 없는 베타 액션입니다: {action}"


def _redirect_params_after_action(query: dict[str, list[str]], ok: bool, message: str) -> str:
    keep_keys = ["beta_anchor", "beta_cand_tab", "beta_jobs", "beta_signals", "beta_trades", "beta_theme"]
    params: Dict[str, str] = {}
    for key in keep_keys:
        value = _query_value(query, key, "")
        if value:
            params[key] = value
    if message:
        params["beta_msg"] = message
        params["beta_ok"] = "1" if ok else "0"
    target = "/beta"
    if params:
        target += "?" + urlencode(params)
    return target


class BetaRequestHandler(BaseHTTPRequestHandler):
    server_version = "AltBetaServer/1.0"

    def _send_html(self, html_text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/":
            self._redirect("/beta")
            return
        if parsed.path == "/health":
            settings = load_settings()
            repository = TradingRepository(settings.storage.db_path)
            repository.initialize()
            status = compute_auto_trading_status(repository, settings.scheduler.loop_sleep_seconds)
            self._send_json({"ok": True, "status": status})
            return
        if parsed.path == "/api/live-payload":
            settings = load_settings()
            self._send_json(_build_live_payload(settings))
            return
        if parsed.path == "/beta":
            action = _query_value(query, "beta_action", "")
            token = _query_value(query, "beta_token", "")
            if action and token:
                ok, message = _handle_beta_action(action, query)
                self._redirect(_redirect_params_after_action(query, ok, message))
                return
            self._send_html(_beta_page_html(query))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def main() -> None:
    parser = argparse.ArgumentParser(description="ALT beta-only web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8505)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, int(args.port)), BetaRequestHandler)
    print(f"ALT beta server listening on http://{args.host}:{args.port}/beta")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
