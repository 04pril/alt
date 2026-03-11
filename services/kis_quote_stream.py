from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Iterable, Sequence

import pandas as pd
from websockets.asyncio.client import connect

from config.settings import RuntimeSettings
from kis_paper import KISPaperClient, extract_kis_code
from runtime_accounts import ACCOUNT_KIS_KR_PAPER
from storage.models import LiveMarketQuoteRecord
from storage.repository import TradingRepository, utc_now_iso


KR_QUOTE_STREAM_URL = "ws://ops.koreainvestment.com:21000/tryitout"
KR_QUOTE_TR_ID = "H0STCNT0"
KR_QUOTE_FIELDS = (
    "MKSC_SHRN_ISCD",
    "TICK_HOUR",
    "STCK_PRPR",
    "PRDY_VRSS_SIGN",
    "PRDY_VRSS",
    "PRDY_CTRT",
    "WGHN_AVRG_STCK_PRC",
    "STCK_OPRC",
    "STCK_HGPR",
    "STCK_LWPR",
    "ASKP1",
    "BIDP1",
    "CNTG_VOL",
    "ACML_VOL",
    "ACML_TR_PBMN",
    "SELN_CNTG_CSNU",
    "SHNU_CNTG_CSNU",
    "NTBY_CNTG_CSNU",
    "CTTR",
    "SELN_CNTG_SMTN",
    "SHNU_CNTG_SMTN",
    "CCLD_DVSN",
    "SHNU_RATE",
    "PRDY_VOL_VRSS_ACML_VOL_RATE",
    "OPRC_HOUR",
    "OPRC_VRSS_PRPR_SIGN",
    "OPRC_VRSS_PRPR",
    "HGPR_HOUR",
    "HGPR_VRSS_PRPR_SIGN",
    "HGPR_VRSS_PRPR",
    "LWPR_HOUR",
    "LWPR_VRSS_PRPR_SIGN",
    "LWPR_VRSS_PRPR",
    "BSOP_DATE",
    "NEW_MKOP_CLS_CODE",
    "TRHT_YN",
    "ASKP_RSQN1",
    "BIDP_RSQN1",
    "TOTAL_ASKP_RSQN",
    "TOTAL_BIDP_RSQN",
    "VOL_TNRT",
    "PRDY_SMNS_HOUR_ACML_VOL",
    "PRDY_SMNS_HOUR_ACML_VOL_RATE",
    "HOUR_CLS_CODE",
    "MRKT_TRTM_CLS_CODE",
    "VI_STND_PRC",
)
# Runtime observation on 2026-03-11: KIS paper KR quote subscriptions
# started returning OPSP0008 / "MAX SUBSCRIBE OVER" once total active
# requests moved past roughly 40 symbols. Keep the total cap at 40.
# KIS paper websocket also rejects parallel sockets with the same appkey,
# so keep KR quotes on a single session.
MAX_KR_QUOTE_SYMBOLS_PER_CONNECTION = 40
MAX_KR_QUOTE_SYMBOLS = 40
TRANSIENT_STREAM_RECONNECT_SECONDS = 1
QUOTE_FLAG_UPDATE_INTERVAL_SECONDS = 2.0
TRANSIENT_STREAM_ERROR_PATTERNS = (
    "no close frame received or sent",
    "keepalive ping timeout",
    "connection closed",
)


def _f(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def _build_subscribe_message(approval_key: str, symbol_code: str, *, tr_type: str = "1") -> str:
    return json.dumps(
        {
            "header": {
                "approval_key": approval_key,
                "custtype": "P",
                "tr_type": tr_type,
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": KR_QUOTE_TR_ID,
                    "tr_key": symbol_code,
                }
            },
        }
    )


def _stream_connect_kwargs() -> dict[str, Any]:
    # KIS uses application-level PINGPONG frames. The websocket library's
    # protocol keepalive ping can spuriously trip 1011 timeout errors here.
    return {
        "ping_interval": None,
        "ping_timeout": None,
        "max_size": None,
        "open_timeout": 20,
    }


def _chunk_symbol_codes(symbol_codes: Sequence[str], *, batch_size: int = MAX_KR_QUOTE_SYMBOLS_PER_CONNECTION) -> tuple[tuple[str, ...], ...]:
    normalized = tuple(str(symbol).strip() for symbol in symbol_codes if str(symbol).strip())
    size = max(int(batch_size), 1)
    return tuple(
        normalized[index : index + size]
        for index in range(0, len(normalized), size)
        if normalized[index : index + size]
    )


def _is_transient_stream_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    if not message:
        return False
    return any(pattern in message for pattern in TRANSIENT_STREAM_ERROR_PATTERNS)


def parse_kr_quote_message(message: str, *, received_at: str | None = None) -> list[LiveMarketQuoteRecord]:
    if not message or message[0] not in {"0", "1"}:
        return []
    parts = str(message).split("|")
    if len(parts) < 4 or parts[1] != KR_QUOTE_TR_ID:
        return []
    raw_fields = str(parts[3] or "").split("^")
    if not raw_fields:
        return []
    rows: list[LiveMarketQuoteRecord] = []
    field_count = len(KR_QUOTE_FIELDS)
    record_time = received_at or utc_now_iso()
    for offset in range(0, len(raw_fields), field_count):
        chunk = raw_fields[offset : offset + field_count]
        if len(chunk) < 6:
            continue
        payload = {name: chunk[index] if index < len(chunk) else "" for index, name in enumerate(KR_QUOTE_FIELDS)}
        symbol_code = str(payload.get("MKSC_SHRN_ISCD") or "").strip()
        current_price = _f(payload.get("STCK_PRPR"))
        change_amount = _f(payload.get("PRDY_VRSS"))
        previous_close = current_price - change_amount if pd.notna(current_price) and pd.notna(change_amount) else float("nan")
        rows.append(
            LiveMarketQuoteRecord(
                symbol_code=symbol_code,
                symbol=symbol_code,
                asset_type="한국주식",
                currency="KRW",
                source="kis_quote_websocket",
                current_price=current_price,
                previous_close=previous_close,
                change_pct=_f(payload.get("PRDY_CTRT")),
                ask_price=_f(payload.get("ASKP1")),
                bid_price=_f(payload.get("BIDP1")),
                volume=_f(payload.get("ACML_VOL")),
                updated_at=record_time,
                raw_json=json.dumps(payload, ensure_ascii=False),
            )
        )
    return [record for record in rows if str(record.symbol_code or "").strip()]


class KISKRQuoteStream:
    def __init__(
        self,
        settings: RuntimeSettings,
        repository: TradingRepository,
        *,
        client_factory=KISPaperClient,
    ):
        self.settings = settings
        self.repository = repository
        self.client_factory = client_factory
        self._lock = threading.Lock()
        self._desired_symbols: tuple[str, ...] = ()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_quote_flag_update_monotonic = 0.0

    def is_enabled(self) -> bool:
        try:
            return bool(self.client_factory().config.is_paper)
        except Exception:
            return False

    def _set_status(self, status: str, notes: str = "") -> None:
        self.repository.set_control_flag("kis_quote_stream_status", status, notes[:200])

    def _desired_snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return self._desired_symbols

    def _candidate_symbols(self) -> tuple[str, ...]:
        symbols: list[str] = []
        positions = self.repository.open_positions(account_id=ACCOUNT_KIS_KR_PAPER)
        if not positions.empty and "symbol" in positions.columns:
            symbols.extend(positions["symbol"].dropna().astype(str).tolist())
        orders = self.repository.open_orders(account_id=ACCOUNT_KIS_KR_PAPER)
        if not orders.empty and "symbol" in orders.columns:
            symbols.extend(orders["symbol"].dropna().astype(str).tolist())
        candidates = self.repository.latest_candidates(
            asset_type="한국주식",
            execution_account_id=ACCOUNT_KIS_KR_PAPER,
            limit=20,
        )
        if not candidates.empty and "symbol" in candidates.columns:
            symbols.extend(candidates["symbol"].dropna().astype(str).tolist())
        universe = self.settings.universes.get("한국주식")
        if universe is not None:
            symbols.extend(list(getattr(universe, "watchlist", []) or []))
            symbols.extend(list(getattr(universe, "top_universe", []) or []))
        normalized: list[str] = []
        for symbol in symbols:
            try:
                symbol_code = extract_kis_code(symbol)
            except Exception:
                continue
            if symbol_code not in normalized:
                normalized.append(symbol_code)
            if len(normalized) >= MAX_KR_QUOTE_SYMBOLS:
                break
        return tuple(normalized)

    def refresh_symbols(self, symbols: Sequence[str] | None = None) -> bool:
        deduped = tuple(dict.fromkeys(str(symbol).strip() for symbol in (symbols or self._candidate_symbols()) if str(symbol).strip()))
        target = deduped[:MAX_KR_QUOTE_SYMBOLS]
        with self._lock:
            changed = target != self._desired_symbols
            self._desired_symbols = target
        self.repository.set_control_flag("kis_quote_stream_symbols", ",".join(target), "KR quote websocket subscriptions")
        return changed

    def start(self) -> bool:
        if not self.is_enabled():
            self._set_status("disabled", "KIS paper configuration unavailable")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_thread, name="kis-quote-stream", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def _run_thread(self) -> None:
        asyncio.run(self._run())

    async def _handle_message(self, ws, message: str) -> None:
        stripped = message.strip()
        if not stripped:
            return
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            header = payload.get("header") or {}
            body = payload.get("body") or {}
            tr_id = str(header.get("tr_id") or "")
            if tr_id == "PINGPONG":
                await ws.send(stripped)
                return
            if body.get("rt_cd") not in (None, "0"):
                self.repository.log_event(
                    "WARNING",
                    "kis_quote_stream",
                    "stream_notice",
                    str(body.get("msg1") or "KIS websocket notice"),
                    {"payload": payload},
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
            return
        rows = parse_kr_quote_message(stripped)
        if rows:
            await asyncio.to_thread(self._persist_quote_rows, rows)

    def _persist_quote_rows(self, rows: list[LiveMarketQuoteRecord]) -> None:
        self.repository.upsert_live_market_quotes(rows)
        now_monotonic = time.monotonic()
        if (
            self._last_quote_flag_update_monotonic <= 0.0
            or (now_monotonic - self._last_quote_flag_update_monotonic) >= QUOTE_FLAG_UPDATE_INTERVAL_SECONDS
            or len(rows) > 1
        ):
            self.repository.set_control_flag(
                "kis_last_websocket_quote_at",
                rows[-1].updated_at,
                f"{len(rows)} quote updates",
            )
            self._last_quote_flag_update_monotonic = now_monotonic

    async def _consume_batch(
        self,
        *,
        approval_key: str,
        desired_symbols: tuple[str, ...],
        symbol_batch: tuple[str, ...],
    ) -> None:
        async with connect(KR_QUOTE_STREAM_URL, **_stream_connect_kwargs()) as ws:
            for symbol_code in symbol_batch:
                await ws.send(_build_subscribe_message(approval_key, symbol_code))
            while not self._stop_event.is_set():
                if desired_symbols != self._desired_snapshot():
                    return
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(message, str):
                    await self._handle_message(ws, message)

    async def _run(self) -> None:
        reconnect_wait = max(int(self.settings.broker.websocket_reconnect_interval_seconds), 5)
        while not self._stop_event.is_set():
            desired_symbols = self._desired_snapshot()
            if not desired_symbols:
                self._set_status("idle", "no KR symbols selected")
                await asyncio.sleep(5)
                continue
            try:
                client = self.client_factory()
                approval_key = client.get_websocket_approval_key()
                symbol_batches = _chunk_symbol_codes(desired_symbols)
                self._set_status("connecting", f"{len(desired_symbols)} symbols / {len(symbol_batches)} sessions")
                tasks = [
                    asyncio.create_task(
                        self._consume_batch(
                            approval_key=approval_key,
                            desired_symbols=desired_symbols,
                            symbol_batch=batch,
                        )
                    )
                    for batch in symbol_batches
                ]
                self._set_status("connected", f"{len(desired_symbols)} symbols / {len(symbol_batches)} sessions")
                self.repository.log_event(
                    "INFO",
                    "kis_quote_stream",
                    "stream_connected",
                    "KIS KR quote websocket connected",
                    {
                        "symbols": list(desired_symbols),
                        "session_count": len(symbol_batches),
                        "batch_sizes": [len(batch) for batch in symbol_batches],
                    },
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if desired_symbols != self._desired_snapshot():
                    continue
            except Exception as exc:
                transient = _is_transient_stream_error(exc)
                self._set_status("reconnecting", str(exc))
                self.repository.log_event(
                    "WARNING" if transient else "ERROR",
                    "kis_quote_stream",
                    "stream_reconnect" if transient else "stream_error",
                    "KIS KR quote websocket reconnecting" if transient else "KIS KR quote websocket error",
                    {"error": str(exc)},
                    account_id=ACCOUNT_KIS_KR_PAPER,
                )
                await asyncio.sleep(TRANSIENT_STREAM_RECONNECT_SECONDS if transient else reconnect_wait)
