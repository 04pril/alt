from __future__ import annotations

import contextlib
import io
import logging
import re
import threading
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from kis_paper import KISPaperClient
from predictor import extract_korean_stock_code, is_korean_stock_symbol
from storage.repository import TradingRepository

_KRX_CACHE_LOCK = threading.Lock()
_KRX_CACHE_TTL_SECONDS = 60.0 * 60.0 * 6.0
_KRX_NAME_MAP_CACHE: tuple[float, Dict[str, Dict[str, str]]] = (0.0, {})


def quiet_external_call(fn):
    previous_disable = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"bs4\..*")
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return fn()
        finally:
            logging.disable(previous_disable)


def normalize_kr_name_key(text: str) -> str:
    normalized = str(text).strip().upper()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^0-9A-Z가-힣]+", "", normalized)
    return normalized


def _read_krx_tables_with_fallback(html_text: str) -> List[pd.DataFrame]:
    parse_attempts: List[dict[str, Any]] = [{}, {"flavor": ["bs4"]}, {"flavor": ["html5lib"]}]
    last_error: Exception | None = None
    for kwargs in parse_attempts:
        try:
            tables = quiet_external_call(lambda: pd.read_html(io.StringIO(html_text), **kwargs))
        except Exception as exc:
            last_error = exc
            continue
        if tables:
            return tables
    if last_error is not None:
        raise last_error
    return []


def load_krx_name_map() -> Dict[str, Dict[str, str]]:
    global _KRX_NAME_MAP_CACHE
    now = time.time()
    with _KRX_CACHE_LOCK:
        if _KRX_NAME_MAP_CACHE[0] > now:
            return dict(_KRX_NAME_MAP_CACHE[1])
    try:
        response = requests.get(
            "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        response.encoding = "euc-kr"
        tables = _read_krx_tables_with_fallback(response.text)
    except Exception:
        return {}
    if not tables:
        return {}

    df = tables[0]
    name_col, market_col, code_col = df.columns[:3]
    frame = df[[name_col, market_col, code_col]].copy()
    frame.columns = ["name", "market", "code"]
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["market"] = frame["market"].astype(str).str.strip()
    frame["code"] = frame["code"].astype(str).str.strip().str.zfill(6)
    frame = frame.drop_duplicates(subset=["name"], keep="first")

    out: Dict[str, Dict[str, str]] = {}
    for row in frame.to_dict("records"):
        name = str(row["name"]).strip()
        market = str(row["market"]).strip()
        code = str(row["code"]).strip()
        payload = {"market": market, "code": code}
        out.setdefault(name, payload)
        normalized_key = normalize_kr_name_key(name)
        if normalized_key:
            out.setdefault(normalized_key, payload)
    with _KRX_CACHE_LOCK:
        _KRX_NAME_MAP_CACHE = (now + _KRX_CACHE_TTL_SECONDS, dict(out))
    return out


def first_valid_float(*values: object, default: float = float("nan")) -> float:
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            return number
    return float(default)


def default_currency_from_symbol(symbol: str) -> str:
    upper = str(symbol or "").upper()
    if upper.endswith(".KS") or upper.endswith(".KQ") or (upper.isdigit() and len(upper) == 6):
        return "KRW"
    if upper.endswith("-USD"):
        return "USD"
    return "USD"


def _chunked(values: Tuple[str, ...], size: int) -> List[Tuple[str, ...]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _build_kr_snapshot(symbol: str, item: Dict[str, object]) -> Dict[str, float | str]:
    current_price = first_valid_float(item.get("nv"))
    previous_close = first_valid_float(item.get("pcv"))
    change_pct = float("nan")
    if np.isfinite(current_price) and np.isfinite(previous_close) and previous_close != 0:
        change_pct = (current_price / previous_close - 1.0) * 100.0
    return {
        "symbol": symbol,
        "currency": "KRW",
        "current_price": float(current_price) if np.isfinite(current_price) else float("nan"),
        "previous_close": float(previous_close) if np.isfinite(previous_close) else float("nan"),
        "change_pct": float(change_pct) if np.isfinite(change_pct) else float("nan"),
    }


def _kr_afterhours_session_mode(now: pd.Timestamp | None = None) -> str:
    local_now = now if now is not None else pd.Timestamp.now(tz="Asia/Seoul")
    if getattr(local_now, "tzinfo", None) is None:
        local_now = local_now.tz_localize("Asia/Seoul")
    else:
        local_now = local_now.tz_convert("Asia/Seoul")
    current_time = local_now.time()
    if pd.Timestamp("15:40", tz="Asia/Seoul").time() <= current_time < pd.Timestamp("16:00", tz="Asia/Seoul").time():
        return "after_close_close_price"
    if pd.Timestamp("16:00", tz="Asia/Seoul").time() <= current_time <= pd.Timestamp("18:00", tz="Asia/Seoul").time():
        return "after_close_single_price"
    return ""


def _build_kr_overtime_snapshot(
    *,
    symbol: str,
    quote: Dict[str, object],
    orderbook: Dict[str, object],
    session_mode: str,
) -> Dict[str, float | str]:
    close_price = first_valid_float(quote.get("close_price"))
    expected_price = first_valid_float(orderbook.get("expected_price"), quote.get("expected_price"))
    best_ask = first_valid_float(orderbook.get("best_ask"))
    best_bid = first_valid_float(orderbook.get("best_bid"))
    current_price = close_price if session_mode == "after_close_close_price" else expected_price
    if not np.isfinite(current_price):
        current_price = first_valid_float(close_price, expected_price)
    previous_close = close_price
    change_pct = float("nan")
    if np.isfinite(current_price) and np.isfinite(previous_close) and previous_close != 0:
        change_pct = (current_price / previous_close - 1.0) * 100.0
    return {
        "symbol": symbol,
        "currency": "KRW",
        "current_price": float(current_price) if np.isfinite(current_price) else float("nan"),
        "previous_close": float(previous_close) if np.isfinite(previous_close) else float("nan"),
        "change_pct": float(change_pct) if np.isfinite(change_pct) else float("nan"),
        "ask_price": float(best_ask) if np.isfinite(best_ask) else float("nan"),
        "bid_price": float(best_bid) if np.isfinite(best_bid) else float("nan"),
        "price_source": "kis_overtime",
        "session_mode": session_mode,
    }


def load_live_kr_quote_snapshots(
    symbols: Tuple[str, ...],
    *,
    db_path: str,
    max_age_seconds: int = 20,
) -> Dict[str, Dict[str, float | str]]:
    if not symbols:
        return {}
    repository = TradingRepository(db_path)
    repository.initialize()
    frame = repository.latest_live_market_quotes(symbols=symbols, max_age_seconds=max_age_seconds)
    if frame.empty:
        return {}
    snapshots: Dict[str, Dict[str, float | str]] = {}
    by_code = {
        str(row.get("symbol_code") or "").strip(): row.to_dict()
        for _, row in frame.iterrows()
        if str(row.get("symbol_code") or "").strip()
    }
    for symbol in symbols:
        code = extract_korean_stock_code(symbol)
        row = by_code.get(code)
        if not row:
            continue
        snapshots[symbol] = {
            "symbol": symbol,
            "currency": str(row.get("currency") or "KRW"),
            "current_price": first_valid_float(row.get("current_price")),
            "previous_close": first_valid_float(row.get("previous_close")),
            "change_pct": first_valid_float(row.get("change_pct")),
            "ask_price": first_valid_float(row.get("ask_price")),
            "bid_price": first_valid_float(row.get("bid_price")),
            "updated_at": str(row.get("updated_at") or ""),
        }
    return snapshots


def fetch_single_kr_quote_snapshot(
    symbol: str,
    *,
    db_path: str,
    prefer_overtime: bool = False,
) -> Dict[str, float | str]:
    overtime_mode = _kr_afterhours_session_mode() if prefer_overtime else ""
    if overtime_mode:
        try:
            client = KISPaperClient()
            quote = client.get_overtime_price(symbol_or_code=symbol)
            orderbook = client.get_overtime_asking_price(symbol_or_code=symbol)
            snapshot = _build_kr_overtime_snapshot(symbol=symbol, quote=quote, orderbook=orderbook, session_mode=overtime_mode)
            if np.isfinite(first_valid_float(snapshot.get("current_price"))):
                return snapshot
        except Exception:
            pass
    live_snapshot = load_live_kr_quote_snapshots((symbol,), db_path=db_path).get(symbol)
    if live_snapshot:
        return live_snapshot
    try:
        kis_quote = KISPaperClient().get_quote(symbol_or_code=symbol)
        current_price = first_valid_float(kis_quote.get("current_price"))
        previous_close = first_valid_float(kis_quote.get("previous_close"))
        change_pct = first_valid_float(kis_quote.get("change_pct"))
        return {
            "symbol": symbol,
            "currency": "KRW",
            "current_price": float(current_price) if np.isfinite(current_price) else float("nan"),
            "previous_close": float(previous_close) if np.isfinite(previous_close) else float("nan"),
            "change_pct": float(change_pct) if np.isfinite(change_pct) else float("nan"),
        }
    except Exception:
        pass

    code = extract_korean_stock_code(symbol)
    response = requests.get(
        "https://polling.finance.naver.com/api/realtime",
        params={"query": f"SERVICE_ITEM:{code}"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    for area in payload.get("result", {}).get("areas", []):
        for item in area.get("datas", []):
            if str(item.get("cd", "")).strip() == code:
                return _build_kr_snapshot(symbol=symbol, item=item)
    return {
        "symbol": symbol,
        "currency": "KRW",
        "current_price": float("nan"),
        "previous_close": float("nan"),
        "change_pct": float("nan"),
    }


def fetch_kr_quote_snapshots(
    symbols: Tuple[str, ...],
    *,
    db_path: str,
    prefer_overtime: bool = False,
) -> Dict[str, Dict[str, float | str]]:
    overtime_mode = _kr_afterhours_session_mode() if prefer_overtime else ""
    if overtime_mode:
        return {
            symbol: fetch_single_kr_quote_snapshot(symbol, db_path=db_path, prefer_overtime=True)
            for symbol in symbols
        }
    snapshots = load_live_kr_quote_snapshots(symbols, db_path=db_path)
    if not symbols:
        return snapshots
    remaining_symbols = tuple(symbol for symbol in symbols if symbol not in snapshots)
    if not remaining_symbols:
        return snapshots
    code_to_symbol = {extract_korean_stock_code(symbol): symbol for symbol in remaining_symbols}
    for chunk in _chunked(tuple(code_to_symbol.keys()), size=40):
        response = requests.get(
            "https://polling.finance.naver.com/api/realtime",
            params={"query": f"SERVICE_RECENT_ITEM:{','.join(chunk)}"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        for area in payload.get("result", {}).get("areas", []):
            for item in area.get("datas", []):
                code = str(item.get("cd", "")).strip()
                symbol = code_to_symbol.get(code)
                if not symbol:
                    continue
                snapshots[symbol] = _build_kr_snapshot(symbol=symbol, item=item)
    for symbol in remaining_symbols:
        snapshots.setdefault(symbol, fetch_single_kr_quote_snapshot(symbol, db_path=db_path))
    return snapshots


def fetch_single_quote_snapshot(symbol: str, *, db_path: str) -> Dict[str, float | str]:
    if is_korean_stock_symbol(symbol):
        try:
            return fetch_single_kr_quote_snapshot(symbol, db_path=db_path)
        except Exception:
            pass

    ticker = yf.Ticker(symbol)
    current_price = float("nan")
    previous_close = float("nan")
    currency = ""

    try:
        fast_info = quiet_external_call(lambda: ticker.fast_info) or {}
        if not isinstance(fast_info, dict):
            fast_info = {}
        current_price = first_valid_float(
            fast_info.get("last_price"),
            fast_info.get("regular_market_price"),
            fast_info.get("lastPrice"),
            fast_info.get("regularMarketPrice"),
        )
        previous_close = first_valid_float(fast_info.get("previous_close"), fast_info.get("previousClose"))
        fast_currency = fast_info.get("currency")
        if isinstance(fast_currency, str):
            currency = fast_currency.upper().strip()
    except Exception:
        pass

    if np.isnan(current_price) or np.isnan(previous_close):
        try:
            hist = quiet_external_call(lambda: ticker.history(period="5d", interval="1d", auto_adjust=False))
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty and "Close" in hist.columns:
            close_series = hist["Close"].dropna()
            if not close_series.empty:
                if np.isnan(current_price):
                    current_price = float(close_series.iloc[-1])
                if np.isnan(previous_close):
                    previous_close = float(close_series.iloc[-2] if len(close_series) >= 2 else close_series.iloc[-1])

    if np.isnan(current_price) or np.isnan(previous_close) or not currency:
        try:
            info = quiet_external_call(lambda: ticker.info) or {}
        except Exception:
            info = {}
        if not isinstance(info, dict):
            info = {}
        if np.isnan(current_price):
            current_price = first_valid_float(info.get("regularMarketPrice"), info.get("currentPrice"), info.get("navPrice"))
        if np.isnan(previous_close):
            previous_close = first_valid_float(
                info.get("previousClose"),
                info.get("regularMarketPreviousClose"),
                info.get("chartPreviousClose"),
            )
        if not currency:
            info_currency = info.get("currency")
            if isinstance(info_currency, str):
                currency = info_currency.upper().strip()

    if not currency:
        currency = default_currency_from_symbol(symbol)

    change_pct = float("nan")
    if np.isfinite(current_price) and np.isfinite(previous_close) and previous_close != 0:
        change_pct = (current_price / previous_close - 1.0) * 100.0
    return {
        "symbol": symbol,
        "currency": currency,
        "current_price": float(current_price) if np.isfinite(current_price) else float("nan"),
        "previous_close": float(previous_close) if np.isfinite(previous_close) else float("nan"),
        "change_pct": float(change_pct) if np.isfinite(change_pct) else float("nan"),
    }


def fetch_quote_snapshots(
    *,
    symbols: Tuple[str, ...],
    db_path: str,
    prefer_overtime_kr: bool = False,
) -> Dict[str, Dict[str, float | str]]:
    snapshots: Dict[str, Dict[str, float | str]] = {}
    kr_symbols = tuple(symbol for symbol in symbols if is_korean_stock_symbol(symbol))
    non_kr_symbols = tuple(symbol for symbol in symbols if not is_korean_stock_symbol(symbol))
    if kr_symbols:
        try:
            snapshots.update(fetch_kr_quote_snapshots(kr_symbols, db_path=db_path, prefer_overtime=prefer_overtime_kr))
        except Exception:
            for symbol in kr_symbols:
                snapshots[symbol] = fetch_single_quote_snapshot(symbol, db_path=db_path)
    for symbol in non_kr_symbols:
        snapshots[symbol] = fetch_single_quote_snapshot(symbol, db_path=db_path)
    return snapshots
