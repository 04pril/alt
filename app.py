from __future__ import annotations

import contextlib
import html
import io
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import streamlit.config as st_config
import streamlit.components.v1 as components
import yfinance as yf

from beta_monitor_clone import render_beta_overview_component
from config.settings import load_settings
from kis_paper import (
    KISPaperClient,
    KISPaperError,
    append_equity_snapshot,
    append_order_log,
    compute_equity_metrics,
    load_equity_curve,
    load_order_log,
)
from monitoring.dashboard_hooks import load_dashboard_data, load_monitor_open_positions, load_monitor_recent_orders
from prediction_store import (
    attach_order_to_prediction,
    filter_prediction_history,
    load_prediction_log,
    load_model_registry,
    prediction_id_for_run,
    refresh_prediction_actuals,
    save_prediction_snapshot,
    summarize_prediction_accuracy,
)
from predictor import extract_korean_stock_code, is_korean_stock_symbol, normalize_symbol, run_forecast
from runtime_accounts import ACCOUNT_KIS_KR_PAPER, ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.repository import TradingRepository
from top100_universe import (
    KR_NAME_ALIASES,
    KR_MANUAL_SYMBOLS,
    KR_TOP100_NAMES,
    US_NAME_QUERY_OVERRIDES,
    US_TOP100_NAME_SYMBOLS,
)
from ui.floating_nav import FloatingNavItem, render_floating_nav, render_navigation_fallback, resolve_current_page_key


WATCHLIST_PRESETS: Dict[str, List[str]] = {
    "코인": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD"],
    "미국주식": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"],
    "한국주식": ["005930", "000660", "035420", "005380", "035720", "068270", "247540", "373220"],
}

PAPER_VIEW_NAME = "모의매매"
PAPER_ORDER_SIDE_LABELS = {"buy": "매수", "sell": "매도", "hold": "관망"}
OPERATIONS_ORDER_STATUS_LABELS = {
    "new": "접수",
    "submitted": "제출",
    "acknowledged": "접수확인",
    "pending_fill": "체결대기",
    "partially_filled": "부분체결",
    "filled": "체결완료",
    "rejected": "거부",
    "cancelled": "취소",
    "expired": "만료",
}
OPERATIONS_ORDER_REASON_LABELS = {
    "entry": "신규진입",
    "manual_exit": "수동청산",
    "stop_loss": "손절",
    "take_profit": "익절",
    "trailing_stop": "추적손절",
    "time_stop": "시간청산",
    "opposite_signal": "반대신호",
    "score_decay": "점수약화",
}
PAPER_ORDER_TYPE_OPTIONS = {"시장가": "market", "지정가": "limit"}
GLOBAL_DATA_PROVIDER_TEXT = (
    "미국주식·코인은 Yahoo Finance, 한국주식 분석은 KIS 우선(실패 시 네이버 fallback), "
    "대시보드 대량 시세는 네이버 배치 조회를 사용합니다."
)
GLOBAL_DISCLAIMER_TEXT = (
    "실험/학습 목적 도구입니다. 예측 정확도와 매매 성과는 다를 수 있으며, 어떤 모델도 미래 가격을 보장하지 않습니다."
)
VIEW_CODE_TO_LABEL = {
    "monitor": "운영 모니터",
    "beta": "운영 모니터 (베타)",
    "dashboard": "대시보드",
    "dashboard_dev": "개발자 대시보드",
    "analysis": "종목 분석",
    "paper": PAPER_VIEW_NAME,
    "scan": "종목 스캔",
}
NAV_ITEMS: List[FloatingNavItem] = [
    FloatingNavItem(key="analysis", label="종목 분석", icon="◌"),
    FloatingNavItem(key="paper", label=PAPER_VIEW_NAME, icon="◎"),
    FloatingNavItem(key="dashboard", label="대시보드", icon="◫"),
    FloatingNavItem(key="monitor", label="운영 모니터", icon="◧"),
    FloatingNavItem(key="scan", label="종목 스캔", icon="▣"),
]

# Change to "classic" to revert the monitor page visual refresh quickly.
MONITOR_UI_VARIANT = "polished"
# Beta monitor can be reverted quickly to the legacy Streamlit layout.
BETA_MONITOR_UI_VARIANT = "clone_v2"
# Keep the public dashboard on the classic flow until a redesign is promoted.
DASHBOARD_UI_VARIANT = "classic"

VALIDATION_LABEL_TO_MODE = {
    "빠름(홀드아웃)": "holdout",
    "엄격(워크포워드 라이트)": "walk_forward",
}

TRADE_METRIC_LABELS = {
    "trades": "거래횟수",
    "win_rate_pct": "승률(%)",
    "expectancy_pct": "기대값/거래(%)",
    "net_cum_return_pct": "누적수익률(%)",
    "max_drawdown_pct": "최대낙폭(%)",
    "profit_factor": "ProfitFactor",
    "exposure_pct": "노출비율(%)",
    "avg_win_pct": "평균이익(%)",
    "avg_loss_pct": "평균손실(%)",
    "round_trip_cost_bps_assumed": "왕복비용(bps)",
    "signal_threshold_pct": "최소신호강도(%)",
    "allow_short": "숏허용(1/0)",
    "avg_position_abs_pct": "평균포지션크기(%)",
}

VALIDATION_ITEM_LABELS = {
    "validation_mode": "검증모드",
    "target_mode": "타깃모드",
    "trade_mode": "매매손익기준",
    "gap_days_total": "총 갭일수(purge+embargo)",
    "purge_days": "purge 일수",
    "embargo_days": "embargo 일수",
    "validation_days": "validation 일수",
    "research_test_days": "연구구간 테스트 일수",
    "final_holdout_days": "최종 홀드아웃 일수",
    "selected_signal_threshold_pct": "선택된 신호강도(%)",
    "allow_short": "숏허용(1/0)",
    "target_daily_vol_pct": "목표일변동성(%)",
    "max_position_size": "최대포지션크기(배)",
    "stop_loss_atr_mult": "ATR 손절 배수",
    "take_profit_atr_mult": "ATR 익절 배수",
    "vol_low_threshold_pct": "저변동성 경계(%)",
    "vol_high_threshold_pct": "고변동성 경계(%)",
}


def parse_symbols(raw_text: str) -> List[str]:
    normalized = raw_text.replace("\n", ",")
    return [token.strip().upper() for token in normalized.split(",") if token.strip()]


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def dedupe_symbol_pairs(values: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    output: List[Tuple[str, str]] = []
    for name, symbol in values:
        key = symbol.upper().strip()
        if key and key not in seen:
            output.append((name, key))
            seen.add(key)
    return output


def looks_like_symbol(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9.\-]{1,12}", text.strip().upper()))


def grade_from_score(score: float) -> str:
    if score >= 75:
        return "강"
    if score >= 60:
        return "관심"
    return "관찰"


def metric_value(metric_df: pd.DataFrame, key: str, default: float = 0.0) -> float:
    row = metric_df.loc[metric_df["metric"] == key, "value"]
    if row.empty:
        return default
    value = float(row.iloc[0])
    if np.isnan(value):
        return default
    return value


def model_metric_value(metrics_df: pd.DataFrame, model: str, key: str, default: float = 0.0) -> float:
    row = metrics_df.loc[metrics_df["model"] == model, key]
    if row.empty:
        return default
    value = float(row.iloc[0])
    if np.isnan(value):
        return default
    return value


def to_trade_view(metric_df: pd.DataFrame) -> pd.DataFrame:
    view = metric_df.copy()
    view["지표"] = view["metric"].map(TRADE_METRIC_LABELS).fillna(view["metric"])
    view = view[["지표", "value"]].rename(columns={"value": "값"})
    return view


def to_model_view(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.rename(
        columns={
            "model": "모델",
            "mae": "MAE",
            "mape_pct": "MAPE(%)",
            "direction_acc_pct": "방향정확도(%)",
        }
    )


def to_validation_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    view = summary_df.copy()
    view["항목"] = view["item"].map(VALIDATION_ITEM_LABELS).fillna(view["item"])
    view = view[["항목", "value"]].rename(columns={"value": "값"})
    return view


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


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_krx_name_map() -> Dict[str, Dict[str, str]]:
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
    return out


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def search_symbol_from_yf(query: str, market_hint: str) -> str | None:
    q = query.strip()
    if not q:
        return None

    try:
        quotes = quiet_external_call(lambda: yf.Search(q, max_results=12).quotes)
    except Exception:
        return None
    if not quotes:
        return None

    def valid_symbol(item: dict) -> bool:
        symbol = str(item.get("symbol", "")).upper()
        exchange = str(item.get("exchange", "")).upper()
        if not symbol:
            return False
        if market_hint == "KR":
            return symbol.endswith(".KS") or symbol.endswith(".KQ")
        if symbol.endswith(".KS") or symbol.endswith(".KQ"):
            return False
        if exchange in {"NMS", "NAS", "NYQ", "NYS", "ASE", "AMEX", "PCX", "BATS", "NGM", "NCM", "NMQ"}:
            return True
        return looks_like_symbol(symbol)

    for item in quotes:
        if valid_symbol(item):
            return str(item.get("symbol")).upper()

    first_symbol = str(quotes[0].get("symbol", "")).upper()
    return first_symbol if first_symbol else None


def resolve_kr_name_to_symbol(name: str) -> str | None:
    if name in KR_MANUAL_SYMBOLS:
        return KR_MANUAL_SYMBOLS[name].upper()

    candidate_names = [name]
    alias = KR_NAME_ALIASES.get(name)
    if alias and alias != name:
        candidate_names.append(alias)

    name_map = load_krx_name_map()
    for candidate in candidate_names:
        keys = [candidate]
        normalized_key = normalize_kr_name_key(candidate)
        if normalized_key:
            keys.append(normalized_key)
        for key in keys:
            if key in name_map:
                code = name_map[key]["code"]
                market = name_map[key]["market"]
                suffix = ".KQ" if "코스닥" in market else ".KS"
                return f"{code}{suffix}".upper()

    return None


def resolve_us_name_to_symbol(name: str, symbol_hint: str | None) -> str | None:
    if symbol_hint:
        return symbol_hint.upper()
    if looks_like_symbol(name):
        return name.upper()

    query = US_NAME_QUERY_OVERRIDES.get(name, name)
    return search_symbol_from_yf(query=query, market_hint="US")


def build_top100_entries(asset_type: str) -> List[Dict[str, str | int | None]]:
    if asset_type == "한국주식":
        entries = []
        for i, name in enumerate(KR_TOP100_NAMES, start=1):
            entries.append({"rank": i, "name": name, "symbol_hint": KR_MANUAL_SYMBOLS.get(name)})
        return entries

    if asset_type == "미국주식":
        entries = []
        for i, (name, symbol) in enumerate(US_TOP100_NAME_SYMBOLS, start=1):
            entries.append({"rank": i, "name": name, "symbol_hint": symbol})
        return entries

    return []


def resolve_top100_entries(asset_type: str, entries: List[Dict[str, str | int | None]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    resolved: List[Tuple[str, str]] = []
    unresolved: List[str] = []

    for entry in entries:
        name = str(entry["name"])
        symbol_hint = entry.get("symbol_hint")

        if asset_type == "한국주식":
            symbol = resolve_kr_name_to_symbol(name=name)
        elif asset_type == "미국주식":
            symbol = resolve_us_name_to_symbol(name=name, symbol_hint=str(symbol_hint) if symbol_hint else None)
        else:
            symbol = None

        if symbol:
            resolved.append((name, symbol))
        else:
            unresolved.append(name)

    return resolved, unresolved


def build_single_top100_entries(scope: str) -> List[Dict[str, str | int | None]]:
    def pack(asset: str, market_label: str) -> List[Dict[str, str | int | None]]:
        return [
            {
                "rank": int(entry["rank"]),
                "name": str(entry["name"]),
                "symbol_hint": entry.get("symbol_hint"),
                "asset_type": asset,
                "market_label": market_label,
            }
            for entry in build_top100_entries(asset_type=asset)
        ]

    kr_entries = pack("한국주식", "국내")
    us_entries = pack("미국주식", "해외")

    if scope == "국내":
        selected = kr_entries
    elif scope == "해외":
        selected = us_entries
    else:
        selected: List[Dict[str, str | int | None]] = []
        max_len = max(len(kr_entries), len(us_entries))
        for idx in range(max_len):
            if idx < len(kr_entries):
                selected.append(kr_entries[idx])
            if idx < len(us_entries):
                selected.append(us_entries[idx])

    output: List[Dict[str, str | int | None]] = []
    for i, entry in enumerate(selected, start=1):
        enriched = dict(entry)
        enriched["display_rank"] = i
        output.append(enriched)
    return output


@st.cache_data(ttl=60 * 10, show_spinner=False)
def resolve_single_top100_entries(scope: str, display_limit: int) -> Tuple[List[Dict[str, str | int | None]], List[str]]:
    entries = build_single_top100_entries(scope=scope)[: max(display_limit, 0)]
    resolved_map: Dict[Tuple[str, str], str] = {}
    unresolved_names: List[str] = []

    for market in ("한국주식", "미국주식"):
        market_entries = [entry for entry in entries if entry["asset_type"] == market]
        if not market_entries:
            continue

        resolved_pairs, unresolved = resolve_top100_entries(asset_type=market, entries=market_entries)
        for name, symbol in resolved_pairs:
            resolved_map[(market, name)] = symbol

        market_label = "국내" if market == "한국주식" else "해외"
        unresolved_names.extend([f"[{market_label}] {name}" for name in unresolved])

    rows: List[Dict[str, str | int | None]] = []
    for entry in entries:
        market = str(entry["asset_type"])
        name = str(entry["name"])
        symbol = resolved_map.get((market, name))
        rows.append(
            {
                "순위": int(entry["display_rank"]),
                "원순위": int(entry["rank"]),
                "시장": str(entry["market_label"]),
                "자산유형": market,
                "종목명": name,
                "심볼": symbol,
                "심볼힌트": entry.get("symbol_hint"),
            }
        )
    return rows, unresolved_names


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
    upper = symbol.upper()
    if upper.endswith(".KS") or upper.endswith(".KQ"):
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


def fetch_single_kr_quote_snapshot(symbol: str) -> Dict[str, float | str]:
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
    areas = payload.get("result", {}).get("areas", [])
    for area in areas:
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


def fetch_kr_quote_snapshots(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, float | str]]:
    snapshots: Dict[str, Dict[str, float | str]] = {}
    if not symbols:
        return snapshots

    code_to_symbol = {extract_korean_stock_code(symbol): symbol for symbol in symbols}
    for chunk in _chunked(tuple(code_to_symbol.keys()), size=40):
        response = requests.get(
            "https://polling.finance.naver.com/api/realtime",
            params={"query": f"SERVICE_RECENT_ITEM:{','.join(chunk)}"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        areas = payload.get("result", {}).get("areas", [])
        for area in areas:
            for item in area.get("datas", []):
                code = str(item.get("cd", "")).strip()
                symbol = code_to_symbol.get(code)
                if not symbol:
                    continue
                snapshots[symbol] = _build_kr_snapshot(symbol=symbol, item=item)

    for symbol in symbols:
        snapshots.setdefault(symbol, fetch_single_kr_quote_snapshot(symbol))
    return snapshots


def fetch_single_quote_snapshot(symbol: str) -> Dict[str, float | str]:
    if is_korean_stock_symbol(symbol):
        try:
            return fetch_single_kr_quote_snapshot(symbol)
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
        previous_close = first_valid_float(
            fast_info.get("previous_close"),
            fast_info.get("previousClose"),
        )
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
                    if len(close_series) >= 2:
                        previous_close = float(close_series.iloc[-2])
                    else:
                        previous_close = float(close_series.iloc[-1])

    if np.isnan(current_price) or np.isnan(previous_close) or not currency:
        try:
            info = quiet_external_call(lambda: ticker.info) or {}
        except Exception:
            info = {}
        if not isinstance(info, dict):
            info = {}

        if np.isnan(current_price):
            current_price = first_valid_float(
                info.get("regularMarketPrice"),
                info.get("currentPrice"),
                info.get("navPrice"),
            )
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


@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote_snapshots(symbols: Tuple[str, ...], refresh_token: int) -> Dict[str, Dict[str, float | str]]:
    _ = refresh_token
    snapshots: Dict[str, Dict[str, float | str]] = {}
    kr_symbols = tuple(symbol for symbol in symbols if is_korean_stock_symbol(symbol))
    non_kr_symbols = tuple(symbol for symbol in symbols if not is_korean_stock_symbol(symbol))

    if kr_symbols:
        try:
            snapshots.update(fetch_kr_quote_snapshots(kr_symbols))
        except Exception:
            for symbol in kr_symbols:
                snapshots[symbol] = fetch_single_quote_snapshot(symbol=symbol)

    for symbol in non_kr_symbols:
        snapshots[symbol] = fetch_single_quote_snapshot(symbol=symbol)
    return snapshots


def format_live_price(value: float, currency: str) -> str:
    if not np.isfinite(value):
        return "N/A"
    if currency in {"KRW", "JPY"}:
        return f"{value:,.0f} {currency}"
    return f"{value:,.2f} {currency}"


def render_top100_snapshot_cards(rows: List[Dict[str, str | int | float | None]]) -> None:
    for row in rows:
        rank = int(row["순위"])
        market = html.escape(str(row["시장"]))
        name = html.escape(str(row["종목명"]))
        symbol = html.escape(str(row["심볼"] or "-"))
        currency = str(row.get("통화", "USD"))
        current_price = float(row.get("현재가", float("nan")))
        change_pct = float(row.get("전일대비(%)", float("nan")))

        price_text = format_live_price(value=current_price, currency=currency)
        if np.isfinite(change_pct):
            change_text = f"{change_pct:+.2f}%"
            if change_pct > 0:
                change_color = "#ef4444"
            elif change_pct < 0:
                change_color = "#3b82f6"
            else:
                change_color = "#9ca3af"
        else:
            change_text = "N/A"
            change_color = "#9ca3af"

        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border:1px solid #2b2f39;border-radius:10px;margin-bottom:8px;background:#10151f;">
              <div style="display:flex;align-items:center;gap:12px;">
                <div style="font-size:20px;font-weight:700;color:#3b82f6;min-width:28px;">{rank}</div>
                <div>
                  <div style="font-size:17px;font-weight:700;color:#f8fafc;line-height:1.15;">{name}</div>
                  <div style="font-size:13px;color:#94a3b8;">{market} · {symbol}</div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:16px;font-weight:700;color:#f8fafc;">{price_text}</div>
                <div style="font-size:14px;font-weight:700;color:{change_color};">{change_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def detect_mobile_client() -> bool:
    user_agent = ""
    try:
        headers = getattr(st.context, "headers", None)
        if headers is not None:
            user_agent = str(headers.get("User-Agent", "") or headers.get("user-agent", ""))
    except Exception:
        user_agent = ""
    if not user_agent:
        return False
    return bool(re.search(r"mobile|android|iphone|ipad|ipod", user_agent, flags=re.IGNORECASE))


def get_streamlit_theme_mode() -> str:
    try:
        mode = str(getattr(st.context.theme, "type", "") or "").lower()
    except Exception:
        mode = ""
    if mode in {"light", "dark"}:
        return mode
    configured = str(st.get_option("theme.base") or "").lower()
    if configured in {"light", "dark"}:
        return configured
    return "light"


def get_active_theme_mode(theme_mode: str | None = None) -> str:
    stored = str(st.session_state.get("ui_theme_mode", "") or "").lower()
    if stored in {"light", "dark"}:
        return stored
    requested = str(theme_mode or "").lower()
    if requested in {"light", "dark"}:
        return requested
    return get_streamlit_theme_mode()


def apply_responsive_css(is_mobile_ui: bool, theme_mode: str) -> None:
    theme_mode = "dark" if theme_mode == "dark" else "light"
    border = "rgba(148, 163, 184, 0.16)"
    text = "inherit"
    muted = "#94a3b8" if theme_mode == "dark" else "#64748b"
    status_running = "#6ee7b7" if theme_mode == "dark" else "#059669"
    status_paused = "#fcd34d" if theme_mode == "dark" else "#d97706"
    status_stopped = "#fca5a5" if theme_mode == "dark" else "#dc2626"
    pio.templates.default = "plotly_dark" if theme_mode == "dark" else "plotly_white"

    css_vars = {
        "theme_mode": theme_mode,
        "border": border,
        "text": text,
        "muted": muted,
        "status_running": status_running,
        "status_paused": status_paused,
        "status_stopped": status_stopped,
    }

    if is_mobile_ui:
        css = """
            <style>
            :root {
              color-scheme: %(theme_mode)s;
            }
            .block-container {
              padding-top: 3.45rem !important;
              padding-bottom: 1.0rem !important;
              padding-left: 0.8rem !important;
              padding-right: 0.8rem !important;
            }
            .alt-page-title {
              margin: 0.2rem 0 1.0rem 0 !important;
              font-size: 3rem !important;
              line-height: 1.0 !important;
              font-weight: 800 !important;
              letter-spacing: -0.04em;
              color: %(text)s;
            }
            .alt-status-badge {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              padding: 0.28rem 0.62rem;
              border-radius: 999px;
              font-size: 0.74rem;
              font-weight: 800;
              letter-spacing: 0.01em;
              border: 1px solid transparent;
            }
            .alt-status-running {
              background: rgba(16, 185, 129, 0.14);
              border-color: rgba(16, 185, 129, 0.35);
              color: %(status_running)s;
            }
            .alt-status-paused {
              background: rgba(245, 158, 11, 0.14);
              border-color: rgba(245, 158, 11, 0.35);
              color: %(status_paused)s;
            }
            .alt-status-stopped {
              background: rgba(239, 68, 68, 0.14);
              border-color: rgba(239, 68, 68, 0.35);
              color: %(status_stopped)s;
            }
            .alt-global-footer {
              margin-top: 2.1rem;
              padding-top: 1.0rem;
              border-top: 1px solid %(border)s;
            }
            .alt-global-footer p {
              margin: 0.2rem 0;
              font-size: 0.76rem;
              line-height: 1.45;
              color: %(muted)s;
            }
            h1 { font-size: 1.4rem !important; }
            h2, h3 { font-size: 1.08rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
            [data-testid="stTabs"] button[role="tab"] {
              font-size: 0.85rem !important;
              padding: 0.45rem 0.5rem !important;
            }
            </style>
        """ % css_vars
        st.markdown(css, unsafe_allow_html=True)
    else:
        css = """
            <style>
            :root {
              color-scheme: %(theme_mode)s;
            }
            .block-container {
              padding-top: 3.75rem !important;
              padding-bottom: 1.2rem !important;
              padding-left: 1.4rem !important;
              padding-right: 1.4rem !important;
            }
            .alt-page-title {
              margin: 0.2rem 0 1.05rem 0 !important;
              font-size: 4rem !important;
              line-height: 0.98 !important;
              font-weight: 800 !important;
              letter-spacing: -0.05em;
              color: %(text)s;
            }
            .alt-status-badge {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              padding: 0.32rem 0.72rem;
              border-radius: 999px;
              font-size: 0.76rem;
              font-weight: 800;
              letter-spacing: 0.01em;
              border: 1px solid transparent;
            }
            .alt-status-running {
              background: rgba(16, 185, 129, 0.14);
              border-color: rgba(16, 185, 129, 0.35);
              color: %(status_running)s;
            }
            .alt-status-paused {
              background: rgba(245, 158, 11, 0.14);
              border-color: rgba(245, 158, 11, 0.35);
              color: %(status_paused)s;
            }
            .alt-status-stopped {
              background: rgba(239, 68, 68, 0.14);
              border-color: rgba(239, 68, 68, 0.35);
              color: %(status_stopped)s;
            }
            .alt-global-footer {
              margin-top: 2.4rem;
              padding-top: 1.15rem;
              border-top: 1px solid %(border)s;
            }
            .alt-global-footer p {
              margin: 0.22rem 0;
              font-size: 0.8rem;
              line-height: 1.5;
              color: %(muted)s;
            }
            </style>
        """ % css_vars
        st.markdown(css, unsafe_allow_html=True)


def format_heartbeat_age_text(seconds: object) -> str:
    value = first_valid_float(seconds)
    if not np.isfinite(value):
        return "heartbeat 없음"
    total_seconds = int(max(value, 0.0))
    if total_seconds < 60:
        return f"{total_seconds}초 전 heartbeat"
    total_minutes = total_seconds // 60
    if total_minutes < 60:
        return f"{total_minutes}분 전 heartbeat"
    total_hours, rem_minutes = divmod(total_minutes, 60)
    if total_hours < 24:
        return f"{total_hours}시간 {rem_minutes}분 전 heartbeat" if rem_minutes else f"{total_hours}시간 전 heartbeat"
    total_days, rem_hours = divmod(total_hours, 24)
    return f"{total_days}일 {rem_hours}시간 전 heartbeat" if rem_hours else f"{total_days}일 전 heartbeat"


def auto_trading_status_text(auto_trading_status: Dict[str, Any]) -> str:
    state = str(auto_trading_status.get("state", "stopped")).lower()
    reason = str(auto_trading_status.get("reason", "") or "")
    heartbeat_text = format_heartbeat_age_text(auto_trading_status.get("heartbeat_age_seconds"))
    heartbeat_kst = str(auto_trading_status.get("heartbeat_at_kst", "") or "")
    if state == "running":
        return f"{heartbeat_kst} · {heartbeat_text}" if heartbeat_kst else heartbeat_text
    if state == "paused":
        base = f"{heartbeat_kst} · {heartbeat_text}" if heartbeat_kst else heartbeat_text
        return f"{base} · 신규 진입 중단"
    if reason and heartbeat_kst:
        return f"{heartbeat_kst} · {reason}"
    return reason or heartbeat_text


def auto_trading_badge_html(auto_trading_status: Dict[str, Any]) -> str:
    state = str(auto_trading_status.get("state", "stopped")).lower()
    label = str(auto_trading_status.get("label", "Stopped") or "Stopped")
    state_class = {
        "running": "alt-status-running",
        "paused": "alt-status-paused",
        "stopped": "alt-status-stopped",
    }.get(state, "alt-status-stopped")
    return f'<span class="alt-status-badge {state_class}">{html.escape(label)}</span>'


def format_display_timestamp(value: object) -> str:
    if value is None:
        return ""
    try:
        timestamp = pd.to_datetime(value, errors="coerce")
    except Exception:
        return str(value)
    if pd.isna(timestamp):
        return str(value)
    if getattr(timestamp, "tzinfo", None) is not None:
        try:
            timestamp = timestamp.tz_convert("Asia/Seoul")
        except Exception:
            try:
                timestamp = timestamp.tz_localize(None)
            except Exception:
                pass
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_frame_timestamps_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = frame.copy()
    for column in view.columns:
        if column.endswith("_at") or column in {"created_at", "updated_at", "resolved_at", "closed_at", "cooldown_until"}:
            view[column] = view[column].map(format_display_timestamp)
    return view


def format_job_health_for_display(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = format_frame_timestamps_for_display(frame)
    if "job" not in view.columns and "job_name" in view.columns:
        view = view.rename(columns={"job_name": "job"})
    if limit is not None:
        return view.head(limit)
    return view


MONITOR_STATUS_LABELS = {
    "completed": "완료",
    "running": "실행 중",
    "failed": "실패",
    "never": "대기",
    "queued": "대기열",
    "cancelled": "취소",
    "submitted": "제출됨",
    "acknowledged": "접수됨",
    "pending_fill": "체결 대기",
    "partially_filled": "부분 체결",
    "filled": "체결 완료",
    "rejected": "거절됨",
}

MONITOR_JOB_LABELS = {
    "broker_account_sync": "계좌 동기화",
    "broker_order_sync": "주문 동기화",
    "broker_position_sync": "포지션 동기화",
    "broker_market_status": "장 상태 확인",
    "outcome_resolution": "성과 정리",
    "daily_report": "일일 보고서",
    "retrain_check": "재학습 점검",
}

MONITOR_FRAME_COLUMN_LABELS = {
    "job": "작업",
    "job_name": "작업",
    "status": "상태",
    "started_at": "시작 시각",
    "finished_at": "종료 시각",
    "scheduled_at": "예정 시각",
    "heartbeat_at": "마지막 갱신",
    "retry_count": "재시도",
    "error_message": "오류 메시지",
    "created_at": "생성 시각",
    "updated_at": "수정 시각",
    "resolved_at": "해결 시각",
    "closed_at": "종료 시각",
    "symbol": "종목",
    "asset_type": "자산",
    "timeframe": "주기",
    "side": "방향",
    "quantity": "수량",
    "entry_price": "진입가",
    "mark_price": "현재가",
    "unrealized_pnl": "평가손익",
    "realized_pnl": "실현손익",
    "prediction_id": "예측 ID",
    "level": "수준",
    "component": "구성요소",
    "event_type": "이벤트",
    "message": "메시지",
    "details_json": "상세 JSON",
    "details": "상세 정보",
}


def _monitor_status_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return MONITOR_STATUS_LABELS.get(text.lower(), text)


def _monitor_job_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return MONITOR_JOB_LABELS.get(text, text)


def _localized_auto_trading_label(auto_trading_status: Dict[str, Any]) -> str:
    state = str(auto_trading_status.get("state", "") or "").lower()
    if state == "running":
        return "가동 중"
    if state == "paused":
        return "일시중지"
    if state == "stopped":
        return "중지"
    label = str(auto_trading_status.get("label", "") or "").strip()
    return label or "중지"


def _build_monitor_auto_trading_badge_html(auto_trading_status: Dict[str, Any]) -> str:
    state = str(auto_trading_status.get("state", "stopped")).lower()
    state_class = {
        "running": "alt-status-running",
        "paused": "alt-status-paused",
        "stopped": "alt-status-stopped",
    }.get(state, "alt-status-stopped")
    label = _localized_auto_trading_label(auto_trading_status)
    return f'<span class="alt-status-badge {state_class}">{html.escape(label)}</span>'


def build_monitor_table_view(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    view = format_frame_timestamps_for_display(frame)
    if limit is not None:
        view = view.head(limit)
    if "job_name" in view.columns:
        view["job_name"] = view["job_name"].map(_monitor_job_label)
    if "job" in view.columns:
        view["job"] = view["job"].map(_monitor_job_label)
    if "status" in view.columns:
        view["status"] = view["status"].map(_monitor_status_label)
    return view.rename(columns={key: value for key, value in MONITOR_FRAME_COLUMN_LABELS.items() if key in view.columns})


def restart_background_worker() -> Tuple[bool, str]:
    workspace = Path(__file__).resolve().parent
    python_executable = Path(sys.executable)

    if sys.platform.startswith("win"):
        command = f"""
$ErrorActionPreference = 'Stop'
$python = '{str(python_executable).replace("'", "''")}'
$cwd = '{str(workspace).replace("'", "''")}'
Get-CimInstance Win32_Process |
  Where-Object {{
    $_.CommandLine -and $_.CommandLine -like '*-m jobs.scheduler*'
  }} |
  ForEach-Object {{
    try {{
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
    }} catch {{
    }}
  }}
Start-Sleep -Milliseconds 800
Start-Process -FilePath $python -ArgumentList '-m','jobs.scheduler' -WorkingDirectory $cwd -WindowStyle Hidden | Out-Null
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        if completed.returncode == 0:
            return True, "worker 재시작 요청을 보냈습니다."
        stderr = (completed.stderr or completed.stdout or "").strip()
        return False, stderr or "worker 재시작에 실패했습니다."

    return False, "worker 재시작 버튼은 현재 Windows 환경에서만 지원합니다."




def run_manual_runtime_job(job_name: str) -> Tuple[bool, str]:
    from jobs.scheduler import _run_guarded
    from jobs.tasks import (
        broker_account_sync_job,
        broker_market_status_job,
        broker_order_sync_job,
        broker_position_sync_job,
        build_task_context,
    )

    job_map = {
        "broker_market_status": broker_market_status_job,
        "broker_position_sync": broker_position_sync_job,
        "broker_order_sync": broker_order_sync_job,
        "broker_account_sync": broker_account_sync_job,
    }
    fn = job_map.get(job_name)
    if fn is None:
        return False, f"지원하지 않는 job 입니다: {job_name}"
    context = build_task_context()
    run_key = f"manual:{pd.Timestamp.utcnow().isoformat()}"
    result = _run_guarded(context, job_name=job_name, run_key=run_key, fn=lambda: fn(context))
    if result is None:
        return False, f"{job_name} 실행에 실패했습니다."
    return True, f"{job_name} 실행 완료"


def stop_background_worker() -> Tuple[bool, str]:
    workspace = Path(__file__).resolve().parent

    if sys.platform.startswith("win"):
        command = """
$ErrorActionPreference = 'Stop'
$killed = 0
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -and $_.CommandLine -like '*-m jobs.scheduler*'
  } |
  ForEach-Object {
    try {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
      $script:killed += 1
    } catch {
    }
  }
Write-Output $killed
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        if completed.returncode == 0:
            killed_text = (completed.stdout or "").strip()
            killed = int(killed_text) if killed_text.isdigit() else 0
            if killed > 0:
                return True, f"worker {killed}개를 중지했습니다."
            return True, "중지할 worker가 없었습니다."
        stderr = (completed.stderr or completed.stdout or "").strip()
        return False, stderr or "worker 중지에 실패했습니다."

    return False, "worker 중지 버튼은 현재 Windows 환경에서만 지원합니다."


def run_manual_scan_job(asset_types: List[str] | None = None) -> Tuple[bool, str]:
    from jobs.scheduler import _run_guarded
    from jobs.tasks import build_task_context, scan_job

    context = build_task_context()
    targets = list(asset_types or context.settings.asset_schedules.keys())
    if not targets:
        return False, "실행할 스캔 자산이 없습니다."

    scanned_counts: Dict[str, int] = {}
    failures: List[str] = []
    for asset_type in targets:
        run_key = f"manual-scan:{asset_type}:{pd.Timestamp.utcnow().isoformat()}"
        result = _run_guarded(
            context,
            job_name=f"scan:{asset_type}",
            run_key=run_key,
            fn=lambda asset_type=asset_type: scan_job(context, [asset_type]),
        )
        if result is None:
            failures.append(asset_type)
            continue
        scanned_counts[asset_type] = int(first_valid_float(result.get(asset_type), default=0.0))

    if failures:
        failed_text = ", ".join(failures)
        return False, f"즉시 스캔 중 일부 실패: {failed_text}"
    summary = ", ".join(f"{asset_type} {count}건" for asset_type, count in scanned_counts.items())
    return True, f"즉시 스캔 완료: {summary or '후보 없음'}"


def render_global_footer() -> None:
    st.markdown(
        f"""
        <div class="alt-global-footer">
          <p>{html.escape(GLOBAL_DATA_PROVIDER_TEXT)}</p>
          <p>{html.escape(GLOBAL_DISCLAIMER_TEXT)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown(
        """
        <div id="alt-page-title-anchor"></div>
        <h1 class="alt-page-title">Alt</h1>
        """,
        unsafe_allow_html=True,
    )


def build_snapshot_view_df(rows: List[Dict[str, str | int | float | None]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["순위", "시장", "종목명", "심볼", "현재가", "전일대비(%)"])

    frame = pd.DataFrame(rows).copy()
    frame["심볼"] = frame["심볼"].fillna("-")

    def _fmt_price(row: pd.Series) -> str:
        return format_live_price(value=float(row.get("현재가", float("nan"))), currency=str(row.get("통화", "USD")))

    frame["현재가"] = frame.apply(_fmt_price, axis=1)
    def _fmt_change(value: object) -> str:
        number = first_valid_float(value)
        return f"{number:+.2f}%" if np.isfinite(number) else "N/A"

    frame["전일대비(%)"] = frame["전일대비(%)"].map(_fmt_change)
    return frame[["순위", "시장", "종목명", "심볼", "현재가", "전일대비(%)"]]


def build_single_picker_options(rows: List[Dict[str, str | int | float | None]]) -> Dict[str, Dict[str, str | int | float | None]]:
    options: Dict[str, Dict[str, str | int | float | None]] = {}
    for row in rows:
        symbol = row.get("심볼")
        if not symbol:
            continue
        change_pct = float(row.get("전일대비(%)", float("nan")))
        change_text = f"{change_pct:+.2f}%" if np.isfinite(change_pct) else "N/A"
        label = f"#{int(row['순위'])} [{row['시장']}] {row['종목명']} ({symbol}) · {change_text}"
        options[label] = row
    return options


def summary_item_value(summary_df: pd.DataFrame, key: str, default: float | str = "") -> float | str:
    row = summary_df.loc[summary_df["item"] == key, "value"]
    if row.empty:
        return default
    value = row.iloc[0]
    if isinstance(default, float):
        try:
            value = float(value)
        except Exception:
            return default
        if np.isnan(value):
            return default
    return value


def format_price_value(value: float) -> str:
    return f"{value:,.2f}" if np.isfinite(value) else "N/A"


def format_pct_value(value: float) -> str:
    return f"{value:+.2f}%" if np.isfinite(value) else "N/A"


def build_trade_sample_view(backtest_frame: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if backtest_frame.empty:
        return pd.DataFrame()

    trade_rows = backtest_frame.loc[backtest_frame["signal"].abs() > 1e-12].copy()
    if trade_rows.empty:
        return pd.DataFrame()

    view = trade_rows.tail(limit).copy()
    for col in ["signal", "predicted_move_pct", "entry_price", "stop_level", "take_level", "exit_price", "gross_return", "net_return"]:
        if col in view.columns:
            if col in {"gross_return", "net_return"}:
                view[col] = view[col].map(lambda v: f"{float(v) * 100.0:+.2f}%" if np.isfinite(float(v)) else "N/A")
            elif col == "signal":
                view[col] = view[col].map(lambda v: f"{float(v):+.2f}" if np.isfinite(float(v)) else "N/A")
            elif col == "predicted_move_pct":
                view[col] = view[col].map(lambda v: format_pct_value(float(v)))
            else:
                view[col] = view[col].map(lambda v: format_price_value(float(v)))

    keep_cols = [c for c in ["signal_label", "signal", "predicted_move_pct", "entry_price", "stop_level", "take_level", "exit_price", "exit_reason", "net_return"] if c in view.columns]
    view = view[keep_cols].rename(
        columns={
            "signal_label": "방향",
            "signal": "포지션",
            "predicted_move_pct": "예상변화율",
            "entry_price": "진입가",
            "stop_level": "손절가",
            "take_level": "익절가",
            "exit_price": "청산가",
            "exit_reason": "청산사유",
            "net_return": "순수익률",
        }
    )
    return view


def chart_window_days_from_label(label: str) -> int | None:
    mapping: Dict[str, int | None] = {
        "최근 3개월": 90,
        "최근 6개월": 180,
        "최근 1년": 365,
        "최근 2년": 730,
        "전체": None,
    }
    return mapping.get(label, 180)


def compute_price_chart_range(result, window_days: int | None) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    all_dates = pd.Index(result.price_data.index).append(pd.Index(result.future_frame.index))
    if all_dates.empty:
        return None, None
    x_max = pd.Timestamp(all_dates.max())
    if window_days is None:
        x_min = pd.Timestamp(all_dates.min())
    else:
        candidate = x_max - pd.Timedelta(days=int(window_days))
        price_min = pd.Timestamp(result.price_data.index.min())
        x_min = max(candidate, price_min)
    return x_min, x_max


def compute_y_axis_range_for_window(result, x_min: pd.Timestamp | None, x_max: pd.Timestamp | None) -> List[float] | None:
    series_list: List[pd.Series] = []
    if x_min is not None and x_max is not None:
        for frame, col in [
            (result.price_data, "Close"),
            (result.test_frame, "ensemble_pred"),
            (result.final_holdout_frame, "ensemble_pred"),
            (result.future_frame, "ensemble_pred"),
            (result.future_frame, "lower_band_1sigma"),
            (result.future_frame, "upper_band_1sigma"),
        ]:
            if frame is None or frame.empty or col not in frame.columns:
                continue
            sl = frame.loc[(frame.index >= x_min) & (frame.index <= x_max), col]
            if not sl.empty:
                series_list.append(pd.to_numeric(sl, errors="coerce"))
    if not series_list:
        return None
    merged = pd.concat(series_list).dropna()
    if merged.empty:
        return None
    y_min = float(merged.min())
    y_max = float(merged.max())
    pad = max((y_max - y_min) * 0.10, max(abs(y_max), 1.0) * 0.015)
    return [y_min - pad, y_max + pad]


def build_prediction_history_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    view = frame.copy()
    for col in ["generated_at", "target_date", "resolved_at", "evaluated_at", "data_cutoff_at"]:
        if col in view.columns:
            view[col] = pd.to_datetime(view[col], errors="coerce")
            view[col] = view[col].dt.strftime("%Y-%m-%d %H:%M").fillna("")
    price_cols = ["current_price", "predicted_price", "actual_price", "abs_error_price", "paper_trade_pnl"]
    ratio_cols = ["predicted_return", "actual_return", "threshold", "position_size", "paper_trade_return"]
    percent_cols = ["ape_pct"]
    for col in price_cols:
        if col in view.columns:
            view[col] = view[col].map(lambda v: format_price_value(float(v)) if np.isfinite(first_valid_float(v)) else "N/A")
    for col in ratio_cols:
        if col in view.columns:
            view[col] = view[col].map(
                lambda v: format_pct_value(float(v) * 100.0) if np.isfinite(first_valid_float(v)) else "N/A"
            )
    for col in percent_cols:
        if col in view.columns:
            view[col] = view[col].map(
                lambda v: format_pct_value(float(v)) if np.isfinite(first_valid_float(v)) else "N/A"
            )
    if "directional_accuracy" in view.columns:
        view["directional_accuracy"] = view["directional_accuracy"].map(
            lambda v: "적중" if first_valid_float(v) >= 0.5 else ("비적중" if np.isfinite(first_valid_float(v)) else "N/A")
        )
    if "status" in view.columns:
        view["status"] = view["status"].map(lambda v: "완료" if str(v) == "resolved" else "미해결")
    if "confidence_score" in view.columns:
        view["confidence_score"] = view["confidence_score"].map(
            lambda v: format_pct_value(float(v) * 100.0) if np.isfinite(first_valid_float(v)) else "N/A"
        )
    keep_cols = [
        "generated_at",
        "prediction_id",
        "forecast_horizon",
        "target_date",
        "status",
        "signal",
        "confidence_score",
        "model_version",
        "predicted_price",
        "actual_price",
        "predicted_return",
        "actual_return",
        "abs_error_price",
        "ape_pct",
        "directional_accuracy",
        "paper_trade_return",
    ]
    keep_cols = [col for col in keep_cols if col in view.columns]
    return view[keep_cols].rename(
        columns={
            "generated_at": "예측시각",
            "prediction_id": "prediction_id",
            "forecast_horizon": "예측일차",
            "target_date": "대상일",
            "status": "상태",
            "signal": "시그널",
            "confidence_score": "confidence",
            "model_version": "모델버전",
            "predicted_price": "예측종가",
            "actual_price": "실제종가",
            "predicted_return": "예상수익률",
            "actual_return": "실제수익률",
            "abs_error_price": "절대오차",
            "ape_pct": "오차율(%)",
            "directional_accuracy": "방향적중",
            "paper_trade_return": "매매수익률",
        }
    )


def render_prediction_tracking_section(result, asset_type: str, korea_market: str) -> None:
    st.subheader("예측 기억 및 비교")
    st.caption("예측은 immutable ledger에 append-only로 저장됩니다. outcome/evaluation은 만기 이후 별도 테이블에 확정됩니다.")
    action_cols = st.columns([1.0, 1.0, 2.2])
    if action_cols[0].button("이번 예측 저장", key=f"save_pred_{result.symbol}"):
        try:
            run_id = save_prediction_snapshot(asset_type=asset_type, korea_market=korea_market, result=result)
        except Exception as exc:
            st.error(f"예측 저장 실패: {exc}")
        else:
            st.success(f"예측을 저장했습니다. run_id={run_id}")
    if action_cols[1].button("실제값 갱신", key=f"refresh_pred_{result.symbol}"):
        try:
            refresh_prediction_actuals(symbol=result.symbol)
        except Exception as exc:
            st.error(f"실제값 갱신 실패: {exc}")
        else:
            st.success("만기된 예측의 실제 종가를 갱신했습니다.")

    try:
        refresh_prediction_actuals(symbol=result.symbol)
    except Exception:
        pass

    history_all = load_prediction_log()
    history_symbol = filter_prediction_history(history_all, symbol=result.symbol)
    summary_all = summarize_prediction_accuracy(history_all)
    summary_symbol = summarize_prediction_accuracy(history_symbol)
    unresolved_symbol = filter_prediction_history(history_symbol, status="unresolved")
    resolved_symbol = filter_prediction_history(history_symbol, status="resolved")

    metrics = st.columns(6)
    metrics[0].metric("저장된 런", f"{int(summary_symbol['saved_runs'])}")
    metrics[1].metric("저장된 예측행", f"{int(summary_symbol['saved_rows'])}")
    metrics[2].metric("실제 비교 완료", f"{int(summary_symbol['matured_rows'])}", f"미해결 {len(unresolved_symbol)}건")
    metrics[3].metric("Price MAE", format_price_value(summary_symbol["mae_price"]))
    metrics[4].metric("MAPE", format_pct_value(summary_symbol["mape_pct"]))
    metrics[5].metric("방향적중도", format_pct_value(summary_symbol["direction_acc_pct"]))
    sub_metrics = st.columns(4)
    sub_metrics[0].metric("Return MAE", format_pct_value(summary_symbol["mae_return_pct"]))
    sub_metrics[1].metric("Price RMSE", format_price_value(summary_symbol["rmse_price"]))
    sub_metrics[2].metric("Return RMSE", format_pct_value(summary_symbol["rmse_return_pct"]))
    sub_metrics[3].metric("Brier", f"{summary_symbol['brier_score']:.4f}" if np.isfinite(summary_symbol["brier_score"]) else "N/A")
    trade_eval_metrics = st.columns(2)
    trade_eval_metrics[0].metric(
        "평균 Target-Aligned 매매수익률",
        format_pct_value(summary_symbol["avg_trade_return_pct"]) if np.isfinite(summary_symbol["avg_trade_return_pct"]) else "N/A",
    )
    trade_eval_metrics[1].metric(
        "평균 Target-Aligned PnL",
        format_price_value(summary_symbol["avg_trade_pnl"]) if np.isfinite(summary_symbol["avg_trade_pnl"]) else "N/A",
    )

    st.caption(
        f"전체 누적 기준: 런 {int(summary_all['saved_runs'])}개 · 예측행 {int(summary_all['saved_rows'])}개 · "
        f"실제 비교 완료 {int(summary_all['matured_rows'])}개"
    )

    if not resolved_symbol.empty:
        matured = resolved_symbol.copy()
        matured["target_date"] = pd.to_datetime(matured["target_date"], errors="coerce")
        compare_fig = go.Figure()
        compare_fig.add_trace(
            go.Scatter(
                x=matured["target_date"],
                y=matured["predicted_price"],
                mode="lines+markers",
                name="저장된 예측종가",
                line=dict(width=2, color="#ef4444"),
            )
        )
        compare_fig.add_trace(
            go.Scatter(
                x=matured["target_date"],
                y=matured["actual_price"],
                mode="lines+markers",
                name="실제 종가",
                line=dict(width=2, color="#e5e7eb"),
            )
        )
        compare_fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(compare_fig, width="stretch")
    else:
        st.info("아직 만기된 예측이 없어 실제 비교 차트가 없습니다.")

    model_registry = load_model_registry()
    if not resolved_symbol.empty:
        model_perf = (
            resolved_symbol.groupby("model_version", dropna=False)
            .agg(
                예측수=("prediction_id", "count"),
                Price_MAE=("abs_error_price", "mean"),
                Return_MAE=("abs_error_return", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) * 100.0),
                방향적중도=("directional_accuracy", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) * 100.0),
                Brier=("brier_score", "mean"),
            )
            .reset_index()
            .rename(columns={"model_version": "모델버전"})
        )
        for col in ("Price_MAE", "Return_MAE", "방향적중도", "Brier"):
            model_perf[col] = pd.to_numeric(model_perf[col], errors="coerce")
        model_perf["Price_MAE"] = model_perf["Price_MAE"].map(
            lambda v: format_price_value(float(v)) if np.isfinite(first_valid_float(v)) else "N/A"
        )
        model_perf["Return_MAE"] = model_perf["Return_MAE"].map(
            lambda v: format_pct_value(float(v)) if np.isfinite(first_valid_float(v)) else "N/A"
        )
        model_perf["방향적중도"] = model_perf["방향적중도"].map(
            lambda v: format_pct_value(float(v)) if np.isfinite(first_valid_float(v)) else "N/A"
        )
        model_perf["Brier"] = model_perf["Brier"].map(
            lambda v: f"{float(v):.4f}" if np.isfinite(first_valid_float(v)) else "N/A"
        )
        st.caption("모델 버전별 성능 비교")
        st.dataframe(model_perf, width="stretch", hide_index=True)

    if not model_registry.empty:
        registry_view = model_registry.copy()
        registry_view["created_at"] = pd.to_datetime(registry_view["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        registry_view["is_champion"] = registry_view["is_champion"].map(lambda v: "Y" if int(first_valid_float(v, default=0.0)) == 1 else "")
        registry_view = registry_view.rename(
            columns={
                "model_version": "모델버전",
                "model_name": "모델명",
                "feature_version": "피처버전",
                "created_at": "등록시각",
                "is_champion": "챔피언",
                "notes": "메모",
            }
        )
        with st.expander("모델 레지스트리", expanded=False):
            st.dataframe(registry_view, width="stretch", hide_index=True)

    unresolved_view = build_prediction_history_view(unresolved_symbol.head(40))
    resolved_view = build_prediction_history_view(resolved_symbol.head(120))
    tab_unresolved, tab_resolved = st.tabs(["미해결 예측", "확정된 예측"])
    with tab_unresolved:
        if unresolved_view.empty:
            st.caption("미해결 예측이 없습니다.")
        else:
            st.dataframe(unresolved_view, width="stretch", hide_index=True)
    with tab_resolved:
        if resolved_view.empty:
            st.caption("이 심볼에 확정된 예측 이력이 없습니다.")
        else:
            st.dataframe(resolved_view, width="stretch", hide_index=True)


def render_operations_monitor(settings=None, dashboard_data: Dict[str, Any] | None = None) -> None:
    settings = settings or load_settings()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()

    st.subheader("운영 모니터링")
    st.caption("background worker가 저장한 예측/포지션/잡 상태를 읽기 전용으로 표시합니다.")
    control_cols = st.columns([1.0, 1.0, 3.0])
    if control_cols[0].button("신규 진입 중단", key="ops_pause"):
        repository.set_control_flag("trading_paused", "1", "set from streamlit monitor")
        st.rerun()
    if control_cols[1].button("신규 진입 재개", key="ops_resume"):
        repository.set_control_flag("trading_paused", "0", "set from streamlit monitor")
        st.rerun()

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    trade_performance = data["trade_performance"]
    auto_trading_status = data.get("auto_trading_status", {})
    metrics = st.columns(6)
    metrics[0].metric("미해결 예측", f"{int(summary.get('unresolved_predictions', 0))}")
    metrics[1].metric("오픈 포지션", f"{int(summary.get('open_positions', 0))}")
    metrics[2].metric("오픈 주문", f"{int(summary.get('open_orders', 0))}")
    metrics[3].metric("Today PnL", format_price_value(float(trade_performance.get("today_pnl", float('nan')))))
    metrics[4].metric("누적 수익률", format_pct_value(float(trade_performance.get("total_return_pct", float("nan")))))
    metrics[5].metric("최대 낙폭", format_pct_value(float(trade_performance.get("max_drawdown_pct", float("nan")))))
    st.caption(
        f"자동 모의매매 {auto_trading_status.get('label', 'Stopped')} · "
        f"{auto_trading_status_text(auto_trading_status)} · db={settings.storage.db_path}"
    )

    job_health = data["job_health"]
    recent_errors = data["recent_errors"]
    open_positions = data["open_positions"]
    open_orders = data["open_orders"]
    candidate_scans = data["candidate_scans"]
    asset_overview = data.get("asset_overview", pd.DataFrame())
    prediction_report = data["prediction_report"]
    equity_curve = data["equity_curve"]

    if not equity_curve.empty:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=pd.to_datetime(equity_curve["created_at"], errors="coerce"),
                    y=pd.to_numeric(equity_curve["equity"], errors="coerce"),
                    mode="lines+markers",
                    name="Equity",
                )
            ]
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_assets, tab_errors = st.tabs(
        ["Job Health", "Open Positions", "Predictions", "Candidates", "Assets", "Recent Errors"]
    )
    with tab_jobs:
        if job_health.empty:
            st.caption("job_runs가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(job_health), width="stretch", hide_index=True)
    with tab_positions:
        if open_positions.empty and open_orders.empty:
            st.caption("오픈 포지션/주문이 없습니다.")
        else:
            if not open_positions.empty:
                st.caption("오픈 포지션")
                st.dataframe(format_frame_timestamps_for_display(open_positions), width="stretch", hide_index=True)
            if not open_orders.empty:
                st.caption("오픈 주문")
                st.dataframe(format_frame_timestamps_for_display(open_orders), width="stretch", hide_index=True)
    with tab_predictions:
        if prediction_report.empty:
            st.caption("예측 ledger가 비어 있습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(prediction_report.head(200)), width="stretch", hide_index=True)
    with tab_candidates:
        if candidate_scans.empty:
            st.caption("candidate_scans가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(candidate_scans.head(200)), width="stretch", hide_index=True)
    with tab_assets:
        if asset_overview.empty:
            st.caption("설정된 자산 유니버스가 없습니다.")
        else:
            st.dataframe(asset_overview, width="stretch", hide_index=True)
    with tab_errors:
        if recent_errors.empty:
            st.caption("최근 ERROR 이벤트가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(recent_errors), width="stretch", hide_index=True)


def build_live_open_positions_view(open_positions: pd.DataFrame, refresh_token: int) -> pd.DataFrame:
    if open_positions.empty or "symbol" not in open_positions.columns:
        return pd.DataFrame()
    symbols = tuple(dict.fromkeys(open_positions["symbol"].dropna().astype(str).tolist()))
    if not symbols:
        return pd.DataFrame()
    snapshots = fetch_quote_snapshots(symbols=symbols, refresh_token=refresh_token)
    rows: List[Dict[str, object]] = []
    for _, row in open_positions.iterrows():
        symbol = str(row.get("symbol") or "")
        snapshot = dict(snapshots.get(symbol) or {})
        currency = str(snapshot.get("currency") or default_currency_from_symbol(symbol))
        side = str(row.get("side") or "")
        quantity = int(first_valid_float(row.get("quantity"), default=0.0))
        entry_price = first_valid_float(row.get("entry_price"))
        current_price = first_valid_float(snapshot.get("current_price"), row.get("mark_price"))
        day_change_pct = first_valid_float(snapshot.get("change_pct"))
        if np.isfinite(entry_price) and np.isfinite(current_price) and entry_price != 0 and quantity > 0:
            signed_return_pct = ((current_price - entry_price) / entry_price) * 100.0
            pnl_pct = signed_return_pct if side == "LONG" else -signed_return_pct
            pnl_value = (current_price - entry_price) * quantity if side == "LONG" else (entry_price - current_price) * quantity
            market_value = current_price * quantity
        else:
            pnl_pct = float("nan")
            pnl_value = first_valid_float(row.get("unrealized_pnl"))
            market_value = float("nan")
        rows.append(
            {
                "종목": symbol,
                "자산": str(row.get("asset_type") or ""),
                "방향": side,
                "수량": quantity,
                "진입가": format_live_price(entry_price, currency),
                "현재가": format_live_price(current_price, currency),
                "전일대비": format_pct_value(day_change_pct),
                "평가금액": format_live_price(market_value, currency),
                "평가손익": format_live_price(pnl_value, currency),
                "수익률": format_pct_value(pnl_pct),
                "갱신시각": format_display_timestamp(row.get("updated_at")),
            }
        )
    return pd.DataFrame(rows)


def build_recent_order_activity_view(recent_orders: pd.DataFrame) -> pd.DataFrame:
    if recent_orders.empty or "symbol" not in recent_orders.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, row in recent_orders.iterrows():
        symbol = str(row.get("symbol") or "")
        currency = default_currency_from_symbol(symbol)
        side = str(row.get("side") or "").lower()
        status = str(row.get("status") or "").lower()
        reason = str(row.get("reason") or "")
        rows.append(
            {
                "종목": symbol,
                "자산": str(row.get("asset_type") or ""),
                "주문": PAPER_ORDER_SIDE_LABELS.get(side, side.upper() or "-"),
                "요청수량": int(first_valid_float(row.get("requested_qty"), default=0.0)),
                "체결수량": int(first_valid_float(row.get("filled_qty"), default=0.0)),
                "주문가": format_live_price(first_valid_float(row.get("requested_price")), currency),
                "상태": OPERATIONS_ORDER_STATUS_LABELS.get(status, status or "-"),
                "사유": OPERATIONS_ORDER_REASON_LABELS.get(reason, reason or "-"),
                "갱신시각": format_display_timestamp(row.get("updated_at")),
            }
        )
    return pd.DataFrame(rows)


MONITOR_SYNC_LABELS = {
    "broker_market_status": "장 상태 확인",
    "broker_account_sync": "계좌 동기화",
    "broker_order_sync": "주문 동기화",
    "broker_position_sync": "포지션 동기화",
}


def _safe_float(value: object) -> float:
    return first_valid_float(value)


def _operations_monitor_styles(theme_mode: str) -> str:
    theme_mode = "dark" if str(theme_mode).lower() == "dark" else "light"
    variant = str(MONITOR_UI_VARIANT or "polished").lower()
    if variant == "classic":
        if theme_mode == "dark":
            palette = {
                "card_border": "rgba(148, 163, 184, 0.14)",
                "card_bg": "linear-gradient(180deg, rgba(15, 23, 42, 0.90) 0%, rgba(9, 14, 28, 0.98) 100%), radial-gradient(circle at top right, rgba(59, 130, 246, 0.14), transparent 55%)",
                "card_shadow": "0 20px 44px rgba(2, 6, 23, 0.26)",
                "card_text": "rgba(241, 245, 249, 0.98)",
                "kicker": "rgba(148, 163, 184, 0.88)",
                "note": "rgba(226, 232, 240, 0.82)",
                "pill_bg": "rgba(15, 23, 42, 0.58)",
                "pill_border": "rgba(148, 163, 184, 0.12)",
                "pill_text": "rgba(226, 232, 240, 0.90)",
                "detail_bg": "rgba(15, 23, 42, 0.58)",
                "detail_span": "rgba(148, 163, 184, 0.92)",
                "chip_bg": "rgba(15, 23, 42, 0.52)",
                "chip_border": "rgba(148, 163, 184, 0.12)",
                "sync_meta": "rgba(226, 232, 240, 0.82)",
                "sync_error": "rgba(252, 165, 165, 0.95)",
            }
        else:
            palette = {
                "card_border": "rgba(148, 163, 184, 0.28)",
                "card_bg": "linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%), radial-gradient(circle at top right, rgba(59, 130, 246, 0.10), transparent 58%)",
                "card_shadow": "0 18px 40px rgba(148, 163, 184, 0.18)",
                "card_text": "#0f172a",
                "kicker": "#94a3b8",
                "note": "#64748b",
                "pill_bg": "rgba(226, 232, 240, 0.82)",
                "pill_border": "rgba(148, 163, 184, 0.36)",
                "pill_text": "#475569",
                "detail_bg": "rgba(226, 232, 240, 0.78)",
                "detail_span": "#64748b",
                "chip_bg": "rgba(226, 232, 240, 0.88)",
                "chip_border": "rgba(148, 163, 184, 0.32)",
                "sync_meta": "#64748b",
                "sync_error": "#b91c1c",
            }
        return f"""
            <style>
            .alt-ops-hero-card,
            .alt-ops-runtime-card,
            .alt-ops-sync-card {{
              border-radius: 22px;
              border: 1px solid {palette["card_border"]};
              background: {palette["card_bg"]};
              box-shadow: {palette["card_shadow"]};
              color: {palette["card_text"]};
            }}
            .alt-ops-hero-card,
            .alt-ops-runtime-card {{
              padding: 1.2rem 1.25rem;
              min-height: 24rem;
              display: flex;
              flex-direction: column;
              gap: 0.95rem;
            }}
            .alt-ops-sync-card {{
              padding: 0.95rem 1rem;
              min-height: 100%;
            }}
            .alt-ops-card-kicker {{
              margin: 0;
              font-size: 0.72rem;
              font-weight: 700;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              color: {palette["kicker"]};
            }}
            .alt-ops-balance-value {{
              margin: 0;
              font-size: clamp(2.2rem, 4vw, 3.65rem);
              line-height: 0.98;
              font-weight: 800;
              letter-spacing: -0.04em;
              color: {palette["card_text"]};
            }}
            .alt-ops-balance-note {{
              margin: 0;
              font-size: 0.95rem;
              color: {palette["note"]};
            }}
            .alt-ops-balance-summary {{
              display: flex;
              flex-wrap: wrap;
              gap: 0.75rem 1rem;
              margin-top: 0;
            }}
            .alt-ops-balance-pill {{
              display: inline-flex;
              align-items: center;
              gap: 0.45rem;
              padding: 0.42rem 0.7rem;
              border-radius: 999px;
              background: {palette["pill_bg"]};
              border: 1px solid {palette["pill_border"]};
              font-size: 0.84rem;
              color: {palette["pill_text"]};
            }}
            .alt-ops-balance-pill strong {{
              font-size: 0.92rem;
              font-weight: 700;
              color: {palette["card_text"]};
            }}
            .alt-ops-detail-grid {{
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 0.75rem;
              margin-top: 0;
            }}
            .alt-ops-detail-item {{
              padding: 0.8rem 0.9rem;
              border-radius: 16px;
              background: {palette["detail_bg"]};
              border: 1px solid {palette["pill_border"]};
            }}
            .alt-ops-detail-item span {{
              display: block;
              margin-bottom: 0.2rem;
              font-size: 0.74rem;
              color: {palette["detail_span"]};
            }}
            .alt-ops-detail-item strong {{
              display: block;
              font-size: 1rem;
              font-weight: 700;
              color: {palette["card_text"]};
            }}
            .alt-ops-runtime-title {{
              margin: 0;
              font-size: 1.28rem;
              font-weight: 700;
              letter-spacing: -0.02em;
              color: {palette["card_text"]};
            }}
            .alt-ops-runtime-body,
            .alt-ops-runtime-meta {{
              margin: 0;
              font-size: 0.88rem;
              line-height: 1.5;
              color: {palette["note"]};
            }}
            .alt-ops-runtime-body strong {{
              color: {palette["card_text"]};
            }}
            .alt-ops-runtime-grid {{
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 0.65rem;
              margin-top: 0;
              padding-top: 0.15rem;
            }}
            .alt-ops-runtime-chip {{
              padding: 0.7rem 0.8rem;
              border-radius: 14px;
              background: {palette["chip_bg"]};
              border: 1px solid {palette["chip_border"]};
            }}
            .alt-ops-runtime-chip span {{
              display: block;
              font-size: 0.73rem;
              color: {palette["detail_span"]};
            }}
            .alt-ops-runtime-chip strong {{
              display: block;
              margin-top: 0.15rem;
              font-size: 0.96rem;
              font-weight: 700;
              color: {palette["card_text"]};
            }}
            .alt-ops-runtime-chip small {{
              display: block;
              margin-top: 0.12rem;
              font-size: 0.72rem;
              color: {palette["note"]};
            }}
            .alt-ops-card-head {{
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 0.8rem;
            }}
            .alt-ops-sync-name {{
              margin: 0.15rem 0 0;
              font-size: 1rem;
              font-weight: 700;
              letter-spacing: -0.02em;
              color: {palette["card_text"]};
            }}
            .alt-ops-sync-meta,
            .alt-ops-sync-error {{
              margin: 0.45rem 0 0;
              font-size: 0.84rem;
              line-height: 1.45;
              color: {palette["sync_meta"]};
            }}
            .alt-ops-sync-error {{
              color: {palette["sync_error"]};
            }}
            .alt-ops-sync-card.alt-ops-sync-failed {{
              border-color: rgba(239, 68, 68, 0.34);
            }}
            .alt-ops-sync-card.alt-ops-sync-running {{
              border-color: rgba(59, 130, 246, 0.34);
            }}
            .alt-ops-sync-card.alt-ops-sync-completed {{
              border-color: rgba(16, 185, 129, 0.30);
            }}
            .alt-ops-sync-card.alt-ops-sync-never {{
              border-color: rgba(148, 163, 184, 0.18);
            }}
            .alt-ops-section-gap {{
              height: 1rem;
            }}
            </style>
        """

    if theme_mode == "dark":
        palette = {
            "panel_border": "rgba(148, 163, 184, 0.16)",
            "panel_border_soft": "rgba(148, 163, 184, 0.12)",
            "panel_bg": "linear-gradient(180deg, rgba(8, 15, 30, 0.92) 0%, rgba(6, 11, 24, 0.98) 100%), radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 55%)",
            "panel_bg_alt": "rgba(9, 17, 34, 0.74)",
            "panel_bg_soft": "rgba(15, 23, 42, 0.64)",
            "panel_bg_strong": "linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(9, 14, 28, 0.94) 100%)",
            "panel_shadow": "0 28px 56px rgba(2, 6, 23, 0.34)",
            "panel_text": "rgba(248, 250, 252, 0.98)",
            "panel_muted": "rgba(203, 213, 225, 0.74)",
            "panel_kicker": "rgba(148, 163, 184, 0.82)",
            "pill_bg": "rgba(15, 23, 42, 0.64)",
            "pill_text": "rgba(226, 232, 240, 0.90)",
            "detail_bg": "rgba(15, 23, 42, 0.58)",
            "detail_border": "rgba(96, 165, 250, 0.10)",
            "sync_meta": "rgba(226, 232, 240, 0.78)",
            "sync_error": "#fca5a5",
            "accent": "#38bdf8",
            "accent_soft": "rgba(56, 189, 248, 0.14)",
            "glow": "rgba(56, 189, 248, 0.16)",
        }
    else:
        palette = {
            "panel_border": "rgba(148, 163, 184, 0.26)",
            "panel_border_soft": "rgba(148, 163, 184, 0.18)",
            "panel_bg": "linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(244, 248, 255, 0.98) 100%), radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 56%)",
            "panel_bg_alt": "rgba(255, 255, 255, 0.84)",
            "panel_bg_soft": "rgba(241, 245, 249, 0.88)",
            "panel_bg_strong": "linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%)",
            "panel_shadow": "0 22px 48px rgba(148, 163, 184, 0.18)",
            "panel_text": "#0f172a",
            "panel_muted": "#64748b",
            "panel_kicker": "#64748b",
            "pill_bg": "rgba(239, 246, 255, 0.92)",
            "pill_text": "#1e293b",
            "detail_bg": "rgba(241, 245, 249, 0.92)",
            "detail_border": "rgba(37, 99, 235, 0.10)",
            "sync_meta": "#64748b",
            "sync_error": "#b91c1c",
            "accent": "#2563eb",
            "accent_soft": "rgba(37, 99, 235, 0.10)",
            "glow": "rgba(37, 99, 235, 0.10)",
        }
    return f"""
        <style>
        .alt-ops-overview-strip,
        .alt-ops-hero-card,
        .alt-ops-runtime-card,
        .alt-ops-sync-card {{
          position: relative;
          overflow: hidden;
          border: 1px solid {palette["panel_border"]};
          background: {palette["panel_bg"]};
          box-shadow: {palette["panel_shadow"]};
          color: {palette["panel_text"]};
          backdrop-filter: blur(18px);
        }}
        .alt-ops-overview-strip::after,
        .alt-ops-hero-card::after,
        .alt-ops-runtime-card::after {{
          content: "";
          position: absolute;
          inset: auto -6% -42% auto;
          width: 14rem;
          height: 14rem;
          background: radial-gradient(circle, {palette["glow"]} 0%, rgba(0, 0, 0, 0) 70%);
          pointer-events: none;
        }}
        .alt-ops-overview-strip {{
          border-radius: 24px;
          padding: 1.15rem 1.2rem 1.2rem;
          margin: 0.15rem 0 0.35rem;
        }}
        .alt-ops-overview-head {{
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 1rem;
          flex-wrap: wrap;
        }}
        .alt-ops-overview-copy {{
          max-width: 34rem;
        }}
        .alt-ops-overview-title {{
          margin: 0.28rem 0 0;
          font-size: clamp(1.2rem, 1.7vw, 1.55rem);
          line-height: 1.2;
          letter-spacing: -0.03em;
          color: {palette["panel_text"]};
        }}
        .alt-ops-overview-note {{
          margin: 0.5rem 0 0;
          font-size: 0.92rem;
          line-height: 1.5;
          color: {palette["panel_muted"]};
        }}
        .alt-ops-overview-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.75rem;
          margin-top: 1rem;
        }}
        .alt-ops-overview-stat {{
          padding: 0.9rem 1rem;
          border-radius: 18px;
          border: 1px solid {palette["panel_border_soft"]};
          background: {palette["panel_bg_alt"]};
        }}
        .alt-ops-overview-stat span,
        .alt-ops-card-kicker {{
          margin: 0;
          display: block;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: {palette["panel_kicker"]};
        }}
        .alt-ops-overview-stat strong {{
          display: block;
          margin-top: 0.45rem;
          font-size: 1.02rem;
          font-weight: 700;
          line-height: 1.3;
          color: {palette["panel_text"]};
        }}
        .alt-ops-overview-stat small {{
          display: block;
          margin-top: 0.25rem;
          font-size: 0.76rem;
          line-height: 1.45;
          color: {palette["panel_muted"]};
        }}
        .alt-ops-hero-card,
        .alt-ops-runtime-card {{
          border-radius: 26px;
          padding: 1.25rem;
          min-height: 30rem;
          height: 100%;
          display: flex;
          flex-direction: column;
          gap: 0.95rem;
        }}
        .alt-ops-sync-card {{
          border-radius: 20px;
          padding: 1rem;
          min-height: 100%;
        }}
        .alt-ops-panel-head,
        .alt-ops-card-head {{
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 0.9rem;
        }}
        .alt-ops-panel-title {{
          margin: 0.26rem 0 0;
          font-size: 1.22rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          color: {palette["panel_text"]};
        }}
        .alt-ops-panel-subtitle,
        .alt-ops-balance-note,
        .alt-ops-runtime-meta,
        .alt-ops-runtime-body,
        .alt-ops-panel-foot {{
          margin: 0;
          font-size: 0.88rem;
          line-height: 1.5;
          color: {palette["panel_muted"]};
        }}
        .alt-ops-balance-hero,
        .alt-ops-runtime-hero {{
          padding: 0.95rem 1rem;
          border-radius: 20px;
          border: 1px solid {palette["panel_border_soft"]};
          background: {palette["panel_bg_strong"]};
        }}
        .alt-ops-balance-value {{
          margin: 0;
          font-size: clamp(2.25rem, 4vw, 3.55rem);
          line-height: 0.96;
          font-weight: 800;
          letter-spacing: -0.05em;
          color: {palette["panel_text"]};
        }}
        .alt-ops-balance-summary {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.65rem;
          margin-top: 0;
        }}
        .alt-ops-balance-pill {{
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.48rem 0.75rem;
          border-radius: 999px;
          border: 1px solid {palette["panel_border_soft"]};
          background: {palette["pill_bg"]};
          color: {palette["pill_text"]};
          font-size: 0.82rem;
          line-height: 1.2;
        }}
        .alt-ops-balance-pill strong {{
          font-size: 0.92rem;
          font-weight: 700;
          color: {palette["panel_text"]};
        }}
        .alt-ops-detail-grid,
        .alt-ops-runtime-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.75rem;
          margin-top: 0;
        }}
        .alt-ops-detail-item,
        .alt-ops-runtime-chip {{
          padding: 0.85rem 0.9rem;
          border-radius: 18px;
          border: 1px solid {palette["detail_border"]};
          background: {palette["detail_bg"]};
        }}
        .alt-ops-detail-item span,
        .alt-ops-runtime-chip span {{
          display: block;
          font-size: 0.74rem;
          color: {palette["panel_kicker"]};
        }}
        .alt-ops-detail-item strong,
        .alt-ops-runtime-chip strong {{
          display: block;
          margin-top: 0.25rem;
          font-size: 1rem;
          font-weight: 700;
          line-height: 1.35;
          color: {palette["panel_text"]};
          word-break: break-word;
        }}
        .alt-ops-runtime-title {{
          margin: 0;
          font-size: 1.16rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: {palette["panel_text"]};
        }}
        .alt-ops-runtime-body strong {{
          color: {palette["panel_text"]};
        }}
        .alt-ops-runtime-chip small {{
          display: block;
          margin-top: 0.25rem;
          font-size: 0.74rem;
          line-height: 1.45;
          color: {palette["sync_meta"]};
        }}
        .alt-ops-runtime-chip.alt-ops-runtime-chip-accent {{
          background: linear-gradient(180deg, {palette["accent_soft"]} 0%, {palette["detail_bg"]} 100%);
        }}
        .alt-ops-panel-foot {{
          margin-top: auto;
        }}
        .alt-ops-sync-name {{
          margin: 0.22rem 0 0;
          font-size: 1rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: {palette["panel_text"]};
        }}
        .alt-ops-sync-card .alt-ops-card-head {{
          align-items: center;
        }}
        .alt-ops-sync-card .alt-ops-card-head .alt-ops-sync-name {{
          margin: 0;
        }}
        .alt-ops-sync-meta,
        .alt-ops-sync-error {{
          margin: 0.45rem 0 0;
          font-size: 0.84rem;
          line-height: 1.45;
          color: {palette["sync_meta"]};
        }}
        .alt-ops-sync-error {{
          color: {palette["sync_error"]};
        }}
        .alt-ops-sync-card.alt-ops-sync-failed {{
          border-color: rgba(239, 68, 68, 0.34);
        }}
        .alt-ops-sync-card.alt-ops-sync-running {{
          border-color: rgba(59, 130, 246, 0.34);
        }}
        .alt-ops-sync-card.alt-ops-sync-completed {{
          border-color: rgba(16, 185, 129, 0.30);
        }}
        .alt-ops-sync-card.alt-ops-sync-never {{
          border-color: rgba(148, 163, 184, 0.18);
        }}
        .alt-ops-section-gap {{
          height: 1rem;
        }}
        @media (max-width: 1080px) {{
          .alt-ops-overview-grid,
          .alt-ops-detail-grid,
          .alt-ops-runtime-grid {{
            grid-template-columns: minmax(0, 1fr);
          }}
          .alt-ops-hero-card,
          .alt-ops-runtime-card {{
            min-height: auto;
          }}
        }}
        </style>
    """


def _sync_status_label(status: str) -> str:
    return _monitor_status_label(status)


def _sync_status_badge_class(status: str) -> str:
    return {
        "completed": "running",
        "running": "running",
        "failed": "stopped",
        "never": "paused",
        "queued": "paused",
        "cancelled": "paused",
    }.get(str(status).lower(), "paused")


def _sync_status_summary_text(row: Dict[str, Any]) -> str:
    finished_text = format_display_timestamp(
        row.get("heartbeat_at") or row.get("finished_at") or row.get("started_at") or row.get("scheduled_at")
    ) or "아직 기록이 없습니다"
    return f"{_sync_status_label(str(row.get('status', 'never')))} · {finished_text}"


def _build_operations_balance_card_html(
    account_snapshot: Dict[str, Any],
    trade_performance: Dict[str, Any],
    runtime_profile: Dict[str, Any],
) -> str:
    equity_text = format_price_value(_safe_float(account_snapshot.get("equity")))
    cash_text = format_price_value(_safe_float(account_snapshot.get("cash")))
    today_pnl_text = format_price_value(_safe_float(trade_performance.get("today_pnl")))
    total_return_text = format_pct_value(_safe_float(trade_performance.get("total_return_pct")))
    gross_text = format_price_value(_safe_float(account_snapshot.get("gross_exposure")))
    net_text = format_price_value(_safe_float(account_snapshot.get("net_exposure")))
    drawdown_text = format_pct_value(
        _safe_float(account_snapshot.get("drawdown_pct", trade_performance.get("max_drawdown_pct")))
    )
    profile_text = str(runtime_profile.get("name") or "baseline")
    updated_text = format_display_timestamp(account_snapshot.get("created_at")) or "스냅샷이 없습니다"
    if str(MONITOR_UI_VARIANT or "polished").lower() == "classic":
        return f"""
            <div class="alt-ops-hero-card">
              <p class="alt-ops-card-kicker">계좌 현황</p>
              <div class="alt-ops-balance-value">{html.escape(equity_text)}</div>
              <p class="alt-ops-balance-note">가장 최근 평가 자산입니다.</p>
              <div class="alt-ops-balance-summary">
                <div class="alt-ops-balance-pill"><span>예수금</span><strong>{html.escape(cash_text)}</strong></div>
                <div class="alt-ops-balance-pill"><span>당일 손익</span><strong>{html.escape(today_pnl_text)}</strong></div>
                <div class="alt-ops-balance-pill"><span>누적 수익률</span><strong>{html.escape(total_return_text)}</strong></div>
              </div>
              <div class="alt-ops-detail-grid">
                <div class="alt-ops-detail-item">
                  <span>총 익스포저</span>
                  <strong>{html.escape(gross_text)}</strong>
                </div>
                <div class="alt-ops-detail-item">
                  <span>순 익스포저</span>
                  <strong>{html.escape(net_text)}</strong>
                </div>
                <div class="alt-ops-detail-item">
                  <span>낙폭</span>
                  <strong>{html.escape(drawdown_text)}</strong>
                </div>
                <div class="alt-ops-detail-item">
                  <span>프로파일</span>
                  <strong>{html.escape(profile_text)}</strong>
                </div>
              </div>
              <p class="alt-ops-runtime-meta">마지막 스냅샷 {html.escape(updated_text)}</p>
            </div>
        """
    return f"""
        <section class="alt-ops-hero-card">
          <div class="alt-ops-panel-head">
            <div>
              <p class="alt-ops-card-kicker">계좌 현황</p>
              <p class="alt-ops-panel-title">실시간 평가 자산</p>
            </div>
            <div class="alt-ops-balance-pill"><span>프로파일</span><strong>{html.escape(profile_text)}</strong></div>
          </div>
          <div class="alt-ops-balance-hero">
            <div class="alt-ops-balance-value">{html.escape(equity_text)}</div>
          </div>
          <div class="alt-ops-balance-summary">
            <div class="alt-ops-balance-pill"><span>예수금</span><strong>{html.escape(cash_text)}</strong></div>
            <div class="alt-ops-balance-pill"><span>당일 손익</span><strong>{html.escape(today_pnl_text)}</strong></div>
            <div class="alt-ops-balance-pill"><span>누적 수익률</span><strong>{html.escape(total_return_text)}</strong></div>
          </div>
          <div class="alt-ops-detail-grid">
            <div class="alt-ops-detail-item">
              <span>총 익스포저</span>
              <strong>{html.escape(gross_text)}</strong>
            </div>
            <div class="alt-ops-detail-item">
              <span>순 익스포저</span>
              <strong>{html.escape(net_text)}</strong>
            </div>
            <div class="alt-ops-detail-item">
              <span>낙폭</span>
              <strong>{html.escape(drawdown_text)}</strong>
            </div>
            <div class="alt-ops-detail-item">
              <span>마지막 스냅샷</span>
              <strong>{html.escape(updated_text)}</strong>
            </div>
          </div>
        </section>
    """


def _build_operations_runtime_card_html(
    auto_trading_status: Dict[str, Any],
    broker_sync_rows: Dict[str, Dict[str, Any]],
    kis_runtime: Dict[str, Any],
    runtime_profile: Dict[str, Any],
) -> str:
    runtime_text = auto_trading_status_text(auto_trading_status)
    badge_html = _build_monitor_auto_trading_badge_html(auto_trading_status)
    account_sync = broker_sync_rows.get("broker_account_sync", {})
    order_sync = broker_sync_rows.get("broker_order_sync", {})
    position_sync = broker_sync_rows.get("broker_position_sync", {})
    market_sync = broker_sync_rows.get("broker_market_status", {})
    profile_source = str(runtime_profile.get("source") or "embedded_defaults")
    ws_event_text = format_display_timestamp(kis_runtime.get("last_websocket_execution_event"))
    ws_event = ws_event_text or "기록 없음"
    ws_note = "KIS 체결 이벤트를 마지막으로 받은 시각" if ws_event_text else "아직 받은 실시간 체결 이벤트가 없습니다"
    pending_orders = int(first_valid_float(kis_runtime.get("pending_submitted_orders"), default=0.0))
    if str(MONITOR_UI_VARIANT or "polished").lower() == "classic":
        return f"""
            <div class="alt-ops-runtime-card">
              <div class="alt-ops-card-head">
                <p class="alt-ops-card-kicker">런타임 상태</p>
                {badge_html}
              </div>
              <p class="alt-ops-runtime-title">백그라운드 워커 상태</p>
              <p class="alt-ops-runtime-body"><strong>{html.escape(runtime_text or 'heartbeat 기록이 없습니다')}</strong></p>
              <p class="alt-ops-runtime-meta">프로파일 출처 {html.escape(profile_source)}</p>
              <div class="alt-ops-runtime-grid">
                <div class="alt-ops-runtime-chip">
                  <span>계좌 동기화</span>
                  <strong>{html.escape(_sync_status_label(str(account_sync.get('status', 'never'))))}</strong>
                  <small>{html.escape(_sync_status_summary_text(account_sync))}</small>
                </div>
                <div class="alt-ops-runtime-chip">
                  <span>주문 동기화</span>
                  <strong>{html.escape(_sync_status_label(str(order_sync.get('status', 'never'))))}</strong>
                  <small>{html.escape(_sync_status_summary_text(order_sync))}</small>
                </div>
                <div class="alt-ops-runtime-chip">
                  <span>포지션 동기화</span>
                  <strong>{html.escape(_sync_status_label(str(position_sync.get('status', 'never'))))}</strong>
                  <small>{html.escape(_sync_status_summary_text(position_sync))}</small>
                </div>
                <div class="alt-ops-runtime-chip">
                  <span>장 상태 확인</span>
                  <strong>{html.escape(_sync_status_label(str(market_sync.get('status', 'never'))))}</strong>
                  <small>{html.escape(_sync_status_summary_text(market_sync))}</small>
                </div>
                <div class="alt-ops-runtime-chip">
                  <span>미체결 제출 주문</span>
                  <strong>{pending_orders}</strong>
                  <small>아직 체결되지 않은 KIS 주문 수</small>
                </div>
                <div class="alt-ops-runtime-chip">
                  <span>최근 실시간 체결 수신</span>
                  <strong>{html.escape(ws_event)}</strong>
                  <small>KIS 체결 이벤트 수신 시각</small>
                </div>
              </div>
            </div>
        """
    return f"""
        <section class="alt-ops-runtime-card">
          <div class="alt-ops-card-head">
            <div>
              <p class="alt-ops-card-kicker">런타임 상태</p>
              <p class="alt-ops-panel-title">자동매매와 브로커 연결</p>
            </div>
            {badge_html}
          </div>
          <div class="alt-ops-runtime-hero">
            <p class="alt-ops-runtime-title">백그라운드 워커</p>
            <p class="alt-ops-runtime-body"><strong>{html.escape(runtime_text or 'heartbeat 기록이 없습니다')}</strong></p>
            <p class="alt-ops-runtime-meta">설정 파일 {html.escape(profile_source)}</p>
          </div>
          <div class="alt-ops-runtime-grid">
            <div class="alt-ops-runtime-chip">
              <span>계좌 동기화</span>
              <strong>{html.escape(_sync_status_label(str(account_sync.get('status', 'never'))))}</strong>
              <small>{html.escape(_sync_status_summary_text(account_sync))}</small>
            </div>
            <div class="alt-ops-runtime-chip">
              <span>주문 동기화</span>
              <strong>{html.escape(_sync_status_label(str(order_sync.get('status', 'never'))))}</strong>
              <small>{html.escape(_sync_status_summary_text(order_sync))}</small>
            </div>
            <div class="alt-ops-runtime-chip">
              <span>포지션 동기화</span>
              <strong>{html.escape(_sync_status_label(str(position_sync.get('status', 'never'))))}</strong>
              <small>{html.escape(_sync_status_summary_text(position_sync))}</small>
            </div>
            <div class="alt-ops-runtime-chip">
              <span>장 상태 확인</span>
              <strong>{html.escape(_sync_status_label(str(market_sync.get('status', 'never'))))}</strong>
              <small>{html.escape(_sync_status_summary_text(market_sync))}</small>
            </div>
            <div class="alt-ops-runtime-chip alt-ops-runtime-chip-accent">
              <span>미체결 제출 주문</span>
              <strong>{pending_orders}</strong>
              <small>아직 체결되지 않은 KIS 주문 수</small>
            </div>
            <div class="alt-ops-runtime-chip alt-ops-runtime-chip-accent">
              <span>최근 실시간 체결 수신</span>
              <strong>{html.escape(ws_event)}</strong>
              <small>{html.escape(ws_note)}</small>
            </div>
          </div>
        </section>
    """


def _build_operations_overview_strip_html(
    auto_trading_status: Dict[str, Any],
    runtime_profile: Dict[str, Any],
    account_snapshot: Dict[str, Any],
) -> str:
    status_label = _localized_auto_trading_label(auto_trading_status)
    status_detail = auto_trading_status_text(auto_trading_status)
    profile_name = str(runtime_profile.get("name") or "baseline")
    profile_source = str(runtime_profile.get("source") or "embedded_defaults")
    snapshot_text = format_display_timestamp(account_snapshot.get("created_at")) or "기록 없음"
    return f"""
        <section class="alt-ops-overview-strip">
          <div class="alt-ops-overview-grid">
            <div class="alt-ops-overview-stat">
              <span>현재 상태</span>
              <strong>{html.escape(status_label)}</strong>
              <small>{html.escape(status_detail)}</small>
            </div>
            <div class="alt-ops-overview-stat">
              <span>활성 프로파일</span>
              <strong>{html.escape(profile_name)}</strong>
              <small>{html.escape(profile_source)}</small>
            </div>
            <div class="alt-ops-overview-stat">
              <span>기준 스냅샷</span>
              <strong>{html.escape(snapshot_text)}</strong>
              <small>가장 최근 계좌 상태 저장 시각</small>
            </div>
          </div>
        </section>
    """


def _build_broker_sync_card_html(job_name: str, row: Dict[str, Any]) -> str:
    status = str(row.get("status", "never") or "never").lower()
    status_class = {
        "completed": "alt-ops-sync-completed",
        "running": "alt-ops-sync-running",
        "failed": "alt-ops-sync-failed",
        "never": "alt-ops-sync-never",
    }.get(status, "alt-ops-sync-never")
    finished_text = format_display_timestamp(
        row.get("heartbeat_at") or row.get("finished_at") or row.get("started_at") or row.get("scheduled_at")
    ) or "아직 기록이 없습니다"
    retry_count = int(first_valid_float(row.get("retry_count"), default=0.0))
    error_text = str(row.get("error_message") or "").strip()
    error_block = (
        f'<p class="alt-ops-sync-error">{html.escape(error_text[:160])}</p>'
        if error_text
        else '<p class="alt-ops-sync-meta">최근 오류가 없습니다</p>'
    )
    if str(MONITOR_UI_VARIANT or "polished").lower() == "classic":
        return f"""
            <div class="alt-ops-sync-card {status_class}">
              <div class="alt-ops-card-head">
                <p class="alt-ops-card-kicker">브로커 동기화</p>
                <span class="alt-status-badge alt-status-{html.escape(_sync_status_badge_class(status))}">
                  {html.escape(_sync_status_label(status))}
                </span>
              </div>
              <p class="alt-ops-sync-name">{html.escape(MONITOR_SYNC_LABELS.get(job_name, job_name))}</p>
              <p class="alt-ops-sync-meta">마지막 갱신 {html.escape(finished_text)} · 재시도 {retry_count}회</p>
              {error_block}
            </div>
        """
    return f"""
        <div class="alt-ops-sync-card {status_class}">
          <div class="alt-ops-card-head">
            <p class="alt-ops-sync-name">{html.escape(MONITOR_SYNC_LABELS.get(job_name, job_name))}</p>
            <span class="alt-status-badge alt-status-{html.escape(_sync_status_badge_class(status))}">
              {html.escape(_sync_status_label(status))}
            </span>
          </div>
          <p class="alt-ops-sync-meta">마지막 갱신 {html.escape(finished_text)} · 재시도 {retry_count}회</p>
          {error_block}
        </div>
    """


def render_live_open_positions_panel(settings) -> None:
    with st.container(border=True):
        st.caption("실시간 보유 포지션")
        view_mode = st.radio(
            "실시간 보유 패널 보기",
            ["보유 포지션", "매수·매도 내역"],
            horizontal=True,
            key="ops_live_positions_mode",
            label_visibility="collapsed",
        )

        @st.fragment(run_every=15)
        def _render_live_open_positions_fragment() -> None:
            if view_mode == "보유 포지션":
                current_positions = load_monitor_open_positions(settings)
                if current_positions.empty:
                    st.caption("현재 보유 중인 포지션이 없습니다.")
                    return
                refresh_token = int(pd.Timestamp.now(tz="UTC").timestamp() // 15)
                live_view = build_live_open_positions_view(current_positions, refresh_token=refresh_token)
                metric_cols = st.columns(3, gap="small")
                metric_cols[0].metric("보유 종목 수", f"{len(current_positions)}", border=True)
                metric_cols[1].metric(
                    "롱 포지션",
                    f"{int((current_positions['side'].astype(str) == 'LONG').sum())}",
                    border=True,
                )
                metric_cols[2].metric(
                    "숏 포지션",
                    f"{int((current_positions['side'].astype(str) == 'SHORT').sum())}",
                    border=True,
                )
                st.dataframe(live_view, width="stretch", hide_index=True, height=260)
                return

            recent_orders = load_monitor_recent_orders(settings, limit=30)
            if recent_orders.empty:
                st.caption("최근 주문 기록이 없습니다.")
                return
            order_view = build_recent_order_activity_view(recent_orders)
            metric_cols = st.columns(3, gap="small")
            metric_cols[0].metric("최근 주문 수", f"{len(recent_orders)}", border=True)
            metric_cols[1].metric(
                "매수 주문",
                f"{int((recent_orders['side'].astype(str).str.lower() == 'buy').sum())}",
                border=True,
            )
            metric_cols[2].metric(
                "매도 주문",
                f"{int((recent_orders['side'].astype(str).str.lower() == 'sell').sum())}",
                border=True,
            )
            st.dataframe(order_view, width="stretch", hide_index=True, height=260)

        _render_live_open_positions_fragment()


def render_operations_monitor(settings=None, dashboard_data: Dict[str, Any] | None = None) -> None:
    settings = settings or load_settings()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()

    st.subheader("운영 모니터")
    st.caption("worker 상태, 실행 보장 지표, broker sync 결과를 확인합니다.")
    control_cols = st.columns([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    if control_cols[0].button("신규 진입 중단", key="ops_pause"):
        repository.set_control_flag("trading_paused", "1", "set from streamlit monitor")
        st.rerun()
    if control_cols[1].button("신규 진입 재개", key="ops_resume"):
        repository.set_control_flag("trading_paused", "0", "set from streamlit monitor")
        st.rerun()
    if control_cols[2].button("Market Sync", key="ops_sync_market"):
        ok, message = run_manual_runtime_job("broker_market_status")
        (st.success if ok else st.error)(message)
    if control_cols[3].button("Order Sync", key="ops_sync_orders"):
        ok, message = run_manual_runtime_job("broker_order_sync")
        (st.success if ok else st.error)(message)
    if control_cols[4].button("Account Sync", key="ops_sync_account"):
        ok, message = run_manual_runtime_job("broker_account_sync")
        (st.success if ok else st.error)(message)

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    trade_performance = data["trade_performance"]
    auto_trading_status = data.get("auto_trading_status", {})
    execution_summary = data.get("execution_summary", {})
    broker_sync_status = data.get("broker_sync_status", pd.DataFrame())
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    runtime_profile = data.get("runtime_profile", {})
    asset_overview = data.get("asset_overview", pd.DataFrame())

    metrics = st.columns(6)
    metrics[0].metric("미해결 예측", f"{int(summary.get('unresolved_predictions', 0))}")
    metrics[1].metric("오픈 포지션", f"{int(summary.get('open_positions', 0))}")
    metrics[2].metric("오픈 주문", f"{int(summary.get('open_orders', 0))}")
    metrics[3].metric("Today PnL", format_price_value(float(trade_performance.get("today_pnl", float("nan")))))
    metrics[4].metric("누적 수익률", format_pct_value(float(trade_performance.get("total_return_pct", float("nan")))))
    metrics[5].metric("최대 낙폭", format_pct_value(float(trade_performance.get("max_drawdown_pct", float("nan")))))
    st.caption(
        f"자동 모의매매 {auto_trading_status.get('label', 'Stopped')} · "
        f"{auto_trading_status_text(auto_trading_status)} · db={settings.storage.db_path}"
    )
    st.caption(
        f"현재 프로파일 · {str(runtime_profile.get('name') or settings.profile_name)}"
        f" · source={str(runtime_profile.get('source') or settings.profile_source)}"
    )
    if not asset_overview.empty and {"자산유형", "실행브로커"}.issubset(asset_overview.columns):
        broker_mode_text = " · ".join(
            f"{str(row['자산유형'])}={str(row['실행브로커'])}"
            for _, row in asset_overview.iterrows()
        )
        st.caption(f"자산별 실행 브로커 · {broker_mode_text}")

    render_live_open_positions_panel(settings)

    execution_cols = st.columns(5)
    execution_cols[0].metric("Today Candidates", f"{int(execution_summary.get('today_candidate_count', 0))}")
    execution_cols[1].metric("Entry Allowed", f"{int(execution_summary.get('today_entry_allowed_count', 0))}")
    execution_cols[2].metric("Entry Rejected", f"{int(execution_summary.get('today_entry_rejected_count', 0))}")
    execution_cols[3].metric("Submitted", f"{int(execution_summary.get('today_submitted_count', 0))}")
    execution_cols[4].metric("Filled", f"{int(execution_summary.get('today_filled_count', 0))}")

    execution_cols_2 = st.columns(5)
    execution_cols_2[0].metric("Submit Requested", f"{int(execution_summary.get('today_submit_requested_count', 0))}")
    execution_cols_2[1].metric("Acknowledged", f"{int(execution_summary.get('today_acknowledged_count', 0))}")
    execution_cols_2[2].metric("Rejected", f"{int(execution_summary.get('today_rejected_count', 0))}")
    execution_cols_2[3].metric("Cancelled", f"{int(execution_summary.get('today_cancelled_count', 0))}")
    execution_cols_2[4].metric("No-op", f"{int(execution_summary.get('today_noop_count', 0))}")

    kis_cols = st.columns(6)
    kis_cols[0].metric("Last Account Sync", format_display_timestamp(kis_runtime.get("last_broker_account_sync")))
    kis_cols[1].metric("Last Order Sync", format_display_timestamp(kis_runtime.get("last_broker_order_sync")))
    last_position_sync = ""
    if not broker_sync_status.empty:
        pos_row = broker_sync_status.loc[broker_sync_status["job_name"].astype(str) == "broker_position_sync"]
        if not pos_row.empty:
            last_position_sync = pos_row.iloc[0].get("heartbeat_at")
    kis_cols[2].metric("Last Position Sync", format_display_timestamp(last_position_sync))
    kis_cols[3].metric("Last WS Event", format_display_timestamp(kis_runtime.get("last_websocket_execution_event")))
    kis_cols[4].metric("Pending Submitted", f"{int(kis_runtime.get('pending_submitted_orders', 0))}")
    kis_cols[5].metric("Broker Rejects Today", f"{int(kis_runtime.get('broker_rejects_today', 0))}")

    job_health = data["job_health"]
    recent_errors = data["recent_errors"]
    open_positions = data["open_positions"]
    open_orders = data["open_orders"]
    candidate_scans = data["candidate_scans"]
    prediction_report = data["prediction_report"]
    equity_curve = data["equity_curve"]
    today_execution_events = data.get("today_execution_events", pd.DataFrame())
    noop_breakdown = execution_summary.get("today_noop_breakdown", pd.DataFrame())

    if not equity_curve.empty:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=pd.to_datetime(equity_curve["created_at"], errors="coerce"),
                    y=pd.to_numeric(equity_curve["equity"], errors="coerce"),
                    mode="lines+markers",
                    name="Equity",
                )
            ]
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_execution, tab_broker, tab_assets, tab_errors = st.tabs(
        ["Job Health", "Open Positions", "Predictions", "Candidates", "Execution", "Broker Sync", "Assets", "Recent Errors"]
    )
    with tab_jobs:
        if job_health.empty:
            st.caption("job_runs가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(job_health), width="stretch", hide_index=True)
    with tab_positions:
        if open_positions.empty and open_orders.empty:
            st.caption("오픈 포지션과 주문이 없습니다.")
        else:
            if not open_positions.empty:
                st.caption("오픈 포지션")
                st.dataframe(format_frame_timestamps_for_display(open_positions), width="stretch", hide_index=True)
            if not open_orders.empty:
                st.caption("오픈 주문")
                st.dataframe(format_frame_timestamps_for_display(open_orders), width="stretch", hide_index=True)
    with tab_predictions:
        if prediction_report.empty:
            st.caption("prediction ledger가 비어 있습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(prediction_report.head(200)), width="stretch", hide_index=True)
    with tab_candidates:
        if candidate_scans.empty:
            st.caption("candidate_scans가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(candidate_scans.head(200)), width="stretch", hide_index=True)
    with tab_execution:
        if not noop_breakdown.empty:
            st.caption("No-op reason breakdown")
            st.dataframe(noop_breakdown, width="stretch", hide_index=True)
        if today_execution_events.empty:
            st.caption("오늘 execution event가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(today_execution_events.head(200)), width="stretch", hide_index=True)
    with tab_broker:
        if broker_sync_status.empty:
            st.caption("broker sync 상태가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(broker_sync_status), width="stretch", hide_index=True)
        if broker_sync_errors.empty:
            st.caption("최근 broker sync 오류가 없습니다.")
        else:
            st.caption("최근 broker sync / execution 이벤트")
            st.dataframe(format_frame_timestamps_for_display(broker_sync_errors.head(100)), width="stretch", hide_index=True)
    with tab_assets:
        if asset_overview.empty:
            st.caption("설정된 자산 유니버스가 없습니다.")
        else:
            st.dataframe(asset_overview, width="stretch", hide_index=True)
    with tab_errors:
        if recent_errors.empty:
            st.caption("최근 ERROR 이벤트가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(recent_errors), width="stretch", hide_index=True)


def render_operations_monitor(
    settings=None,
    dashboard_data: Dict[str, Any] | None = None,
    theme_mode: str | None = None,
) -> None:
    settings = settings or load_settings()
    theme_mode = get_active_theme_mode(theme_mode)
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    accounts_overview = data.get("accounts_overview", {})
    total_portfolio_overview = data.get("total_portfolio_overview", {})
    auto_trading_status = data.get("auto_trading_status", {})
    execution_summary = data.get("execution_summary", {})
    broker_sync_status = data.get("broker_sync_status", pd.DataFrame())
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    runtime_profile = data.get("runtime_profile", {})
    asset_overview = data.get("asset_overview", pd.DataFrame())
    job_health = data["job_health"]
    recent_errors = data["recent_errors"]
    recent_events = data.get("recent_events", pd.DataFrame())
    open_positions = data["open_positions"]
    open_orders = data["open_orders"]
    candidate_scans = data["candidate_scans"]
    prediction_report = data["prediction_report"]
    equity_curve = data["equity_curve"]
    equity_curves_by_account = data.get("equity_curves_by_account", {})
    today_execution_events = data.get("today_execution_events", pd.DataFrame())
    recent_orders = load_monitor_recent_orders(settings, limit=30)

    sync_rows: Dict[str, Dict[str, Any]] = {}
    if not broker_sync_status.empty and "job_name" in broker_sync_status.columns:
        for _, row in broker_sync_status.iterrows():
            sync_rows[str(row.get("job_name") or "")] = row.to_dict()

    st.subheader("운영 모니터링")
    st.caption("worker 상태, execution pipeline, broker sync 결과를 한 화면에서 확인합니다.")
    st.markdown(_operations_monitor_styles(theme_mode), unsafe_allow_html=True)

    feedback = st.session_state.pop("broker_sync_feedback", None)
    if isinstance(feedback, dict) and feedback.get("message"):
        if feedback.get("ok"):
            st.success(str(feedback["message"]))
        else:
            st.error(str(feedback["message"]))

    hero_cols = st.columns([1.45, 1.0], gap="large")
    with hero_cols[0]:
        st.markdown(
            _build_operations_balance_card_html(
                account_snapshot=account_snapshot,
                trade_performance=trade_performance,
                runtime_profile=runtime_profile,
            ),
            unsafe_allow_html=True,
        )
    with hero_cols[1]:
        st.markdown(
            _build_operations_runtime_card_html(
                auto_trading_status=auto_trading_status,
                broker_sync_rows=sync_rows,
                kis_runtime=kis_runtime,
                runtime_profile=runtime_profile,
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    action_cols = st.columns([1.0, 1.6], gap="large")
    with action_cols[0]:
        with st.container(border=True):
            st.caption("Auto Trading Control")
            pause_type = "secondary"
            resume_type = "primary" if str(auto_trading_status.get("state", "")).lower() == "paused" else "secondary"
            control_cols = st.columns(2, gap="small")
            if control_cols[0].button("Pause Entries", key="ops_pause", use_container_width=True, type=pause_type):
                repository.set_control_flag("trading_paused", "1", "set from streamlit monitor")
                st.rerun()
            if control_cols[1].button("Resume Entries", key="ops_resume", use_container_width=True, type=resume_type):
                repository.set_control_flag("trading_paused", "0", "set from streamlit monitor")
                st.rerun()
            st.caption(f"Current state: {auto_trading_status.get('label', 'Stopped')}")
            st.caption(auto_trading_status_text(auto_trading_status))
    with action_cols[1]:
        with st.container(border=True):
            st.caption("Broker Sync Actions")
            sync_button_cols = st.columns(4, gap="small")
            sync_jobs = [
                ("Market Sync", "broker_market_status", "ops_sync_market"),
                ("Account Sync", "broker_account_sync", "ops_sync_account"),
                ("Order Sync", "broker_order_sync", "ops_sync_orders"),
                ("Position Sync", "broker_position_sync", "ops_sync_position"),
            ]
            for column, (label, job_name, key) in zip(sync_button_cols, sync_jobs):
                if column.button(label, key=key, use_container_width=True):
                    ok, message = run_manual_runtime_job(job_name)
                    st.session_state["broker_sync_feedback"] = {"ok": ok, "message": message}
                    st.rerun()
            st.caption("계좌, 주문, 포지션, 장 상태를 즉시 다시 읽어서 모니터를 갱신합니다.")

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.caption("브로커 동기화 상태")
        st.caption("장 상태, 계좌, 주문, 포지션 동기화 결과를 한 묶음으로 봅니다.")
        sync_cols = st.columns(4, gap="small")
        sync_order = [
            "broker_market_status",
            "broker_account_sync",
            "broker_order_sync",
            "broker_position_sync",
        ]
        for column, job_name in zip(sync_cols, sync_order):
            column.markdown(
                _build_broker_sync_card_html(job_name, sync_rows.get(job_name, {"status": "never"})),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    render_live_open_positions_panel(settings)

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    primary_metrics = st.columns(3, gap="small")
    primary_metrics[0].metric("Unresolved Predictions", f"{int(summary.get('unresolved_predictions', 0))}", border=True)
    primary_metrics[1].metric("Open Positions", f"{int(summary.get('open_positions', 0))}", border=True)
    primary_metrics[2].metric("Open Orders", f"{int(summary.get('open_orders', 0))}", border=True)

    execution_cols = st.columns(5, gap="small")
    execution_cols[0].metric("Candidates", f"{int(execution_summary.get('today_candidate_count', 0))}", border=True)
    execution_cols[1].metric("Allowed", f"{int(execution_summary.get('today_entry_allowed_count', 0))}", border=True)
    execution_cols[2].metric("Rejected", f"{int(execution_summary.get('today_entry_rejected_count', 0))}", border=True)
    execution_cols[3].metric("Submitted", f"{int(execution_summary.get('today_submitted_count', 0))}", border=True)
    execution_cols[4].metric("Filled", f"{int(execution_summary.get('today_filled_count', 0))}", border=True)

    execution_cols_2 = st.columns(5, gap="small")
    execution_cols_2[0].metric("Submit Requested", f"{int(execution_summary.get('today_submit_requested_count', 0))}", border=True)
    execution_cols_2[1].metric("Acknowledged", f"{int(execution_summary.get('today_acknowledged_count', 0))}", border=True)
    execution_cols_2[2].metric("Cancelled", f"{int(execution_summary.get('today_cancelled_count', 0))}", border=True)
    execution_cols_2[3].metric("No-op", f"{int(execution_summary.get('today_noop_count', 0))}", border=True)
    execution_cols_2[4].metric("Broker Rejects", f"{int(first_valid_float(kis_runtime.get('broker_rejects_today'), default=0.0))}", border=True)

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    status_cols = st.columns([1.35, 0.85], gap="large")
    with status_cols[0]:
        with st.container(border=True, height=360):
            st.caption("Recent Job Status")
            if job_health.empty:
                st.caption("No job history yet.")
            else:
                st.markdown(
                    f"**{auto_trading_status.get('label', 'Stopped')}**  \n"
                    f"{auto_trading_status_text(auto_trading_status)}"
                )
                job_preview = format_job_health_for_display(job_health, limit=5)
                preview_cols = [c for c in ["job", "status", "started_at", "finished_at", "retry_count", "error_message"] if c in job_preview.columns]
                st.dataframe(job_preview[preview_cols], width="stretch", hide_index=True, height=250)
    with status_cols[1]:
        with st.container(border=True, height=360):
            st.caption("Recent Broker Errors")
            if broker_sync_errors.empty:
                st.caption("No recent broker sync error.")
            else:
                st.dataframe(
                    format_frame_timestamps_for_display(broker_sync_errors.head(5)),
                    width="stretch",
                    hide_index=True,
                    height=250,
                )

    if not equity_curve.empty:
        st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.caption("Account Equity Curve")
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=pd.to_datetime(equity_curve["created_at"], errors="coerce"),
                        y=pd.to_numeric(equity_curve["equity"], errors="coerce"),
                        mode="lines+markers",
                        name="Equity",
                    )
                ]
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_execution, tab_broker, tab_assets, tab_errors = st.tabs(
        ["Job Health", "Open Positions", "Predictions", "Candidates", "Execution", "Broker Sync", "Assets", "Recent Errors"]
    )
    with tab_jobs:
        if job_health.empty:
            st.caption("No job history yet.")
        else:
            st.dataframe(format_job_health_for_display(job_health), width="stretch", hide_index=True)
    with tab_positions:
        if open_positions.empty and open_orders.empty:
            st.caption("No open position or open order.")
        else:
            if not open_positions.empty:
                st.caption("Open Positions")
                st.dataframe(format_frame_timestamps_for_display(open_positions), width="stretch", hide_index=True)
            if not open_orders.empty:
                st.caption("Open Orders")
                st.dataframe(format_frame_timestamps_for_display(open_orders), width="stretch", hide_index=True)
    with tab_predictions:
        if prediction_report.empty:
            st.caption("Prediction ledger is empty.")
        else:
            st.dataframe(format_frame_timestamps_for_display(prediction_report.head(200)), width="stretch", hide_index=True)
    with tab_candidates:
        if candidate_scans.empty:
            st.caption("Candidate scans are empty.")
        else:
            st.dataframe(format_frame_timestamps_for_display(candidate_scans.head(200)), width="stretch", hide_index=True)
    with tab_execution:
        if not noop_breakdown.empty:
            st.caption("No-op reason breakdown")
            st.dataframe(noop_breakdown, width="stretch", hide_index=True)
        if today_execution_events.empty:
            st.caption("No execution event today.")
        else:
            st.dataframe(format_frame_timestamps_for_display(today_execution_events.head(200)), width="stretch", hide_index=True)
    with tab_broker:
        if broker_sync_status.empty:
            st.caption("No broker sync history.")
        else:
            st.dataframe(format_frame_timestamps_for_display(broker_sync_status), width="stretch", hide_index=True)
        if broker_sync_errors.empty:
            st.caption("No recent broker sync error.")
        else:
            st.caption("Recent broker sync / execution events")
            st.dataframe(format_frame_timestamps_for_display(broker_sync_errors.head(100)), width="stretch", hide_index=True)
    with tab_assets:
        if asset_overview.empty:
            st.caption("No configured asset universe.")
        else:
            st.dataframe(asset_overview, width="stretch", hide_index=True)
    with tab_errors:
        if recent_errors.empty:
            st.caption("No recent ERROR event.")
        else:
            st.dataframe(format_frame_timestamps_for_display(recent_errors), width="stretch", hide_index=True)


def render_operations_monitor(
    settings=None,
    dashboard_data: Dict[str, Any] | None = None,
    theme_mode: str | None = None,
) -> None:
    settings = settings or load_settings()
    theme_mode = get_active_theme_mode(theme_mode)
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    accounts_overview = data.get("accounts_overview", {})
    total_portfolio_overview = data.get("total_portfolio_overview", {})
    account_snapshot = dict(summary.get("latest_account") or {})
    trade_performance = data["trade_performance"]
    auto_trading_status = data.get("auto_trading_status", {})
    execution_summary = data.get("execution_summary", {})
    broker_sync_status = data.get("broker_sync_status", pd.DataFrame())
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    runtime_profile = data.get("runtime_profile", {})
    asset_overview = data.get("asset_overview", pd.DataFrame())
    job_health = data["job_health"]
    recent_errors = data["recent_errors"]
    recent_events = data.get("recent_events", pd.DataFrame())
    open_positions = data["open_positions"]
    open_orders = data["open_orders"]
    candidate_scans = data["candidate_scans"]
    prediction_report = data["prediction_report"]
    equity_curve = data["equity_curve"]
    equity_curves_by_account = data.get("equity_curves_by_account", {})
    today_execution_events = data.get("today_execution_events", pd.DataFrame())
    recent_orders = load_monitor_recent_orders(settings, limit=30)
    noop_breakdown = execution_summary.get("today_noop_breakdown", pd.DataFrame())

    sync_rows: Dict[str, Dict[str, Any]] = {}
    if not broker_sync_status.empty and "job_name" in broker_sync_status.columns:
        for _, row in broker_sync_status.iterrows():
            sync_rows[str(row.get("job_name") or "")] = row.to_dict()

    st.subheader("운영 모니터")
    st.markdown(_operations_monitor_styles(theme_mode), unsafe_allow_html=True)

    feedback = st.session_state.pop("broker_sync_feedback", None)
    if isinstance(feedback, dict) and feedback.get("message"):
        if feedback.get("ok"):
            st.success(str(feedback["message"]))
        else:
            st.error(str(feedback["message"]))

    scope_specs = [
        ("한국주식(KIS)", ACCOUNT_KIS_KR_PAPER),
        ("미국주식(SIM)", ACCOUNT_SIM_US_EQUITY),
        ("코인(SIM)", ACCOUNT_SIM_CRYPTO),
        ("전체(참고용)", "__total__"),
    ]
    scope_labels = [label for label, _ in scope_specs]
    default_scope_value = ACCOUNT_KIS_KR_PAPER if ACCOUNT_KIS_KR_PAPER in accounts_overview else "__total__"
    default_scope_label = next((label for label, value in scope_specs if value == default_scope_value), scope_labels[0])
    selected_scope_label = st.radio(
        "계좌 보기 기준",
        scope_labels,
        horizontal=True,
        key="ops_account_scope_label",
        index=scope_labels.index(default_scope_label),
        label_visibility="collapsed",
    )
    selected_scope_value = dict(scope_specs)[selected_scope_label]

    def _scoped_frame(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
        if selected_scope_value == "__total__" or frame.empty:
            return frame.copy()
        for column in columns:
            if column in frame.columns:
                return frame.loc[frame[column].fillna("").astype(str) == str(selected_scope_value)].copy()
        return frame.iloc[0:0].copy()

    def _build_scope_summary(events_frame: pd.DataFrame) -> Dict[str, Any]:
        if selected_scope_value == "__total__":
            return execution_summary
        counts = {
            f"today_{key}_count": int((events_frame["event_type"].astype(str) == key).sum()) if not events_frame.empty else 0
            for key in [
                "candidate",
                "entry_allowed",
                "entry_rejected",
                "submit_requested",
                "submitted",
                "acknowledged",
                "filled",
                "rejected",
                "cancelled",
                "noop",
            ]
        }
        if events_frame.empty or "details" not in events_frame.columns:
            counts["today_noop_breakdown"] = pd.DataFrame(columns=["reason", "count"])
            return counts
        noop_breakdown = (
            events_frame.loc[events_frame["event_type"].astype(str) == "noop", "details"]
            .map(lambda item: str((item or {}).get("reason") or "unknown"))
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="count")
        )
        counts["today_noop_breakdown"] = noop_breakdown
        return counts

    selected_account = dict(accounts_overview.get(selected_scope_value) or {}) if selected_scope_value != "__total__" else {}
    selected_snapshot = (
        dict(selected_account.get("latest_snapshot") or {})
        if selected_account
        else {
            "created_at": total_portfolio_overview.get("last_sync_time") or summary.get("latest_job_at"),
            "equity": total_portfolio_overview.get("equity"),
            "cash": total_portfolio_overview.get("cash"),
            "gross_exposure": np.nan,
            "net_exposure": np.nan,
            "drawdown_pct": total_portfolio_overview.get("drawdown_pct"),
        }
    )
    selected_trade_performance = selected_account.get("trade_performance", total_portfolio_overview.get("trade_performance", data["trade_performance"]))
    selected_open_positions = _scoped_frame(open_positions, "account_id")
    selected_open_orders = _scoped_frame(open_orders, "account_id")
    selected_recent_orders = _scoped_frame(recent_orders, "account_id")
    selected_prediction_report = _scoped_frame(prediction_report, "execution_account_id")
    selected_candidate_scans = _scoped_frame(candidate_scans, "execution_account_id")
    selected_execution_events = _scoped_frame(today_execution_events, "account_id")
    selected_recent_events = _scoped_frame(recent_events, "account_id")
    selected_recent_errors = _scoped_frame(recent_errors, "account_id")
    selected_broker_sync_errors = _scoped_frame(broker_sync_errors, "account_id")
    selected_equity_curve = (
        equity_curve
        if selected_scope_value == "__total__"
        else equity_curves_by_account.get(selected_scope_value, pd.DataFrame())
    )
    selected_execution_summary = _build_scope_summary(selected_execution_events)
    noop_breakdown = selected_execution_summary.get("today_noop_breakdown", pd.DataFrame())
    unresolved_predictions = (
        int((selected_prediction_report["status"].astype(str) == "unresolved").sum())
        if not selected_prediction_report.empty and "status" in selected_prediction_report.columns
        else 0
    )
    selected_asset_overview = asset_overview.copy()
    scope_asset_type = str(selected_account.get("asset_scope") or "")
    if selected_scope_value != "__total__" and scope_asset_type and not asset_overview.empty and "자산유형" in asset_overview.columns:
        selected_asset_overview = asset_overview.loc[asset_overview["자산유형"].astype(str) == scope_asset_type].copy()

    def _format_currency_totals(values: Dict[str, Any]) -> str:
        parts = []
        for currency, amount in (values or {}).items():
            try:
                numeric = float(amount)
            except Exception:
                continue
            if not np.isfinite(numeric):
                continue
            parts.append(f"{format_price_value(numeric)} {currency}")
        return " / ".join(parts) if parts else "N/A"

    def _render_scoped_live_panel() -> None:
        with st.container(border=True):
            panel_title = "실시간 보유 포지션" if selected_scope_value != "__total__" else "전체 보유 현황"
            st.caption(panel_title)
            view_mode = st.radio(
                "실시간 보유 패널 보기",
                ["보유 포지션", "매수·매도 내역"],
                horizontal=True,
                key=f"ops_live_positions_mode_{selected_scope_value}",
                label_visibility="collapsed",
            )
            if view_mode == "보유 포지션":
                if selected_open_positions.empty:
                    st.caption("현재 보유 중인 포지션이 없습니다.")
                    return
                refresh_token = int(pd.Timestamp.now(tz="UTC").timestamp() // 15)
                live_view = build_live_open_positions_view(selected_open_positions, refresh_token=refresh_token)
                metric_cols = st.columns(3, gap="small")
                metric_cols[0].metric("보유 종목 수", f"{len(selected_open_positions)}", border=True)
                metric_cols[1].metric(
                    "롱 포지션",
                    f"{int((selected_open_positions['side'].astype(str) == 'LONG').sum())}",
                    border=True,
                )
                metric_cols[2].metric(
                    "숏 포지션",
                    f"{int((selected_open_positions['side'].astype(str) == 'SHORT').sum())}",
                    border=True,
                )
                st.dataframe(live_view, width="stretch", hide_index=True, height=260)
                return
            if selected_recent_orders.empty:
                st.caption("최근 주문 기록이 없습니다.")
                return
            order_view = build_recent_order_activity_view(selected_recent_orders)
            metric_cols = st.columns(3, gap="small")
            metric_cols[0].metric("최근 주문 수", f"{len(selected_recent_orders)}", border=True)
            metric_cols[1].metric(
                "매수 주문",
                f"{int((selected_recent_orders['side'].astype(str).str.lower() == 'buy').sum())}",
                border=True,
            )
            metric_cols[2].metric(
                "매도 주문",
                f"{int((selected_recent_orders['side'].astype(str).str.lower() == 'sell').sum())}",
                border=True,
            )
            st.dataframe(order_view, width="stretch", hide_index=True, height=260)

    if str(MONITOR_UI_VARIANT or "polished").lower() == "polished":
        st.markdown(
            _build_operations_overview_strip_html(
                auto_trading_status=auto_trading_status,
                runtime_profile=runtime_profile,
                account_snapshot=selected_snapshot,
            ),
            unsafe_allow_html=True,
        )

    st.caption("계좌별 운영 현황")
    overview_cols = st.columns(4, gap="small")
    account_card_specs = [
        ("한국주식(KIS)", accounts_overview.get(ACCOUNT_KIS_KR_PAPER, {}), False),
        ("미국주식(SIM)", accounts_overview.get(ACCOUNT_SIM_US_EQUITY, {}), False),
        ("코인(SIM)", accounts_overview.get(ACCOUNT_SIM_CRYPTO, {}), False),
        ("전체(참고용)", total_portfolio_overview, True),
    ]
    for column, (label, payload, is_total) in zip(overview_cols, account_card_specs):
        with column:
            with st.container(border=True):
                st.caption(label)
                if is_total:
                    st.metric("평가 자산", _format_currency_totals(payload.get("equity_by_currency", {})), border=True)
                    st.caption("주문 가능 잔고 기준이 아닙니다.")
                    st.caption(f"보유 포지션 {int(payload.get('open_positions', 0))} · 대기 주문 {int(payload.get('pending_orders', 0))}")
                    st.caption(format_display_timestamp(payload.get("last_sync_time")) or "최근 동기화 기록 없음")
                else:
                    currency = str(payload.get("currency") or "KRW")
                    st.metric("평가 자산", f"{format_price_value(float(payload.get('equity', 0.0) or 0.0))} {currency}", border=True)
                    st.caption(f"예수금 {format_price_value(float(payload.get('cash', 0.0) or 0.0))} {currency}")
                    st.caption(
                        f"낙폭 {format_pct_value(float(payload.get('drawdown_pct', 0.0) or 0.0))} · 포지션 {int(payload.get('open_positions', 0))} · 주문 {int(payload.get('pending_orders', 0))}"
                    )
                    st.caption(
                        f"마지막 동기화 {format_display_timestamp(payload.get('last_sync_time')) or '기록 없음'} · {_sync_status_label(str(payload.get('last_sync_status') or 'never'))}"
                    )

    if selected_scope_value == "__total__":
        st.warning(str(total_portfolio_overview.get("warning") or "전체 합산 뷰는 참고용입니다."))
    else:
        st.info(
            f"현재 보기 기준: {selected_scope_label} · 브로커 {selected_account.get('broker_mode', '-')}"
            f" · 기준 통화 {selected_account.get('currency', '-')}"
            " · RiskEngine과 주문 가능 잔고 판단은 이 계좌만 사용합니다."
        )

    hero_cols = st.columns([1.15, 1.0], gap="large")
    with hero_cols[0]:
        if selected_scope_value == "__total__":
            with st.container(border=True):
                st.caption("전체 포트폴리오 합산")
                top_metrics = st.columns(3, gap="small")
                top_metrics[0].metric("평가 자산", _format_currency_totals(total_portfolio_overview.get("equity_by_currency", {})), border=True)
                total_current_pnl = {
                    currency: float(total_portfolio_overview.get("realized_pnl_by_currency", {}).get(currency, 0.0))
                    + float(total_portfolio_overview.get("unrealized_pnl_by_currency", {}).get(currency, 0.0))
                    for currency in set(total_portfolio_overview.get("realized_pnl_by_currency", {}))
                    | set(total_portfolio_overview.get("unrealized_pnl_by_currency", {}))
                }
                top_metrics[1].metric("현재 손익", _format_currency_totals(total_current_pnl), border=True)
                top_metrics[2].metric("총 익스포저", _format_currency_totals(total_portfolio_overview.get("gross_exposure_by_currency", {})), border=True)
                st.caption(f"열린 포지션 {int(total_portfolio_overview.get('open_positions', 0))} · 대기 주문 {int(total_portfolio_overview.get('pending_orders', 0))}")
                st.caption(
                    f"최근 동기화 {format_display_timestamp(total_portfolio_overview.get('last_sync_time')) or '기록 없음'}"
                    f" · {str(runtime_profile.get('name') or settings.profile_name)}"
                )
        else:
            st.markdown(
                _build_operations_balance_card_html(
                    account_snapshot=selected_snapshot,
                    trade_performance=selected_trade_performance,
                    runtime_profile=runtime_profile,
                ),
                unsafe_allow_html=True,
            )
        st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)
        _render_scoped_live_panel()
    with hero_cols[1]:
        if selected_scope_value != "__total__":
            st.caption(
                f"선택 계좌 마지막 동기화 {format_display_timestamp(selected_account.get('last_sync_time')) or '기록 없음'}"
                f" · {_sync_status_label(str(selected_account.get('last_sync_status') or 'never'))}"
            )
        else:
            st.caption("브로커 동기화 잡 상태는 계좌 공통 런타임 기준입니다.")
        st.markdown(
            _build_operations_runtime_card_html(
                auto_trading_status=auto_trading_status,
                broker_sync_rows=sync_rows,
                kis_runtime=kis_runtime,
                runtime_profile=runtime_profile,
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    action_cols = st.columns([1.0, 1.6], gap="large")
    with action_cols[0]:
        with st.container(border=True):
            st.caption("자동매매 제어")
            pause_type = "secondary"
            resume_type = "primary" if str(auto_trading_status.get("state", "")).lower() == "paused" else "secondary"
            control_cols = st.columns(2, gap="small")
            if control_cols[0].button("진입 일시중지", key="ops_pause", use_container_width=True, type=pause_type):
                repository.set_control_flag("trading_paused", "1", "set from streamlit monitor")
                st.rerun()
            if control_cols[1].button("진입 재개", key="ops_resume", use_container_width=True, type=resume_type):
                repository.set_control_flag("trading_paused", "0", "set from streamlit monitor")
                st.rerun()
            st.caption(f"현재 상태: {_localized_auto_trading_label(auto_trading_status)}")
            st.caption(auto_trading_status_text(auto_trading_status))

    with action_cols[1]:
        with st.container(border=True):
            st.caption("브로커 동기화 실행")
            sync_button_cols = st.columns(4, gap="small")
            sync_jobs = [
                ("장 상태 확인", "broker_market_status", "ops_sync_market"),
                ("계좌 동기화", "broker_account_sync", "ops_sync_account"),
                ("주문 동기화", "broker_order_sync", "ops_sync_orders"),
                ("포지션 동기화", "broker_position_sync", "ops_sync_position"),
            ]
            for column, (label, job_name, key) in zip(sync_button_cols, sync_jobs):
                if column.button(label, key=key, use_container_width=True):
                    ok, message = run_manual_runtime_job(job_name)
                    st.session_state["broker_sync_feedback"] = {"ok": ok, "message": message}
                    st.rerun()

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.caption("브로커 동기화 상태")
        sync_cols = st.columns(4, gap="small")
        sync_order = [
            "broker_market_status",
            "broker_account_sync",
            "broker_order_sync",
            "broker_position_sync",
        ]
        for column, job_name in zip(sync_cols, sync_order):
            column.markdown(
                _build_broker_sync_card_html(job_name, sync_rows.get(job_name, {"status": "never"})),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    primary_metrics = st.columns(3, gap="small")
    primary_metrics[0].metric("미정산 예측", f"{unresolved_predictions if selected_scope_value != '__total__' else int(summary.get('unresolved_predictions', 0))}", border=True)
    primary_metrics[1].metric("보유 포지션", f"{len(selected_open_positions) if selected_scope_value != '__total__' else int(summary.get('open_positions', 0))}", border=True)
    primary_metrics[2].metric("대기 주문", f"{len(selected_open_orders) if selected_scope_value != '__total__' else int(summary.get('open_orders', 0))}", border=True)

    execution_cols = st.columns(5, gap="small")
    execution_cols[0].metric("오늘 후보", f"{len(selected_candidate_scans) if selected_scope_value != '__total__' else int(execution_summary.get('today_candidate_count', 0))}", border=True)
    execution_cols[1].metric("진입 허용", f"{int(selected_execution_summary.get('today_entry_allowed_count', 0))}", border=True)
    execution_cols[2].metric("진입 거절", f"{int(selected_execution_summary.get('today_entry_rejected_count', 0))}", border=True)
    execution_cols[3].metric("주문 제출", f"{int(selected_execution_summary.get('today_submitted_count', 0))}", border=True)
    execution_cols[4].metric("체결 완료", f"{int(selected_execution_summary.get('today_filled_count', 0))}", border=True)

    execution_cols_2 = st.columns(5, gap="small")
    execution_cols_2[0].metric("제출 요청", f"{int(selected_execution_summary.get('today_submit_requested_count', 0))}", border=True)
    execution_cols_2[1].metric("접수 완료", f"{int(selected_execution_summary.get('today_acknowledged_count', 0))}", border=True)
    execution_cols_2[2].metric("취소", f"{int(selected_execution_summary.get('today_cancelled_count', 0))}", border=True)
    execution_cols_2[3].metric("미실행", f"{int(selected_execution_summary.get('today_noop_count', 0))}", border=True)
    execution_cols_2[4].metric(
        "브로커 거절",
        f"{int(first_valid_float(kis_runtime.get('broker_rejects_today'), default=0.0)) if selected_scope_value in {'__total__', ACCOUNT_KIS_KR_PAPER} else 0}",
        border=True,
    )

    st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)

    status_cols = st.columns([1.35, 0.85], gap="large")
    with status_cols[0]:
        with st.container(border=True, height=360):
            st.caption("최근 작업 상태")
            if job_health.empty:
                st.caption("아직 작업 이력이 없습니다.")
            else:
                st.markdown(
                    f"**{_localized_auto_trading_label(auto_trading_status)}**  \n"
                    f"{auto_trading_status_text(auto_trading_status)}"
                )
                job_preview = build_monitor_table_view(job_health, limit=5)
                preview_cols = [c for c in ["작업", "상태", "시작 시각", "종료 시각", "재시도", "오류 메시지"] if c in job_preview.columns]
                st.dataframe(job_preview[preview_cols] if preview_cols else job_preview, width="stretch", hide_index=True, height=250)

    with status_cols[1]:
        with st.container(border=True, height=360):
            st.caption("최근 브로커 오류")
            if selected_broker_sync_errors.empty:
                st.caption("최근 브로커 동기화 오류가 없습니다.")
            else:
                st.dataframe(
                    build_monitor_table_view(selected_broker_sync_errors, limit=5),
                    width="stretch",
                    hide_index=True,
                    height=250,
                )

    if not selected_equity_curve.empty:
        st.markdown('<div class="alt-ops-section-gap"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.caption("계좌 자산 추이" if selected_scope_value != "__total__" else "전체 포트폴리오 추이(참고용)")
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=pd.to_datetime(selected_equity_curve["created_at"], errors="coerce"),
                        y=pd.to_numeric(selected_equity_curve["equity"], errors="coerce"),
                        mode="lines+markers",
                        name="자산",
                    )
                ]
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_execution, tab_broker, tab_assets, tab_errors = st.tabs(
        ["작업 이력", "보유 현황", "예측 기록", "후보 종목", "실행 이벤트", "브로커 동기화", "자산 설정", "최근 오류"]
    )
    with tab_jobs:
        if job_health.empty:
            st.caption("아직 작업 이력이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(job_health), width="stretch", hide_index=True)

    with tab_positions:
        if selected_open_positions.empty and selected_open_orders.empty:
            st.caption("보유 포지션이나 대기 주문이 없습니다.")
        else:
            if not selected_open_positions.empty:
                st.caption("보유 포지션")
                st.dataframe(build_monitor_table_view(selected_open_positions), width="stretch", hide_index=True)
            if not selected_open_orders.empty:
                st.caption("대기 주문")
                st.dataframe(build_monitor_table_view(selected_open_orders), width="stretch", hide_index=True)

    with tab_predictions:
        if selected_prediction_report.empty:
            st.caption("예측 기록이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(selected_prediction_report, limit=200), width="stretch", hide_index=True)

    with tab_candidates:
        if selected_candidate_scans.empty:
            st.caption("후보 종목 기록이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(selected_candidate_scans, limit=200), width="stretch", hide_index=True)

    with tab_execution:
        if not noop_breakdown.empty:
            noop_view = noop_breakdown.rename(columns={"reason": "사유", "count": "건수"})
            st.caption("미실행 사유 요약")
            st.dataframe(noop_view, width="stretch", hide_index=True)
        if selected_execution_events.empty:
            st.caption("오늘 실행 이벤트가 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(selected_execution_events, limit=200), width="stretch", hide_index=True)

    with tab_broker:
        if broker_sync_status.empty:
            st.caption("브로커 동기화 이력이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(broker_sync_status), width="stretch", hide_index=True)
        if selected_broker_sync_errors.empty:
            st.caption("최근 브로커 동기화 오류가 없습니다.")
        else:
            st.caption("최근 브로커 동기화/실행 이벤트")
            st.dataframe(build_monitor_table_view(selected_broker_sync_errors, limit=100), width="stretch", hide_index=True)

    with tab_assets:
        if selected_asset_overview.empty:
            st.caption("설정된 자산 구성이 없습니다.")
        else:
            st.dataframe(selected_asset_overview, width="stretch", hide_index=True)

    with tab_errors:
        if selected_recent_errors.empty:
            st.caption("최근 오류 이벤트가 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(selected_recent_errors), width="stretch", hide_index=True)


def build_scan_row(symbol: str, result, forecast_days: int) -> Dict[str, float | str]:
    eval_metrics = result.final_holdout_metrics if not result.final_holdout_metrics.empty else result.metrics
    eval_trade_metrics = (
        result.final_holdout_trade_metrics if not result.final_holdout_trade_metrics.empty else result.trade_metrics
    )

    latest_close = float(result.latest_close)
    next_pred = float(result.future_frame["ensemble_pred"].iloc[0])
    last_pred = float(result.future_frame["ensemble_pred"].iloc[-1])
    expected_return = (last_pred / latest_close - 1.0) * 100.0

    mae = model_metric_value(eval_metrics, "Ensemble", "mae", default=0.0)
    direction_acc = model_metric_value(eval_metrics, "Ensemble", "direction_acc_pct", default=0.0)
    mape = model_metric_value(eval_metrics, "Ensemble", "mape_pct", default=0.0)
    mae_pct = (mae / max(latest_close, 1e-9)) * 100.0

    win_rate_pct = metric_value(eval_trade_metrics, "win_rate_pct")
    expectancy_pct = metric_value(eval_trade_metrics, "expectancy_pct")
    net_cum_return_pct = metric_value(eval_trade_metrics, "net_cum_return_pct")
    max_dd_pct = metric_value(eval_trade_metrics, "max_drawdown_pct")
    trades = metric_value(eval_trade_metrics, "trades")

    trend_score = float(np.clip((expected_return + 12.0) / 24.0, 0.0, 1.0) * 100.0)
    error_score = float(np.clip(100.0 - mae_pct * 12.0, 0.0, 100.0))
    drawdown_penalty = float(np.clip(abs(min(max_dd_pct, 0.0)) * 2.0, 0.0, 40.0))
    trade_score = float(np.clip(50.0 + expectancy_pct * 7.0 + (win_rate_pct - 50.0) * 0.8 - drawdown_penalty, 0, 100))
    score = 0.35 * direction_acc + 0.20 * trend_score + 0.15 * error_score + 0.30 * trade_score

    return {
        "심볼": symbol,
        "등급": grade_from_score(score),
        "유망도점수": score,
        "예상수익률(%)": expected_return,
        "최종홀드아웃_기대값(%)": expectancy_pct,
        "최종홀드아웃_누적수익률(%)": net_cum_return_pct,
        "최종홀드아웃_방향정확도(%)": direction_acc,
        "최종홀드아웃_승률(%)": win_rate_pct,
        "최종홀드아웃_MDD(%)": max_dd_pct,
        "최종홀드아웃_거래횟수": trades,
        "최종홀드아웃_MAE/가격(%)": mae_pct,
        "최종홀드아웃_MAPE(%)": mape,
        "최근종가": latest_close,
        "내일예측종가": next_pred,
        f"{forecast_days}일예측종가": last_pred,
    }


def render_single_result(result, forecast_days: int, is_mobile_ui: bool, asset_type: str, korea_market: str) -> None:
    latest_close = result.latest_close
    last_pred = float(result.future_frame["ensemble_pred"].iloc[-1])
    next_pred = float(result.future_frame["ensemble_pred"].iloc[0])
    expected_return = (last_pred / latest_close - 1.0) * 100.0
    trade_mode = str(summary_item_value(result.validation_summary, "trade_mode", "close_to_close"))
    next_plan = result.future_frame.iloc[0]
    planned_signal = float(next_plan.get("planned_signal", 0.0))
    position_size = float(next_plan.get("position_size", 0.0))
    entry_estimate = float(next_plan.get("entry_estimate", float("nan")))
    stop_level = float(next_plan.get("stop_level", float("nan")))
    take_level = float(next_plan.get("take_level", float("nan")))
    atr_now = float(next_plan.get("atr_14", float("nan")))
    next_move_pct = float(next_plan.get("ensemble_pred_return_pct", float("nan")))
    if planned_signal > 1e-12:
        plan_direction = "LONG"
    elif planned_signal < -1e-12:
        plan_direction = "SHORT"
    else:
        plan_direction = "FLAT"

    st.subheader("핵심 요약")
    if is_mobile_ui:
        top_row_1 = st.columns(2)
        top_row_2 = st.columns(2)
        top_row_1[0].metric("최근 종가", f"{latest_close:,.2f}")
        top_row_1[1].metric("내일 예측 종가", f"{next_pred:,.2f}")
        top_row_2[0].metric(f"{forecast_days}일 기대수익률", f"{expected_return:+.2f}%")
        top_row_2[1].metric("검증모드", "워크포워드" if result.validation_mode == "walk_forward" else "홀드아웃")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최근 종가", f"{latest_close:,.2f}")
        c2.metric("내일 예측 종가", f"{next_pred:,.2f}")
        c3.metric(f"{forecast_days}일 기대 수익률", f"{expected_return:+.2f}%")
        c4.metric("검증모드", "워크포워드" if result.validation_mode == "walk_forward" else "홀드아웃")

    st.caption(f"분석 심볼: `{result.symbol}`")
    st.subheader("다음 거래 계획")
    if is_mobile_ui:
        plan_top = st.columns(2)
        plan_mid = st.columns(2)
        plan_bot = st.columns(2)
        plan_top[0].metric("방향", plan_direction, f"포지션 {position_size:.2f}배")
        plan_top[1].metric("예상 변화율", format_pct_value(next_move_pct), trade_mode)
        plan_mid[0].metric("예상 진입가", format_price_value(entry_estimate))
        plan_mid[1].metric("손절가", format_price_value(stop_level))
        plan_bot[0].metric("익절가", format_price_value(take_level))
        plan_bot[1].metric("ATR(14)", format_price_value(atr_now))
    else:
        plan_cols = st.columns(6)
        plan_cols[0].metric("방향", plan_direction, f"포지션 {position_size:.2f}배")
        plan_cols[1].metric("예상 변화율", format_pct_value(next_move_pct), trade_mode)
        plan_cols[2].metric("예상 진입가", format_price_value(entry_estimate))
        plan_cols[3].metric("손절가", format_price_value(stop_level))
        plan_cols[4].metric("익절가", format_price_value(take_level))
        plan_cols[5].metric("ATR(14)", format_price_value(atr_now))

    if plan_direction == "FLAT":
        st.caption("현재 기준으로는 최소 신호 강도 미만이라 신규 진입보다 관망에 가깝습니다.")
    elif trade_mode == "open_to_close":
        st.caption("`open→close` 모드에서는 다음 시가를 알 수 없어서 예상 진입가는 현재 종가 기준 추정치로 표시합니다.")

    symbol_key = re.sub(r"[^A-Za-z0-9]+", "_", str(result.symbol))
    chart_control_cols = st.columns([1.3, 1.0, 1.2] if not is_mobile_ui else [1.0, 1.0, 1.0])
    chart_range_label = chart_control_cols[0].selectbox(
        "가격 차트 범위",
        ["최근 3개월", "최근 6개월", "최근 1년", "최근 2년", "전체"],
        index=1,
        key=f"chart_range_{symbol_key}",
    )
    show_rangeslider = chart_control_cols[1].checkbox(
        "하단 줌 바",
        value=not is_mobile_ui,
        key=f"chart_rangeslider_{symbol_key}",
    )
    show_full_context = chart_control_cols[2].checkbox(
        "전체 기간도 보기",
        value=False,
        key=f"chart_full_context_{symbol_key}",
    )

    x_min, x_max = compute_price_chart_range(result=result, window_days=chart_window_days_from_label(chart_range_label))
    y_range = compute_y_axis_range_for_window(result=result, x_min=x_min, x_max=x_max)

    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=result.trade_backtest.index,
            y=(result.trade_backtest["equity_curve"] - 1.0) * 100.0,
            mode="lines",
            name="연구구간",
            line=dict(width=2, color="#3b82f6"),
        )
    )
    eq_fig.add_trace(
        go.Scatter(
            x=result.final_holdout_trade_backtest.index,
            y=(result.final_holdout_trade_backtest["equity_curve"] - 1.0) * 100.0,
            mode="lines",
            name="최종홀드아웃",
            line=dict(width=2, dash="dot", color="#f59e0b"),
        )
    )
    eq_fig.update_layout(
        height=230 if is_mobile_ui else 260,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        yaxis_title="누적수익률(%)",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.price_data.index,
            y=result.price_data["Close"],
            mode="lines",
            name="실제 종가",
            line=dict(width=2, color="#e5e7eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.test_frame["ensemble_pred"],
            mode="lines",
            name="연구구간 예측",
            line=dict(width=2, color="#3b82f6"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.final_holdout_frame.index,
            y=result.final_holdout_frame["ensemble_pred"],
            mode="lines",
            name="최종홀드아웃 예측",
            line=dict(width=2, dash="dot", color="#fca5a5"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_frame.index,
            y=result.future_frame["ensemble_pred"],
            mode="lines+markers",
            name="미래 예측",
            line=dict(width=3, color="#ef4444"),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_frame.index,
            y=result.future_frame["upper_band_1sigma"],
            mode="lines",
            name="상단밴드(1σ)",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_frame.index,
            y=result.future_frame["lower_band_1sigma"],
            mode="lines",
            name="하단밴드(1σ)",
            fill="tonexty",
            fillcolor="rgba(32, 201, 151, 0.18)",
            line=dict(width=0),
        )
    )
    fig.update_layout(
        height=460 if is_mobile_ui else 560,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(
            range=[x_min, x_max] if x_min is not None and x_max is not None else None,
            rangeslider=dict(visible=show_rangeslider, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
        ),
        yaxis=dict(range=y_range),
    )
    st.caption("기본은 최근 구간 중심으로 확대해서 보여줍니다. 하단 줌 바나 범위 버튼으로 직접 조절할 수 있습니다.")
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(eq_fig, width="stretch")

    if show_full_context:
        full_fig = go.Figure(fig)
        full_fig.update_layout(
            height=320 if is_mobile_ui else 420,
            xaxis=dict(
                range=None,
                rangeslider=dict(visible=True, thickness=0.06),
                rangeselector=dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ]
                ),
            ),
            yaxis=dict(range=None),
        )
        with st.expander("전체 기간 차트", expanded=False):
            st.plotly_chart(full_fig, width="stretch")

    trade_sample_view = build_trade_sample_view(result.trade_backtest)
    holdout_trade_sample_view = build_trade_sample_view(result.final_holdout_trade_backtest)

    with st.expander("매매 성과", expanded=True):
        if is_mobile_ui:
            st.caption("연구구간")
            st.dataframe(to_trade_view(result.trade_metrics), width="stretch", hide_index=True)
            if not trade_sample_view.empty:
                st.caption("연구구간 최근 거래 샘플")
                st.dataframe(trade_sample_view, width="stretch")
            st.caption("최종 홀드아웃")
            st.dataframe(to_trade_view(result.final_holdout_trade_metrics), width="stretch", hide_index=True)
            if not holdout_trade_sample_view.empty:
                st.caption("최종 홀드아웃 최근 거래 샘플")
                st.dataframe(holdout_trade_sample_view, width="stretch")
        else:
            bt_left, bt_right = st.columns(2)
            with bt_left:
                st.caption("연구구간")
                st.dataframe(to_trade_view(result.trade_metrics), width="stretch", hide_index=True)
                if not trade_sample_view.empty:
                    st.caption("최근 거래 샘플")
                    st.dataframe(trade_sample_view, width="stretch")
            with bt_right:
                st.caption("최종 홀드아웃")
                st.dataframe(to_trade_view(result.final_holdout_trade_metrics), width="stretch", hide_index=True)
                if not holdout_trade_sample_view.empty:
                    st.caption("최근 거래 샘플")
                    st.dataframe(holdout_trade_sample_view, width="stretch")

        st.caption("변동성 레짐별 성과")
        if result.regime_metrics.empty:
            st.warning("레짐 분해 데이터가 부족합니다.")
        else:
            regime_view = result.regime_metrics.rename(
                columns={
                    "segment": "구간",
                    "regime": "레짐",
                    "days": "일수",
                    "coverage_pct": "커버리지(%)",
                    "trades": "거래횟수",
                    "win_rate_pct": "승률(%)",
                    "expectancy_pct": "기대값/거래(%)",
                    "net_cum_return_pct": "누적수익률(%)",
                    "max_drawdown_pct": "최대낙폭(%)",
                    "avg_signal_abs": "평균신호강도(|signal|)",
                    "avg_volatility_pct": "평균변동성(%)",
                }
            )
            st.dataframe(regime_view, width="stretch", hide_index=True)

    with st.expander("모델 상세", expanded=False):
        st.caption("검증 설정")
        st.dataframe(to_validation_view(result.validation_summary), width="stretch", hide_index=True)
        if is_mobile_ui:
            st.caption("Validation")
            st.dataframe(to_model_view(result.validation_metrics), width="stretch", hide_index=True)
            st.caption("연구구간")
            st.dataframe(to_model_view(result.metrics), width="stretch", hide_index=True)
            st.caption("최종 홀드아웃")
            st.dataframe(to_model_view(result.final_holdout_metrics), width="stretch", hide_index=True)
        else:
            left, mid, right = st.columns(3)
            with left:
                st.caption("Validation")
                st.dataframe(to_model_view(result.validation_metrics), width="stretch", hide_index=True)
            with mid:
                st.caption("연구구간")
                st.dataframe(to_model_view(result.metrics), width="stretch", hide_index=True)
            with right:
                st.caption("최종 홀드아웃")
                st.dataframe(to_model_view(result.final_holdout_metrics), width="stretch", hide_index=True)

        weight_table = result.metrics[result.metrics["model"].isin(result.weights.keys())][["model"]].copy().assign(
            weight_pct=lambda df: df["model"].map(result.weights).fillna(0.0) * 100.0
        )
        weight_table = weight_table.rename(columns={"model": "모델", "weight_pct": "연구구간 가중치(%)"})
        st.caption("앙상블 가중치")
        st.dataframe(weight_table, width="stretch", hide_index=True)

    with st.expander("예측 기억 및 실제 비교", expanded=False):
        render_prediction_tracking_section(result=result, asset_type=asset_type, korea_market=korea_market)

    future_view = result.future_frame.rename(
        columns={
            "ensemble_pred": "앙상블예측",
            "entry_estimate": "예상진입가",
            "stop_level": "손절가",
            "take_level": "익절가",
            "planned_signal": "계획신호",
            "position_size": "포지션크기",
            "atr_14": "ATR(14)",
            "Ridge_pred": "Ridge예측",
            "RandomForest_pred": "RandomForest예측",
            "GradientBoosting_pred": "GradientBoosting예측",
            "lower_band_1sigma": "하단밴드(1σ)",
            "upper_band_1sigma": "상단밴드(1σ)",
        }
    )
    with st.expander("미래 예측 상세", expanded=False):
        st.dataframe(future_view, width="stretch")


def set_analysis_target(asset_type: str, raw_symbol: str, korea_market: str) -> None:
    st.session_state["analysis_asset_type"] = asset_type
    st.session_state["analysis_raw_symbol"] = raw_symbol
    st.session_state["analysis_korea_market"] = korea_market
    st.session_state["pending_analysis_target"] = {
        "asset_type": asset_type,
        "raw_symbol": raw_symbol,
        "korea_market": korea_market,
    }
    st.session_state["analysis_result"] = None
    st.session_state["analysis_result_symbol"] = ""
    st.session_state["analysis_result_forecast_days"] = 0
    st.session_state["analysis_auto_run"] = True


def build_dashboard_market_rows(asset_type: str, display_limit: int, refresh_token: int) -> Tuple[List[Dict[str, str | int | float | None]], List[str]]:
    if asset_type == "코인":
        entries = [
            {
                "순위": idx,
                "원순위": idx,
                "시장": "코인",
                "자산유형": "코인",
                "종목명": symbol,
                "심볼": symbol,
                "심볼힌트": symbol,
            }
            for idx, symbol in enumerate(WATCHLIST_PRESETS["코인"][:display_limit], start=1)
        ]
        symbols = tuple(str(row["심볼"]) for row in entries)
        snapshots = fetch_quote_snapshots(symbols=symbols, refresh_token=refresh_token) if symbols else {}
        out: List[Dict[str, str | int | float | None]] = []
        for row in entries:
            symbol = str(row["심볼"])
            snap = snapshots.get(symbol, {})
            enriched = dict(row)
            enriched["현재가"] = snap.get("current_price", float("nan"))
            enriched["전일대비(%)"] = snap.get("change_pct", float("nan"))
            enriched["통화"] = snap.get("currency", default_currency_from_symbol(symbol))
            out.append(enriched)
        return out, []

    scope = "국내" if asset_type == "한국주식" else "해외"
    rows, unresolved = resolve_single_top100_entries(scope=scope, display_limit=display_limit)
    quote_symbols = tuple(str(row["심볼"]) for row in rows if row["심볼"])
    snapshots = fetch_quote_snapshots(symbols=quote_symbols, refresh_token=refresh_token) if quote_symbols else {}
    out: List[Dict[str, str | int | float | None]] = []
    for row in rows:
        symbol = str(row["심볼"]) if row["심볼"] else ""
        snap = snapshots.get(symbol, {})
        enriched = dict(row)
        enriched["현재가"] = snap.get("current_price", float("nan"))
        enriched["전일대비(%)"] = snap.get("change_pct", float("nan"))
        enriched["통화"] = snap.get("currency", default_currency_from_symbol(symbol) if symbol else "")
        out.append(enriched)
    return out, unresolved


def build_analysis_target(asset_type: str, raw_symbol: str, korea_market: str) -> Tuple[str, str, str]:
    symbol = normalize_symbol(asset_type=asset_type, raw_symbol=raw_symbol, korea_market=korea_market)
    resolved_market = "KOSDAQ" if symbol.endswith(".KQ") else korea_market
    return asset_type, symbol, resolved_market


def run_analysis_target(
    run_asset_type: str,
    run_raw_symbol: str,
    run_korea_market: str,
    years: int,
    test_days: int,
    forecast_days: int,
    validation_mode: str,
    retrain_every: int,
    round_trip_cost_bps: float,
    min_signal_strength_pct: float,
    final_holdout_days: int,
    purge_days: int,
    embargo_days: int,
    target_mode: str,
    validation_days: int,
    allow_short: bool,
    trade_mode: str,
    target_daily_vol_pct: float,
    max_position_size: float,
    stop_loss_atr_mult: float,
    take_profit_atr_mult: float,
):
    return run_cached(
        symbol=run_raw_symbol,
        years=years,
        test_days=test_days,
        forecast_days=forecast_days,
        validation_mode=validation_mode,
        retrain_every=retrain_every,
        round_trip_cost_bps=round_trip_cost_bps,
        min_signal_strength_pct=min_signal_strength_pct,
        final_holdout_days=final_holdout_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
        target_mode=target_mode,
        validation_days=validation_days,
        allow_short=allow_short,
        trade_mode=trade_mode,
        target_daily_vol_pct=target_daily_vol_pct,
        max_position_size=max_position_size,
        stop_loss_atr_mult=stop_loss_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult,
    )


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def build_paper_symbol_catalog() -> Tuple[List[Dict[str, str]], List[str]]:
    entries = build_top100_entries(asset_type="한국주식")
    resolved_pairs, unresolved = resolve_top100_entries(asset_type="한국주식", entries=entries)
    resolved_map = {name: symbol for name, symbol in resolved_pairs}
    catalog: List[Dict[str, str]] = []
    for entry in entries:
        name = str(entry["name"])
        symbol = resolved_map.get(name)
        if not symbol:
            continue
        catalog.append(
            {
                "rank": str(entry["rank"]),
                "name": name,
                "symbol": symbol,
                "label": f"{int(entry['rank']):>3}. {name} ({symbol})",
            }
        )
    return catalog, unresolved


@st.cache_data(ttl=15, show_spinner=False)
def fetch_paper_account_snapshot(refresh_token: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    _ = refresh_token
    client = KISPaperClient()
    snapshot = client.get_account_snapshot()
    return dict(snapshot.summary), snapshot.holdings.copy()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_paper_quote_snapshot(symbol: str, refresh_token: int) -> Dict[str, Any]:
    _ = refresh_token
    client = KISPaperClient()
    return client.get_quote(symbol)


def build_paper_trade_plan(
    *,
    symbol: str,
    quote_snapshot: Dict[str, Any],
    account_summary: Dict[str, Any],
    holdings: pd.DataFrame,
    analysis_result: Any | None,
) -> Dict[str, Any]:
    symbol_code = extract_korean_stock_code(symbol)
    current_price = first_valid_float(quote_snapshot.get("current_price"))
    cash = max(0.0, first_valid_float(account_summary.get("cash"), default=0.0))
    holding_qty = 0
    if not holdings.empty and "symbol_code" in holdings.columns:
        matched = holdings.loc[holdings["symbol_code"].astype(str) == str(symbol_code)]
        if not matched.empty:
            holding_qty = int(first_valid_float(matched.iloc[0].get("보유수량"), default=0.0))

    empty_plan = {
        "side": "hold",
        "side_label": PAPER_ORDER_SIDE_LABELS["hold"],
        "predicted_move_pct": float("nan"),
        "planned_signal": 0.0,
        "position_size": 0.0,
        "entry_estimate": current_price,
        "stop_level": float("nan"),
        "take_level": float("nan"),
        "ensemble_pred": float("nan"),
        "recommended_budget": 0.0,
        "recommended_qty": 0,
        "holding_qty": holding_qty,
        "current_price": current_price,
        "note": "예측 결과가 아직 없습니다.",
    }
    if analysis_result is None or getattr(analysis_result, "future_frame", pd.DataFrame()).empty:
        return empty_plan

    row = analysis_result.future_frame.iloc[0]
    planned_signal = first_valid_float(row.get("planned_signal"), default=0.0)
    position_size = abs(first_valid_float(row.get("position_size"), default=abs(planned_signal)))
    predicted_move_pct = first_valid_float(row.get("ensemble_pred_return_pct"))
    ensemble_pred = first_valid_float(row.get("ensemble_pred"))
    entry_estimate = first_valid_float(row.get("entry_estimate"), default=current_price)
    stop_level = first_valid_float(row.get("stop_level"))
    take_level = first_valid_float(row.get("take_level"))
    side = "buy" if planned_signal > 1e-9 else "sell" if planned_signal < -1e-9 else "hold"
    note = ""

    recommended_budget = 0.0
    recommended_qty = 0
    order_ref_price = current_price if np.isfinite(current_price) and current_price > 0 else entry_estimate
    if side == "buy" and np.isfinite(order_ref_price) and order_ref_price > 0:
        recommended_budget = cash * min(max(abs(planned_signal), 0.0), 1.0)
        recommended_qty = int(np.floor(recommended_budget / order_ref_price))
        if recommended_qty <= 0 and cash >= order_ref_price:
            recommended_qty = 1
        if recommended_qty <= 0:
            note = "예수금 대비 추천 비중이 너무 작아 주문 수량이 0주입니다."
    elif side == "sell":
        if holding_qty <= 0:
            side = "hold"
            note = "하락 신호지만 보유 수량이 없어 숏 주문은 생략합니다."
        else:
            recommended_qty = max(1, int(np.floor(holding_qty * min(max(abs(planned_signal), 0.0), 1.0))))
            recommended_qty = min(recommended_qty, holding_qty)
            recommended_budget = recommended_qty * (order_ref_price if np.isfinite(order_ref_price) else 0.0)
    else:
        note = "신호가 임계값을 넘지 않아 관망입니다."

    return {
        "side": side,
        "side_label": PAPER_ORDER_SIDE_LABELS[side],
        "predicted_move_pct": predicted_move_pct,
        "planned_signal": planned_signal,
        "position_size": position_size,
        "entry_estimate": entry_estimate,
        "stop_level": stop_level,
        "take_level": take_level,
        "ensemble_pred": ensemble_pred,
        "recommended_budget": recommended_budget,
        "recommended_qty": int(recommended_qty),
        "holding_qty": holding_qty,
        "current_price": current_price,
        "note": note,
    }


def render_paper_trading_page(analysis_inputs: Dict[str, Any], is_mobile_ui: bool) -> None:
    st.subheader("모의매매")
    st.caption("한국투자증권 모의계좌(KIS REST)로 국내주식 모의 주문을 넣고, 계좌 성과를 추적합니다.")

    try:
        kis_config = KISPaperClient().config
    except KISPaperError as exc:
        st.error(f"KIS 모의매매 설정 오류: {exc}")
        return

    st.info(
        f"모의 계좌 `{kis_config.account_masked}` · 상품코드 `{kis_config.product_code}` · 도메인 `{kis_config.base_url}`",
        icon="ℹ️",
    )

    catalog, unresolved = build_paper_symbol_catalog()
    latest_analysis_symbol = str(st.session_state.get("analysis_result_symbol", "") or "")
    if latest_analysis_symbol.endswith((".KS", ".KQ")) and all(item["symbol"] != latest_analysis_symbol for item in catalog):
        catalog = [
            {
                "rank": "0",
                "name": "최근 분석 종목",
                "symbol": latest_analysis_symbol,
                "label": f"최근 분석 종목 ({latest_analysis_symbol})",
            }
        ] + catalog
    if not catalog:
        st.error("모의매매에 사용할 국내 종목 목록을 준비하지 못했습니다.")
        return

    option_map = {item["label"]: item for item in catalog}
    option_labels = list(option_map.keys())
    if st.session_state.get("paper_symbol_label") not in option_map:
        default_symbol = latest_analysis_symbol if latest_analysis_symbol.endswith((".KS", ".KQ")) else "005930.KS"
        default_label = next((label for label, item in option_map.items() if item["symbol"] == default_symbol), option_labels[0])
        st.session_state["paper_symbol_label"] = default_label
    selected_entry = option_map[st.session_state["paper_symbol_label"]]
    selected_market = "KOSDAQ" if str(selected_entry["symbol"]).endswith(".KQ") else "KOSPI"
    if not str(st.session_state.get("paper_manual_symbol", "")).strip():
        st.session_state["paper_market"] = selected_market

    if is_mobile_ui:
        st.selectbox("거래 종목", options=option_labels, key="paper_symbol_label")
        st.text_input("직접 입력(선택사항)", key="paper_manual_symbol", placeholder="예: 005930 또는 005930.KS")
        st.radio("시장", ["KOSPI", "KOSDAQ"], horizontal=True, key="paper_market")
        ctrl_cols = st.columns(3)
    else:
        sel_col, input_col, market_col = st.columns([3.5, 2.0, 1.3])
        sel_col.selectbox("거래 종목", options=option_labels, key="paper_symbol_label")
        input_col.text_input("직접 입력(선택사항)", key="paper_manual_symbol", placeholder="예: 005930 또는 005930.KS")
        market_col.radio("시장", ["KOSPI", "KOSDAQ"], horizontal=True, key="paper_market")
        ctrl_cols = st.columns([1.0, 1.0, 1.2])

    selected_entry = option_map[st.session_state["paper_symbol_label"]]
    paper_market = str(st.session_state.get("paper_market", selected_market))
    manual_symbol = str(st.session_state.get("paper_manual_symbol", "") or "").strip()
    if manual_symbol:
        paper_symbol = normalize_symbol(asset_type="한국주식", raw_symbol=manual_symbol, korea_market=paper_market)
        paper_name = manual_symbol
    else:
        paper_symbol = str(selected_entry["symbol"])
        paper_name = str(selected_entry["name"])
        paper_market = "KOSDAQ" if paper_symbol.endswith(".KQ") else "KOSPI"

    if ctrl_cols[0].button("계좌 새로고침", key="paper_refresh_account"):
        st.session_state["paper_account_refresh_token"] += 1
    if ctrl_cols[1].button("현재가 새로고침", key="paper_refresh_quote"):
        st.session_state["paper_quote_refresh_token"] += 1
    run_paper_analysis = ctrl_cols[2].button("예측 갱신", key="paper_run_analysis", type="primary")

    try:
        account_summary, account_holdings = fetch_paper_account_snapshot(
            refresh_token=int(st.session_state.get("paper_account_refresh_token", 0))
        )
        append_equity_snapshot(account_summary, account_holdings)
        quote_snapshot = fetch_paper_quote_snapshot(
            symbol=paper_symbol,
            refresh_token=int(st.session_state.get("paper_quote_refresh_token", 0)),
        )
    except Exception as exc:
        st.error(f"KIS 모의계좌 조회 실패: {exc}")
        return

    global_analysis_result = st.session_state.get("analysis_result")
    active_analysis_result = None
    if run_paper_analysis:
        try:
            with st.spinner(f"{paper_symbol} 예측 모델을 갱신하는 중..."):
                paper_result = run_analysis_target(
                    run_asset_type="한국주식",
                    run_raw_symbol=paper_symbol,
                    run_korea_market=paper_market,
                    **analysis_inputs,
                )
        except Exception as exc:
            st.error(f"모의매매 예측 실행 실패: {exc}")
        else:
            st.session_state["paper_analysis_result"] = paper_result
            st.session_state["paper_analysis_symbol"] = paper_symbol
            st.session_state["paper_analysis_forecast_days"] = int(analysis_inputs["forecast_days"])
            try:
                st.session_state["paper_prediction_run_id"] = save_prediction_snapshot(
                    asset_type="한국주식",
                    korea_market=paper_market,
                    result=paper_result,
                    notes="paper_analysis",
                )
            except Exception:
                st.session_state["paper_prediction_run_id"] = ""
            active_analysis_result = paper_result

    if active_analysis_result is None:
        saved_paper_symbol = str(st.session_state.get("paper_analysis_symbol", "") or "")
        saved_paper_result = st.session_state.get("paper_analysis_result")
        if saved_paper_result is not None and saved_paper_symbol == paper_symbol:
            active_analysis_result = saved_paper_result
        elif global_analysis_result is not None and latest_analysis_symbol == paper_symbol:
            active_analysis_result = global_analysis_result

    linked_prediction_run_id = ""
    if active_analysis_result is not None:
        if str(st.session_state.get("paper_analysis_symbol", "") or "") == paper_symbol:
            linked_prediction_run_id = str(st.session_state.get("paper_prediction_run_id", "") or "")
        elif latest_analysis_symbol == paper_symbol:
            linked_prediction_run_id = str(st.session_state.get("analysis_prediction_run_id", "") or "")

    plan = build_paper_trade_plan(
        symbol=paper_symbol,
        quote_snapshot=quote_snapshot,
        account_summary=account_summary,
        holdings=account_holdings,
        analysis_result=active_analysis_result,
    )

    current_price = first_valid_float(quote_snapshot.get("current_price"))
    change_pct = first_valid_float(quote_snapshot.get("change_pct"))
    total_eval = first_valid_float(account_summary.get("total_eval"))
    pnl_value = first_valid_float(account_summary.get("pnl"))
    cash_value = first_valid_float(account_summary.get("cash"))
    holding_count = int(first_valid_float(account_summary.get("holding_count"), default=0.0))

    metric_cols = st.columns(4)
    metric_cols[0].metric("거래 종목", paper_name, paper_symbol)
    metric_cols[1].metric("현재가", format_live_price(current_price, "KRW"), f"{change_pct:+.2f}%" if np.isfinite(change_pct) else None)
    metric_cols[2].metric("총평가금액", format_live_price(total_eval, "KRW"), f"{pnl_value:+,.0f} KRW" if np.isfinite(pnl_value) else None)
    metric_cols[3].metric("예수금", format_live_price(cash_value, "KRW"), f"보유 {holding_count}종목")

    plan_col, order_col = st.columns([1.3, 1.0]) if not is_mobile_ui else (st.container(), st.container())

    with plan_col:
        st.markdown("**예측 기반 주문안**")
        st.metric("추천 액션", plan["side_label"], f"{plan['predicted_move_pct']:+.2f}%" if np.isfinite(plan["predicted_move_pct"]) else None)
        plan_metrics = st.columns(3)
        plan_metrics[0].metric("예상 진입가", format_live_price(plan["entry_estimate"], "KRW"))
        plan_metrics[1].metric("손절가", format_live_price(plan["stop_level"], "KRW"))
        plan_metrics[2].metric("익절가", format_live_price(plan["take_level"], "KRW"))
        plan_metrics_2 = st.columns(3)
        plan_metrics_2[0].metric("예상 종가", format_live_price(plan["ensemble_pred"], "KRW"))
        plan_metrics_2[1].metric("추천 수량", f"{int(plan['recommended_qty'])}주")
        plan_metrics_2[2].metric("보유 수량", f"{int(plan['holding_qty'])}주")
        st.caption(
            f"계좌 예수금 기준 추천 배정금액: {format_live_price(plan['recommended_budget'], 'KRW')} · 신호 크기 {plan['planned_signal']:+.3f}"
        )
        if plan["note"]:
            st.info(plan["note"])
        if active_analysis_result is None:
            st.caption("아직 이 종목의 예측 결과가 없어 주문안이 제한적입니다. `예측 갱신`을 눌러 계산하세요.")

    with order_col:
        st.markdown("**모의 주문 실행**")
        suggested_side = plan["side"] if plan["side"] in {"buy", "sell"} else "buy"
        side_labels = ["매수", "매도"]
        default_side_index = 0 if suggested_side == "buy" else 1
        side_label = st.selectbox("주문 방향", options=side_labels, index=default_side_index, key="paper_order_side")
        side = "buy" if side_label == "매수" else "sell"
        order_type_label = st.selectbox("주문 유형", options=list(PAPER_ORDER_TYPE_OPTIONS.keys()), index=0, key="paper_order_type")
        order_type = PAPER_ORDER_TYPE_OPTIONS[order_type_label]
        default_qty = max(1, int(plan["recommended_qty"] or 1))
        order_qty = int(
            st.number_input("주문 수량", min_value=1, value=default_qty, step=1, key="paper_order_qty")
        )
        default_limit_price = int(round(plan["entry_estimate"])) if np.isfinite(plan["entry_estimate"]) else int(round(current_price or 0.0))
        limit_price = float(default_limit_price)
        if order_type == "limit":
            limit_price = float(
                st.number_input(
                    "지정가",
                    min_value=0,
                    value=max(0, default_limit_price),
                    step=1,
                    key="paper_limit_price",
                )
            )

        can_submit = True
        if side == "sell" and order_qty > int(plan["holding_qty"]):
            can_submit = False
            st.warning("매도 수량이 현재 보유 수량보다 많습니다.")
        if side == "buy" and np.isfinite(current_price):
            estimated_cost = order_qty * (limit_price if order_type == "limit" else current_price)
            if estimated_cost > cash_value > 0:
                st.warning("예수금보다 큰 주문입니다. KIS가 주문을 거절할 수 있습니다.")

        consent = st.checkbox("KIS 모의계좌에 실제 API 주문을 전송하는 것에 동의합니다.", key="paper_order_consent")
        if st.button(
            "추천대로 모의주문 실행" if plan["recommended_qty"] > 0 else "모의주문 실행",
            key="paper_submit_order",
            type="primary",
            disabled=(not consent) or (not can_submit),
        ):
            try:
                prediction_id = prediction_id_for_run(linked_prediction_run_id, forecast_horizon=1) if linked_prediction_run_id else None
                client = KISPaperClient()
                order_result = client.place_cash_order(
                    symbol_or_code=paper_symbol,
                    side=side,
                    quantity=order_qty,
                    order_type=order_type,
                    price=limit_price,
                )
                append_order_log(
                    {
                        "prediction_id": prediction_id,
                        "requested_at": order_result["requested_at"],
                        "symbol": paper_symbol,
                        "symbol_code": order_result["symbol_code"],
                        "name": paper_name,
                        "side": side,
                        "order_type": order_type,
                        "quantity": order_qty,
                        "requested_price": None if order_type == "market" else limit_price,
                        "quote_price": current_price,
                        "predicted_move_pct": plan["predicted_move_pct"],
                        "entry_estimate": plan["entry_estimate"],
                        "stop_level": plan["stop_level"],
                        "take_level": plan["take_level"],
                        "order_no": order_result["order_no"],
                        "message": order_result["message"],
                    }
                )
                if prediction_id and order_result.get("order_no"):
                    attach_order_to_prediction(prediction_id=prediction_id, order_id=str(order_result["order_no"]))
            except Exception as exc:
                st.error(f"모의 주문 실패: {exc}")
            else:
                st.success(f"주문 접수 완료: {order_result['message'] or '모의투자 주문이 전송되었습니다.'}")
                st.session_state["paper_account_refresh_token"] += 1
                st.session_state["paper_quote_refresh_token"] += 1
                st.rerun()

    st.markdown("**보유 종목**")
    if account_holdings.empty:
        st.info("현재 모의계좌 보유 종목이 없습니다.")
    else:
        st.dataframe(account_holdings, width="stretch", hide_index=True)

    equity_curve = load_equity_curve()
    equity_metrics = compute_equity_metrics(equity_curve)
    order_log = load_order_log(limit=200)

    perf_cols = st.columns(4)
    perf_cols[0].metric("에쿼티 샘플", f"{int(equity_metrics['samples'])}")
    perf_cols[1].metric(
        "누적 수익률",
        f"{equity_metrics['total_return_pct']:+.2f}%" if np.isfinite(equity_metrics["total_return_pct"]) else "N/A",
    )
    perf_cols[2].metric(
        "최대 낙폭",
        f"{equity_metrics['max_drawdown_pct']:.2f}%" if np.isfinite(equity_metrics["max_drawdown_pct"]) else "N/A",
    )
    perf_cols[3].metric("주문 로그", f"{len(order_log)}건")

    perf_cols_2 = st.columns(4)
    perf_cols_2[0].metric("샤프", f"{equity_metrics['sharpe']:.2f}" if np.isfinite(equity_metrics["sharpe"]) else "N/A")
    perf_cols_2[1].metric("소르티노", f"{equity_metrics['sortino']:.2f}" if np.isfinite(equity_metrics["sortino"]) else "N/A")
    perf_cols_2[2].metric("Calmar", f"{equity_metrics['calmar']:.2f}" if np.isfinite(equity_metrics["calmar"]) else "N/A")
    perf_cols_2[3].metric(
        "노출비중",
        format_pct_value(equity_metrics["exposure_pct"]) if np.isfinite(equity_metrics["exposure_pct"]) else "N/A",
    )

    perf_cols_3 = st.columns(4)
    perf_cols_3[0].metric(
        "기간 승률",
        format_pct_value(equity_metrics["win_rate_pct"]) if np.isfinite(equity_metrics["win_rate_pct"]) else "N/A",
    )
    perf_cols_3[1].metric(
        "Profit Factor",
        f"{equity_metrics['profit_factor']:.2f}" if np.isfinite(equity_metrics["profit_factor"]) else "N/A",
    )
    perf_cols_3[2].metric(
        "연속 손실",
        f"{int(equity_metrics['max_consecutive_losses'])}" if np.isfinite(equity_metrics["max_consecutive_losses"]) else "N/A",
    )
    perf_cols_3[3].metric(
        "최근 평가금액",
        format_live_price(equity_metrics["latest_equity"], "KRW"),
        None,
    )

    if not equity_curve.empty:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=equity_curve["timestamp"],
                    y=equity_curve["total_eval"],
                    mode="lines+markers",
                    name="총평가금액",
                )
            ]
        )
        fig.update_layout(
            title="모의계좌 에쿼티 곡선",
            height=280 if is_mobile_ui else 340,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title="KRW",
            xaxis_title="시각",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("주문 로그", expanded=not order_log.empty):
        if order_log.empty:
            st.caption("아직 기록된 주문이 없습니다.")
        else:
            order_view = order_log.copy()
            if "requested_at" in order_view.columns:
                order_view["requested_at"] = pd.to_datetime(order_view["requested_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
            order_view = order_view.rename(
                columns={
                    "requested_at": "시각",
                    "prediction_id": "prediction_id",
                    "order_id": "order_id",
                    "name": "종목명",
                    "symbol": "심볼",
                    "side": "방향",
                    "order_type": "유형",
                    "quantity": "수량",
                    "requested_price": "주문가",
                    "quote_price": "현재가",
                    "predicted_move_pct": "예상수익률(%)",
                    "entry_estimate": "예상 진입가",
                    "stop_level": "손절가",
                    "take_level": "익절가",
                    "order_no": "주문번호",
                    "message": "메시지",
                }
            )
            if "방향" in order_view.columns:
                order_view["방향"] = order_view["방향"].map(PAPER_ORDER_SIDE_LABELS).fillna(order_view["방향"])
            if "유형" in order_view.columns:
                order_view["유형"] = order_view["유형"].map({"market": "시장가", "limit": "지정가"}).fillna(order_view["유형"])
            preferred_cols = [
                "시각",
                "prediction_id",
                "order_id",
                "종목명",
                "심볼",
                "방향",
                "유형",
                "수량",
                "주문가",
                "현재가",
                "예상수익률(%)",
                "예상 진입가",
                "손절가",
                "익절가",
                "주문번호",
                "메시지",
            ]
            order_view = order_view[[col for col in preferred_cols if col in order_view.columns]]
            st.dataframe(order_view, width="stretch", hide_index=True)

    if unresolved:
        with st.expander("국내 Top100 심볼 해석 실패"):
            st.dataframe(pd.DataFrame({"종목명": unresolved}), width="stretch", hide_index=True)


st.set_page_config(page_title="멀티마켓 가격 예측기", layout="wide")
auto_mobile_detected = detect_mobile_client()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def run_cached(
    symbol: str,
    years: int,
    test_days: int,
    forecast_days: int,
    validation_mode: str,
    retrain_every: int,
    round_trip_cost_bps: float,
    min_signal_strength_pct: float,
    final_holdout_days: int,
    purge_days: int,
    embargo_days: int,
    target_mode: str,
    validation_days: int,
    allow_short: bool,
    trade_mode: str,
    target_daily_vol_pct: float,
    max_position_size: float,
    stop_loss_atr_mult: float,
    take_profit_atr_mult: float,
):
    return run_forecast(
        symbol=symbol,
        years=years,
        test_days=test_days,
        forecast_days=forecast_days,
        validation_mode=validation_mode,
        retrain_every=retrain_every,
        round_trip_cost_bps=round_trip_cost_bps,
        min_signal_strength_pct=min_signal_strength_pct,
        final_holdout_days=final_holdout_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
        target_mode=target_mode,
        validation_days=validation_days,
        allow_short=allow_short,
        trade_mode=trade_mode,
        target_daily_vol_pct=target_daily_vol_pct,
        max_position_size=max_position_size,
        stop_loss_atr_mult=stop_loss_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult,
    )



def default_symbol_for_asset(asset_type: str) -> str:
    if asset_type == "코인":
        return "BTC-USD"
    if asset_type == "미국주식":
        return "AAPL"
    return "005930"


pending_analysis_target = st.session_state.pop("pending_analysis_target", None)
if pending_analysis_target:
    st.session_state["sidebar_asset_type"] = str(pending_analysis_target["asset_type"])
    st.session_state["sidebar_raw_symbol"] = str(pending_analysis_target["raw_symbol"])
    st.session_state["sidebar_korea_market"] = str(pending_analysis_target["korea_market"])
    st.session_state["sidebar_asset_type_prev"] = str(pending_analysis_target["asset_type"])


st.session_state.setdefault("ui_mode_choice", "자동")
st.session_state.setdefault("ui_theme_mode", get_streamlit_theme_mode())
st.session_state.setdefault("ui_theme_toggle_event", "")
st.session_state.setdefault("sidebar_asset_type", "한국주식")
st.session_state.setdefault("sidebar_raw_symbol", default_symbol_for_asset(st.session_state["sidebar_asset_type"]))
st.session_state.setdefault("sidebar_korea_market", "KOSPI")
st.session_state.setdefault("sidebar_asset_type_prev", st.session_state["sidebar_asset_type"])

st.session_state.setdefault("analysis_auto_run", False)
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("analysis_result_symbol", "")
st.session_state.setdefault("analysis_result_forecast_days", 0)
st.session_state.setdefault("analysis_prediction_run_id", "")
st.session_state.setdefault("paper_analysis_result", None)
st.session_state.setdefault("paper_analysis_symbol", "")
st.session_state.setdefault("paper_analysis_forecast_days", 0)
st.session_state.setdefault("paper_prediction_run_id", "")
st.session_state.setdefault("paper_account_refresh_token", 0)
st.session_state.setdefault("paper_quote_refresh_token", 0)
st.session_state.setdefault("paper_symbol_label", "")
st.session_state.setdefault("paper_manual_symbol", "")
st.session_state.setdefault("paper_market", "KOSPI")
st.session_state.setdefault("dashboard_quote_refresh_token", 0)
st.session_state.setdefault("dashboard_initialized", not auto_mobile_detected)

with st.sidebar:
    st.subheader("기본 설정")
    ui_mode_choice = st.selectbox("화면 모드", ["자동", "PC", "모바일"], key="ui_mode_choice")
    if ui_mode_choice == "자동":
        is_mobile_ui = auto_mobile_detected
    else:
        is_mobile_ui = ui_mode_choice == "모바일"
    st.caption(f"현재 적용 UI: {'모바일' if is_mobile_ui else 'PC'}")

    asset_type = st.selectbox("자산 유형", ["한국주식", "미국주식", "코인"], key="sidebar_asset_type")
    previous_asset_type = st.session_state.get("sidebar_asset_type_prev")
    if previous_asset_type != asset_type:
        st.session_state["sidebar_raw_symbol"] = default_symbol_for_asset(asset_type)
        if asset_type != "한국주식":
            st.session_state["sidebar_korea_market"] = "KOSPI"
        st.session_state["sidebar_asset_type_prev"] = asset_type

    if asset_type == "한국주식":
        korea_market = st.radio("한국 시장", ["KOSPI", "KOSDAQ"], horizontal=True, key="sidebar_korea_market")
        st.caption("숫자 6자리 입력 시 자동으로 .KS / .KQ를 붙입니다.")
    else:
        korea_market = "KOSPI"
        st.session_state["sidebar_korea_market"] = "KOSPI"
        if asset_type == "코인":
            st.caption("예: BTC-USD, ETH-USD, SOL-USD")
        else:
            st.caption("예: AAPL, NVDA, MSFT")

    raw_symbol = st.text_input("심볼", key="sidebar_raw_symbol")
    years = st.slider("학습 데이터 기간(년)", min_value=2, max_value=10, value=5)
    forecast_days = st.slider("미래 예측(일)", min_value=7, max_value=60, value=14)

    with st.expander("고급 설정", expanded=False):
        test_days = st.slider("연구구간 테스트 일수", min_value=20, max_value=260, value=60)
        final_holdout_days = st.slider("최종 홀드아웃 일수", min_value=20, max_value=220, value=40)

        validation_label = st.selectbox("검증 모드", list(VALIDATION_LABEL_TO_MODE.keys()), index=0)
        validation_mode = VALIDATION_LABEL_TO_MODE[validation_label]
        if validation_mode == "walk_forward":
            retrain_every = st.slider("워크포워드 재학습 주기(일)", min_value=1, max_value=20, value=5)
        else:
            retrain_every = 5
        validation_days = st.slider("Validation(가중치 산출) 일수", min_value=20, max_value=180, value=40)

        target_mode_label = st.selectbox("예측 타깃", ["수익률(return)", "가격(price)"], index=0)
        target_mode = "return" if target_mode_label.startswith("수익률") else "price"

        purge_days = st.slider("Purging 일수", min_value=0, max_value=10, value=2)
        embargo_days = st.slider("Embargo 일수", min_value=0, max_value=10, value=1)

        round_trip_cost_bps = st.slider("왕복 거래비용 가정(bps)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
        min_signal_strength_pct = st.slider(
            "최소 신호 강도(%)",
            min_value=0.0,
            max_value=3.0,
            value=0.2,
            step=0.05,
        )
        allow_short = st.checkbox("숏(공매도) 허용", value=False)
        trade_mode_label = st.selectbox("매매 손익 계산 기준", ["close→close(권장)", "open→close(기존)"], index=0)
        trade_mode = "close_to_close" if trade_mode_label.startswith("close") else "open_to_close"
        target_daily_vol_pct = st.slider("목표 일변동성(%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        max_position_size = st.slider("최대 포지션 크기(배)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
        stop_loss_atr_mult = st.slider("ATR 손절 배수", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        take_profit_atr_mult = st.slider("ATR 익절 배수", min_value=0.0, max_value=8.0, value=3.0, step=0.1)
        st.caption(
            "포지션 크기는 신호강도와 변동성 타깃에 따라 자동 조절됩니다. ATR 손절/익절은 일봉 근사이며, 같은 날 둘 다 닿으면 손절 우선으로 처리합니다."
        )

    st.caption("운영 도구")
    if st.button("worker 재시작", key="sidebar_restart_worker", width="stretch"):
        ok, message = restart_background_worker()
        st.session_state["worker_restart_feedback"] = {"ok": ok, "message": message}
    feedback = st.session_state.get("worker_restart_feedback")
    if isinstance(feedback, dict) and feedback.get("message"):
        if feedback.get("ok"):
            st.success(str(feedback["message"]))
        else:
            st.error(str(feedback["message"]))

analysis_inputs = {
    "years": years,
    "test_days": test_days,
    "forecast_days": forecast_days,
    "validation_mode": validation_mode,
    "retrain_every": retrain_every,
    "round_trip_cost_bps": round_trip_cost_bps,
    "min_signal_strength_pct": min_signal_strength_pct,
    "final_holdout_days": final_holdout_days,
    "purge_days": purge_days,
    "embargo_days": embargo_days,
    "target_mode": target_mode,
    "validation_days": validation_days,
    "allow_short": allow_short,
    "trade_mode": trade_mode,
    "target_daily_vol_pct": target_daily_vol_pct,
    "max_position_size": max_position_size,
    "stop_loss_atr_mult": stop_loss_atr_mult,
    "take_profit_atr_mult": take_profit_atr_mult,
}


def render_remote_access_help() -> None:
    with st.expander("원격 접속(Tailscale) 안내"):
        st.markdown(
            """
            1. 이 PC와 원격 기기에 Tailscale을 설치하고 같은 Tailnet에 로그인합니다.
            2. 서버 PC에서 앱 실행: `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8505 --server.headless true`
            3. 서버 PC Tailscale IP 확인: `tailscale ip -4`
            4. 원격 기기 브라우저에서 접속: `http://<TAILSCALE_IP>:8505`
            """
        )


theme_mode = get_streamlit_theme_mode()
stored_theme_mode = str(st.session_state.get("ui_theme_mode", theme_mode) or theme_mode).lower()
if stored_theme_mode in {"light", "dark"} and stored_theme_mode != theme_mode:
    st_config.set_option("theme.base", stored_theme_mode)
    theme_mode = stored_theme_mode
apply_responsive_css(is_mobile_ui=is_mobile_ui, theme_mode=theme_mode)
runtime_settings = load_settings()
dashboard_data = load_dashboard_data(runtime_settings)
page_objects: Dict[str, Any] = {}


def switch_to_page(page_key: str) -> None:
    page = page_objects.get(page_key)
    if page is not None:
        st.switch_page(page)



def render_monitor_page() -> None:
    render_operations_monitor(
        settings=runtime_settings,
        dashboard_data=dashboard_data,
        theme_mode=get_active_theme_mode(),
    )


def _beta_monitor_styles(theme_mode: str) -> str:
    dark = get_active_theme_mode(theme_mode) == "dark"
    css_vars = {
        "surface": "#1e2130" if dark else "#ffffff",
        "surface2": "#252a3a" if dark else "#f9fafb",
        "bg2": "#181b23" if dark else "#eef0f3",
        "border": "#2e3347" if dark else "#e5e7eb",
        "border2": "#2a2f42" if dark else "#eff0f2",
        "text": "#f1f3f9" if dark else "#111827",
        "text2": "#b8c0d0" if dark else "#4b5563",
        "text3": "#8892a8" if dark else "#9ca3af",
        "up": "#f87171" if dark else "#ef4444",
        "down": "#60a5fa" if dark else "#3b82f6",
        "warn": "#fbbf24" if dark else "#f59e0b",
        "ok": "#34d399" if dark else "#10b981",
        "accent": "#818cf8" if dark else "#6366f1",
    }
    return """
        <style>
        .beta-strip {
          background: %(surface2)s;
          border: 1px solid %(border)s;
          border-radius: 12px;
          font-size: 11px;
          padding: 7px 16px;
          display: flex;
          align-items: center;
          gap: 14px;
          color: %(text2)s;
          margin-bottom: 14px;
        }
        .beta-strip-item { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
        .beta-dot { width: 7px; height: 7px; border-radius: 999px; flex-shrink: 0; }
        .beta-dot.ok { background: %(ok)s; box-shadow: 0 0 0 2px color-mix(in srgb, %(ok)s 25%%, transparent); }
        .beta-dot.warn { background: %(warn)s; }
        .beta-dot.bad { background: %(up)s; box-shadow: 0 0 0 2px color-mix(in srgb, %(up)s 25%%, transparent); }
        .beta-dot.neutral { background: %(text3)s; }
        .beta-strip-time { margin-left: auto; color: %(text3)s; font-family: "SF Mono", Consolas, monospace; }
        .beta-account-row {
          display: grid;
          grid-template-columns: 1.2fr 1fr 1fr;
          gap: 10px;
          margin-bottom: 12px;
        }
        .beta-acct-card {
          position: relative;
          background: %(surface)s;
          border: 1px solid %(border)s;
          border-radius: 12px;
          padding: 15px 17px;
        }
        .beta-acct-card::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          border-radius: 12px 12px 0 0;
          background: var(--beta-accent, %(border)s);
        }
        .beta-acct-card.primary { --beta-accent: %(ok)s; }
        .beta-acct-card.runtime { --beta-accent: %(accent)s; }
        .beta-acct-card.summary { --beta-accent: %(warn)s; }
        .beta-acct-head { display: flex; align-items: center; gap: 8px; margin-bottom: 11px; }
        .beta-broker-badge {
          font-size: 10px;
          font-weight: 800;
          padding: 2px 8px;
          border-radius: 5px;
          letter-spacing: 0.04em;
          background: %(bg2)s;
          color: %(text2)s;
        }
        .beta-acct-scope { font-size: 11px; color: %(text3)s; }
        .beta-kpi-value { font-size: 29px; font-weight: 800; line-height: 1; color: %(text)s; }
        .beta-kpi-sub { font-size: 11px; color: %(text3)s; margin-top: 5px; }
        .beta-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 14px;
        }
        .beta-pill {
          background: %(surface2)s;
          border: 1px solid %(border2)s;
          border-radius: 999px;
          padding: 5px 10px;
          font-size: 11px;
          color: %(text2)s;
        }
        .beta-pill strong { color: %(text)s; margin-left: 4px; }
        .beta-summary-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-top: 12px;
        }
        .beta-summary-item {
          background: %(surface2)s;
          border: 1px solid %(border2)s;
          border-radius: 9px;
          padding: 10px 11px;
        }
        .beta-summary-item span { display: block; font-size: 10px; color: %(text3)s; margin-bottom: 3px; }
        .beta-summary-item strong { display: block; font-size: 16px; color: %(text)s; }
        .beta-stat-bar {
          background: %(surface)s;
          border: 1px solid %(border)s;
          border-radius: 12px;
          padding: 10px 12px;
          margin-bottom: 12px;
          display: grid;
          grid-template-columns: repeat(8, minmax(0, 1fr));
          gap: 4px;
        }
        .beta-stat-item {
          padding: 6px 8px;
          border-right: 1px solid %(border2)s;
        }
        .beta-stat-item:last-child { border-right: none; }
        .beta-stat-item span { display: block; font-size: 10px; color: %(text3)s; margin-bottom: 3px; }
        .beta-stat-item strong { display: block; font-size: 18px; color: %(text)s; }
        .beta-card-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 12px;
        }
        .beta-card-title { font-size: 13px; font-weight: 700; color: %(text)s; }
        .beta-card-meta { font-size: 11px; color: %(text3)s; }
        .beta-signal-list,
        .beta-job-list,
        .beta-error-list {
          display: flex;
          flex-direction: column;
          gap: 7px;
        }
        .beta-signal-item,
        .beta-job-row,
        .beta-error-row {
          background: %(surface2)s;
          border: 1px solid %(border2)s;
          border-radius: 9px;
          padding: 9px 11px;
        }
        .beta-job-row.issue { border-left: 3px solid %(warn)s; }
        .beta-signal-item.ok { border-left: 3px solid %(ok)s; }
        .beta-signal-item.warn { border-left: 3px solid %(warn)s; }
        .beta-signal-item.bad { border-left: 3px solid %(up)s; }
        .beta-line-top {
          display: flex;
          align-items: center;
          gap: 8px;
          justify-content: space-between;
          margin-bottom: 4px;
        }
        .beta-line-title { font-weight: 700; color: %(text)s; }
        .beta-line-meta { color: %(text3)s; font-size: 11px; font-family: "SF Mono", Consolas, monospace; }
        .beta-line-body { color: %(text2)s; font-size: 11px; line-height: 1.45; }
        .beta-chip {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 10px;
          font-weight: 700;
        }
        .beta-chip.ok { background: color-mix(in srgb, %(ok)s 15%%, transparent); color: %(ok)s; }
        .beta-chip.warn { background: color-mix(in srgb, %(warn)s 15%%, transparent); color: %(warn)s; }
        .beta-chip.bad { background: color-mix(in srgb, %(up)s 12%%, transparent); color: %(up)s; }
        .beta-chip.info { background: color-mix(in srgb, %(accent)s 14%%, transparent); color: %(accent)s; }
        .beta-chip.neutral { background: %(bg2)s; color: %(text3)s; }
        .beta-cand-table { width: 100%%; border-collapse: collapse; font-size: 12px; }
        .beta-cand-table th {
          font-size: 10px;
          font-weight: 600;
          color: %(text3)s;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          text-align: left;
          padding: 0 8px 8px 0;
          border-bottom: 1px solid %(border2)s;
        }
        .beta-cand-table td {
          padding: 8px 8px 8px 0;
          border-bottom: 1px solid %(border2)s;
          vertical-align: middle;
          color: %(text2)s;
        }
        .beta-cand-table tr:last-child td { border-bottom: none; }
        .beta-cand-symbol { font-weight: 700; color: %(text)s; }
        .beta-empty {
          padding: 22px 0;
          text-align: center;
          color: %(text3)s;
          font-size: 12px;
        }
        .beta-sync-list { display: flex; flex-direction: column; gap: 8px; }
        .beta-sync-row {
          display: flex;
          align-items: flex-start;
          gap: 9px;
          padding: 8px 0;
          border-bottom: 1px solid %(border2)s;
        }
        .beta-sync-row:last-child { border-bottom: none; }
        .beta-sync-main { flex: 1; }
        .beta-sync-main strong { display: block; color: %(text)s; font-size: 12px; }
        .beta-sync-main span { display: block; color: %(text3)s; font-size: 11px; margin-top: 2px; }
        @media (max-width: 1100px) {
          .beta-account-row,
          .beta-stat-bar {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }
        </style>
    """ % css_vars


def _beta_monitor_tone(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"running", "completed", "filled", "ok", "success", "info"}:
        return "ok"
    if normalized in {"paused", "queued", "pending_fill", "submitted", "acknowledged", "retry"}:
        return "warn"
    if normalized in {"failed", "error", "rejected", "cancelled", "stopped"}:
        return "bad"
    if normalized in {"candidate", "long", "active"}:
        return "info"
    return "neutral"


def _beta_monitor_chip(label: str, tone: str) -> str:
    return f'<span class="beta-chip {html.escape(tone)}">{html.escape(label)}</span>'


def _beta_monitor_status_strip_html(
    auto_trading_status: Dict[str, Any],
    runtime_profile: Dict[str, Any],
    sync_rows: Dict[str, Dict[str, Any]],
    broker_sync_errors: pd.DataFrame,
) -> str:
    worker_tone = _beta_monitor_tone(str(auto_trading_status.get("state", "stopped")))
    issue_count = 0
    for job_name in ("broker_market_status", "broker_account_sync", "broker_order_sync", "broker_position_sync"):
        if str(sync_rows.get(job_name, {}).get("status", "never")).lower() not in {"completed", "running"}:
            issue_count += 1
    error_count = int(len(broker_sync_errors)) if isinstance(broker_sync_errors, pd.DataFrame) else 0
    profile_name = str(runtime_profile.get("name") or "-")
    last_refresh = format_display_timestamp(pd.Timestamp.now(tz="Asia/Seoul"))
    broker_tone = "bad" if error_count else ("warn" if issue_count else "ok")
    broker_text = f"브로커 오류 {error_count}건" if error_count else ("동기화 점검 필요" if issue_count else "브로커 이상 없음")
    return f"""
        <div class="beta-strip">
          <div class="beta-strip-item"><span class="beta-dot {worker_tone}"></span><span>워커 {_localized_auto_trading_label(auto_trading_status)}</span></div>
          <div class="beta-strip-item"><span class="beta-dot ok"></span><span>프로파일 {html.escape(profile_name)}</span></div>
          <div class="beta-strip-item"><span class="beta-dot {broker_tone}"></span><span>{html.escape(broker_text)}</span></div>
          <div class="beta-strip-item"><span class="beta-dot {'warn' if issue_count else 'neutral'}"></span><span>동기화 이슈 {issue_count}건</span></div>
          <span class="beta-strip-time">{html.escape(last_refresh)}</span>
        </div>
    """


def _beta_monitor_account_row_html(
    account_snapshot: Dict[str, Any],
    trade_performance: Dict[str, Any],
    auto_trading_status: Dict[str, Any],
    execution_summary: Dict[str, Any],
    summary: Dict[str, Any],
    runtime_profile: Dict[str, Any],
    kis_runtime: Dict[str, Any],
) -> str:
    equity_text = format_price_value(_safe_float(account_snapshot.get("equity")))
    cash_text = format_price_value(_safe_float(account_snapshot.get("cash")))
    pnl_text = format_price_value(_safe_float(trade_performance.get("today_pnl")))
    return_text = format_pct_value(_safe_float(trade_performance.get("total_return_pct")))
    heartbeat_text = auto_trading_status_text(auto_trading_status)
    pending_orders = int(first_valid_float(kis_runtime.get("pending_submitted_orders"), default=0.0))
    updated_text = format_display_timestamp(account_snapshot.get("created_at")) or "기록 없음"
    profile_text = str(runtime_profile.get("name") or "-")
    last_ws_event = format_display_timestamp(kis_runtime.get("last_websocket_execution_event")) or "기록 없음"
    unresolved = int(summary.get("unresolved_predictions", 0))
    candidates = int(execution_summary.get("today_candidate_count", 0))
    rejected = int(execution_summary.get("today_entry_rejected_count", 0))
    return f"""
        <div class="beta-account-row">
          <div class="beta-acct-card primary">
            <div class="beta-acct-head">
              <span class="beta-broker-badge">ACCOUNT</span>
              <span class="beta-acct-scope">실시간 평가 자산</span>
            </div>
            <div class="beta-kpi-value">{html.escape(equity_text)}</div>
            <div class="beta-kpi-sub">마지막 스냅샷 {html.escape(updated_text)}</div>
            <div class="beta-pill-row">
              <div class="beta-pill">예수금 <strong>{html.escape(cash_text)}</strong></div>
              <div class="beta-pill">당일 손익 <strong>{html.escape(pnl_text)}</strong></div>
              <div class="beta-pill">누적 수익률 <strong>{html.escape(return_text)}</strong></div>
            </div>
          </div>
          <div class="beta-acct-card runtime">
            <div class="beta-acct-head">
              <span class="beta-broker-badge">RUNTIME</span>
              <span class="beta-acct-scope">자동매매 런타임</span>
            </div>
            <div class="beta-kpi-value">{html.escape(_localized_auto_trading_label(auto_trading_status))}</div>
            <div class="beta-kpi-sub">{html.escape(heartbeat_text)}</div>
            <div class="beta-summary-grid">
              <div class="beta-summary-item"><span>프로파일</span><strong>{html.escape(profile_text)}</strong></div>
              <div class="beta-summary-item"><span>미체결 제출 주문</span><strong>{pending_orders}</strong></div>
              <div class="beta-summary-item"><span>최근 실시간 체결</span><strong>{html.escape(last_ws_event)}</strong></div>
              <div class="beta-summary-item"><span>브로커 reject</span><strong>{int(first_valid_float(kis_runtime.get('broker_rejects_today'), default=0.0))}</strong></div>
            </div>
          </div>
          <div class="beta-acct-card summary">
            <div class="beta-acct-head">
              <span class="beta-broker-badge">SUMMARY</span>
              <span class="beta-acct-scope">오늘 처리 요약</span>
            </div>
            <div class="beta-summary-grid" style="margin-top:0;">
              <div class="beta-summary-item"><span>후보 스캔</span><strong>{candidates}</strong></div>
              <div class="beta-summary-item"><span>진입 거절</span><strong>{rejected}</strong></div>
              <div class="beta-summary-item"><span>미정산 예측</span><strong>{unresolved}</strong></div>
              <div class="beta-summary-item"><span>대기 주문</span><strong>{int(summary.get('open_orders', 0))}</strong></div>
            </div>
          </div>
        </div>
    """


def _beta_monitor_stat_bar_html(
    summary: Dict[str, Any],
    execution_summary: Dict[str, Any],
    kis_runtime: Dict[str, Any],
) -> str:
    items = [
        ("보유 포지션", int(summary.get("open_positions", 0))),
        ("대기 주문", int(summary.get("open_orders", 0))),
        ("후보 스캔", int(execution_summary.get("today_candidate_count", 0))),
        ("진입 허용", int(execution_summary.get("today_entry_allowed_count", 0))),
        ("진입 거절", int(execution_summary.get("today_entry_rejected_count", 0))),
        ("주문 제출", int(execution_summary.get("today_submitted_count", 0))),
        ("체결 완료", int(execution_summary.get("today_filled_count", 0))),
        ("브로커 거절", int(first_valid_float(kis_runtime.get("broker_rejects_today"), default=0.0))),
    ]
    cells = "".join(
        f'<div class="beta-stat-item"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in items
    )
    return f'<div class="beta-stat-bar">{cells}</div>'


def _beta_monitor_signal_items_html(
    today_execution_events: pd.DataFrame,
    execution_summary: Dict[str, Any],
    candidate_scans: pd.DataFrame,
) -> str:
    items: List[str] = []
    if not today_execution_events.empty:
        preview = today_execution_events.copy()
        if "created_at" in preview.columns:
            preview = preview.sort_values("created_at", ascending=False)
        for _, row in preview.head(4).iterrows():
            level = str(row.get("level", "") or "").lower()
            tone = "bad" if level == "error" else "warn" if level == "warning" else "ok"
            items.append(
                f"""
                <div class="beta-signal-item {tone}">
                  <div class="beta-line-top">
                    <span class="beta-line-title">{html.escape(str(row.get('event_type', 'execution')))}</span>
                    <span class="beta-line-meta">{html.escape(format_display_timestamp(row.get('created_at')))}</span>
                  </div>
                  <div class="beta-line-body">{html.escape(str(row.get('message', '') or '')[:180])}</div>
                </div>
                """
            )
    elif not candidate_scans.empty:
        preview = candidate_scans.copy()
        if "created_at" in preview.columns:
            preview = preview.sort_values("created_at", ascending=False)
        for _, row in preview.head(4).iterrows():
            signal = str(row.get("signal", "") or "").upper()
            tone = "ok" if signal == "LONG" else "warn" if signal in {"HOLD", "WATCH"} else "bad" if signal == "SHORT" else "neutral"
            items.append(
                f"""
                <div class="beta-signal-item {tone}">
                  <div class="beta-line-top">
                    <span class="beta-line-title">{html.escape(str(row.get('symbol', '-')))}</span>
                    {_beta_monitor_chip(signal or 'FLAT', tone)}
                  </div>
                  <div class="beta-line-body">예상 수익 {html.escape(format_pct_value(_safe_float(row.get('expected_return')) * 100.0))} · 신뢰도 {html.escape(format_pct_value(_safe_float(row.get('confidence')) * 100.0))}</div>
                </div>
                """
            )
    else:
        noops = int(execution_summary.get("today_noop_count", 0))
        items.append(
            f"""
            <div class="beta-signal-item warn">
              <div class="beta-line-top">
                <span class="beta-line-title">오늘 실행 이벤트가 없습니다</span>
                {_beta_monitor_chip('NOOP', 'warn')}
              </div>
              <div class="beta-line-body">미실행 누적 {noops}건</div>
            </div>
            """
        )
    return f'<div class="beta-signal-list">{"".join(items)}</div>'


def _beta_monitor_candidate_table_html(candidate_scans: pd.DataFrame) -> str:
    if candidate_scans.empty:
        return '<div class="beta-empty">오늘 후보 스캔 기록이 없습니다.</div>'
    preview = candidate_scans.copy()
    sort_cols = [column for column in ["created_at", "rank"] if column in preview.columns]
    if sort_cols:
        preview = preview.sort_values(sort_cols, ascending=[False, True][: len(sort_cols)])
    rows: List[str] = []
    for _, row in preview.head(8).iterrows():
        signal = str(row.get("signal", "") or "").upper()
        tone = "ok" if signal == "LONG" else "warn" if signal in {"HOLD", "WATCH"} else "bad" if signal == "SHORT" else "neutral"
        expected_return = _safe_float(row.get("expected_return")) * 100.0
        confidence = _safe_float(row.get("confidence")) * 100.0
        score = _safe_float(row.get("score"))
        status_label = signal or str(row.get("status", "flat") or "flat").upper()
        rows.append(
            f"""
            <tr>
              <td><span class="beta-cand-symbol">{html.escape(str(row.get('symbol', '-')))}</span></td>
              <td>{_beta_monitor_chip(status_label, tone)}</td>
              <td>{html.escape(format_pct_value(expected_return))}</td>
              <td>{html.escape(format_pct_value(confidence))}</td>
              <td>{html.escape(f"{score:.2f}" if np.isfinite(score) else 'N/A')}</td>
            </tr>
            """
        )
    return f"""
        <table class="beta-cand-table">
          <thead>
            <tr>
              <th>종목</th>
              <th>시그널</th>
              <th>예상 수익</th>
              <th>신뢰도</th>
              <th>점수</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
    """


def _beta_monitor_sync_list_html(sync_rows: Dict[str, Dict[str, Any]]) -> str:
    rows: List[str] = []
    sync_order = [
        "broker_market_status",
        "broker_account_sync",
        "broker_order_sync",
        "broker_position_sync",
    ]
    for job_name in sync_order:
        row = sync_rows.get(job_name, {})
        status = str(row.get("status", "never") or "never").lower()
        tone = _beta_monitor_tone(status)
        rows.append(
            f"""
            <div class="beta-sync-row">
              <span class="beta-dot {tone}"></span>
              <div class="beta-sync-main">
                <strong>{html.escape(MONITOR_SYNC_LABELS.get(job_name, job_name))}</strong>
                <span>{html.escape(_sync_status_summary_text(row) if row else '기록이 없습니다')}</span>
              </div>
              {_beta_monitor_chip(_sync_status_label(status), tone)}
            </div>
            """
        )
    return f'<div class="beta-sync-list">{"".join(rows)}</div>'


def _beta_monitor_job_list_html(job_health: pd.DataFrame) -> str:
    if job_health.empty:
        return '<div class="beta-empty">최근 작업 이력이 없습니다.</div>'
    preview = job_health.copy()
    if "finished_at" in preview.columns:
        preview = preview.sort_values("finished_at", ascending=False, na_position="last")
    rows: List[str] = []
    for _, row in preview.head(5).iterrows():
        status = str(row.get("status", "never") or "never").lower()
        tone = _beta_monitor_tone(status)
        run_key = str(row.get("run_key", "") or "")[-8:]
        timestamp = format_display_timestamp(row.get("finished_at") or row.get("started_at") or row.get("scheduled_at"))
        error_text = str(row.get("error_message", "") or "").strip()
        issue_class = " issue" if error_text else ""
        rows.append(
            f"""
            <div class="beta-job-row{issue_class}">
              <div class="beta-line-top">
                <span class="beta-line-title">{html.escape(_monitor_job_label(row.get('job_name')))}</span>
                {_beta_monitor_chip(_monitor_status_label(status), tone)}
              </div>
              <div class="beta-line-body">run {html.escape(run_key or '-')} · {html.escape(timestamp or '기록 없음')} · 재시도 {int(first_valid_float(row.get('retry_count'), default=0.0))}회</div>
              {f'<div class="beta-line-body">{html.escape(error_text[:180])}</div>' if error_text else ''}
            </div>
            """
        )
    return f'<div class="beta-job-list">{"".join(rows)}</div>'


def _beta_monitor_error_list_html(broker_sync_errors: pd.DataFrame, recent_errors: pd.DataFrame) -> str:
    preview = broker_sync_errors if not broker_sync_errors.empty else recent_errors
    if preview.empty:
        return '<div class="beta-empty">최근 오류가 없습니다.</div>'
    if "created_at" in preview.columns:
        preview = preview.sort_values("created_at", ascending=False)
    rows: List[str] = []
    for _, row in preview.head(5).iterrows():
        timestamp = format_display_timestamp(row.get("created_at"))
        rows.append(
            f"""
            <div class="beta-error-row">
              <div class="beta-line-top">
                <span class="beta-line-title">{html.escape(str(row.get('component', 'runtime') or 'runtime'))}</span>
                {_beta_monitor_chip(str(row.get('level', 'error') or 'error').upper(), 'bad')}
              </div>
              <div class="beta-line-body">{html.escape(str(row.get('event_type', '') or 'event'))} · {html.escape(timestamp or '기록 없음')}</div>
              <div class="beta-line-body">{html.escape(str(row.get('message', '') or '')[:220])}</div>
            </div>
            """
        )
    return f'<div class="beta-error-list">{"".join(rows)}</div>'


def render_beta_monitor_page() -> None:
    if str(BETA_MONITOR_UI_VARIANT or "clone_v2").lower() != "legacy":
        _render_beta_monitor_clone_page()
        return

    settings = runtime_settings
    theme_mode = get_active_theme_mode()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    account_snapshot = dict(summary.get("latest_account") or {})
    trade_performance = data["trade_performance"]
    auto_trading_status = data.get("auto_trading_status", {})
    execution_summary = data.get("execution_summary", {})
    broker_sync_status = data.get("broker_sync_status", pd.DataFrame())
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    runtime_profile = data.get("runtime_profile", {})
    asset_overview = data.get("asset_overview", pd.DataFrame())
    job_health = data["job_health"]
    recent_errors = data["recent_errors"]
    open_positions = data["open_positions"]
    open_orders = data["open_orders"]
    candidate_scans = data["candidate_scans"]
    prediction_report = data["prediction_report"]
    equity_curve = data["equity_curve"]
    today_execution_events = data.get("today_execution_events", pd.DataFrame())
    noop_breakdown = execution_summary.get("today_noop_breakdown", pd.DataFrame())

    sync_rows: Dict[str, Dict[str, Any]] = {}
    if not broker_sync_status.empty and "job_name" in broker_sync_status.columns:
        for _, row in broker_sync_status.iterrows():
            sync_rows[str(row.get("job_name") or "")] = row.to_dict()

    st.subheader("운영 모니터 (베타)")
    st.caption("기존 운영 모니터는 그대로 두고, 베타 디자인으로 기능을 옮긴 화면입니다.")
    st.markdown(_beta_monitor_styles(theme_mode), unsafe_allow_html=True)

    worker_feedback = st.session_state.pop("beta_worker_restart_feedback", None)
    if isinstance(worker_feedback, dict) and worker_feedback.get("message"):
        if worker_feedback.get("ok"):
            st.success(str(worker_feedback["message"]))
        else:
            st.error(str(worker_feedback["message"]))

    sync_feedback = st.session_state.pop("beta_broker_sync_feedback", None)
    if isinstance(sync_feedback, dict) and sync_feedback.get("message"):
        if sync_feedback.get("ok"):
            st.success(str(sync_feedback["message"]))
        else:
            st.error(str(sync_feedback["message"]))

    st.markdown(
        _beta_monitor_status_strip_html(
            auto_trading_status=auto_trading_status,
            runtime_profile=runtime_profile,
            sync_rows=sync_rows,
            broker_sync_errors=broker_sync_errors,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _beta_monitor_account_row_html(
            account_snapshot=account_snapshot,
            trade_performance=trade_performance,
            auto_trading_status=auto_trading_status,
            execution_summary=execution_summary,
            summary=summary,
            runtime_profile=runtime_profile,
            kis_runtime=kis_runtime,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _beta_monitor_stat_bar_html(
            summary=summary,
            execution_summary=execution_summary,
            kis_runtime=kis_runtime,
        ),
        unsafe_allow_html=True,
    )

    content_cols = st.columns([1.45, 0.95], gap="large")
    with content_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">실행 이벤트</div>'
                '<div class="beta-card-meta">최근 처리 흐름</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _beta_monitor_signal_items_html(
                    today_execution_events=today_execution_events,
                    execution_summary=execution_summary,
                    candidate_scans=candidate_scans,
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">실시간 포지션 / 주문 활동</div>'
                '<div class="beta-card-meta">15초 간격 갱신</div></div>',
                unsafe_allow_html=True,
            )
            view_mode = st.radio(
                "베타 포지션 보기",
                ["보유 포지션", "최근 주문 활동"],
                horizontal=True,
                key="beta_live_positions_mode",
                label_visibility="collapsed",
            )

            @st.fragment(run_every=15)
            def _render_beta_live_fragment() -> None:
                if view_mode == "보유 포지션":
                    current_positions = load_monitor_open_positions(settings)
                    if current_positions.empty:
                        st.caption("현재 보유 중인 포지션이 없습니다.")
                        return
                    refresh_token = int(pd.Timestamp.now(tz="UTC").timestamp() // 15)
                    live_view = build_live_open_positions_view(current_positions, refresh_token=refresh_token)
                    metric_cols = st.columns(3, gap="small")
                    metric_cols[0].metric("보유 종목 수", f"{len(current_positions)}", border=True)
                    metric_cols[1].metric(
                        "LONG 포지션",
                        f"{int((current_positions['side'].astype(str) == 'LONG').sum())}",
                        border=True,
                    )
                    metric_cols[2].metric(
                        "SHORT 포지션",
                        f"{int((current_positions['side'].astype(str) == 'SHORT').sum())}",
                        border=True,
                    )
                    st.dataframe(live_view, width="stretch", hide_index=True, height=260)
                    return

                recent_orders = load_monitor_recent_orders(settings, limit=30)
                if recent_orders.empty:
                    st.caption("최근 주문 기록이 없습니다.")
                    return
                order_view = build_recent_order_activity_view(recent_orders)
                metric_cols = st.columns(3, gap="small")
                metric_cols[0].metric("최근 주문 수", f"{len(recent_orders)}", border=True)
                metric_cols[1].metric(
                    "매수 주문",
                    f"{int((recent_orders['side'].astype(str).str.lower() == 'buy').sum())}",
                    border=True,
                )
                metric_cols[2].metric(
                    "매도 주문",
                    f"{int((recent_orders['side'].astype(str).str.lower() == 'sell').sum())}",
                    border=True,
                )
                st.dataframe(order_view, width="stretch", hide_index=True, height=260)

            _render_beta_live_fragment()

        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">오늘 후보 목록</div>'
                '<div class="beta-card-meta">최근 스캔 기준</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(_beta_monitor_candidate_table_html(candidate_scans), unsafe_allow_html=True)

    with content_cols[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">트레이딩 제어</div>'
                '<div class="beta-card-meta">현재 상태와 제어</div></div>',
                unsafe_allow_html=True,
            )
            tone = _beta_monitor_tone(str(auto_trading_status.get("state", "stopped")))
            st.markdown(
                _beta_monitor_chip(_localized_auto_trading_label(auto_trading_status), tone),
                unsafe_allow_html=True,
            )
            st.caption(auto_trading_status_text(auto_trading_status))
            restart_col, pause_col, resume_col = st.columns(3, gap="small")
            if restart_col.button("worker 재시작", key="beta_restart_worker", use_container_width=True):
                ok, message = restart_background_worker()
                st.session_state["beta_worker_restart_feedback"] = {"ok": ok, "message": message}
                st.rerun()
            if pause_col.button("진입 일시중지", key="beta_pause", use_container_width=True):
                repository.set_control_flag("trading_paused", "1", "set from beta monitor")
                st.rerun()
            if resume_col.button("진입 재개", key="beta_resume", use_container_width=True):
                repository.set_control_flag("trading_paused", "0", "set from beta monitor")
                st.rerun()

        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">런타임 상태</div>'
                '<div class="beta-card-meta">동기화와 브로커 연결</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(_beta_monitor_sync_list_html(sync_rows), unsafe_allow_html=True)
            st.caption(f"설정 소스: {runtime_profile.get('source', 'embedded_defaults')}")
            st.caption(
                "최근 실시간 체결: "
                + (format_display_timestamp(kis_runtime.get("last_websocket_execution_event")) or "기록 없음")
            )
            sync_cols_top = st.columns(2, gap="small")
            sync_cols_bottom = st.columns(2, gap="small")
            sync_jobs = [
                ("장 상태 확인", "broker_market_status", "beta_sync_market"),
                ("계좌 동기화", "broker_account_sync", "beta_sync_account"),
                ("주문 동기화", "broker_order_sync", "beta_sync_order"),
                ("포지션 동기화", "broker_position_sync", "beta_sync_position"),
            ]
            for column, (label, job_name, key) in zip(sync_cols_top + sync_cols_bottom, sync_jobs):
                if column.button(label, key=key, use_container_width=True):
                    ok, message = run_manual_runtime_job(job_name)
                    st.session_state["beta_broker_sync_feedback"] = {"ok": ok, "message": message}
                    st.rerun()

        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">자산 추이</div>'
                '<div class="beta-card-meta">최근 계좌 스냅샷</div></div>',
                unsafe_allow_html=True,
            )
            if equity_curve.empty:
                st.caption("계좌 스냅샷 기록이 없습니다.")
            else:
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=pd.to_datetime(equity_curve["created_at"], errors="coerce"),
                            y=pd.to_numeric(equity_curve["equity"], errors="coerce"),
                            mode="lines+markers",
                            name="Equity",
                        )
                    ]
                )
                fig.update_layout(height=220, margin=dict(l=12, r=12, t=12, b=12), hovermode="x unified")
                st.plotly_chart(fig, width="stretch")

    bottom_cols = st.columns(2, gap="large")
    with bottom_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">최근 작업 상태</div>'
                '<div class="beta-card-meta">최근 5개 작업</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(_beta_monitor_job_list_html(job_health), unsafe_allow_html=True)

    with bottom_cols[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="beta-card-head"><div class="beta-card-title">최근 브로커 오류</div>'
                '<div class="beta-card-meta">브로커/런타임 예외</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _beta_monitor_error_list_html(broker_sync_errors=broker_sync_errors, recent_errors=recent_errors),
                unsafe_allow_html=True,
            )

    st.markdown("---")
    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_execution, tab_broker, tab_assets, tab_errors = st.tabs(
        ["작업 이력", "보유 현황", "예측 기록", "후보 종목", "실행 이벤트", "브로커 동기화", "자산 설정", "최근 오류"]
    )
    with tab_jobs:
        if job_health.empty:
            st.caption("아직 작업 이력이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(job_health), width="stretch", hide_index=True)

    with tab_positions:
        if open_positions.empty and open_orders.empty:
            st.caption("보유 포지션과 대기 주문이 없습니다.")
        else:
            if not open_positions.empty:
                st.caption("보유 포지션")
                st.dataframe(build_monitor_table_view(open_positions), width="stretch", hide_index=True)
            if not open_orders.empty:
                st.caption("대기 주문")
                st.dataframe(build_monitor_table_view(open_orders), width="stretch", hide_index=True)

    with tab_predictions:
        if prediction_report.empty:
            st.caption("예측 기록이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(prediction_report, limit=200), width="stretch", hide_index=True)

    with tab_candidates:
        if candidate_scans.empty:
            st.caption("후보 종목 기록이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(candidate_scans, limit=200), width="stretch", hide_index=True)

    with tab_execution:
        if not noop_breakdown.empty:
            noop_view = noop_breakdown.rename(columns={"reason": "사유", "count": "건수"})
            st.caption("미실행 사유 요약")
            st.dataframe(noop_view, width="stretch", hide_index=True)
        if today_execution_events.empty:
            st.caption("오늘 실행 이벤트가 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(today_execution_events, limit=200), width="stretch", hide_index=True)

    with tab_broker:
        if broker_sync_status.empty:
            st.caption("브로커 동기화 이력이 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(broker_sync_status), width="stretch", hide_index=True)
        if broker_sync_errors.empty:
            st.caption("최근 브로커 동기화 오류가 없습니다.")
        else:
            st.caption("최근 브로커 동기화/실행 이벤트")
            st.dataframe(build_monitor_table_view(broker_sync_errors, limit=100), width="stretch", hide_index=True)

    with tab_assets:
        if asset_overview.empty:
            st.caption("설정된 자산 구성이 없습니다.")
        else:
            st.dataframe(asset_overview, width="stretch", hide_index=True)

    with tab_errors:
        if recent_errors.empty:
            st.caption("최근 오류 이벤트가 없습니다.")
        else:
            st.dataframe(build_monitor_table_view(recent_errors), width="stretch", hide_index=True)


def _beta_query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, list):
        return str(value[0] if value else default)
    return str(value or default)


def _handle_beta_monitor_clone_action(settings) -> None:
    action = _beta_query_value("beta_action", "")
    token = _beta_query_value("beta_token", "")
    if not action or not token:
        return
    if str(st.session_state.get("beta_last_action_token", "") or "") == token:
        return

    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    ok = True
    message = ""

    try:
        if action == "toggle_theme":
            current_mode = get_active_theme_mode()
            next_theme_mode = "dark" if current_mode == "light" else "light"
            st.session_state["beta_last_action_token"] = token
            st.session_state["beta_action_feedback"] = {
                "ok": True,
                "message": f"{'다크' if next_theme_mode == 'dark' else '라이트'} 모드로 전환했습니다.",
            }
            st.session_state["ui_theme_mode"] = next_theme_mode
            st.session_state["ui_theme_toggle_event"] = token
            st_config.set_option("theme.base", next_theme_mode)
            st.rerun()
        elif action == "restart_worker":
            ok, message = restart_background_worker()
        elif action == "pause_entries":
            repository.set_control_flag("trading_paused", "1", "set from beta clone monitor")
            message = "자동 진입을 일시중지했습니다."
        elif action == "resume_entries":
            repository.set_control_flag("trading_paused", "0", "set from beta clone monitor")
            message = "자동 진입을 재개했습니다."
        elif action == "halt_all":
            repository.set_control_flag("trading_paused", "1", "set from beta clone monitor")
            ok, stop_message = stop_background_worker()
            message = f"전체 정지 요청 완료. {stop_message}" if ok else stop_message
        elif action == "sync_market":
            ok, message = run_manual_runtime_job("broker_market_status")
        elif action == "sync_account":
            ok, message = run_manual_runtime_job("broker_account_sync")
        elif action == "sync_order":
            ok, message = run_manual_runtime_job("broker_order_sync")
        elif action == "sync_position":
            ok, message = run_manual_runtime_job("broker_position_sync")
        elif action == "scan_now":
            ok, message = run_manual_scan_job()
        elif action == "order_check":
            ok, message = run_manual_runtime_job("broker_order_sync")
        else:
            ok = False
            message = f"알 수 없는 베타 액션입니다: {action}"
    except Exception as exc:
        ok = False
        message = str(exc)

    st.session_state["beta_last_action_token"] = token
    st.session_state["beta_action_feedback"] = {"ok": ok, "message": message}


def _load_beta_monitor_template() -> str:
    template_path = Path.home() / "Desktop" / "monitor_redesign_v2.html"
    return template_path.read_text(encoding="utf-8", errors="ignore")


def _replace_outer_block(source: str, start_marker: str) -> Tuple[int, int]:
    start = source.find(start_marker)
    if start < 0:
        raise ValueError(f"marker not found: {start_marker}")
    tag_start = source.rfind("<", 0, start + 1)
    if tag_start < 0:
        raise ValueError(f"tag start not found for {start_marker}")
    if source.startswith("<nav", tag_start):
        depth = 0
        index = tag_start
        while index < len(source):
            next_open = source.find("<nav", index)
            next_close = source.find("</nav>", index)
            if next_close < 0:
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                index = next_open + 4
                continue
            depth -= 1 if depth else 0
            end = next_close + len("</nav>")
            if depth == 0:
                return tag_start, end
            index = end
        raise ValueError(f"nav close not found for {start_marker}")
    depth = 0
    index = tag_start
    while index < len(source):
        next_open = source.find("<div", index)
        next_close = source.find("</div>", index)
        if next_close < 0:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            index = next_open + 4
            continue
        end = next_close + len("</div>")
        depth -= 1
        if depth == 0:
            return tag_start, end
        index = end
    raise ValueError(f"div close not found for {start_marker}")


def _replace_block(source: str, marker: str, replacement: str) -> str:
    start, end = _replace_outer_block(source, marker)
    return source[:start] + replacement + source[end:]



def _render_beta_monitor_clone_page() -> None:
    settings = runtime_settings
    _handle_beta_monitor_clone_action(settings)
    theme_mode = get_active_theme_mode()
    current_candidate_tab = _beta_query_value("beta_cand_tab", "")
    jobs_expanded = _beta_query_value("beta_jobs", "") == "all"
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    st.markdown(
        """
        <style>
        .block-container {
          max-width: none !important;
          padding-top: 0 !important;
          padding-right: 0 !important;
          padding-left: 0 !important;
          padding-bottom: 0.75rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = load_dashboard_data(settings)
    recent_orders = load_monitor_recent_orders(settings, limit=40)
    open_positions = data.get("open_positions", pd.DataFrame())
    accounts_overview = data.get("accounts_overview", {})
    total_portfolio_overview = data.get("total_portfolio_overview", {})
    kr_asset_types = {
        asset_type
        for asset_type, schedule in settings.asset_schedules.items()
        if str(getattr(schedule, "timezone", "")) == "Asia/Seoul"
    }
    quote_snapshots: Dict[str, Dict[str, Any]] = {}
    quote_symbols: List[str] = []
    if not open_positions.empty and "symbol" in open_positions.columns:
        quote_symbols.extend(open_positions["symbol"].dropna().astype(str).tolist())
    if any(
        str((accounts_overview.get(account_id) or {}).get("currency") or "").upper() == "USD"
        for account_id in (ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO)
    ):
        quote_symbols.append("KRW=X")
    if quote_symbols:
        symbols = tuple(dict.fromkeys(quote_symbols))
        if symbols:
            refresh_token = int(pd.Timestamp.now(tz="UTC").timestamp() // 15)
            quote_snapshots = fetch_quote_snapshots(symbols=symbols, refresh_token=refresh_token)

    render_beta_overview_component(
        data=data,
        theme_mode=theme_mode,
        initial_anchor=_beta_query_value("beta_anchor", "beta-overview"),
        feedback=st.session_state.get("beta_action_feedback") if isinstance(st.session_state.get("beta_action_feedback"), dict) else None,
        accounts_overview=accounts_overview,
        total_portfolio_overview=total_portfolio_overview,
        quote_snapshots=quote_snapshots,
        kr_asset_types=kr_asset_types,
        recent_orders=recent_orders,
        krx_name_map=load_krx_name_map(),
        current_candidate_tab=current_candidate_tab,
        jobs_expanded=jobs_expanded,
    )

def render_dashboard_page() -> None:
    st.subheader(f"{asset_type} 대시보드")
    st.caption("리스트에서 종목을 고르면 종목 분석 화면으로 이동해 바로 예측을 실행합니다.")

    dashboard_initialized = bool(st.session_state.get("dashboard_initialized", not is_mobile_ui))

    if is_mobile_ui:
        search_keyword = st.text_input("종목 검색", value="", placeholder="종목명 또는 심볼", key="dashboard_search_mobile")
        sort_mode = st.selectbox("정렬", ["기본 순위", "전일대비 상승순", "전일대비 하락순"], key="dashboard_sort_mobile")
        display_limit = st.slider("표시 개수", min_value=10, max_value=100, value=20, step=10, key="dashboard_limit_mobile")
        load_mobile_dashboard = st.button("대시보드 불러오기", key="dashboard_load_mobile", type="primary")
        if st.button("시세 새로고침", key="dashboard_refresh_mobile"):
            st.session_state["dashboard_initialized"] = True
            st.session_state["dashboard_quote_refresh_token"] += 1
        if load_mobile_dashboard:
            st.session_state["dashboard_initialized"] = True
            st.rerun()
        dashboard_initialized = bool(st.session_state.get("dashboard_initialized", False))
    else:
        dash_ctrl_1, dash_ctrl_2, dash_ctrl_3, dash_ctrl_4 = st.columns([1.8, 1.2, 1.0, 0.9])
        search_keyword = dash_ctrl_1.text_input(
            "종목 검색",
            value="",
            placeholder="예: 삼성전자, NVDA, SOXL, BTC-USD",
            key="dashboard_search_desktop",
        )
        sort_mode = dash_ctrl_2.selectbox(
            "정렬",
            ["기본 순위", "전일대비 상승순", "전일대비 하락순"],
            key="dashboard_sort_desktop",
        )
        display_limit = dash_ctrl_3.slider("표시 개수", min_value=10, max_value=100, value=60, step=10, key="dashboard_limit_desktop")
        if dash_ctrl_4.button("시세 새로고침", key="dashboard_refresh_desktop"):
            st.session_state["dashboard_quote_refresh_token"] += 1
        dashboard_initialized = True

    if not dashboard_initialized:
        st.info("모바일에서는 첫 화면 로딩을 줄이기 위해 대시보드 시세를 바로 불러오지 않습니다.")
        st.caption("필요할 때 `대시보드 불러오기`를 누르면 목록과 현재가를 가져옵니다.")
        render_remote_access_help()
        return

    with st.spinner("대시보드 시세를 불러오는 중..."):
        dashboard_rows, unresolved_dashboard_names = build_dashboard_market_rows(
            asset_type=asset_type,
            display_limit=display_limit,
            refresh_token=int(st.session_state["dashboard_quote_refresh_token"]),
        )

    dashboard_filtered = dashboard_rows
    keyword = search_keyword.strip().lower()
    if keyword:
        dashboard_filtered = [
            row
            for row in dashboard_filtered
            if keyword in str(row.get("종목명", "")).lower()
            or keyword in str(row.get("심볼", "")).lower()
            or keyword in str(row.get("시장", "")).lower()
        ]

    if sort_mode == "전일대비 상승순":
        dashboard_filtered = sorted(
            dashboard_filtered,
            key=lambda row: (
                not np.isfinite(first_valid_float(row.get("전일대비(%)"))),
                -first_valid_float(row.get("전일대비(%)")),
            ),
        )
    elif sort_mode == "전일대비 하락순":
        dashboard_filtered = sorted(
            dashboard_filtered,
            key=lambda row: (
                not np.isfinite(first_valid_float(row.get("전일대비(%)"))),
                first_valid_float(row.get("전일대비(%)")),
            ),
        )
    else:
        dashboard_filtered = sorted(dashboard_filtered, key=lambda row: int(row["순위"]))

    st.caption("현재가와 전일대비는 자산별 실시간 또는 지연 시세 기준입니다.")

    if dashboard_filtered:
        for idx, row in enumerate(dashboard_filtered):
            symbol = str(row.get("심볼") or "")
            market = str(row.get("시장") or "")
            row_asset_type = str(row.get("자산유형") or asset_type)
            current_price = float(row.get("현재가", float("nan")))
            currency = str(row.get("통화", default_currency_from_symbol(symbol) if symbol else "USD"))
            change_pct = float(row.get("전일대비(%)", float("nan")))
            price_text = format_live_price(current_price, currency)
            if np.isfinite(change_pct):
                change_color = "#ef4444" if change_pct > 0 else "#3b82f6" if change_pct < 0 else "#9ca3af"
                change_html = f"<span style='color:{change_color};font-weight:700;'>{change_pct:+.2f}%</span>"
            else:
                change_html = "<span style='color:#9ca3af;'>N/A</span>"

            row_market = "KOSDAQ" if symbol.endswith(".KQ") else "KOSPI"
            button_key = f"dashboard_analyze_{idx}_{symbol or row.get('종목명', '')}"

            with st.container(border=True):
                if is_mobile_ui:
                    st.markdown(f"**#{int(row['순위'])} {row['종목명']}**")
                    st.caption(f"{market} · `{symbol or '-'}`")
                    st.markdown(f"{price_text} · {change_html}", unsafe_allow_html=True)
                    if st.button("분석", key=button_key, disabled=not bool(symbol), type="primary"):
                        set_analysis_target(row_asset_type, symbol, row_market)
                        switch_to_page("analysis")
                else:
                    info_col, price_col, action_col = st.columns([4.0, 2.0, 1.0])
                    with info_col:
                        st.markdown(f"**#{int(row['순위'])} {row['종목명']}**")
                        st.caption(f"{market} · `{symbol or '-'}`")
                    with price_col:
                        st.markdown(f"**{price_text}**")
                        st.markdown(change_html, unsafe_allow_html=True)
                    with action_col:
                        st.write("")
                        if st.button("분석", key=button_key, disabled=not bool(symbol), type="primary"):
                            set_analysis_target(row_asset_type, symbol, row_market)
                            switch_to_page("analysis")
    else:
        st.warning("검색 조건에 맞는 종목이 없습니다.")

    with st.expander("테이블로 보기", expanded=not is_mobile_ui):
        st.dataframe(build_snapshot_view_df(dashboard_filtered), width="stretch", hide_index=True)

    if unresolved_dashboard_names:
        with st.expander("심볼 자동 해석 실패 종목"):
            st.dataframe(pd.DataFrame({"종목명": unresolved_dashboard_names}), width="stretch", hide_index=True)

    render_remote_access_help()



_render_dashboard_page_classic = render_dashboard_page


def _dashboard_redesign_styles(theme_mode: str) -> str:
    dark = get_active_theme_mode(theme_mode) == "dark"
    css_vars = {
        "bg": "#081121" if dark else "#f3f6fb",
        "surface": "rgba(15, 23, 42, 0.88)" if dark else "rgba(255, 255, 255, 0.92)",
        "surface_alt": "rgba(15, 23, 42, 0.62)" if dark else "rgba(241, 245, 249, 0.85)",
        "border": "rgba(148, 163, 184, 0.22)" if dark else "rgba(15, 23, 42, 0.10)",
        "text": "#e5eefb" if dark else "#0f172a",
        "muted": "#8fa2bf" if dark else "#64748b",
        "muted_strong": "#b7c6dc" if dark else "#475569",
        "accent": "#38bdf8" if dark else "#2563eb",
        "accent_alt": "#22c55e" if dark else "#0f766e",
        "danger": "#fb7185" if dark else "#dc2626",
        "warn": "#fbbf24" if dark else "#d97706",
        "shadow": "0 24px 60px rgba(2, 8, 23, 0.28)" if dark else "0 24px 48px rgba(15, 23, 42, 0.08)",
    }
    return """
        <style>
        .alt-dash-strip {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex-wrap: wrap;
          margin: 0 0 1rem 0;
          padding: 0.85rem 1rem;
          border-radius: 18px;
          border: 1px solid %(border)s;
          background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 32%%),
            linear-gradient(135deg, %(surface)s 0%%, %(surface_alt)s 100%%);
          box-shadow: %(shadow)s;
          color: %(text)s;
        }
        .alt-dash-strip-item {
          display: inline-flex;
          align-items: center;
          gap: 0.42rem;
          font-size: 0.82rem;
          font-weight: 600;
          color: %(muted_strong)s;
        }
        .alt-dash-dot {
          width: 0.56rem;
          height: 0.56rem;
          border-radius: 999px;
          box-shadow: 0 0 0 0.2rem rgba(56, 189, 248, 0.12);
        }
        .alt-dash-dot.ok { background: %(accent_alt)s; }
        .alt-dash-dot.warn { background: %(warn)s; }
        .alt-dash-dot.bad { background: %(danger)s; }
        .alt-dash-dot.neutral { background: %(muted)s; box-shadow: none; }
        .alt-dash-strip-time {
          margin-left: auto;
          color: %(muted)s;
          font-size: 0.78rem;
          font-family: "SFMono-Regular", Consolas, monospace;
        }
        .alt-dash-kpis {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.9rem;
          margin-bottom: 1rem;
        }
        .alt-dash-kpi {
          border: 1px solid %(border)s;
          border-radius: 22px;
          padding: 1rem 1.1rem;
          min-height: 9.8rem;
          background:
            linear-gradient(180deg, rgba(56, 189, 248, 0.10), transparent 45%%),
            %(surface)s;
          box-shadow: %(shadow)s;
        }
        .alt-dash-kpi-label {
          font-size: 0.78rem;
          letter-spacing: 0.03em;
          color: %(muted)s;
          margin-bottom: 0.55rem;
        }
        .alt-dash-kpi-value {
          font-size: 2rem;
          line-height: 1;
          letter-spacing: -0.04em;
          font-weight: 800;
          color: %(text)s;
        }
        .alt-dash-kpi-sub {
          margin-top: 0.5rem;
          font-size: 0.86rem;
          color: %(muted_strong)s;
        }
        .alt-dash-kpi-meta {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.55rem;
          margin-top: 0.95rem;
        }
        .alt-dash-kpi-meta div {
          padding: 0.55rem 0.68rem;
          border-radius: 14px;
          background: %(surface_alt)s;
          border: 1px solid %(border)s;
        }
        .alt-dash-kpi-meta span {
          display: block;
          font-size: 0.72rem;
          color: %(muted)s;
          margin-bottom: 0.14rem;
        }
        .alt-dash-kpi-meta strong {
          display: block;
          font-size: 0.96rem;
          color: %(text)s;
        }
        .alt-dash-statbar {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 0.75rem;
          margin-bottom: 1rem;
        }
        .alt-dash-stat {
          border: 1px solid %(border)s;
          border-radius: 18px;
          padding: 0.9rem 0.95rem;
          background: %(surface_alt)s;
        }
        .alt-dash-stat label {
          display: block;
          font-size: 0.74rem;
          color: %(muted)s;
          margin-bottom: 0.32rem;
        }
        .alt-dash-stat strong {
          display: block;
          color: %(text)s;
          font-size: 1.15rem;
          letter-spacing: -0.03em;
        }
        .alt-dash-card-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.75rem;
        }
        .alt-dash-card-title {
          margin: 0;
          font-size: 1rem;
          font-weight: 800;
          letter-spacing: -0.02em;
          color: %(text)s;
        }
        .alt-dash-card-note {
          color: %(muted)s;
          font-size: 0.78rem;
        }
        .alt-dash-chip {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 1.8rem;
          padding: 0.2rem 0.65rem;
          border-radius: 999px;
          border: 1px solid transparent;
          font-size: 0.76rem;
          font-weight: 700;
        }
        .alt-dash-chip.ok { background: rgba(34, 197, 94, 0.12); color: %(accent_alt)s; border-color: rgba(34, 197, 94, 0.22); }
        .alt-dash-chip.warn { background: rgba(251, 191, 36, 0.12); color: %(warn)s; border-color: rgba(251, 191, 36, 0.25); }
        .alt-dash-chip.bad { background: rgba(251, 113, 133, 0.12); color: %(danger)s; border-color: rgba(251, 113, 133, 0.24); }
        .alt-dash-chip.info { background: rgba(56, 189, 248, 0.12); color: %(accent)s; border-color: rgba(56, 189, 248, 0.24); }
        .alt-dash-chip.neutral { background: %(surface_alt)s; color: %(muted_strong)s; border-color: %(border)s; }
        .alt-dash-list {
          display: grid;
          gap: 0.7rem;
        }
        .alt-dash-list-item {
          border: 1px solid %(border)s;
          border-radius: 16px;
          background: %(surface_alt)s;
          padding: 0.8rem 0.9rem;
        }
        .alt-dash-list-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          margin-bottom: 0.35rem;
        }
        .alt-dash-list-title {
          font-weight: 700;
          color: %(text)s;
        }
        .alt-dash-list-meta {
          color: %(muted)s;
          font-size: 0.76rem;
        }
        .alt-dash-list-body {
          color: %(muted_strong)s;
          font-size: 0.84rem;
          line-height: 1.5;
        }
        @media (max-width: 1100px) {
          .alt-dash-kpis,
          .alt-dash-statbar {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }
        </style>
    """ % css_vars


def _dashboard_tone_from_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"completed", "running", "filled", "success", "ok"}:
        return "ok"
    if normalized in {"paused", "queued", "pending_fill", "submitted", "acknowledged"}:
        return "warn"
    if normalized in {"failed", "rejected", "cancelled", "error", "stopped"}:
        return "bad"
    if normalized in {"candidate", "long", "info"}:
        return "info"
    return "neutral"


def _dashboard_chip_html(label: str, tone: str) -> str:
    return f'<span class="alt-dash-chip {html.escape(tone)}">{html.escape(label)}</span>'


def _dashboard_change_html(change_pct: float) -> str:
    if not np.isfinite(change_pct):
        return '<span class="alt-dash-chip neutral">N/A</span>'
    tone = "bad" if change_pct < 0 else "ok" if change_pct > 0 else "neutral"
    return _dashboard_chip_html(f"{change_pct:+.2f}%", tone)


def _dashboard_status_strip_html(
    asset_type: str,
    auto_trading_status: Dict[str, Any],
    runtime_profile: Dict[str, Any],
    unresolved_count: int,
    row_count: int,
) -> str:
    state = str(auto_trading_status.get("state", "stopped")).lower()
    tone = _dashboard_tone_from_status(state)
    dot_class = tone if tone in {"ok", "warn", "bad"} else "neutral"
    profile_name = str(runtime_profile.get("name") or "-")
    last_refresh = format_display_timestamp(pd.Timestamp.now(tz="Asia/Seoul"))
    return f"""
        <div class="alt-dash-strip">
          <div class="alt-dash-strip-item"><span class="alt-dash-dot {dot_class}"></span>{html.escape(_localized_auto_trading_label(auto_trading_status))}</div>
          <div class="alt-dash-strip-item"><strong>{html.escape(asset_type)}</strong> 스냅샷 {row_count}개</div>
          <div class="alt-dash-strip-item">프로파일 {html.escape(profile_name)}</div>
          <div class="alt-dash-strip-item">미해결 이름 {unresolved_count}개</div>
          <div class="alt-dash-strip-time">갱신 {html.escape(last_refresh)}</div>
        </div>
    """

def render_analysis_page() -> None:
    st.subheader("종목 분석")
    st.caption("대시보드에서 종목을 고르면 여기로 넘어오고, 현재 사이드바 설정 기준으로 예측을 실행합니다.")

    run_asset_type, normalized_symbol, run_korea_market = build_analysis_target(
        asset_type=asset_type,
        raw_symbol=raw_symbol,
        korea_market=korea_market,
    )

    info_col_1, info_col_2, info_col_3 = st.columns([2.2, 1.2, 1.2])
    info_col_1.metric("분석 대상", normalized_symbol)
    info_col_2.metric("자산 유형", run_asset_type)
    info_col_3.metric("예측 기간", f"{forecast_days}일")

    run_analysis = st.button("예측 실행", type="primary", key="analysis_run")
    should_auto_run = bool(st.session_state.get("analysis_auto_run", False))
    if should_auto_run:
        st.session_state["analysis_auto_run"] = False

    if run_analysis or should_auto_run:
        try:
            with st.spinner(f"{normalized_symbol} 데이터를 불러오고 모델을 학습하는 중..."):
                analysis_result = run_analysis_target(
                    run_asset_type=run_asset_type,
                    run_raw_symbol=normalized_symbol,
                    run_korea_market=run_korea_market,
                    years=years,
                    test_days=test_days,
                    forecast_days=forecast_days,
                    validation_mode=validation_mode,
                    retrain_every=retrain_every,
                    round_trip_cost_bps=round_trip_cost_bps,
                    min_signal_strength_pct=min_signal_strength_pct,
                    final_holdout_days=final_holdout_days,
                    purge_days=purge_days,
                    embargo_days=embargo_days,
                    target_mode=target_mode,
                    validation_days=validation_days,
                    allow_short=allow_short,
                    trade_mode=trade_mode,
                    target_daily_vol_pct=target_daily_vol_pct,
                    max_position_size=max_position_size,
                    stop_loss_atr_mult=stop_loss_atr_mult,
                    take_profit_atr_mult=take_profit_atr_mult,
                )
        except Exception as exc:
            st.error(f"실행 중 오류: {exc}")
        else:
            st.session_state["analysis_result"] = analysis_result
            st.session_state["analysis_result_symbol"] = normalized_symbol
            st.session_state["analysis_result_forecast_days"] = forecast_days
            try:
                st.session_state["analysis_prediction_run_id"] = save_prediction_snapshot(
                    asset_type=run_asset_type,
                    korea_market=run_korea_market,
                    result=analysis_result,
                    notes="analysis_page_auto_save",
                )
            except Exception:
                st.session_state["analysis_prediction_run_id"] = ""

    saved_result = st.session_state.get("analysis_result")
    saved_symbol = str(st.session_state.get("analysis_result_symbol", ""))
    saved_forecast_days = int(st.session_state.get("analysis_result_forecast_days", forecast_days) or forecast_days)

    if saved_result is None:
        st.info("심볼을 정한 뒤 예측 실행을 누르세요. 대시보드의 분석 버튼으로 넘어오면 자동 실행됩니다.")
    else:
        if saved_symbol != normalized_symbol:
            st.info(f"현재 설정은 `{normalized_symbol}` 이고, 아래 결과는 최근 실행한 `{saved_symbol}` 기준입니다.")
        render_single_result(
            result=saved_result,
            forecast_days=saved_forecast_days,
            is_mobile_ui=is_mobile_ui,
            asset_type=run_asset_type,
            korea_market=run_korea_market,
        )



def render_paper_page() -> None:
    render_paper_trading_page(analysis_inputs=analysis_inputs, is_mobile_ui=is_mobile_ui)



def render_scan_page() -> None:
    st.subheader("종목 스캔")
    st.caption("Top100 후보나 직접 입력한 심볼을 한 번에 돌려 유망도 순위를 확인합니다.")

    use_top100 = asset_type in {"한국주식", "미국주식"}
    if use_top100:
        entries = build_top100_entries(asset_type=asset_type)
        option_map = {
            f"{entry['rank']:>3}. {entry['name']}" + (f" ({entry['symbol_hint']})" if entry["symbol_hint"] else ""): entry
            for entry in entries
        }
        option_labels = list(option_map.keys())
        default_pick_count = 10 if is_mobile_ui else 20
        default_labels = option_labels[: min(default_pick_count, len(option_labels))]
        selected_labels = st.multiselect(
            "Top100 종목 선택",
            options=option_labels,
            default=default_labels,
            key=f"scan_top100_names_{asset_type}",
        )
        selected_entries = [option_map[label] for label in selected_labels]
    else:
        selected_entries = []

    preset_symbols = WATCHLIST_PRESETS[asset_type]
    manual_symbol_mode = st.checkbox("기본 심볼 후보 사용", value=(asset_type == "코인"), key=f"scan_manual_mode_{asset_type}")
    if manual_symbol_mode:
        selected_symbols = st.multiselect(
            "기본 심볼 후보",
            options=preset_symbols,
            default=preset_symbols[: min(6, len(preset_symbols))],
            key=f"scan_preset_{asset_type}",
        )
    else:
        selected_symbols = []

    extra_symbols_raw = st.text_input(
        "추가 심볼 (쉼표/줄바꿈 구분)",
        value="",
        key=f"scan_extra_{asset_type}",
    )
    top_n = st.slider(
        "상위 카드 표시 개수",
        min_value=3,
        max_value=10 if is_mobile_ui else 15,
        value=5 if is_mobile_ui else 7,
        key="scan_top_n",
    )
    run_scan = st.button("유망 종목 스캔 실행", type="primary", key="scan_run")

    if not run_scan:
        st.info("종목을 선택하고 스캔 실행을 누르세요.")
        return

    resolved_pairs: List[Tuple[str, str]] = []
    unresolved_names: List[str] = []

    if selected_entries:
        name_pairs, unresolved_names = resolve_top100_entries(asset_type=asset_type, entries=selected_entries)
        resolved_pairs.extend(name_pairs)

    extra_symbols = parse_symbols(extra_symbols_raw)
    resolved_pairs.extend((sym, sym) for sym in selected_symbols)
    resolved_pairs.extend((sym, sym) for sym in extra_symbols)
    resolved_pairs = dedupe_symbol_pairs(resolved_pairs)

    if unresolved_names:
        st.warning("일부 종목은 티커를 자동으로 찾지 못했습니다. 아래 목록을 직접 심볼 입력으로 보완해 주세요.")
        st.dataframe(pd.DataFrame({"미해결 종목명": unresolved_names}), width="stretch", hide_index=True)

    if not resolved_pairs:
        st.error("스캔할 심볼이 없습니다. Top100 선택 또는 심볼 직접 입력을 확인해 주세요.")
        return

    if validation_mode == "walk_forward" and len(resolved_pairs) >= 8:
        st.warning("워크포워드 모드는 느릴 수 있습니다. 종목 수가 많으면 분석 시간이 길어집니다.")

    rows: List[Dict[str, float | str]] = []
    errors: List[Dict[str, str]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for idx, (display_name, scan_raw_symbol) in enumerate(resolved_pairs, start=1):
        try:
            symbol = normalize_symbol(asset_type=asset_type, raw_symbol=scan_raw_symbol, korea_market=korea_market)
            status.write(f"{idx}/{len(resolved_pairs)} 분석 중: `{display_name}` -> `{symbol}`")
            scan_result = run_cached(
                symbol=symbol,
                years=years,
                test_days=test_days,
                forecast_days=forecast_days,
                validation_mode=validation_mode,
                retrain_every=retrain_every,
                round_trip_cost_bps=round_trip_cost_bps,
                min_signal_strength_pct=min_signal_strength_pct,
                final_holdout_days=final_holdout_days,
                purge_days=purge_days,
                embargo_days=embargo_days,
                target_mode=target_mode,
                validation_days=validation_days,
                allow_short=allow_short,
                trade_mode=trade_mode,
                target_daily_vol_pct=target_daily_vol_pct,
                max_position_size=max_position_size,
                stop_loss_atr_mult=stop_loss_atr_mult,
                take_profit_atr_mult=take_profit_atr_mult,
            )
            row = build_scan_row(symbol=symbol, result=scan_result, forecast_days=forecast_days)
            row["종목명"] = display_name
            rows.append(row)
        except Exception as exc:
            errors.append({"입력": display_name, "심볼": scan_raw_symbol, "오류": str(exc)})
        progress.progress(idx / len(resolved_pairs))

    status.empty()
    progress.empty()

    if rows:
        summary = (
            pd.DataFrame(rows)
            .sort_values(
                ["유망도점수", "최종홀드아웃_기대값(%)", "예상수익률(%)"],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )
        summary.insert(0, "순위", np.arange(1, len(summary) + 1))
        top_df = summary.head(min(top_n, len(summary)))

        st.subheader("유망 후보 카드")
        card_cols = st.columns(min((1 if is_mobile_ui else 3), len(top_df)))
        for i in range(len(top_df)):
            row = top_df.iloc[i]
            with card_cols[i % len(card_cols)]:
                st.metric(
                    f"#{int(row['순위'])} {row['종목명']}",
                    f"{row['유망도점수']:.1f}점",
                    f"{row['예상수익률(%)']:+.2f}%",
                )
                st.caption(f"{row['심볼']} · 홀드아웃 기대값 {row['최종홀드아웃_기대값(%)']:+.3f}%")
                st.caption(
                    f"방향정확도 {row['최종홀드아웃_방향정확도(%)']:.1f}% · 승률 {row['최종홀드아웃_승률(%)']:.1f}% · MDD {row['최종홀드아웃_MDD(%)']:.2f}%"
                )

        score_fig = go.Figure(
            data=[
                go.Bar(
                    x=top_df["종목명"],
                    y=top_df["유망도점수"],
                    text=[f"{value:.1f}" for value in top_df["유망도점수"]],
                    textposition="outside",
                )
            ]
        )
        score_fig.update_layout(
            title="상위 후보 유망도 점수",
            height=300 if is_mobile_ui else 380,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title="점수",
            xaxis_title="종목",
        )
        st.plotly_chart(score_fig, width="stretch")

        st.subheader("전체 스캔 결과")
        st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.warning("정상 분석된 종목이 없습니다. 심볼 형식이나 티커 연결 상태를 확인해 주세요.")

    if errors:
        st.subheader("실패 항목")
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)


def render_dashboard_developer_page() -> None:
    st.caption("Developer mode")
    _render_dashboard_page_classic()


page_objects = {
    "monitor": st.Page(render_monitor_page, title=VIEW_CODE_TO_LABEL["monitor"], url_path="monitor", default=True),
    "beta": st.Page(render_beta_monitor_page, title=VIEW_CODE_TO_LABEL["beta"], url_path="beta"),
    "dashboard": st.Page(render_dashboard_page, title=VIEW_CODE_TO_LABEL["dashboard"], url_path="dashboard"),
    "dashboard_dev": st.Page(
        render_dashboard_developer_page,
        title=VIEW_CODE_TO_LABEL["dashboard_dev"],
        url_path="dashboard-dev",
    ),
    "analysis": st.Page(render_analysis_page, title=VIEW_CODE_TO_LABEL["analysis"], url_path="analysis"),
    "paper": st.Page(render_paper_page, title=VIEW_CODE_TO_LABEL["paper"], url_path="paper"),
    "scan": st.Page(render_scan_page, title=VIEW_CODE_TO_LABEL["scan"], url_path="scan"),
}
page_sequence = [
    page_objects["monitor"],
    page_objects["beta"],
    page_objects["dashboard"],
    page_objects["dashboard_dev"],
    page_objects["analysis"],
    page_objects["paper"],
    page_objects["scan"],
]
selected_page = st.navigation(page_sequence, position="hidden")
current_page_key = resolve_current_page_key(page_objects, selected_page, default_key="monitor")

if current_page_key != "beta":
    render_page_header()

    nav_events: Dict[str, str | None] = {"selected_page": None, "theme_toggle_event": None}
    nav_error = None
    try:
        nav_events = render_floating_nav(
            current_page=current_page_key,
            items=NAV_ITEMS,
            status=dashboard_data.get("auto_trading_status", {}),
            theme_mode=theme_mode,
            hide_on_scroll=True,
            scroll_threshold=88,
        )
    except Exception as exc:
        nav_error = str(exc)

    theme_toggle_event = nav_events.get("theme_toggle_event")
    if theme_toggle_event and theme_toggle_event != st.session_state.get("ui_theme_toggle_event", ""):
        st.session_state["ui_theme_toggle_event"] = theme_toggle_event
        next_theme_mode = "dark" if theme_mode == "light" else "light"
        st.session_state["ui_theme_mode"] = next_theme_mode
        st_config.set_option("theme.base", next_theme_mode)
        st.rerun()

    nav_selection = nav_events.get("selected_page")
    if nav_selection and nav_selection != current_page_key:
        switch_to_page(nav_selection)

    if nav_error:
        st.caption("??? ????? ????????? ?????? ??? ??? ??? UI???????????.")
        render_navigation_fallback(current_page=current_page_key, items=NAV_ITEMS, page_map=page_objects)

selected_page.run()

if current_page_key != "beta":
    render_global_footer()
