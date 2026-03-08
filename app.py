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
import yfinance as yf

from config.settings import load_settings
from kis_paper import KISPaperError
from monitoring.dashboard_hooks import load_dashboard_data
from prediction_memory import (
    filter_prediction_history,
    load_prediction_log,
    load_model_registry,
    prediction_id_for_run,
    refresh_prediction_actuals,
    save_prediction_snapshot,
    summarize_prediction_accuracy,
)
from predictor import extract_korean_stock_code, is_korean_stock_symbol, normalize_symbol, run_forecast
from services.manual_kis_service import (
    compute_manual_equity_metrics,
    load_kis_account_snapshot,
    load_kis_config,
    load_kis_quote,
    load_manual_equity_curve,
    load_manual_order_history,
    submit_manual_kis_order,
)
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
    "dashboard": "대시보드",
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


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_krx_name_map() -> Dict[str, Dict[str, str]]:
    try:
        response = requests.get(
            "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        response.encoding = "euc-kr"
        tables = quiet_external_call(lambda: pd.read_html(io.StringIO(response.text)))
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


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
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
        kis_quote = load_kis_quote(symbol)
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
        broker_order_sync_job,
        broker_position_sync_job,
        build_task_context,
    )

    runners = {
        "broker_account_sync": broker_account_sync_job,
        "broker_order_sync": broker_order_sync_job,
        "broker_position_sync": broker_position_sync_job,
    }
    runner = runners.get(job_name)
    if runner is None:
        return False, f"지원하지 않는 job입니다: {job_name}"
    try:
        context = build_task_context()
        run_key = f"manual:{pd.Timestamp.now(tz='UTC').isoformat()}"
        result = _run_guarded(context, job_name, run_key, lambda: runner(context))
    except Exception as exc:
        return False, str(exc)
    if result is None:
        return False, f"{job_name} 실행을 시작하지 못했습니다."
    return True, f"{job_name} 실행 완료"




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
    control_cols = st.columns([1.0, 1.0, 1.1, 1.1, 1.1, 2.6])
    if control_cols[0].button("신규 진입 중단", key="ops_pause"):
        repository.set_control_flag("trading_paused", "1", "set from streamlit monitor")
        st.rerun()
    if control_cols[1].button("신규 진입 재개", key="ops_resume"):
        repository.set_control_flag("trading_paused", "0", "set from streamlit monitor")
        st.rerun()
    if control_cols[2].button("계좌 Sync", key="ops_broker_account_sync"):
        ok, message = run_manual_runtime_job("broker_account_sync")
        st.session_state["broker_sync_feedback"] = {"ok": ok, "message": message}
        st.rerun()
    if control_cols[3].button("주문 Sync", key="ops_broker_order_sync"):
        ok, message = run_manual_runtime_job("broker_order_sync")
        st.session_state["broker_sync_feedback"] = {"ok": ok, "message": message}
        st.rerun()
    if control_cols[4].button("포지션 Sync", key="ops_broker_position_sync"):
        ok, message = run_manual_runtime_job("broker_position_sync")
        st.session_state["broker_sync_feedback"] = {"ok": ok, "message": message}
        st.rerun()

    data = dashboard_data or load_dashboard_data(settings)
    summary = data["summary"]
    trade_performance = data["trade_performance"]
    auto_trading_status = data.get("auto_trading_status", {})
    broker_sync_status = data.get("broker_sync_status", {})
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    feedback = st.session_state.pop("broker_sync_feedback", None)
    if isinstance(feedback, dict) and feedback.get("message"):
        if feedback.get("ok"):
            st.success(str(feedback["message"]))
        else:
            st.error(str(feedback["message"]))
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
    if broker_sync_status:
        sync_rows = pd.DataFrame(
            [
                {
                    "job": key,
                    "status": row.get("status", "never"),
                    "finished_at": row.get("finished_at", pd.NaT),
                    "retry_count": row.get("retry_count", 0),
                    "error_message": row.get("error_message", ""),
                }
                for key, row in broker_sync_status.items()
            ]
        )
        if not sync_rows.empty:
            st.dataframe(format_frame_timestamps_for_display(sync_rows), width="stretch", hide_index=True)

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

    tab_jobs, tab_positions, tab_predictions, tab_candidates, tab_assets, tab_errors, tab_broker = st.tabs(
        ["Job Health", "Open Positions", "Predictions", "Candidates", "Assets", "Recent Errors", "Broker Sync"]
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
    with tab_broker:
        if broker_sync_errors.empty:
            st.caption("최근 broker sync 오류가 없습니다.")
        else:
            st.dataframe(format_frame_timestamps_for_display(broker_sync_errors), width="stretch", hide_index=True)


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
    return load_kis_account_snapshot()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_paper_quote_snapshot(symbol: str, refresh_token: int) -> Dict[str, Any]:
    _ = refresh_token
    return load_kis_quote(symbol)


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
            holding_qty = int(first_valid_float(matched.iloc[0].get("quantity"), matched.iloc[0].get("보유수량"), default=0.0))

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
        kis_config = load_kis_config()
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
                order_result = submit_manual_kis_order(
                    symbol=paper_symbol,
                    side=side,
                    quantity=order_qty,
                    order_type=order_type,
                    requested_price=limit_price,
                    prediction_id=prediction_id,
                    metadata={
                        "name": paper_name,
                        "quote_price": current_price,
                        "predicted_move_pct": plan["predicted_move_pct"],
                        "entry_estimate": plan["entry_estimate"],
                        "stop_level": plan["stop_level"],
                        "take_level": plan["take_level"],
                    },
                )
            except Exception as exc:
                st.error(f"모의 주문 실패: {exc}")
            else:
                st.success(
                    f"주문 접수 완료: {order_result.get('status') or 'submitted'}"
                    + (f" / broker={order_result.get('broker_order_id')}" if order_result.get("broker_order_id") else "")
                )
                st.session_state["paper_account_refresh_token"] += 1
                st.session_state["paper_quote_refresh_token"] += 1
                st.rerun()

    st.markdown("**보유 종목**")
    if account_holdings.empty:
        st.info("현재 모의계좌 보유 종목이 없습니다.")
    else:
        st.dataframe(account_holdings, width="stretch", hide_index=True)

    equity_curve = load_manual_equity_curve()
    equity_metrics = compute_manual_equity_metrics()
    order_log = load_manual_order_history(limit=200)

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
            2. 서버 PC에서 앱 실행: `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`
            3. 서버 PC Tailscale IP 확인: `tailscale ip -4`
            4. 원격 기기 브라우저에서 접속: `http://<TAILSCALE_IP>:8501`
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
    render_operations_monitor(settings=runtime_settings, dashboard_data=dashboard_data)



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


page_objects = {
    "monitor": st.Page(render_monitor_page, title=VIEW_CODE_TO_LABEL["monitor"], url_path="monitor", default=True),
    "dashboard": st.Page(render_dashboard_page, title=VIEW_CODE_TO_LABEL["dashboard"], url_path="dashboard"),
    "analysis": st.Page(render_analysis_page, title=VIEW_CODE_TO_LABEL["analysis"], url_path="analysis"),
    "paper": st.Page(render_paper_page, title=VIEW_CODE_TO_LABEL["paper"], url_path="paper"),
    "scan": st.Page(render_scan_page, title=VIEW_CODE_TO_LABEL["scan"], url_path="scan"),
}
page_sequence = [
    page_objects["monitor"],
    page_objects["dashboard"],
    page_objects["analysis"],
    page_objects["paper"],
    page_objects["scan"],
]
selected_page = st.navigation(page_sequence, position="hidden")
current_page_key = resolve_current_page_key(page_objects, selected_page, default_key="monitor")

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
    st.caption("상단 커스텀 내비게이션을 로드하지 못해 기본 탐색 UI로 전환했습니다.")
    render_navigation_fallback(current_page=current_page_key, items=NAV_ITEMS, page_map=page_objects)
selected_page.run()
render_global_footer()
