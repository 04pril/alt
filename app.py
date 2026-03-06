from __future__ import annotations

import contextlib
import html
import io
import logging
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from predictor import extract_korean_stock_code, is_korean_stock_symbol, normalize_symbol, run_forecast
from top100_universe import (
    KR_NAME_ALIASES,
    KR_MANUAL_SYMBOLS,
    KR_TOP100_NAMES,
    US_NAME_QUERY_OVERRIDES,
    US_TOP100_NAME_SYMBOLS,
)


WATCHLIST_PRESETS: Dict[str, List[str]] = {
    "코인": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD"],
    "미국주식": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"],
    "한국주식": ["005930", "000660", "035420", "005380", "035720", "068270", "247540", "373220"],
}

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


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_krx_name_map() -> Dict[str, Dict[str, str]]:
    try:
        df = quiet_external_call(
            lambda: pd.read_html("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13")[0]
        )
    except Exception:
        return {}

    name_col, market_col, code_col = df.columns[:3]
    frame = df[[name_col, market_col, code_col]].copy()
    frame.columns = ["name", "market", "code"]
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["market"] = frame["market"].astype(str).str.strip()
    frame["code"] = frame["code"].astype(str).str.strip().str.zfill(6)
    frame = frame.drop_duplicates(subset=["name"], keep="first")
    return frame.set_index("name")[["market", "code"]].to_dict("index")


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
        if candidate in name_map:
            code = name_map[candidate]["code"]
            market = name_map[candidate]["market"]
            suffix = ".KQ" if "코스닥" in market else ".KS"
            return f"{code}{suffix}".upper()

    return search_symbol_from_yf(query=alias or name, market_hint="KR")


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


def first_valid_float(*values: object) -> float:
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            return number
    return float("nan")


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


def apply_responsive_css(is_mobile_ui: bool) -> None:
    if is_mobile_ui:
        st.markdown(
            """
            <style>
            .block-container {
              padding-top: 0.8rem !important;
              padding-bottom: 1.0rem !important;
              padding-left: 0.8rem !important;
              padding-right: 0.8rem !important;
            }
            h1 { font-size: 1.4rem !important; }
            h2, h3 { font-size: 1.08rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
            [data-testid="stTabs"] button[role="tab"] {
              font-size: 0.85rem !important;
              padding: 0.45rem 0.5rem !important;
            }
            [data-testid="stDataFrame"] {
              font-size: 0.84rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .block-container {
              padding-top: 1.0rem !important;
              padding-bottom: 1.2rem !important;
              padding-left: 1.4rem !important;
              padding-right: 1.4rem !important;
            }
            </style>
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


def render_single_result(result, forecast_days: int, is_mobile_ui: bool) -> None:
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

    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=result.trade_backtest.index,
            y=(result.trade_backtest["equity_curve"] - 1.0) * 100.0,
            mode="lines",
            name="연구구간",
            line=dict(width=2),
        )
    )
    eq_fig.add_trace(
        go.Scatter(
            x=result.final_holdout_trade_backtest.index,
            y=(result.final_holdout_trade_backtest["equity_curve"] - 1.0) * 100.0,
            mode="lines",
            name="최종홀드아웃",
            line=dict(width=2, dash="dot"),
        )
    )
    eq_fig.update_layout(
        height=260 if is_mobile_ui else 300,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis_title="누적수익률(%)",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.price_data.index,
            y=result.price_data["Close"],
            mode="lines",
            name="실제 종가",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.test_frame["ensemble_pred"],
            mode="lines",
            name="연구구간 예측",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.final_holdout_frame.index,
            y=result.final_holdout_frame["ensemble_pred"],
            mode="lines",
            name="최종홀드아웃 예측",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_frame.index,
            y=result.future_frame["ensemble_pred"],
            mode="lines+markers",
            name="미래 예측",
            line=dict(width=3),
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
        height=430 if is_mobile_ui else 620,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=20, r=20, t=10, b=20),
    )
    if is_mobile_ui:
        st.plotly_chart(eq_fig, width="stretch")
        st.plotly_chart(fig, width="stretch")
    else:
        chart_left, chart_right = st.columns([1.0, 1.35])
        with chart_left:
            st.plotly_chart(eq_fig, width="stretch")
        with chart_right:
            st.plotly_chart(fig, width="stretch")

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
    st.session_state["sidebar_asset_type"] = asset_type
    st.session_state["sidebar_raw_symbol"] = raw_symbol
    st.session_state["sidebar_korea_market"] = korea_market
    st.session_state["sidebar_asset_type_prev"] = asset_type
    st.session_state["active_view"] = "종목 분석"
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


st.set_page_config(page_title="멀티마켓 가격 예측기", layout="wide")

st.title("코인 · 미국주식 · 한국주식 가격 예측기")
st.caption("Yahoo Finance 데이터를 기반으로 앙상블 모델 예측 결과를 시각화합니다.")
st.info("이 도구는 실험/학습 목적입니다. 어떤 모델도 미래 가격을 보장하지 않습니다.", icon="ℹ️")
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


st.session_state.setdefault("ui_mode_choice", "자동")
st.session_state.setdefault("sidebar_asset_type", "코인")
st.session_state.setdefault("sidebar_raw_symbol", default_symbol_for_asset(st.session_state["sidebar_asset_type"]))
st.session_state.setdefault("sidebar_korea_market", "KOSPI")
st.session_state.setdefault("sidebar_asset_type_prev", st.session_state["sidebar_asset_type"])
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "종목 분석" if auto_mobile_detected else "대시보드"
st.session_state.setdefault("analysis_auto_run", False)
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("analysis_result_symbol", "")
st.session_state.setdefault("analysis_result_forecast_days", 0)
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

    asset_type = st.selectbox("자산 유형", ["코인", "미국주식", "한국주식"], key="sidebar_asset_type")
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

apply_responsive_css(is_mobile_ui=is_mobile_ui)
st.radio("화면", ["대시보드", "종목 분석", "종목 스캔"], horizontal=True, key="active_view")

if st.session_state["active_view"] == "대시보드":
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
        with st.expander("원격 접속(Tailscale) 안내"):
            st.markdown(
                """
                1. 이 PC와 원격 기기에 Tailscale을 설치하고 같은 Tailnet에 로그인합니다.
                2. 서버 PC에서 앱 실행: `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`
                3. 서버 PC Tailscale IP 확인: `tailscale ip -4`
                4. 원격 기기 브라우저에서 접속: `http://<TAILSCALE_IP>:8501`
                """
            )
    else:
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
                            st.rerun()
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
                                st.rerun()
        else:
            st.warning("검색 조건에 맞는 종목이 없습니다.")

        with st.expander("테이블로 보기", expanded=not is_mobile_ui):
            st.dataframe(build_snapshot_view_df(dashboard_filtered), width="stretch", hide_index=True)

        if unresolved_dashboard_names:
            with st.expander("심볼 자동 해석 실패 종목"):
                st.dataframe(pd.DataFrame({"종목명": unresolved_dashboard_names}), width="stretch", hide_index=True)

        with st.expander("원격 접속(Tailscale) 안내"):
            st.markdown(
                """
                1. 이 PC와 원격 기기에 Tailscale을 설치하고 같은 Tailnet에 로그인합니다.
                2. 서버 PC에서 앱 실행: `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`
                3. 서버 PC Tailscale IP 확인: `tailscale ip -4`
                4. 원격 기기 브라우저에서 접속: `http://<TAILSCALE_IP>:8501`
                """
            )

elif st.session_state["active_view"] == "종목 분석":
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

    saved_result = st.session_state.get("analysis_result")
    saved_symbol = str(st.session_state.get("analysis_result_symbol", ""))
    saved_forecast_days = int(st.session_state.get("analysis_result_forecast_days", forecast_days) or forecast_days)

    if saved_result is None:
        st.info("심볼을 정한 뒤 예측 실행을 누르세요. 대시보드의 분석 버튼으로 넘어오면 자동 실행됩니다.")
    else:
        if saved_symbol != normalized_symbol:
            st.info(f"현재 설정은 `{normalized_symbol}` 이고, 아래 결과는 최근 실행한 `{saved_symbol}` 기준입니다.")
        render_single_result(result=saved_result, forecast_days=saved_forecast_days, is_mobile_ui=is_mobile_ui)

else:
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

    if run_scan:
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
        else:
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
    else:
        st.info("종목을 선택하고 스캔 실행을 누르세요.")
