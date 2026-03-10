from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from monitoring.live_display import build_live_accounts_overview, build_live_total_portfolio_overview
from runtime_accounts import (
    ACCOUNT_KIS_KR_PAPER,
    ACCOUNT_SIM_CRYPTO,
    ACCOUNT_SIM_US_EQUITY,
)

_TEMPLATE_OVERRIDE = str(os.getenv("ALT_BETA_TEMPLATE_PATH", "") or "").strip()
TEMPLATE_PATH = Path(_TEMPLATE_OVERRIDE).expanduser() if _TEMPLATE_OVERRIDE else Path.home() / "Desktop" / "monitor_redesign_v2.html"
BETA_PATH = "/beta"
BETA_LIVE_PAYLOAD_ELEMENT_ID = "beta-live-payload"
KST = ZoneInfo("Asia/Seoul")
ACCOUNT_VIEW_SPECS: Sequence[Tuple[str, str, str, str, str]] = (
    (ACCOUNT_KIS_KR_PAPER, "KIS", "한국주식 계좌", "kis", "KRW"),
    (ACCOUNT_SIM_US_EQUITY, "SIM", "미국주식 계좌", "sim", "USD"),
    (ACCOUNT_SIM_CRYPTO, "SIM", "코인 계좌", "sim", "USD"),
)
NAV_ITEMS: Sequence[Tuple[str, str]] = (
    ("운영 모니터", "beta-overview"),
    ("보유 현황", "positions"),
    ("후보 종목", "candidates"),
    ("실행 이벤트", "events"),
    ("브로커 동기화", "sync"),
    ("작업 이력", "jobs"),
    ("자산 설정", "assets"),
    ("최근 오류", "errors"),
)
COLUMN_LABELS = {
    "symbol": "종목",
    "asset_type": "자산",
    "side": "방향",
    "quantity": "수량",
    "entry_price": "진입가",
    "current_price": "현재가",
    "requested_qty": "요청 수량",
    "filled_qty": "체결 수량",
    "requested_price": "주문가",
    "status": "상태",
    "reason": "사유",
    "event_type": "이벤트",
    "component": "컴포넌트",
    "message": "메시지",
    "expected_return": "기대 수익",
    "confidence": "신뢰도",
    "score": "점수",
    "job_name": "작업",
    "run_key": "실행 키",
    "retry_count": "재시도",
    "error_message": "오류",
    "account_id": "계좌",
    "execution_account_id": "실행 계좌",
    "signal": "신호",
    "created_at": "생성 시각",
    "updated_at": "갱신 시각",
    "scheduled_at": "예약 시각",
    "started_at": "시작 시각",
    "finished_at": "종료 시각",
    "heartbeat_at": "마지막 갱신",
    "label": "전략",
    "timeframe": "타임프레임",
    "enabled": "활성",
    "experimental": "실험",
    "today_candidate_count": "후보",
    "today_entry_allowed_count": "허용",
    "today_entry_rejected_count": "거절",
    "today_submit_requested_count": "제출 요청",
    "today_submitted_count": "제출",
    "today_filled_count": "체결",
    "today_noop_count": "미실행",
    "open_positions": "보유 포지션",
    "pending_orders": "대기 주문",
    "today_noop_top_reason": "주요 미실행 사유",
    "today_reject_top_reason": "주요 거절 사유",
    "strategy_id": "전략 ID",
}
STATUS_LABELS = {
    "completed": "완료",
    "running": "실행 중",
    "queued": "대기",
    "retry": "재시도",
    "failed": "실패",
    "rejected": "거절",
    "cancelled": "취소",
    "submitted": "제출",
    "acknowledged": "접수",
    "pending_fill": "체결 대기",
    "partially_filled": "부분 체결",
    "filled": "체결 완료",
    "paused": "일시중지",
    "stopped": "중지",
    "never": "기록 없음",
}
JOB_LABELS = {
    "broker_market_status": "장 상태 확인",
    "broker_account_sync": "계좌 동기화",
    "broker_order_sync": "주문 동기화",
    "broker_position_sync": "포지션 동기화",
    "signal_scan": "시그널 스캔",
}
HIDDEN_DETAIL_COLUMNS = {
    "rowid",
    "raw_json",
    "details_json",
    "prediction_id",
    "scan_id",
    "job_run_id",
    "run_key",
    "event_id",
}
ENTRY_RESULT_EVENT_TYPES = {
    "entry_allowed",
    "entry_rejected",
    "submit_requested",
    "submitted",
    "acknowledged",
    "filled",
    "rejected",
    "cancelled",
    "noop",
    "entry_orders_created",
}
ENTRY_RESULT_LABELS = {
    "entry_allowed": "진입 허용",
    "entry_rejected": "진입 거절",
    "submit_requested": "주문 요청",
    "submitted": "주문 제출",
    "acknowledged": "주문 접수",
    "filled": "체결 완료",
    "rejected": "주문 거절",
    "cancelled": "주문 취소",
    "noop": "미실행",
    "entry_orders_created": "주문 생성",
}
ENTRY_REASON_LABELS = {
    "outside_preclose_window": "진입 가능 시간이 아님",
    "market_closed": "장 마감 상태",
    "insufficient_buying_power": "주문 가능 금액 부족",
    "duplicate_pending_entry": "같은 종목 대기 주문 있음",
    "cooldown_active": "쿨다운 진행 중",
    "no_quote": "호가 없음",
    "position_exists": "기존 포지션 보유 중",
    "entry_limit_reached": "오늘 진입 한도 도달",
    "entry_paused": "진입 일시중지 상태",
}


def _replace_block(source: str, marker: str, replacement: str) -> str:
    start = source.find(marker)
    if start < 0:
        return source
    tag_start = source.rfind("<", 0, start + 1)
    if tag_start < 0:
        return source
    if source.startswith("<nav", tag_start):
        open_tag, close_tag = "<nav", "</nav>"
    else:
        open_tag, close_tag = "<div", "</div>"
    depth, index = 0, tag_start
    while index < len(source):
        next_open = source.find(open_tag, index)
        next_close = source.find(close_tag, index)
        if next_close < 0:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            index = next_open + len(open_tag)
            continue
        end = next_close + len(close_tag)
        depth -= 1
        if depth == 0:
            return source[:tag_start] + replacement + source[end:]
        index = end
    return source


def _replace_template_script(source: str, replacement: str) -> str:
    pattern = re.compile(r"<script\b[^>]*>[\s\S]*?</script>\s*</body>", re.IGNORECASE)
    if pattern.search(source):
        return pattern.sub(replacement + "\n</body>", source, count=1)
    body_match = re.search(r"</body>", source, re.IGNORECASE)
    if not body_match:
        return source + replacement
    return source[: body_match.start()] + replacement + "\n" + source[body_match.start() :]


def _ensure_template_base_href(source: str, href: str = "/") -> str:
    base_pattern = re.compile(r"<base\b[^>]*href=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)
    if base_pattern.search(source):
        return base_pattern.sub(f'<base href="{href}">', source, count=1)
    head_pattern = re.compile(r"<head[^>]*>", re.IGNORECASE)
    head_match = head_pattern.search(source)
    if not head_match:
        return source
    insert_at = head_match.end()
    return source[:insert_at] + f'\n<base href="{href}">' + source[insert_at:]


def _account_label(account_id: object) -> str:
    text = str(account_id or "").strip()
    for candidate_id, broker, scope, _, _ in ACCOUNT_VIEW_SPECS:
        if candidate_id == text:
            return f"{broker} {scope}"
    return text or "-"


def _f(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def _i(value: object) -> int:
    number = _f(value)
    return int(number) if np.isfinite(number) else 0


def _fmt_time(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "기록 없음"
    return str(parsed.tz_convert(KST).strftime("%Y-%m-%d %H:%M:%S"))


def _fmt_money(value: object, currency: str | None = None) -> str:
    number = _f(value)
    if not np.isfinite(number):
        return "N/A"
    suffix = f" {currency}" if currency else ""
    return f"{number:,.2f}{suffix}"


def _fx_rate_to_krw(currency: str, quote_snapshots: Dict[str, Dict[str, Any]]) -> float:
    normalized = str(currency or "").upper()
    if normalized in {"", "KRW"}:
        return 1.0
    if normalized == "USD":
        return _f((quote_snapshots.get("KRW=X") or {}).get("current_price"))
    return float("nan")


def _money_display_pair(value: object, currency: str, quote_snapshots: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    amount = _f(value)
    normalized = str(currency or "").upper()
    if not np.isfinite(amount):
        return "N/A", ""
    fx_rate = _fx_rate_to_krw(normalized, quote_snapshots)
    if normalized not in {"", "KRW"} and np.isfinite(fx_rate):
        return _fmt_money(amount * fx_rate, "KRW"), _fmt_money(amount, normalized)
    return _fmt_money(amount, normalized or None), ""


def _money_value_display(value: object, currency: str, quote_snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    amount = _f(value)
    normalized = str(currency or "").upper()
    primary_value = amount
    primary_currency = normalized or "KRW"
    secondary_text = ""
    if not np.isfinite(amount):
        return {
            "primary_value": float("nan"),
            "primary_currency": primary_currency,
            "primary_text": "N/A",
            "secondary_text": "",
            "compact_text": "N/A",
        }
    fx_rate = _fx_rate_to_krw(normalized, quote_snapshots)
    if normalized not in {"", "KRW"} and np.isfinite(fx_rate):
        primary_value = amount * fx_rate
        primary_currency = "KRW"
        secondary_text = _fmt_money(amount, normalized)
    primary_text = _fmt_money(primary_value, primary_currency)
    return {
        "primary_value": float(primary_value),
        "primary_currency": primary_currency,
        "primary_text": primary_text,
        "secondary_text": secondary_text,
        "compact_text": _fmt_compact_money(primary_value),
    }


def _fmt_compact_money(value: object) -> str:
    number = _f(value)
    if not np.isfinite(number):
        return "N/A"
    sign = "-" if number < 0 else ""
    amount = abs(number)
    if amount >= 100_000_000:
        compact = f"{amount / 100_000_000:,.1f}".rstrip("0").rstrip(".") + "억"
    elif amount >= 10_000:
        compact = f"{amount / 10_000:,.0f}만"
    else:
        compact = f"{amount:,.0f}"
    return sign + compact


def _format_currency_mix(values: Dict[str, float]) -> str:
    rows = []
    for currency, amount in values.items():
        numeric = _f(amount)
        if not np.isfinite(numeric):
            continue
        rows.append(_fmt_money(numeric, currency))
    return " / ".join(rows) if rows else "N/A"


def _fmt_pct(value: object, *, ratio: bool = False) -> str:
    number = _f(value)
    if not np.isfinite(number):
        return "N/A"
    if ratio or abs(number) <= 1.5:
        number *= 100.0
    return f"{number:+.2f}%"


def _fmt_qty(value: object) -> str:
    number = _f(value)
    if not np.isfinite(number):
        return "-"
    rounded = round(number)
    if abs(number - rounded) < 1e-9:
        return f"{int(rounded):,}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")


def _side_badge(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return '<span class="side-badge sl">롱</span>'
    if side in {"SHORT", "SELL"}:
        return '<span class="side-badge ss">숏</span>'
    return '<span class="side-badge sf">관망</span>'


def _chip(label: str, tone: str) -> str:
    tone_class = {
        "ok": "c-ok",
        "warn": "c-retry",
        "bad": "c-fail",
        "idle": "c-idle",
    }.get(tone, "c-idle")
    return f'<span class="chip {tone_class}">{html.escape(label)}</span>'


def _tone(status: object) -> str:
    text = str(status or "").strip().lower()
    if text in {"completed", "running", "filled", "success", "ok"}:
        return "ok"
    if text in {"queued", "retry", "pending_fill", "submitted", "acknowledged", "paused"}:
        return "warn"
    if text in {"failed", "rejected", "cancelled", "error", "stopped"}:
        return "bad"
    return "idle"


def _status_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return STATUS_LABELS.get(text, text or "-")


def _job_label(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("scan:"):
        suffix = text.split(":", 1)[1].strip()
        return f"시그널 스캔 · {suffix}" if suffix else "시그널 스캔"
    return JOB_LABELS.get(text, text or "-")


def _metric_note(base_text: str, extra_text: str | None = None) -> str:
    if extra_text:
        return f"{base_text} · {extra_text}"
    return base_text


def _recent_trade_reason_label(notes: object) -> str:
    text = str(notes or "").strip().lower()
    if text.startswith("closed_by_"):
        text = text[len("closed_by_") :]
    mapping = {
        "manual_exit": "수동청산",
        "stop_loss": "손절",
        "take_profit": "익절",
        "trailing_stop": "추적손절",
        "time_stop": "시간청산",
        "opposite_signal": "반대신호",
        "score_decay": "점수약화",
    }
    return mapping.get(text, text.replace("_", " ").strip() or "-")


def _holding_duration_text(started_at: object, closed_at: object) -> str:
    opened = pd.to_datetime(started_at, errors="coerce", utc=True)
    closed = pd.to_datetime(closed_at, errors="coerce", utc=True)
    if pd.isna(opened) or pd.isna(closed):
        return "-"
    seconds = max(int((closed - opened).total_seconds()), 0)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    if days > 0:
        return f"{days}일 {hours}시간"
    if hours > 0:
        return f"{hours}시간 {minutes}분"
    return f"{minutes}분"


def _trade_money_text(value: object, currency: str, quote_snapshots: Dict[str, Dict[str, Any]]) -> str:
    primary, secondary = _money_display_pair(value, currency, quote_snapshots)
    return f"{primary} / {secondary}" if secondary else primary


def _position_quote_snapshot(
    symbol: str,
    asset_type: str,
    quote_snapshots: Dict[str, Dict[str, Any]],
    kr_asset_types: set[str],
) -> Dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    candidates = [str(symbol or "").strip()]
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    market_code, _ = _market_meta(symbol, asset_type, kr_asset_types)
    if market_code == "KR":
        if normalized.endswith(".KS") or normalized.endswith(".KQ"):
            candidates.append(normalized[:-3])
        elif normalized.isdigit() and len(normalized) == 6:
            candidates.extend([f"{normalized}.KS", f"{normalized}.KQ"])
    for candidate in candidates:
        quote = quote_snapshots.get(candidate)
        if isinstance(quote, dict) and quote:
            return dict(quote)
    return {}


def _optional_money_text(value: object, currency: str) -> str:
    amount = _f(value)
    if not np.isfinite(amount) or abs(amount) <= 1e-12:
        return "-"
    return _fmt_money(amount, currency)


def _build_realized_trade_table_frame(
    recent_realized_trades: pd.DataFrame,
    quote_snapshots: Dict[str, Dict[str, Any]],
    *,
    kr_symbol_names: Dict[str, str] | None = None,
) -> pd.DataFrame:
    if recent_realized_trades.empty:
        return pd.DataFrame()
    currency_map = {account_id: currency for account_id, _, _, _, currency in ACCOUNT_VIEW_SPECS}
    rows: list[dict[str, str]] = []
    ordered = recent_realized_trades.sort_values("closed_at", ascending=False, na_position="last") if "closed_at" in recent_realized_trades.columns else recent_realized_trades
    for _, row in ordered.iterrows():
        account_id = str(row.get("account_id") or "")
        currency = str(currency_map.get(account_id) or "KRW")
        rows.append(
            {
                "종료시각": _fmt_time(row.get("closed_at")),
                "계좌": _account_label(account_id),
                "종목": _symbol_display_text(
                    row.get("symbol"),
                    asset_type=row.get("asset_type"),
                    kr_symbol_names=kr_symbol_names,
                ),
                "방향": "롱" if str(row.get("side") or "").upper() == "LONG" else "숏" if str(row.get("side") or "").upper() == "SHORT" else "-",
                "보유기간": _holding_duration_text(row.get("created_at"), row.get("closed_at")),
                "진입가": _trade_money_text(row.get("entry_price"), currency, quote_snapshots),
                "청산가": _trade_money_text(row.get("mark_price"), currency, quote_snapshots),
                "실현손익": _trade_money_text(row.get("realized_pnl"), currency, quote_snapshots),
                "청산사유": _recent_trade_reason_label(row.get("notes")),
            }
        )
    return pd.DataFrame(rows)


def _build_total_equity_curve(
    equity_curves_by_account: Dict[str, pd.DataFrame],
    quote_snapshots: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, str, str | None]:
    currency_map = {account_id: currency for account_id, _, _, _, currency in ACCOUNT_VIEW_SPECS}
    series_list: list[pd.Series] = []
    warnings: list[str] = []
    for account_id, frame in (equity_curves_by_account or {}).items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or "created_at" not in frame.columns or "equity" not in frame.columns:
            continue
        timestamps = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
        values = pd.to_numeric(frame["equity"], errors="coerce")
        mask = timestamps.notna() & values.notna()
        if not mask.any():
            continue
        series = pd.Series(values.loc[mask].to_numpy(dtype=float), index=timestamps.loc[mask]).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        currency = str(currency_map.get(str(account_id), "KRW") or "KRW").upper()
        if currency == "USD":
            fx_rate = _fx_rate_to_krw(currency, quote_snapshots)
            if not np.isfinite(fx_rate):
                warnings.append("USD 환율 미수신 계좌 제외")
                continue
            series = series * fx_rate
        elif currency != "KRW":
            warnings.append(f"{currency} 계좌 제외")
            continue
        series_list.append(series.rename(str(account_id)))
    if not series_list:
        return pd.DataFrame(), "", None
    union_index = series_list[0].index
    for series in series_list[1:]:
        union_index = union_index.union(series.index)
    union_index = union_index.sort_values()
    aligned = [series.reindex(union_index).ffill().fillna(0.0) for series in series_list]
    total = aligned[0].copy()
    for series in aligned[1:]:
        total = total.add(series, fill_value=0.0)
    note = "KRW 환산 합산"
    if warnings:
        note += " · " + ", ".join(dict.fromkeys(warnings))
    return pd.DataFrame({"created_at": union_index, "equity": total.values}), note, "KRW"


def _candidate_bucket(symbol: str, asset_type: str, kr_asset_types: set[str]) -> str:
    market_code, _ = _market_meta(symbol, asset_type, kr_asset_types)
    if market_code == "KR":
        return "kr"
    if market_code == "CR":
        return "crypto"
    return "us"


def _build_kr_symbol_name_map(krx_name_map: Dict[str, Dict[str, str]] | None) -> Dict[str, str]:
    symbol_map: Dict[str, str] = {}
    for name, payload in (krx_name_map or {}).items():
        if not isinstance(payload, dict):
            continue
        code = str(payload.get("code") or "").strip()
        market = str(payload.get("market") or "").strip().upper()
        clean_name = str(name or "").strip()
        if not code or not clean_name:
            continue
        suffix = ".KQ" if "KOSDAQ" in market else ".KS"
        symbol = f"{code.zfill(6)}{suffix}".upper()
        bare_code = code.zfill(6)
        symbol_map.setdefault(bare_code, clean_name)
        symbol_map.setdefault(symbol, clean_name)
    return symbol_map


def _kr_symbol_candidates(symbol: object) -> Tuple[str, ...]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ()
    candidates = [normalized]
    if normalized.endswith(".KS") or normalized.endswith(".KQ"):
        candidates.append(normalized[:-3])
    elif normalized.isdigit() and len(normalized) == 6:
        candidates.extend([f"{normalized}.KS", f"{normalized}.KQ"])
    seen: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def _kr_symbol_code(symbol: object) -> str:
    normalized = str(symbol or "").strip().upper()
    if normalized.endswith(".KS") or normalized.endswith(".KQ"):
        return normalized[:-3]
    if normalized.isdigit() and len(normalized) == 6:
        return normalized
    return ""


def _symbol_display_text(
    symbol: object,
    *,
    asset_type: object = "",
    kr_symbol_names: Dict[str, str] | None = None,
    name_only: bool = False,
) -> str:
    text = str(symbol or "").strip()
    normalized = text.upper()
    display_name = ""
    for candidate in _kr_symbol_candidates(text):
        display_name = str((kr_symbol_names or {}).get(candidate) or "").strip()
        if display_name:
            break
    code = _kr_symbol_code(text)
    if display_name:
        return display_name if name_only else f"{display_name} ({code or normalized})"
    if code:
        return code
    return text or "-"


def _candidate_symbol_markup(
    symbol: str,
    asset_type: str,
    *,
    kr_asset_types: set[str],
    kr_symbol_names: Dict[str, str],
) -> str:
    bucket = _candidate_bucket(symbol, asset_type, kr_asset_types)
    if bucket == "kr":
        normalized = str(symbol or "").upper()
        display_name = kr_symbol_names.get(normalized, "")
        display_code = normalized.replace(".KS", "").replace(".KQ", "") or "-"
        if display_name:
            return (
                f'<div class="cand-kr-wrap"><span class="cand-sym">{html.escape(display_name)}</span>'
                f'<span class="cand-code">{html.escape(display_code)}</span></div>'
            )
        return f'<div class="cand-kr-wrap"><span class="cand-sym">{html.escape(display_code)}</span></div>'
    _, market = _market_meta(symbol, asset_type, kr_asset_types)
    return (
        f'<span class="cand-sym">{html.escape(symbol or "-")}</span>'
        f'<span class="cand-mkt">{html.escape(market)}</span>'
    )


def _details_payload(row: pd.Series) -> Dict[str, Any]:
    details = row.get("details")
    if isinstance(details, dict):
        return details
    raw = row.get("details_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _entry_result_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return ENTRY_RESULT_LABELS.get(text, text.replace("_", " ").strip().title() or "이벤트")


def _entry_reason_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return ENTRY_REASON_LABELS.get(text, text.replace("_", " ").strip() or "-")


def _entry_result_tone(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"entry_allowed", "submit_requested", "submitted", "acknowledged", "filled", "entry_orders_created"}:
        return "ok"
    if text in {"entry_rejected", "rejected", "cancelled"}:
        return "fail"
    if text == "noop":
        return "skip"
    return "skip"


def _entry_result_body(event_type: str, details: Dict[str, Any], row: pd.Series) -> str:
    parts: list[str] = []
    reason = str(details.get("reason") or "").strip()
    expected_return = _f(details.get("expected_return"))
    confidence = _f(details.get("confidence"))
    message = str(row.get("message") or "").strip()

    if reason:
        parts.append(f"사유 {_entry_reason_label(reason)}")
    if np.isfinite(expected_return):
        parts.append(f"예상 수익 {_fmt_pct(expected_return, ratio=True)}")
    if np.isfinite(confidence):
        parts.append(f"신뢰도 {_fmt_pct(confidence, ratio=True)}")
    if not parts and message:
        parts.append(message[:100])
    if not parts and event_type == "entry_orders_created":
        parts.append(f"생성 건수 {_i(details.get('count'))}건")
    return " · ".join(parts) if parts else "세부 정보 없음"


def _build_entry_result_rows(
    today_execution_events: pd.DataFrame,
    *,
    kr_symbol_names: Dict[str, str] | None = None,
    limit: int | None = 4,
) -> list[str]:
    if today_execution_events.empty:
        return []
    rows: list[str] = []
    for _, row in _sort_frame(today_execution_events).iterrows():
        event_type = str(row.get("event_type") or "").strip().lower()
        if event_type not in ENTRY_RESULT_EVENT_TYPES:
            continue
        details = _details_payload(row)
        symbol = str(
            details.get("symbol")
            or details.get("ticker")
            or details.get("code")
            or row.get("symbol")
            or ""
        ).strip()
        if not symbol:
            continue
        display_symbol = _symbol_display_text(
            symbol,
            asset_type=details.get("asset_type") or row.get("asset_type") or "",
            kr_symbol_names=kr_symbol_names,
        )
        body = _entry_result_body(event_type, details, row)
        tone_class = _entry_result_tone(event_type)
        chip_tone = {"ok": "ok", "fail": "bad", "skip": "warn"}.get(tone_class, "idle")
        rows.append(
            '<div class="signal-item s-'
            + tone_class
            + '"><div class="sig-main"><div class="sig-top"><div class="sig-sym">'
            + html.escape(display_symbol)
            + '</div><div class="sig-status">'
            + _chip(_entry_result_label(event_type), chip_tone)
            + '</div></div><div class="sig-body">'
            + html.escape(body)
            + '</div></div><div class="sig-time">'
            + html.escape(_fmt_time(row.get("created_at")))
            + "</div></div>"
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _market_meta(symbol: str, asset_type: str, kr_asset_types: set[str]) -> Tuple[str, str]:
    normalized = str(symbol or "").upper()
    if asset_type in kr_asset_types or normalized.endswith(".KS") or normalized.endswith(".KQ") or normalized.isdigit():
        return "KR", f"Korea · {normalized.replace('.KS', '').replace('.KQ', '')}"
    if "-" in normalized and any(token in normalized for token in ("USD", "USDT", "KRW")):
        return "CR", "Crypto"
    return "US", asset_type or "US Equity"


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in ("created_at", "updated_at", "finished_at", "started_at", "scheduled_at", "heartbeat_at"):
        if column in frame.columns:
            return frame.sort_values(column, ascending=False, na_position="last")
    return frame


def _column_label(column: object) -> str:
    text = str(column or "")
    if text in COLUMN_LABELS:
        return COLUMN_LABELS[text]
    if any("\u3131" <= char <= "\ud7a3" for char in text):
        return text
    return text.replace("_", " ").strip().title() or "-"


def _compact(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "").strip()
    return text[:120] + "..." if len(text) > 120 else text


def _format_cell(column: str, value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if column == "enabled":
        return _chip("활성" if bool(value) else "비활성", "ok" if bool(value) else "mid")
    if column == "experimental":
        return _chip("실험" if bool(value) else "일반", "warn" if bool(value) else "mid")
    if column in {"job_name"}:
        return html.escape(_job_label(value))
    if column in {"account_id", "execution_account_id"}:
        return html.escape(_account_label(value))
    if column == "symbol":
        return html.escape(_symbol_display_text(value))
    if column == "status":
        return _chip(_status_label(value), _tone(value))
    if column in {"side", "signal"}:
        return _side_badge(value)
    if column in {"created_at", "updated_at", "scheduled_at", "started_at", "finished_at", "heartbeat_at"} or column.endswith("_at"):
        return html.escape(_fmt_time(value))
    if column in {"quantity", "requested_qty", "filled_qty"}:
        return html.escape(_fmt_qty(value))
    if column in {"entry_price", "current_price", "requested_price", "market_value", "unrealized_pnl", "equity", "cash", "gross_exposure", "daily_pnl"}:
        return html.escape(_fmt_money(value))
    if column in {"expected_return", "change_pct", "drawdown_pct"}:
        return html.escape(_fmt_pct(value, ratio=True))
    if column in {"confidence", "score"}:
        number = _f(value)
        return html.escape(f"{number:.2f}" if np.isfinite(number) else "-")
    if column == "reason":
        return f'<span class="rej-reason">{html.escape(str(value or "-"))}</span>'
    return html.escape(_compact(value) or "-")


def _table_html(
    frame: pd.DataFrame,
    columns: Sequence[str],
    empty_text: str,
    limit: int = 12,
    *,
    kr_symbol_names: Dict[str, str] | None = None,
) -> str:
    if frame.empty:
        return f'<div class="empty-block">{html.escape(empty_text)}</div>'
    hidden = HIDDEN_DETAIL_COLUMNS
    view = _sort_frame(frame).head(limit).copy()
    selected = [column for column in columns if column in view.columns and column not in hidden]
    for column in view.columns:
        if column not in selected and column not in hidden:
            selected.append(column)
        if len(selected) >= 8:
            break
    if not selected:
        return f'<div class="empty-block">{html.escape(empty_text)}</div>'
    header_html = "".join(f"<th>{html.escape(_column_label(column))}</th>" for column in selected)
    body_html = []
    for _, row in view.iterrows():
        cells = "".join(
            f"<td>{html.escape(_symbol_display_text(row.get(column), asset_type=row.get('asset_type'), kr_symbol_names=kr_symbol_names)) if column == 'symbol' else _format_cell(column, row.get(column))}</td>"
            for column in selected
        )
        body_html.append(f"<tr>{cells}</tr>")
    return '<div class="detail-table-wrap"><table class="detail-table"><thead><tr>' + header_html + "</tr></thead><tbody>" + "".join(body_html) + "</tbody></table></div>"


def _detail_card(title: str, body: str, note: str = "", *, card_class: str = "detail-card") -> str:
    note_html = f'<div class="card-meta">{html.escape(note)}</div>' if note else ""
    classes = html.escape((card_class or "detail-card").strip())
    return '<div class="card ' + classes + '"><div class="card-hd"><div class="card-title">' + html.escape(title) + "</div>" + note_html + "</div>" + body + "</div>"


def _section_html(section_id: str, title: str, cards: Sequence[str]) -> str:
    return f'<section class="detail-section" id="{html.escape(section_id)}" data-section-anchor="{html.escape(section_id)}"><div class="detail-section-title">{html.escape(title)}</div><div class="detail-grid">{"".join(cards)}</div></section>'


def _beta_href(
    *,
    anchor: str,
    action: str | None = None,
    token: str | None = None,
    candidate_tab: str | None = None,
    jobs: str | None = None,
    signals: str | None = None,
    theme: str | None = None,
) -> str:
    params: Dict[str, str] = {"beta_anchor": anchor}
    if action:
        params["beta_action"] = action
    if token:
        params["beta_token"] = token
    if candidate_tab:
        params["beta_cand_tab"] = candidate_tab
    if jobs:
        params["beta_jobs"] = jobs
    if signals:
        params["beta_signals"] = signals
    if theme:
        params["beta_theme"] = theme
    return BETA_PATH + "?" + urlencode(params)


def _beta_link_button(
    *,
    css_class: str,
    href: str,
    label: str | None = None,
    body_html: str | None = None,
    action: str | None = None,
    anchor: str | None = None,
    preserve_candidate_tab: bool = True,
    theme_toggle: bool = False,
) -> str:
    attrs = [
        'type="button"',
        f'class="{html.escape(css_class)}"',
        f'data-beta-href="{html.escape(href)}"',
    ]
    if action:
        attrs.append(f'data-action="{html.escape(action)}"')
    if anchor:
        attrs.append(f'data-anchor="{html.escape(anchor)}"')
    if preserve_candidate_tab:
        attrs.append('data-beta-link="preserve-candidate-tab"')
    if theme_toggle:
        attrs.append('data-theme-toggle="1"')
    inner = body_html if body_html is not None else html.escape(label or "")
    return f"<button {' '.join(attrs)}>{inner}</button>"


def _action_button(
    label: str,
    action: str,
    css_class: str,
    anchor: str,
    *,
    token: str,
    candidate_tab: str | None,
    jobs: str | None,
    signals: str | None,
    theme_mode: str,
) -> str:
    href = _beta_href(anchor=anchor, action=action, token=token, candidate_tab=candidate_tab, jobs=jobs, signals=signals, theme=theme_mode)
    return _beta_link_button(css_class=css_class, href=href, label=label, action=action, anchor=anchor)


def _nav_button(label: str, target: str, active: bool = False) -> str:
    active_class = " active" if active else ""
    return f'<div class="nav-tab{active_class}" data-nav-target="{html.escape(target)}">{html.escape(label)}</div>'


def _theme_button(theme_mode: str, anchor: str, *, token: str, candidate_tab: str | None, jobs: str | None, signals: str | None) -> str:
    next_icon = "☾" if theme_mode == "light" else "☀"
    next_label = "다크" if theme_mode == "light" else "라이트"
    next_theme = "dark" if theme_mode == "light" else "light"
    href = _beta_href(anchor=anchor, action="toggle_theme", token=token, candidate_tab=candidate_tab, jobs=jobs, signals=signals, theme=next_theme)
    return _beta_link_button(
        css_class="theme-toggle",
        href=href,
        body_html=f'<span id="theme-icon">{next_icon}</span><span id="theme-label">{next_label}</span>',
        action="toggle_theme",
        anchor=anchor,
        theme_toggle=True,
    )


def _latest_scan_time(job_health: pd.DataFrame) -> str:
    if job_health.empty or "job_name" not in job_health.columns:
        return "기록 없음"
    scan_rows = job_health.loc[job_health["job_name"].astype(str).str.startswith("scan:", na=False)].copy()
    if scan_rows.empty:
        return "기록 없음"
    row = _sort_frame(scan_rows).iloc[0]
    return _fmt_time(row.get("finished_at") or row.get("started_at") or row.get("scheduled_at"))


def _extra_styles() -> str:
    return """
.empty-block{padding:24px 12px;text-align:center;color:var(--text3);background:var(--surface2);border:1px dashed var(--border2);border-radius:10px}
.content-grid>*,.left-col,.right-col,.card{min-width:0}
.chart-box-live{display:block;height:auto;min-height:0;align-items:stretch;justify-content:flex-start;overflow:hidden;padding:12px}
.chart-box-live svg{width:100%;max-width:100%;height:156px;display:block;flex:0 0 auto}
.chart-legend{margin-top:8px;font-size:11px;color:var(--text3);text-align:right}
.acct-value.compact,.sum-value.compact{font-size:24px;letter-spacing:-0.03em}
.acct-mini,.sum-mini{margin-top:4px;font-size:10px;color:var(--text3);line-height:1.35}
.theme-toggle,.cand-tab,.btn,.btn-mini,.wp-restart,.job-more{text-decoration:none;appearance:none;-webkit-appearance:none;font:inherit;cursor:pointer}
.cand-tab,.btn,.btn-mini,.wp-restart,.job-more{display:inline-flex;align-items:center;justify-content:center}
.signal-item{align-items:flex-start}
.sig-main{min-width:0;flex:1;display:flex;flex-direction:column;gap:4px}
.sig-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.sig-body{font-size:11px;color:var(--text2);line-height:1.45}
.sig-status{margin-left:0}
.sig-time{min-width:132px;margin-left:12px;align-self:center}
.cand-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.cand-tab{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface2);color:var(--text2);font-size:11px;font-weight:700;cursor:pointer}
.cand-tab.active{background:var(--surface);color:var(--text);border-color:var(--accent);box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 20%,transparent) inset}
.cand-tab-count{font-size:10px;color:var(--text3)}
.cand-pane{display:none}
.cand-pane.active{display:block}
.cand-table-wrap{overflow:auto;max-height:360px}
.cand-kr-wrap{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cand-code{font-size:10px;color:var(--text3);font-family:"SF Mono",Consolas,monospace}
.job-card-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.job-more{padding:7px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface2);color:var(--text2);font-size:11px;font-weight:700}
.feedback-banner{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:10px;border:1px solid var(--border);background:var(--surface);margin-bottom:12px;font-size:12px;font-weight:600}
.feedback-banner.ok{border-color:rgba(16,185,129,.28);color:var(--ok)}
.feedback-banner.bad{border-color:rgba(239,68,68,.28);color:var(--up)}
.side-badge.sf{background:color-mix(in srgb,var(--surface2) 72%,transparent);border:1px solid var(--border2);color:var(--text2)}
.equity-area{fill:color-mix(in srgb,var(--accent) 18%,transparent)}
.equity-line{stroke:var(--accent)}
.detail-section{margin-top:14px}
.detail-section-title{font-size:12px;font-weight:800;letter-spacing:.08em;color:var(--text3);text-transform:uppercase;margin:0 0 8px 2px}
.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.detail-events-grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(300px,.85fr);gap:12px}
.detail-events-main,.detail-events-side,.detail-anchor-card{min-width:0}
.detail-events-side{display:grid;gap:12px;align-content:start}
.detail-card.compact .detail-table-wrap{max-height:208px;overflow:auto}
.detail-card.compact .detail-table th,.detail-card.compact .detail-table td{padding:8px 8px;font-size:10px}
.detail-card.compact .empty-block{padding:16px 10px}
.detail-table-wrap{overflow:auto}
.detail-table{width:100%;border-collapse:collapse;font-size:11px}
.detail-table th{padding:10px 8px;border-bottom:1px solid var(--border);color:var(--text3);font-size:10px;font-weight:700;text-align:left;white-space:nowrap}
.detail-table td{padding:10px 8px;border-bottom:1px solid var(--border2);vertical-align:top;color:var(--text);line-height:1.45}
.detail-table tr:last-child td{border-bottom:none}
@media (max-width: 1180px){.account-row{grid-template-columns:1fr}.content-grid{grid-template-columns:1fr}.detail-grid,.detail-events-grid{grid-template-columns:1fr}.signal-item{flex-direction:column}.sig-time{min-width:auto;margin-left:0;align-self:flex-start}}
"""


def _equity_svg(equity_curve: pd.DataFrame, theme_mode: str, *, legend_currency: str | None = None, legend_note: str = "") -> str:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return '<div class="chart-box">계좌 스냅샷이 없습니다.</div>'
    values = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna().tail(24)
    if values.empty:
        return '<div class="chart-box">계좌 스냅샷이 없습니다.</div>'
    width, height, padding = 620.0, 190.0, 18.0
    minimum, maximum = float(values.min()), float(values.max())
    spread = max(maximum - minimum, 1.0)
    points = []
    for index, value in enumerate(values.tolist()):
        x = padding + (index / max(len(values) - 1, 1)) * (width - padding * 2)
        y = height - padding - ((float(value) - minimum) / spread) * (height - padding * 2)
        points.append(f"{x:.2f},{y:.2f}")
    area = " ".join(points + [f"{width - padding:.2f},{height - padding:.2f}", f"{padding:.2f},{height - padding:.2f}"])
    polyline = " ".join(points)
    latest_value = _fmt_money(values.iloc[-1], legend_currency)
    suffix = f" · {legend_note}" if legend_note else ""
    return f'<div class="chart-box chart-box-live"><svg viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="xMidYMid meet"><polygon class="equity-area" points="{area}"></polygon><polyline class="equity-line" points="{polyline}" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline></svg><div class="chart-legend">최근 평가 자산 {html.escape(latest_value)}{html.escape(suffix)}</div></div>'


def _account_card(
    snapshot: Dict[str, Any],
    fallback_account: Dict[str, Any],
    trade_performance: Dict[str, Any],
    broker: str,
    scope: str,
    css_class: str,
    *,
    currency: str = "KRW",
    quote_snapshots: Dict[str, Dict[str, Any]] | None = None,
    footer_note: str = "",
) -> str:
    quote_snapshots = quote_snapshots or {}
    equity = _f(snapshot.get("equity"))
    cash = _f(snapshot.get("cash"))
    daily = _f(snapshot.get("daily_pnl"))
    if not np.isfinite(equity):
        equity = _f(fallback_account.get("equity"))
    if not np.isfinite(cash):
        cash = _f(fallback_account.get("cash"))
    if not np.isfinite(daily):
        daily = _f(trade_performance.get("today_pnl"))
    exposure = abs(_f(snapshot.get("gross_exposure")))
    drawdown = _f(snapshot.get("drawdown_pct"))
    daily_pct = daily / equity if np.isfinite(equity) and equity > 0 else float("nan")
    daily_class = " up" if np.isfinite(daily) and daily > 0 else " dn" if np.isfinite(daily) and daily < 0 else ""
    updated = _fmt_time(snapshot.get("created_at") or fallback_account.get("created_at"))
    equity_display = _money_value_display(equity, currency, quote_snapshots)
    daily_display = _money_value_display(daily, currency, quote_snapshots)
    cash_display = _money_value_display(cash, currency, quote_snapshots)
    exposure_display = _money_value_display(exposure, currency, quote_snapshots)
    equity_sub = f"전일 대비 {_fmt_pct(daily_pct)}"
    if equity_display["secondary_text"]:
        equity_sub = f"{equity_sub} · {equity_display['secondary_text']}"
    daily_sub = "실현 + 미실현 합산"
    if daily_display["secondary_text"]:
        daily_sub += f" · {daily_display['secondary_text']}"
    cash_sub = "진입 가능 현금"
    if cash_display["secondary_text"]:
        cash_sub += f" · {cash_display['secondary_text']}"
    exposure_sub = exposure_display["secondary_text"] or exposure_display["primary_text"]
    footer_html = f'<div class="acct-mini" style="margin-top:6px;">{html.escape(footer_note)}</div>' if footer_note else ""
    return f'<div class="acct-card {html.escape(css_class)}"><div class="acct-header"><span class="acct-broker-badge">{html.escape(broker)}</span><span class="acct-scope">{html.escape(scope)}</span></div><div class="acct-metrics"><div class="acct-metric"><div class="acct-label">평가 자산</div><div class="acct-value">{html.escape(equity_display["primary_text"])}</div><div class="acct-sub">{html.escape(equity_sub)}</div></div><div class="acct-metric"><div class="acct-label">오늘 손익</div><div class="acct-value sm{daily_class}">{html.escape(daily_display["primary_text"])}</div><div class="acct-sub">{html.escape(daily_sub)}</div></div><div class="acct-metric"><div class="acct-label">예수금</div><div class="acct-value sm">{html.escape(cash_display["primary_text"])}</div><div class="acct-sub">{html.escape(cash_sub)}</div></div></div><hr class="acct-divider"><div class="acct-metrics"><div class="acct-metric"><div class="acct-label">총 익스포저</div><div class="acct-value sm">{html.escape(exposure_display["primary_text"])}</div><div class="acct-mini">{html.escape(exposure_sub)}</div></div><div class="acct-metric"><div class="acct-label">낙폭</div><div class="acct-value warn-c sm">{html.escape(_fmt_pct(drawdown))}</div></div><div class="acct-metric"><div class="acct-label">마지막 스냅샷</div><div class="acct-value sm" style="font-size:11px;font-weight:600;">{html.escape(updated)}</div>{footer_html}</div></div></div>'


def _account_card_compact(
    snapshot: Dict[str, Any],
    fallback_account: Dict[str, Any],
    trade_performance: Dict[str, Any],
    broker: str,
    scope: str,
    css_class: str,
    *,
    currency: str = "KRW",
    quote_snapshots: Dict[str, Dict[str, Any]] | None = None,
    footer_note: str = "",
) -> str:
    quote_snapshots = quote_snapshots or {}
    equity = _f(snapshot.get("equity"))
    cash = _f(snapshot.get("cash"))
    realized_pnl = _f(snapshot.get("daily_pnl"))
    unrealized_pnl = _f(snapshot.get("unrealized_pnl"))
    if not np.isfinite(equity):
        equity = _f(fallback_account.get("equity"))
    if not np.isfinite(cash):
        cash = _f(fallback_account.get("cash"))
    if not np.isfinite(realized_pnl):
        realized_pnl = _f(trade_performance.get("today_pnl"))
    if not np.isfinite(unrealized_pnl):
        unrealized_pnl = 0.0
    current_pnl = realized_pnl + unrealized_pnl if np.isfinite(realized_pnl) else unrealized_pnl
    exposure = abs(_f(snapshot.get("gross_exposure")))
    drawdown = _f(snapshot.get("drawdown_pct"))
    pnl_pct = current_pnl / equity if np.isfinite(equity) and equity > 0 else float("nan")
    daily_class = " up" if np.isfinite(current_pnl) and current_pnl > 0 else " dn" if np.isfinite(current_pnl) and current_pnl < 0 else ""
    updated = _fmt_time(snapshot.get("created_at") or fallback_account.get("created_at"))
    equity_display = _money_value_display(equity, currency, quote_snapshots)
    pnl_display = _money_value_display(current_pnl, currency, quote_snapshots)
    cash_display = _money_value_display(cash, currency, quote_snapshots)
    exposure_display = _money_value_display(exposure, currency, quote_snapshots)
    equity_note = _metric_note(equity_display["primary_text"], f"현재 손익 {_fmt_pct(pnl_pct)}" if np.isfinite(pnl_pct) else None)
    if equity_display["secondary_text"]:
        equity_note += f" · {equity_display['secondary_text']}"
    daily_note = _metric_note(pnl_display["primary_text"], "실현 + 미실현 합산")
    if pnl_display["secondary_text"]:
        daily_note += f" · {pnl_display['secondary_text']}"
    cash_note = _metric_note(cash_display["primary_text"], "진입 가능한 현금")
    if cash_display["secondary_text"]:
        cash_note += f" · {cash_display['secondary_text']}"
    exposure_note = exposure_display["secondary_text"] or exposure_display["primary_text"]
    footer_html = f'<div class="acct-mini" style="margin-top:6px;">{html.escape(footer_note)}</div>' if footer_note else ""
    return f'<div class="acct-card {html.escape(css_class)}"><div class="acct-header"><span class="acct-broker-badge">{html.escape(broker)}</span><span class="acct-scope">{html.escape(scope)}</span></div><div class="acct-metrics"><div class="acct-metric"><div class="acct-label">평가 자산</div><div class="acct-value compact">{html.escape(str(equity_display["compact_text"]))}</div><div class="acct-mini">{html.escape(equity_note)}</div></div><div class="acct-metric"><div class="acct-label">현재 손익</div><div class="acct-value compact{daily_class}">{html.escape(str(pnl_display["compact_text"]))}</div><div class="acct-mini">{html.escape(daily_note)}</div></div><div class="acct-metric"><div class="acct-label">예수금</div><div class="acct-value compact">{html.escape(str(cash_display["compact_text"]))}</div><div class="acct-mini">{html.escape(cash_note)}</div></div></div><hr class="acct-divider"><div class="acct-metrics"><div class="acct-metric"><div class="acct-label">총 익스포저</div><div class="acct-value sm">{html.escape(str(exposure_display["compact_text"]))}</div><div class="acct-mini">{html.escape(exposure_note)}</div></div><div class="acct-metric"><div class="acct-label">낙폭</div><div class="acct-value warn-c sm">{html.escape(_fmt_pct(drawdown))}</div></div><div class="acct-metric"><div class="acct-label">마지막 스냅샷</div><div class="acct-value sm" style="font-size:11px;font-weight:600;">{html.escape(updated)}</div>{footer_html}</div></div></div>'


def _positions_card(
    frame: pd.DataFrame,
    title: str,
    broker_label: str,
    broker_class: str,
    account_equity: float,
    quote_snapshots: Dict[str, Dict[str, Any]],
    kr_asset_types: set[str],
    *,
    kr_symbol_names: Dict[str, str] | None = None,
) -> str:
    rows: list[str] = []
    if frame.empty:
        rows.append('<tr><td colspan="6"><div class="empty-block">보유 포지션이 없습니다.</div></td></tr>')
    else:
        for _, row in frame.head(8).iterrows():
            symbol = str(row.get("symbol") or "")
            icon, market = _market_meta(symbol, str(row.get("asset_type") or ""), kr_asset_types)
            display_symbol = _symbol_display_text(
                symbol,
                asset_type=row.get("asset_type"),
                kr_symbol_names=kr_symbol_names,
                name_only=True,
            )
            quote = _position_quote_snapshot(symbol, str(row.get("asset_type") or ""), quote_snapshots, kr_asset_types)
            currency = "KRW" if icon == "KR" else str(quote.get("currency") or "USD")
            quantity = _f(row.get("quantity"))
            current_price = _f(quote.get("current_price"))
            if not np.isfinite(current_price):
                current_price = _f(row.get("mark_price"))
            entry_price = _f(row.get("entry_price"))
            side = str(row.get("side") or "").upper()
            pnl_value = _f(row.get("unrealized_pnl"))
            pnl_pct = float("nan")
            if quantity > 0 and np.isfinite(current_price) and np.isfinite(entry_price) and entry_price > 0:
                pnl_value = (current_price - entry_price) * quantity if side == "LONG" else (entry_price - current_price) * quantity
                pnl_pct = ((current_price - entry_price) / entry_price) * (1.0 if side == "LONG" else -1.0)
            exposure = abs(_f(row.get("exposure_value")))
            if not np.isfinite(exposure):
                exposure = max(current_price, 0.0) * max(quantity, 0.0)
            exposure_pct = (exposure / max(account_equity, 1.0)) * 100.0 if account_equity > 0 else 0.0
            direction_class = "long" if side == "LONG" else "short"
            pnl_class = "pnl-p" if np.isfinite(pnl_value) and pnl_value > 0 else "pnl-n" if np.isfinite(pnl_value) and pnl_value < 0 else "pos-amt"
            market_value_text, market_value_sub = _money_display_pair(max(current_price, 0.0) * max(quantity, 0.0), currency, quote_snapshots)
            pnl_text, pnl_sub = _money_display_pair(pnl_value, currency, quote_snapshots)
            current_price_text = _fmt_money(current_price, currency)
            stop_loss_text = _optional_money_text(row.get("stop_loss"), currency)
            take_profit_text = _optional_money_text(row.get("take_profit"), currency)
            market_value_sub_html = f'<div class="pos-amt">{html.escape(market_value_sub)}</div>' if market_value_sub else ""
            current_price_html = f'<div class="pos-amt">현재가 {html.escape(current_price_text)}</div>' if current_price_text != "N/A" else ""
            pnl_sub_html = f'<div class="pos-amt">{html.escape(pnl_sub)}</div>' if pnl_sub else ""
            rows.append(f'<tr><td><div class="sym-cell"><div class="sym-icon">{html.escape(icon)}</div><div><div class="sym-name">{html.escape(display_symbol)}</div><div class="sym-mkt">{html.escape(market)}</div></div></div></td><td><div class="side-cell">{_side_badge(side)}<div class="exp-bar-wrap"><div class="exp-bar {direction_class}" style="width:{min(max(exposure_pct, 0.0), 100.0):.0f}%"></div></div><span style="font-size:10px;color:var(--text3);">{exposure_pct:.0f}%</span></div></td><td><div class="pos-qty">{html.escape(_fmt_qty(quantity))}</div></td><td><div class="pos-qty">{html.escape(market_value_text)}</div>{market_value_sub_html}{current_price_html}</td><td style="font-size:11px;color:var(--text3);">{html.escape(stop_loss_text)} / {html.escape(take_profit_text)}</td><td><span class="{pnl_class}">{html.escape(pnl_text)}</span>{pnl_sub_html}<div class="pos-amt">{html.escape(_fmt_pct(pnl_pct))}</div></td></tr>')
    return '<div class="card"><div class="card-hd"><div style="display:flex;align-items:center;gap:8px;"><span class="broker-tag ' + html.escape(broker_class) + '">' + html.escape(broker_label) + '</span><div class="card-title">' + html.escape(title) + '</div></div><div class="card-meta">' + str(len(frame)) + ' 포지션</div></div><table class="pos-table"><thead><tr><th>종목</th><th>방향 · 익스포저</th><th>수량</th><th>평가금액</th><th>손절 / 익절</th><th>미실현 손익</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def _candidate_table_html(
    frame: pd.DataFrame,
    *,
    kr_asset_types: set[str],
    kr_symbol_names: Dict[str, str],
    empty_text: str,
    limit: int = 12,
) -> str:
    rows: list[str] = []
    for _, row in _sort_frame(frame).head(limit).iterrows():
        symbol = str(row.get("symbol") or "-")
        asset_type = str(row.get("asset_type") or "")
        signal = str(row.get("signal") or "FLAT")
        confidence = _f(row.get("confidence"))
        confidence_pct = confidence * 100.0 if np.isfinite(confidence) else 0.0
        score = _f(row.get("score"))
        fill_class = " warn-fill" if confidence_pct < 55 else ""
        rows.append(
            "<tr>"
            + "<td>"
            + _candidate_symbol_markup(
                symbol,
                asset_type,
                kr_asset_types=kr_asset_types,
                kr_symbol_names=kr_symbol_names,
            )
            + "</td>"
            + f"<td>{_side_badge(signal)}</td>"
            + f'<td class="cand-ret {"dn" if signal.upper() == "SHORT" else "up"}">{html.escape(_fmt_pct(row.get("expected_return"), ratio=True))}</td>'
            + f'<td><span class="cand-conf">{html.escape(f"{confidence:.2f}" if np.isfinite(confidence) else "-")}</span><div class="conf-bar"><div class="conf-fill{fill_class}" style="width:{min(max(confidence_pct, 0.0), 100.0):.0f}%"></div></div></td>'
            + f'<td class="cand-score">{html.escape(f"{score:.2f}" if np.isfinite(score) else "-")}</td>'
            + "</tr>"
        )
    if not rows:
        rows.append(f'<tr><td colspan="5"><div class="empty-block">{html.escape(empty_text)}</div></td></tr>')
    return '<div class="cand-table-wrap"><table class="cand-table"><thead><tr><th>종목</th><th>방향</th><th>기대 수익</th><th>신뢰도</th><th>점수</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def _candidate_tabs_html(
    candidate_scans: pd.DataFrame,
    *,
    kr_asset_types: set[str],
    kr_symbol_names: Dict[str, str],
    current_tab: str | None,
    jobs: str | None,
) -> str:
    if candidate_scans.empty:
        bucket_frames = {"kr": candidate_scans, "us": candidate_scans, "crypto": candidate_scans}
    else:
        bucket_keys = candidate_scans.apply(
            lambda row: _candidate_bucket(str(row.get("symbol") or ""), str(row.get("asset_type") or ""), kr_asset_types),
            axis=1,
        )
        bucket_frames = {
            "kr": candidate_scans.loc[bucket_keys == "kr"].copy(),
            "us": candidate_scans.loc[bucket_keys == "us"].copy(),
            "crypto": candidate_scans.loc[bucket_keys == "crypto"].copy(),
        }

    tab_specs = (
        ("kr", "국내주식", "오늘 국내주식 후보가 없습니다."),
        ("us", "해외주식", "오늘 해외주식 후보가 없습니다."),
        ("crypto", "코인", "오늘 코인 후보가 없습니다."),
    )
    fallback_key = next((key for key, _, _ in tab_specs if not bucket_frames[key].empty), "kr")
    active_key = current_tab if current_tab in {key for key, _, _ in tab_specs} else fallback_key
    tab_buttons = []
    tab_panes = []
    for key, label, empty_text in tab_specs:
        count = len(bucket_frames[key])
        active_class = " active" if key == active_key else ""
        tab_buttons.append(
            f'<button type="button" class="cand-tab{active_class}" data-cand-tab="{html.escape(key)}">'
            + html.escape(label)
            + f'<span class="cand-tab-count">{count}</span></button>'
        )
        tab_panes.append(
            f'<div class="cand-pane{active_class}" data-cand-pane="{html.escape(key)}">'
            + _candidate_table_html(
                bucket_frames[key],
                kr_asset_types=kr_asset_types,
                kr_symbol_names=kr_symbol_names,
                empty_text=empty_text,
            )
            + "</div>"
        )
    return '<div class="cand-tabs">' + "".join(tab_buttons) + '</div><div class="cand-panes">' + "".join(tab_panes) + "</div>"


def _frame_for_account(frame: pd.DataFrame, account_id: str, *, column: str = "account_id") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(frame.columns))
    candidate_columns = [column]
    if column != "execution_account_id":
        candidate_columns.append("execution_account_id")
    for candidate_column in candidate_columns:
        if candidate_column in frame.columns:
            mask = frame[candidate_column].fillna("").astype(str) == str(account_id)
            return frame.loc[mask].copy()
    return pd.DataFrame(columns=list(frame.columns))


def _build_status_strip_html(
    *,
    auto_trading_status: Dict[str, Any],
    kis_account_snapshot: Dict[str, Any],
    kis_runtime: Dict[str, Any],
    broker_sync_errors: pd.DataFrame,
    scan_time: str,
) -> str:
    worker_state = str(auto_trading_status.get("state") or "").lower()
    entry_label = "일시중지" if worker_state == "paused" else "가동 중"
    worker_dot = "sdot-green" if worker_state == "running" else "sdot-yellow" if worker_state == "paused" else "sdot-red"
    kis_dot = "sdot-green" if kis_account_snapshot or kis_runtime.get("last_broker_account_sync") else "sdot-gray"
    error_dot = "sdot-red" if len(broker_sync_errors) else "sdot-gray"
    entry_dot = "sdot-yellow" if worker_state == "paused" else "sdot-green"
    now_text = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
    return (
        f'<div id="beta-live-status-strip" class="status-strip">'
        f'<div class="strip-item"><span class="sdot {worker_dot}"></span><span>워커 {html.escape(str(auto_trading_status.get("label") or auto_trading_status.get("state") or "-"))}</span></div>'
        f'<div class="strip-item"><span class="sdot {kis_dot}"></span><span>KIS {"연결됨 · 모의" if kis_account_snapshot or kis_runtime.get("last_broker_account_sync") else "확인 필요"}</span></div>'
        f'<div class="strip-item"><span class="sdot sdot-green"></span><span>시그널 스캔 {html.escape(scan_time)}</span></div>'
        f'<div class="strip-item"><span class="sdot {error_dot}"></span><span>브로커 오류 {len(broker_sync_errors)}건</span></div>'
        f'<div class="strip-item"><span class="sdot {entry_dot}"></span><span>진입 {entry_label}</span></div>'
        f'<span class="strip-time">{html.escape(now_text)}</span>'
        "</div>"
    )


def _build_account_row_html(
    *,
    accounts_overview: Dict[str, Dict[str, Any]],
    fallback_account: Dict[str, Any],
    quote_snapshots: Dict[str, Dict[str, Any]],
) -> str:
    account_cards: list[str] = []
    for account_id, broker_label, scope_label, css_class, default_currency in ACCOUNT_VIEW_SPECS:
        account_row = dict(accounts_overview.get(account_id) or {})
        snapshot = dict(account_row.get("latest_snapshot") or {})
        for key in ("equity", "cash", "gross_exposure", "net_exposure", "realized_pnl", "unrealized_pnl", "drawdown_pct"):
            if key not in snapshot and key in account_row:
                snapshot[key] = account_row.get(key)
        if "daily_pnl" not in snapshot and "realized_pnl" in account_row:
            snapshot["daily_pnl"] = account_row.get("realized_pnl")
        currency = str(account_row.get("currency") or default_currency or "KRW")
        sync_status = _status_label(account_row.get("last_sync_status") or "never")
        sync_time = _fmt_time(account_row.get("last_sync_time") or snapshot.get("created_at"))
        footer_note = f"동기화 {sync_status} · {sync_time}"
        account_cards.append(
            _account_card_compact(
                snapshot,
                fallback_account,
                account_row.get("trade_performance", {}),
                broker_label,
                scope_label,
                css_class,
                currency=currency,
                quote_snapshots=quote_snapshots,
                footer_note=footer_note,
            )
        )
    return f'<div id="beta-live-account-row" class="account-row">{"".join(account_cards)}</div>'


def _build_positions_region_html(
    *,
    open_positions: pd.DataFrame,
    accounts_overview: Dict[str, Dict[str, Any]],
    quote_snapshots: Dict[str, Dict[str, Any]],
    kr_asset_types: set[str],
    kr_symbol_names: Dict[str, str],
) -> str:
    kis_account = dict(accounts_overview.get(ACCOUNT_KIS_KR_PAPER) or {})
    us_account = dict(accounts_overview.get(ACCOUNT_SIM_US_EQUITY) or {})
    crypto_account = dict(accounts_overview.get(ACCOUNT_SIM_CRYPTO) or {})
    kis_account_snapshot = dict(kis_account.get("latest_snapshot") or {})
    us_account_snapshot = dict(us_account.get("latest_snapshot") or {})
    crypto_account_snapshot = dict(crypto_account.get("latest_snapshot") or {})
    kis_positions = _frame_for_account(open_positions, ACCOUNT_KIS_KR_PAPER)
    us_positions = _frame_for_account(open_positions, ACCOUNT_SIM_US_EQUITY)
    crypto_positions = _frame_for_account(open_positions, ACCOUNT_SIM_CRYPTO)
    return (
        '<div id="beta-live-positions">'
        + _positions_card(
            kis_positions,
            "한국주식 포지션",
            "KIS",
            "broker-kis",
            max(_f(kis_account_snapshot.get("equity")), 0.0),
            quote_snapshots,
            kr_asset_types,
            kr_symbol_names=kr_symbol_names,
        )
        + _positions_card(
            us_positions,
            "미국주식 포지션",
            "SIM",
            "broker-sim",
            max(_f(us_account_snapshot.get("equity")), 0.0),
            quote_snapshots,
            kr_asset_types,
            kr_symbol_names=kr_symbol_names,
        )
        + _positions_card(
            crypto_positions,
            "코인 포지션",
            "SIM",
            "broker-sim",
            max(_f(crypto_account_snapshot.get("equity")), 0.0),
            quote_snapshots,
            kr_asset_types,
            kr_symbol_names=kr_symbol_names,
        )
        + "</div>"
    )


def build_beta_live_payload(
    *,
    data: Dict[str, Any],
    accounts_overview: Dict[str, Dict[str, Any]],
    quote_snapshots: Dict[str, Dict[str, Any]],
    kr_asset_types: set[str],
    krx_name_map: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    summary = data["summary"]
    fallback_account = dict(summary.get("latest_account") or {})
    auto_trading_status = data.get("auto_trading_status", {})
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    job_health = data.get("job_health", pd.DataFrame())
    open_positions = data.get("open_positions", pd.DataFrame())
    kr_symbol_names = _build_kr_symbol_name_map(krx_name_map)
    kis_account_snapshot = dict((accounts_overview.get(ACCOUNT_KIS_KR_PAPER) or {}).get("latest_snapshot") or {})
    scan_time = _latest_scan_time(job_health)
    return {
        "version": str(pd.Timestamp.utcnow().value),
        "status_strip_html": _build_status_strip_html(
            auto_trading_status=auto_trading_status,
            kis_account_snapshot=kis_account_snapshot,
            kis_runtime=kis_runtime,
            broker_sync_errors=broker_sync_errors,
            scan_time=scan_time,
        ),
        "account_row_html": _build_account_row_html(
            accounts_overview=accounts_overview,
            fallback_account=fallback_account,
            quote_snapshots=quote_snapshots,
        ),
        "positions_html": _build_positions_region_html(
            open_positions=open_positions,
            accounts_overview=accounts_overview,
            quote_snapshots=quote_snapshots,
            kr_asset_types=kr_asset_types,
            kr_symbol_names=kr_symbol_names,
        ),
    }


def render_beta_live_payload_host(payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<div style="display:none!important;visibility:hidden!important;height:0;overflow:hidden" aria-hidden="true">'
        f'<script id="{html.escape(BETA_LIVE_PAYLOAD_ELEMENT_ID)}" type="application/json">{payload_json}</script>'
        "</div>"
    )


def render_beta_overview_component(
    *,
    data: Dict[str, Any],
    theme_mode: str,
    initial_anchor: str,
    feedback: Dict[str, Any] | None,
    accounts_overview: Dict[str, Dict[str, Any]],
    total_portfolio_overview: Dict[str, Any],
    quote_snapshots: Dict[str, Dict[str, Any]],
    kr_asset_types: set[str],
    recent_orders: pd.DataFrame,
    krx_name_map: Dict[str, Dict[str, str]] | None = None,
    current_candidate_tab: str = "",
    jobs_expanded: bool = False,
    signals_expanded: bool = False,
) -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8", errors="ignore")
    template = re.sub(r'<html[^>]*data-theme="[^"]+"', f'<html lang="ko" data-theme="{theme_mode}"', template, count=1)
    template = _ensure_template_base_href(template, "/")
    template = template.replace("</style>", _extra_styles() + "</style>", 1)

    summary = data["summary"]
    fallback_account = dict(summary.get("latest_account") or {})
    trade_performance = total_portfolio_overview.get("trade_performance", data.get("trade_performance", {}))
    auto_trading_status = data.get("auto_trading_status", {})
    execution_summary = data.get("execution_summary", {})
    broker_sync_status = data.get("broker_sync_status", pd.DataFrame())
    broker_sync_errors = data.get("broker_sync_errors", pd.DataFrame())
    kis_runtime = data.get("kis_runtime", {})
    runtime_profile = data.get("runtime_profile", {})
    job_health = data.get("job_health", pd.DataFrame())
    recent_errors = data.get("recent_errors", pd.DataFrame())
    recent_events = data.get("recent_events", pd.DataFrame())
    open_positions = data.get("open_positions", pd.DataFrame())
    open_orders = data.get("open_orders", pd.DataFrame())
    recent_realized_trades = data.get("recent_realized_trades", pd.DataFrame())
    kr_strategy_overview = data.get("kr_strategy_overview", pd.DataFrame())
    kr_strategy_recent_events = data.get("kr_strategy_recent_events", pd.DataFrame())
    candidate_scans = data.get("candidate_scans", pd.DataFrame())
    prediction_report = data.get("prediction_report", pd.DataFrame())
    equity_curve = data.get("equity_curve", pd.DataFrame())
    today_execution_events = data.get("today_execution_events", pd.DataFrame())
    asset_overview = data.get("asset_overview", pd.DataFrame())
    noop_breakdown = execution_summary.get("today_noop_breakdown", pd.DataFrame())
    kr_symbol_names = _build_kr_symbol_name_map(krx_name_map)
    jobs_query_value = "all" if jobs_expanded else None
    signals_query_value = "all" if signals_expanded else None
    action_token = str(pd.Timestamp.utcnow().value)
    kis_account = dict(accounts_overview.get(ACCOUNT_KIS_KR_PAPER) or {})
    us_account = dict(accounts_overview.get(ACCOUNT_SIM_US_EQUITY) or {})
    crypto_account = dict(accounts_overview.get(ACCOUNT_SIM_CRYPTO) or {})
    kis_account_snapshot = dict(kis_account.get("latest_snapshot") or {})
    us_account_snapshot = dict(us_account.get("latest_snapshot") or {})
    crypto_account_snapshot = dict(crypto_account.get("latest_snapshot") or {})

    broker_sync_rows: Dict[str, Dict[str, Any]] = {}
    if not broker_sync_status.empty and "job_name" in broker_sync_status.columns:
        for _, row in broker_sync_status.iterrows():
            broker_sync_rows[str(row.get("job_name") or "")] = row.to_dict()

    nav_html = '<nav class="top-nav"><div class="brand">ALT</div><div class="nav-tabs">' + "".join(_nav_button(label, target, active=index == 0) for index, (label, target) in enumerate(NAV_ITEMS)) + '</div><div class="nav-right">' + _theme_button(theme_mode, initial_anchor or "beta-overview", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value) + "</div></nav>"
    template = _replace_block(template, 'class="top-nav"', nav_html)

    worker_state = str(auto_trading_status.get("state") or "").lower()
    scan_time = _latest_scan_time(job_health)
    template = _replace_block(
        template,
        'class="status-strip"',
        _build_status_strip_html(
            auto_trading_status=auto_trading_status,
            kis_account_snapshot=kis_account_snapshot,
            kis_runtime=kis_runtime,
            broker_sync_errors=broker_sync_errors,
            scan_time=scan_time,
        ),
    )
    template = _replace_block(
        template,
        'class="account-row"',
        _build_account_row_html(
            accounts_overview=accounts_overview,
            fallback_account=fallback_account,
            quote_snapshots=quote_snapshots,
        ),
    )

    stat_items = [("보유 포지션", _i(summary.get("open_positions", 0))), ("대기 주문", _i(summary.get("open_orders", 0))), ("미해결", _i(summary.get("unresolved_predictions", 0))), ("오늘 후보", _i(execution_summary.get("today_candidate_count", 0))), ("진입 허용", _i(execution_summary.get("today_entry_allowed_count", 0))), ("진입 거절", _i(execution_summary.get("today_entry_rejected_count", 0))), ("주문 제출", _i(execution_summary.get("today_submitted_count", 0))), ("체결 완료", _i(execution_summary.get("today_filled_count", 0))), ("브로커 오류", len(broker_sync_errors)), ("프로파일", str(runtime_profile.get("name") or "-")), ("KR 전략", str(runtime_profile.get("kr_active_strategy_labels") or runtime_profile.get("kr_active_strategies") or "-"))]
    stat_html = []
    for index, (label, value) in enumerate(stat_items):
        classes = ["stat-item"]
        if index in {3, 6, 8}:
            classes.append("group-sep")
        if isinstance(value, int):
            value_class = "warn-val" if label in {"미해결", "진입 거절"} and value else "zero" if value == 0 else ""
            stat_value_html = f'<div class="stat-val {value_class}">{value}</div>'
        else:
            stat_value_html = f'<div class="stat-val text-val">{html.escape(str(value))}</div>'
        stat_html.append(f'<div class="{" ".join(classes)}">{stat_value_html}<div class="stat-lbl">{html.escape(label)}</div></div>')
    template = _replace_block(template, 'class="stat-bar"', '<div class="stat-bar">' + "".join(stat_html) + "</div>")

    all_signal_rows = _build_entry_result_rows(
        today_execution_events,
        kr_symbol_names=kr_symbol_names,
        limit=None,
    )
    visible_signal_rows = all_signal_rows if signals_expanded else all_signal_rows[:4]
    signal_count = len(visible_signal_rows)
    signal_meta = f"표시 {signal_count}건"
    if len(all_signal_rows) > signal_count:
        signal_meta = f"표시 {signal_count} / 전체 {len(all_signal_rows)}건"
    if not visible_signal_rows:
        visible_signal_rows.append('<div class="empty-block">오늘 표시할 종목별 진입 결과가 없습니다.</div>')
    signals_toggle_html = ""
    if len(all_signal_rows) > 4:
        signals_href = _beta_href(
            anchor="events",
            candidate_tab=current_candidate_tab or None,
            jobs=jobs_query_value,
            signals=None if signals_expanded else "all",
            theme=theme_mode,
        )
        signals_toggle_html = _beta_link_button(
            css_class="job-more",
            href=signals_href,
            label="접기" if signals_expanded else "상세보기",
        )
    candidate_tabs_html = _candidate_tabs_html(
        candidate_scans,
        kr_asset_types=kr_asset_types,
        kr_symbol_names=kr_symbol_names,
        current_tab=current_candidate_tab or None,
        jobs=jobs_query_value,
    )

    event_rows = []
    for _, row in _sort_frame(recent_events).head(7).iterrows():
        level = str(row.get("level") or "INFO").upper()
        level_class = "li" if level == "INFO" else "lw" if level == "WARNING" else "le"
        event_rows.append(f'<div><span class="lt">{html.escape(_fmt_time(row.get("created_at")))}</span> <span class="{level_class}">[{html.escape(level)}]</span> {html.escape(str(row.get("message") or ""))}</div>')

    positions_region_html = _build_positions_region_html(
        open_positions=open_positions,
        accounts_overview=accounts_overview,
        quote_snapshots=quote_snapshots,
        kr_asset_types=kr_asset_types,
        kr_symbol_names=kr_symbol_names,
    )
    left_html = '<div id="beta-overview" data-section-anchor="beta-overview"></div>' + positions_region_html + '<div class="card" data-section-anchor="events"><div class="card-hd job-card-head"><div class="card-title">진입 처리 결과</div>' + signals_toggle_html + '</div><div class="card-meta">' + html.escape(signal_meta) + '</div><div class="signal-list">' + "".join(visible_signal_rows) + '</div></div>' + '<div class="card" data-section-anchor="candidates"><div class="card-hd"><div class="card-title">오늘 후보 목록</div><div class="card-meta">' + f'{len(candidate_scans)}종목 · 허용 {_i(execution_summary.get("today_entry_allowed_count", 0))} / 거절 {_i(execution_summary.get("today_entry_rejected_count", 0))}' + '</div></div>' + candidate_tabs_html + '</div>' + '<div class="card" data-section-anchor="events"><div class="card-hd"><div class="card-title">실행 이벤트</div><div class="card-meta">최근 7건</div></div><div class="log-body">' + ("".join(event_rows) if event_rows else '<div class="empty-block">최근 이벤트가 없습니다.</div>') + "</div></div>"

    sync_rows = []
    for label, action, job_name in [("계좌", "sync_account", "broker_account_sync"), ("주문", "sync_order", "broker_order_sync"), ("포지션", "sync_position", "broker_position_sync"), ("시세", "sync_market", "broker_market_status")]:
        row = broker_sync_rows.get(job_name, {})
        status = str(row.get("status") or "never").lower()
        dot_class = "s-g" if status in {"completed", "running"} else "s-y" if status in {"queued", "retry", "paused"} else "s-r"
        button_label = "장 상태 확인" if job_name == "broker_market_status" else "재동기화"
        sync_rows.append('<div class="sync-row"><div class="sync-left"><span class="s-dot ' + dot_class + '"></span><span class="sync-lbl">' + html.escape(label) + '</span></div><span class="sync-time">' + html.escape(_fmt_time(row.get("heartbeat_at") or row.get("finished_at") or row.get("started_at"))) + '</span>' + _action_button(button_label, action, "btn-mini", "sync", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + '</div>')

    strategy_rows = []
    for _, row in _sort_frame(kr_strategy_overview).iterrows():
        experimental_badge = '<span class="badge b-warn">experimental</span>' if bool(row.get("experimental")) else '<span class="badge b-ok">stable</span>'
        enabled_badge = '<span class="badge b-ok">활성</span>' if bool(row.get("enabled")) else '<span class="badge b-mod">비활성</span>'
        strategy_rows.append(
            '<div class="sync-row"><div class="sync-left"><span class="sync-lbl">'
            + html.escape(str(row.get("label") or row.get("display_name") or row.get("strategy_id") or "-"))
            + '</span></div><span class="sync-time">'
            + html.escape(
                f"{str(row.get('session_mode') or 'regular')} · "
                f"후보 {_i(row.get('today_candidate_count'))} / 제출 {_i(row.get('today_submitted_count'))} / 체결 {_i(row.get('today_filled_count'))}"
            )
            + "</span>"
            + enabled_badge
            + experimental_badge
            + "</div>"
        )
    if not strategy_rows:
        strategy_rows.append('<div class="empty-block">표시할 KR 전략 상태가 없습니다.</div>')

    worker_tone = "s-g" if worker_state == "running" else "s-y" if worker_state == "paused" else "s-r"
    quote_stream_status = str(kis_runtime.get("quote_stream_status") or "").lower()
    quote_stream_at = _fmt_time(kis_runtime.get("last_websocket_quote_at"))
    quote_stream_badge = "수신중" if quote_stream_status == "connected" else "재연결" if quote_stream_status == "reconnecting" else "대기"
    quote_stream_tone = "b-ok" if quote_stream_status == "connected" else "b-warn"
    total_curve, total_curve_note, total_curve_currency = _build_total_equity_curve(data.get("equity_curves_by_account", {}), quote_snapshots)
    if total_curve.empty:
        total_curve = equity_curve
        total_curve_note = "계좌 스냅샷 참고용"
        total_curve_currency = None
    right_html = '<div class="card" data-section-anchor="beta-overview"><div class="card-hd"><div class="card-title">트레이딩 제어</div></div>' + f'<div class="worker-pill"><span class="s-dot {worker_tone}"></span><span class="wp-label">워커</span><span class="wp-val">{html.escape(str(auto_trading_status.get("label") or auto_trading_status.get("state") or "-"))}</span>{_action_button("재시작", "restart_worker", "wp-restart", "beta-overview", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode)}</div>' + '<div class="ctrl-sect"><div class="ctrl-lbl">진입</div><div class="ctrl-btns">' + _action_button("일시정지", "pause_entries", "btn btn-stop", "beta-overview", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + _action_button("재개", "resume_entries", "btn btn-go", "beta-overview", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + '</div></div><div class="ctrl-sect"><div class="ctrl-lbl">전체 매매</div><div class="ctrl-btns">' + _action_button("전체 정지", "halt_all", "btn btn-stop", "beta-overview", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + '</div></div></div>' + '<div id="sync" class="card" data-section-anchor="sync"><div class="card-hd"><div class="card-title">런타임 상태</div></div><div class="sync-section"><div class="sync-title">동기화</div>' + "".join(sync_rows) + '<div class="sync-row"><div class="sync-left"><span class="s-dot s-g"></span><span class="sync-lbl">시그널 스캔</span></div><span class="sync-time">' + html.escape(scan_time) + '</span>' + _action_button("즉시 스캔", "scan_now", "btn-mini", "sync", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + '</div></div><div class="sync-section"><div class="sync-title">KR 전략</div><div class="card-meta">' + html.escape(str(runtime_profile.get("kr_default_strategy_label") or runtime_profile.get("kr_default_strategy_id") or "-")) + " · 세션 " + html.escape(str(runtime_profile.get("kr_default_strategy_session_mode") or "-")) + "</div>" + "".join(strategy_rows) + '</div><div class="sync-section"><div class="sync-title">KIS 브로커</div><div class="sync-row"><div class="sync-left"><span class="sync-lbl">대기 주문</span></div><span style=\"color:var(--warn);font-weight:700;font-size:12px;margin-right:6px;\">' + str(_i(kis_runtime.get("pending_submitted_orders"))) + '건</span>' + _action_button("주문 확인", "order_check", "btn-mini", "sync", token=action_token, candidate_tab=current_candidate_tab or None, jobs=jobs_query_value, signals=signals_query_value, theme_mode=theme_mode) + '</div>' + f'<div class="sync-row"><div class="sync-left"><span class="sync-lbl">실시간 시세</span></div><span class="sync-time">{html.escape(quote_stream_at)}</span><span class="badge {quote_stream_tone}">{html.escape(quote_stream_badge)}</span></div>' + f'<div class="sync-row"><div class="sync-left"><span class="sync-lbl">API 토큰</span></div><span class="badge {"b-ok" if kis_account_snapshot else "b-warn"}">{"유효" if kis_account_snapshot else "확인 필요"}</span></div>' + f'<div class="sync-row"><div class="sync-left"><span class="sync-lbl">거래 모드</span></div><span class="badge {"b-mod" if kis_account_snapshot else "b-warn"}">{"모의" if kis_account_snapshot else "비활성"}</span></div></div></div>' + '<div class="card" data-section-anchor="beta-overview"><div class="card-hd"><div class="card-title">자산 추이</div><div class="card-meta">계좌 합산 · 최근 24개 구간</div></div>' + _equity_svg(total_curve, theme_mode, legend_currency=total_curve_currency, legend_note=total_curve_note) + "</div>"
    template = _replace_block(template, 'class="content-grid"', '<div class="content-grid"><div class="left-col">' + left_html + '</div><div class="right-col">' + right_html + "</div></div>")

    all_job_rows = []
    for _, row in _sort_frame(job_health).head(16).iterrows():
        status = str(row.get("status") or "never").lower()
        issue_text = str(row.get("error_message") or "").strip()
        fallback_text = issue_text[:24] if issue_text else f"재시도 {_i(row.get('retry_count'))}회"
        issue_class = " has-issue" if issue_text else ""
        all_job_rows.append(f'<div class="job-row{issue_class}"><div class="job-name">{html.escape(_job_label(row.get("job_name")))}</div>{_chip(_status_label(status), _tone(status))}<div class="job-time">{html.escape(_fmt_time(row.get("finished_at") or row.get("started_at") or row.get("scheduled_at")))}</div><div class="job-dur">{html.escape(fallback_text)}</div></div>')
    visible_job_rows = all_job_rows if jobs_expanded else all_job_rows[:5]
    jobs_toggle_html = ""
    if len(all_job_rows) > 5:
        jobs_href = _beta_href(
            anchor="jobs",
            candidate_tab=current_candidate_tab or None,
            jobs=None if jobs_expanded else "all",
            signals=signals_query_value,
            theme=theme_mode,
        )
        jobs_toggle_html = _beta_link_button(
            css_class="job-more",
            href=jobs_href,
            label="접기" if jobs_expanded else "상세보기",
        )

    error_preview = broker_sync_errors if not broker_sync_errors.empty else recent_errors
    error_rows = []
    for _, row in _sort_frame(error_preview).head(5).iterrows():
        error_rows.append('<div class="err-row"><div class="err-head">' + f'<span class="err-sym">{html.escape(str(row.get("component") or "runtime"))}</span>' + f'<span class="err-type">{html.escape(str(row.get("event_type") or "error"))}</span>' + f'<span class="err-time">{html.escape(_fmt_time(row.get("created_at")))}</span>' + '</div><div class="err-msg">' + html.escape(str(row.get("message") or "")) + "</div></div>")
    template = _replace_block(template, 'class="bottom-grid"', '<div class="bottom-grid"><div id="jobs" class="card" data-section-anchor="jobs"><div class="card-hd job-card-head"><div class="card-title">최근 작업 상태</div>' + jobs_toggle_html + '</div><div class="job-list">' + ("".join(visible_job_rows) if visible_job_rows else '<div class="empty-block">최근 작업 이력이 없습니다.</div>') + '</div></div><div class="card" data-section-anchor="errors"><div class="card-hd"><div class="card-title">최근 브로커 오류</div><div class="card-meta">' + _chip(f"{len(broker_sync_errors)}건", "bad" if len(broker_sync_errors) else "ok") + '</div></div><div class="err-list">' + ("".join(error_rows) if error_rows else '<div class="empty-block">최근 오류가 없습니다.</div>') + "</div></div></div>")

    events_section = (
        '<section class="detail-section" id="events" data-section-anchor="events">'
        '<div class="detail-section-title">실행 이벤트</div>'
        '<div class="detail-events-grid">'
        '<div class="detail-events-main">'
        + _detail_card(
            "실행 이벤트 상세",
            _table_html(
                today_execution_events,
                ["created_at", "account_id", "event_type", "component", "level", "message"],
                "실행 이벤트가 없습니다.",
                limit=14,
                kr_symbol_names=kr_symbol_names,
            ),
            note=f"최근 {min(len(today_execution_events), 14)}건",
        )
        + '</div><div class="detail-events-side">'
        + '<div class="detail-anchor-card">'
        + _detail_card(
            "미진입 사유 요약",
            _table_html(noop_breakdown, ["reason", "count"], "미진입 집계가 없습니다.", limit=6),
            note=f"사유 {len(noop_breakdown)}건",
            card_class="detail-card compact",
        )
        + "</div>"
        + '<div id="kr-strategy" class="detail-anchor-card" data-section-anchor="kr-strategy">'
        + _detail_card(
            "KR 전략 집계",
            _table_html(
                kr_strategy_overview,
                ["label", "session_mode", "timeframe", "enabled", "experimental", "today_candidate_count", "today_entry_allowed_count", "today_entry_rejected_count", "today_submit_requested_count", "today_submitted_count", "today_acknowledged_count", "today_pending_fill_count", "today_filled_count", "today_rejected_count", "today_cancelled_count", "today_noop_breakdown"],
                "KR 전략 집계가 없습니다.",
                limit=6,
                kr_symbol_names=kr_symbol_names,
            ),
            note=f"전략 {len(kr_strategy_overview)}개",
            card_class="detail-card compact",
        )
        + "</div>"
        + '<div class="detail-anchor-card">'
        + _detail_card(
            "KR 전략 최근 이벤트",
            _table_html(
                kr_strategy_recent_events,
                ["created_at", "strategy_id", "session_mode", "symbol", "event_type", "reason", "message"],
                "최근 KR 전략 이벤트가 없습니다.",
                limit=8,
                kr_symbol_names=kr_symbol_names,
            ),
            note=f"최근 {min(len(kr_strategy_recent_events), 8)}건",
            card_class="detail-card compact",
        )
        + "</div>"
        + '<div id="assets" class="detail-anchor-card" data-section-anchor="assets">'
        + _detail_card(
            "운영 자산 설정",
            _table_html(asset_overview, [], "자산 설정 정보가 없습니다.", limit=8),
            note=f"자산 {len(asset_overview)}종",
            card_class="detail-card compact",
        )
        + "</div>"
        + '<div id="errors" class="detail-anchor-card" data-section-anchor="errors">'
        + _detail_card(
            "최근 오류",
            _table_html(recent_errors, ["created_at", "component", "event_type", "level", "message"], "최근 오류가 없습니다.", limit=8),
            note=f"최근 {min(len(recent_errors), 8)}건",
            card_class="detail-card compact",
        )
        + "</div></div></div></section>"
    )
    detail_sections = [
        _section_html(
            "positions",
            "보유 현황",
            [
                _detail_card(
                    "최근 주문 활동",
                    _table_html(
                        recent_orders,
                        ["updated_at", "account_id", "symbol", "asset_type", "side", "requested_qty", "filled_qty", "requested_price", "status", "reason"],
                        "최근 주문 활동이 없습니다.",
                        limit=12,
                        kr_symbol_names=kr_symbol_names,
                    ),
                    note=f"최근 {min(len(recent_orders), 12)}건",
                ),
                _detail_card(
                    "대기 주문",
                    _table_html(
                        open_orders,
                        ["updated_at", "account_id", "symbol", "asset_type", "side", "requested_qty", "filled_qty", "requested_price", "status", "reason"],
                        "대기 주문이 없습니다.",
                        limit=12,
                        kr_symbol_names=kr_symbol_names,
                    ),
                    note=f"현재 {_i(summary.get('open_orders', 0))}건",
                ),
                _detail_card(
                    "최근 거래 손익",
                    _table_html(
                        _build_realized_trade_table_frame(
                            recent_realized_trades,
                            quote_snapshots,
                            kr_symbol_names=kr_symbol_names,
                        ),
                        ["종료시각", "계좌", "종목", "방향", "보유기간", "진입가", "청산가", "실현손익", "청산사유"],
                        "최근 실현손익 거래가 없습니다.",
                        limit=12,
                        kr_symbol_names=kr_symbol_names,
                    ),
                    note=f"최근 {min(len(recent_realized_trades), 12)}건",
                ),
            ],
        ),
        events_section,
    ]
    template = template.replace("</div><!-- /main -->", '<div class="detail-stack">' + "".join(detail_sections) + "</div></div><!-- /main -->", 1)

    feedback_html = ""
    if isinstance(feedback, dict) and feedback.get("message"):
        tone_class = "ok" if feedback.get("ok") else "bad"
        feedback_html = f'<div class="feedback-banner {tone_class}">{html.escape(str(feedback.get("message")))}</div>'
    template = template.replace('<div class="main">', f'<div class="main">{feedback_html}', 1)

    detail_frames = [recent_orders, open_orders, recent_realized_trades, today_execution_events, noop_breakdown, asset_overview, recent_errors]
    extra_rows = sum(min(len(frame), 16) for frame in detail_frames if isinstance(frame, pd.DataFrame))
    component_height = min(7200, max(3600, 3200 + extra_rows * 26))

    script_html = f"""
<script>
(() => {{
  const initialAnchor = {json.dumps(initial_anchor or "beta-overview")};
  const serverThemeMode = {json.dumps(theme_mode if theme_mode in {"light", "dark"} else "light")};
  const livePayloadElementId = {json.dumps(BETA_LIVE_PAYLOAD_ELEMENT_ID)};
  const themeStorageKey = "alt-beta-theme";
  let disposed = false;
  let livePayloadVersion = "";
  const root = document.documentElement;
  const actionButtons = Array.from(document.querySelectorAll("[data-beta-href]"));
  const navButtons = Array.from(document.querySelectorAll("[data-nav-target]"));
  const candidateTabs = Array.from(document.querySelectorAll("[data-cand-tab]"));
  const candidatePanes = Array.from(document.querySelectorAll("[data-cand-pane]"));
  const themeToggleButton = document.querySelector("[data-theme-toggle='1']");
  const themeIcon = document.getElementById("theme-icon");
  const themeLabel = document.getElementById("theme-label");
  const navIds = navButtons.map((button) => button.dataset.navTarget).filter(Boolean);
  const pendingTimers = [];
  let resizeObserver = null;
  let parentScrollHandler = null;
  let livePayloadTimer = null;
  const readStoredTheme = () => {{
    try {{
      const storedTheme = window.localStorage.getItem(themeStorageKey);
      if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
    }} catch (error) {{
    }}
    return serverThemeMode;
  }};
  const rewriteCandidateTabOnLinks = (targetKey) => {{
    actionButtons.forEach((link) => {{
      if (link.dataset.betaLink !== "preserve-candidate-tab") return;
      const href = link.getAttribute("data-beta-href");
      if (!href) return;
      const url = new URL(href, "http://beta.local");
      if (targetKey) url.searchParams.set("beta_cand_tab", targetKey);
      else url.searchParams.delete("beta_cand_tab");
      link.setAttribute("data-beta-href", url.pathname + url.search + url.hash);
    }});
  }};
  const rewriteThemeOnLinks = (themeMode) => {{
    actionButtons.forEach((link) => {{
      const href = link.getAttribute("data-beta-href");
      if (!href) return;
      const url = new URL(href, "http://beta.local");
      if (themeMode === "light" || themeMode === "dark") url.searchParams.set("beta_theme", themeMode);
      else url.searchParams.delete("beta_theme");
      link.setAttribute("data-beta-href", url.pathname + url.search + url.hash);
    }});
  }};
  const updateThemeToggle = (themeMode) => {{
    if (!themeToggleButton) return;
    const nextTheme = themeMode === "dark" ? "light" : "dark";
    if (themeIcon) themeIcon.textContent = themeMode === "dark" ? "☀" : "☾";
    if (themeLabel) themeLabel.textContent = themeMode === "dark" ? "라이트" : "다크";
    themeToggleButton.dataset.themeMode = nextTheme;
  }};
  const applyTheme = (themeMode) => {{
    const normalizedTheme = themeMode === "dark" ? "dark" : "light";
    root.setAttribute("data-theme", normalizedTheme);
    document.body.style.colorScheme = normalizedTheme;
    rewriteThemeOnLinks(normalizedTheme);
    updateThemeToggle(normalizedTheme);
  }};
  const cleanupNavigation = () => {{
    disposed = true;
    if (resizeObserver) resizeObserver.disconnect();
    if (parentScrollHandler) {{
      try {{
        window.parent.removeEventListener("scroll", parentScrollHandler);
      }} catch (error) {{
      }}
      parentScrollHandler = null;
    }}
    pendingTimers.forEach((timerId) => clearTimeout(timerId));
    if (livePayloadTimer) {{
      clearInterval(livePayloadTimer);
      livePayloadTimer = null;
    }}
  }};
  const notifyHeight = () => {{
    if (disposed) return;
    const height = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight);
    try {{
      window.parent.postMessage({{ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height }}, "*");
    }} catch (error) {{
      disposed = true;
    }}
  }};
  const navigateParent = (href) => {{
    if (!href) return;
    cleanupNavigation();
    let absoluteUrl = href;
    try {{
      absoluteUrl = new URL(href, window.parent.location.href).toString();
    }} catch (error) {{
    }}
    try {{
      if (window.parent && window.parent.location) {{
        window.parent.location.assign(absoluteUrl);
        return;
      }}
    }} catch (error) {{
    }}
    try {{
      if (window.top && window.top.location) {{
        window.top.location.href = absoluteUrl;
        return;
      }}
    }} catch (error) {{
    }}
    window.location.assign(absoluteUrl);
  }};
  const targetTopInParentViewport = (target) => {{
    if (!target) return null;
    try {{
      if (!window.frameElement || !window.parent) return null;
      const frameRect = window.frameElement.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      return frameRect.top + targetRect.top;
    }} catch (error) {{
      return null;
    }}
  }};
  const scrollParentToTarget = (target, smooth = true) => {{
    if (!target) return false;
    try {{
      if (!window.frameElement || !window.parent || !window.parent.scrollTo) return false;
      const frameRect = window.frameElement.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const currentScrollY = window.parent.scrollY || window.parent.pageYOffset || 0;
      const nextTop = Math.max(currentScrollY + frameRect.top + targetRect.top - 20, 0);
      window.parent.scrollTo({{ top: nextTop, behavior: smooth ? "smooth" : "auto" }});
      return true;
    }} catch (error) {{
      return false;
    }}
  }};
  const activateTab = (targetId) => {{
    navButtons.forEach((button) => button.classList.toggle("active", button.dataset.navTarget === targetId));
  }};
  const activateCandidateTab = (targetKey) => {{
    candidateTabs.forEach((button) => button.classList.toggle("active", button.dataset.candTab === targetKey));
    candidatePanes.forEach((pane) => pane.classList.toggle("active", pane.dataset.candPane === targetKey));
    rewriteCandidateTabOnLinks(targetKey);
    notifyHeight();
  }};
  const readLivePayload = () => {{
    try {{
      if (!window.parent || !window.parent.document) return null;
      const node = window.parent.document.getElementById(livePayloadElementId);
      if (!node) return null;
      const raw = node.textContent || "";
      if (!raw.trim()) return null;
      return JSON.parse(raw);
    }} catch (error) {{
      return null;
    }}
  }};
  const replaceLiveOuterHtml = (elementId, nextHtml) => {{
    if (!nextHtml) return;
    const node = document.getElementById(elementId);
    if (!node) return;
    node.outerHTML = nextHtml;
  }};
  const applyLivePayload = (payload) => {{
    if (!payload || !payload.version || payload.version === livePayloadVersion) return;
    livePayloadVersion = payload.version;
    replaceLiveOuterHtml("beta-live-status-strip", payload.status_strip_html);
    replaceLiveOuterHtml("beta-live-account-row", payload.account_row_html);
    replaceLiveOuterHtml("beta-live-positions", payload.positions_html);
    notifyHeight();
    updateActiveFromScroll();
  }};
  const startLivePayloadPolling = () => {{
    if (livePayloadTimer) clearInterval(livePayloadTimer);
    livePayloadTimer = setInterval(() => {{
      if (disposed) return;
      if (document.visibilityState === "hidden") return;
      applyLivePayload(readLivePayload());
    }}, 1200);
  }};
  actionButtons.forEach((button) => {{
    button.addEventListener("click", (event) => {{
      if (button.dataset.themeToggle === "1") {{
        event.preventDefault();
        const nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        try {{
          window.localStorage.setItem(themeStorageKey, nextTheme);
        }} catch (error) {{
        }}
        applyTheme(nextTheme);
        notifyHeight();
        return;
      }}
      event.preventDefault();
      navigateParent(button.getAttribute("data-beta-href") || "");
    }});
  }});
  navButtons.forEach((button) => {{
    button.addEventListener("click", () => {{
      const target = document.getElementById(button.dataset.navTarget || "");
      if (!target) return;
      activateTab(button.dataset.navTarget || "beta-overview");
      if (!scrollParentToTarget(target, true)) {{
        target.scrollIntoView({{ behavior: "smooth", block: "start" }});
      }}
    }});
  }});
  candidateTabs.forEach((button) => {{
    button.addEventListener("click", (event) => {{
      event.preventDefault();
      const targetKey = button.dataset.candTab || "";
      if (!targetKey) return;
      activateCandidateTab(targetKey);
    }});
  }});
  const updateActiveFromScroll = () => {{
    let active = "beta-overview";
    navIds.forEach((targetId) => {{
      const section = document.getElementById(targetId);
      const parentTop = targetTopInParentViewport(section);
      const localTop = section ? section.getBoundingClientRect().top : null;
      const effectiveTop = parentTop ?? localTop;
      if (typeof effectiveTop === "number" && effectiveTop <= 120) active = targetId;
    }});
    activateTab(active);
  }};
  window.addEventListener("scroll", updateActiveFromScroll, {{ passive: true }});
  window.addEventListener("resize", notifyHeight);
  try {{
    if (window.parent && window.parent !== window) {{
      parentScrollHandler = () => {{
        if (!disposed) updateActiveFromScroll();
      }};
      window.parent.addEventListener("scroll", parentScrollHandler, {{ passive: true }});
    }}
  }} catch (error) {{
    parentScrollHandler = null;
  }}
  if (window.ResizeObserver) {{
    resizeObserver = new ResizeObserver(() => {{
      if (!disposed) notifyHeight();
    }});
    resizeObserver.observe(document.body);
  }}
  window.addEventListener("pagehide", () => {{
    cleanupNavigation();
  }});
  applyTheme(readStoredTheme());
  requestAnimationFrame(() => {{
    const initialCandidateTab = candidateTabs.find((button) => button.classList.contains("active"))?.dataset.candTab || candidateTabs[0]?.dataset.candTab;
    if (initialCandidateTab) activateCandidateTab(initialCandidateTab);
    const target = initialAnchor ? document.getElementById(initialAnchor) : null;
    if (target) {{
      if (!scrollParentToTarget(target, false)) {{
        target.scrollIntoView({{ block: "start" }});
      }}
      activateTab(target.id || "beta-overview");
    }} else {{
      activateTab("beta-overview");
    }}
    applyLivePayload(readLivePayload());
    startLivePayloadPolling();
    notifyHeight();
    updateActiveFromScroll();
  }});
  pendingTimers.push(setTimeout(notifyHeight, 120));
  pendingTimers.push(setTimeout(notifyHeight, 480));
}})();
</script>
"""
    template = _replace_template_script(template, script_html)
    components.html(template, height=component_height, scrolling=False)
