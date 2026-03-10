from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd


def _f(value: object, default: float = float("nan")) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float(default)


def _symbol_aliases(symbol: object) -> tuple[str, ...]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ()
    aliases = [normalized]
    if normalized.endswith(".KS") or normalized.endswith(".KQ"):
        aliases.append(normalized[:-3])
    elif normalized.isdigit() and len(normalized) == 6:
        aliases.extend([f"{normalized}.KS", f"{normalized}.KQ"])
    deduped: list[str] = []
    for alias in aliases:
        if alias and alias not in deduped:
            deduped.append(alias)
    return tuple(deduped)


def resolve_quote_snapshot(symbol: object, quote_snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for alias in _symbol_aliases(symbol):
        snapshot = quote_snapshots.get(alias)
        if snapshot:
            return dict(snapshot)
    return {}


def _frame_for_account(frame: pd.DataFrame, account_id: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(frame.columns))
    for column in ("account_id", "execution_account_id"):
        if column in frame.columns:
            mask = frame[column].fillna("").astype(str) == str(account_id)
            return frame.loc[mask].copy()
    return pd.DataFrame(columns=list(frame.columns))


def build_live_accounts_overview(
    accounts_overview: Dict[str, Dict[str, Any]],
    open_positions: pd.DataFrame,
    quote_snapshots: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    live_accounts: Dict[str, Dict[str, Any]] = {}
    for account_id, payload in (accounts_overview or {}).items():
        account_row = deepcopy(dict(payload or {}))
        snapshot = deepcopy(dict(account_row.get("latest_snapshot") or {}))
        positions = _frame_for_account(open_positions, str(account_id))
        cash = _f(snapshot.get("cash"), _f(account_row.get("cash"), 0.0))
        snapshot_equity = _f(snapshot.get("equity"), _f(account_row.get("equity")))
        snapshot_net_exposure = _f(snapshot.get("net_exposure"), _f(account_row.get("net_exposure")))
        realized_pnl = _f(snapshot.get("daily_pnl"), _f(account_row.get("realized_pnl"), 0.0))
        gross_exposure = 0.0
        net_exposure = 0.0
        unrealized_pnl = 0.0
        live_quotes = 0
        for _, row in positions.iterrows():
            symbol = str(row.get("symbol") or "")
            quote = resolve_quote_snapshot(symbol, quote_snapshots)
            current_price = _f(quote.get("current_price"), _f(row.get("mark_price")))
            entry_price = _f(row.get("entry_price"))
            quantity = _f(row.get("quantity"), 0.0)
            side = str(row.get("side") or "").upper()
            sign = -1.0 if side == "SHORT" else 1.0
            if not np.isfinite(current_price) or quantity <= 0:
                current_price = _f(row.get("mark_price"))
            if np.isfinite(current_price) and quantity > 0:
                live_quotes += 1
                gross_exposure += abs(current_price * quantity)
                net_exposure += sign * current_price * quantity
                if np.isfinite(entry_price):
                    unrealized_pnl += sign * (current_price - entry_price) * quantity
                else:
                    unrealized_pnl += _f(row.get("unrealized_pnl"), 0.0)
            else:
                exposure = _f(row.get("exposure_value"), 0.0)
                gross_exposure += abs(exposure)
                net_exposure += exposure
                unrealized_pnl += _f(row.get("unrealized_pnl"), 0.0)
        equity_anchor = cash
        if np.isfinite(snapshot_equity) and np.isfinite(snapshot_net_exposure):
            equity_anchor = snapshot_equity - snapshot_net_exposure
        equity = equity_anchor + net_exposure
        snapshot["cash"] = cash
        snapshot["equity"] = equity
        snapshot["gross_exposure"] = gross_exposure
        snapshot["net_exposure"] = net_exposure
        snapshot["unrealized_pnl"] = unrealized_pnl
        snapshot.setdefault("daily_pnl", realized_pnl)
        account_row["cash"] = cash
        account_row["equity"] = equity
        account_row["gross_exposure"] = gross_exposure
        account_row["net_exposure"] = net_exposure
        account_row["unrealized_pnl"] = unrealized_pnl
        account_row["latest_snapshot"] = snapshot
        account_row["live_quote_count"] = live_quotes
        live_accounts[str(account_id)] = account_row
    return live_accounts


def build_live_total_portfolio_overview(accounts_overview: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cash_by_currency: Dict[str, float] = {}
    equity_by_currency: Dict[str, float] = {}
    gross_exposure_by_currency: Dict[str, float] = {}
    realized_pnl_by_currency: Dict[str, float] = {}
    unrealized_pnl_by_currency: Dict[str, float] = {}
    for item in (accounts_overview or {}).values():
        currency = str((item or {}).get("currency") or "KRW").upper()
        cash_by_currency[currency] = float(cash_by_currency.get(currency, 0.0) + _f((item or {}).get("cash"), 0.0))
        equity_by_currency[currency] = float(equity_by_currency.get(currency, 0.0) + _f((item or {}).get("equity"), 0.0))
        gross_exposure_by_currency[currency] = float(
            gross_exposure_by_currency.get(currency, 0.0) + abs(_f((item or {}).get("gross_exposure"), 0.0))
        )
        realized_pnl_by_currency[currency] = float(
            realized_pnl_by_currency.get(currency, 0.0) + _f((item or {}).get("realized_pnl"), 0.0)
        )
        unrealized_pnl_by_currency[currency] = float(
            unrealized_pnl_by_currency.get(currency, 0.0) + _f((item or {}).get("unrealized_pnl"), 0.0)
        )
    currencies = [code for code, value in equity_by_currency.items() if abs(float(value)) > 1e-9]
    single_currency = currencies[0] if len(currencies) == 1 else ""
    drawdowns = [_f((item or {}).get("drawdown_pct"), 0.0) for item in (accounts_overview or {}).values()]
    latest_sync_time = max((str((item or {}).get("last_sync_time") or "") for item in (accounts_overview or {}).values()), default="")
    statuses = {str((item or {}).get("last_sync_status") or "never") for item in (accounts_overview or {}).values()}
    if "failed" in statuses:
        sync_status = "failed"
    elif "completed" in statuses:
        sync_status = "completed"
    else:
        sync_status = "never"
    return {
        "cash": float(cash_by_currency.get(single_currency, float("nan"))) if single_currency else float("nan"),
        "equity": float(equity_by_currency.get(single_currency, float("nan"))) if single_currency else float("nan"),
        "cash_by_currency": cash_by_currency,
        "equity_by_currency": equity_by_currency,
        "gross_exposure_by_currency": gross_exposure_by_currency,
        "realized_pnl_by_currency": realized_pnl_by_currency,
        "unrealized_pnl_by_currency": unrealized_pnl_by_currency,
        "display_currency": single_currency,
        "drawdown_pct": float(min(drawdowns) if drawdowns else 0.0),
        "open_positions": int(sum(int((item or {}).get("open_positions", 0) or 0) for item in (accounts_overview or {}).values())),
        "pending_orders": int(sum(int((item or {}).get("pending_orders", 0) or 0) for item in (accounts_overview or {}).values())),
        "last_sync_time": latest_sync_time,
        "last_sync_status": sync_status,
        "warning": (
            "전체 합산 뷰는 참고용이며 주문 가능 잔고 기준이 아닙니다. "
            "달러/원화 혼용 계좌는 불러온 FX 기준으로만 표시해야 합니다."
        ),
    }
