from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_settings
from runtime_accounts import ACCOUNT_SIM_CRYPTO, ACCOUNT_SIM_US_EQUITY
from storage.models import AccountSnapshotRecord
from storage.repository import TradingRepository, make_id, utc_now_iso

DEFAULT_RESET_KRW = 30_000_000.0
DEFAULT_USD_KRW = 1450.0


def _latest_usdkrw_rate(explicit_rate: float | None = None) -> float:
    if explicit_rate and explicit_rate > 0:
        return float(explicit_rate)
    ticker = yf.Ticker("KRW=X")
    try:
        fast_info = ticker.fast_info or {}
        for key in ("last_price", "regular_market_price", "lastPrice", "regularMarketPrice"):
            value = fast_info.get(key)
            if value:
                return float(value)
    except Exception:
        pass
    try:
        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if not history.empty and "Close" in history.columns:
            close = history["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        pass
    return float(DEFAULT_USD_KRW)


def _reset_account_state(
    repository: TradingRepository,
    *,
    account_id: str,
    reset_amount_krw: float,
    usdkrw_rate: float,
) -> dict[str, float | str]:
    account = repository.get_broker_account(account_id) or {}
    currency = str(account.get("currency") or "KRW").upper()
    if currency == "USD":
        cash = float(reset_amount_krw) / max(float(usdkrw_rate), 1.0)
    else:
        cash = float(reset_amount_krw)

    with repository.connect() as conn:
        conn.execute("DELETE FROM fills WHERE account_id = ?", (account_id,))
        conn.execute("DELETE FROM orders WHERE account_id = ?", (account_id,))
        conn.execute("DELETE FROM positions WHERE account_id = ?", (account_id,))
        conn.execute("DELETE FROM account_snapshots WHERE account_id = ?", (account_id,))

    snapshot = AccountSnapshotRecord(
        snapshot_id=make_id("snap"),
        created_at=utc_now_iso(),
        cash=cash,
        equity=cash,
        gross_exposure=0.0,
        net_exposure=0.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        daily_pnl=0.0,
        drawdown_pct=0.0,
        open_positions=0,
        open_orders=0,
        paused=int(repository.get_control_flag("trading_paused", "0") == "1"),
        source="manual_reset",
        raw_json=json.dumps(
            {
                "account_id": account_id,
                "broker": "sim",
                "reset_base_krw": float(reset_amount_krw),
                "currency": currency,
                "usdkrw_rate": float(usdkrw_rate),
            },
            ensure_ascii=False,
        ),
        account_id=account_id,
    )
    repository.insert_account_snapshot(snapshot)
    repository.log_event(
        "INFO",
        "reset_sim_accounts",
        "manual_reset",
        f"{account_id} reset to base capital",
        {
            "account_id": account_id,
            "currency": currency,
            "cash": cash,
            "reset_base_krw": float(reset_amount_krw),
            "usdkrw_rate": float(usdkrw_rate),
        },
        account_id=account_id,
    )
    return {
        "account_id": account_id,
        "currency": currency,
        "cash": float(cash),
        "reset_base_krw": float(reset_amount_krw),
        "usdkrw_rate": float(usdkrw_rate),
    }


def reset_accounts(account_ids: Iterable[str], *, reset_amount_krw: float, explicit_rate: float | None = None) -> list[dict[str, float | str]]:
    settings = load_settings()
    repository = TradingRepository(settings.storage.db_path)
    repository.initialize()
    usdkrw_rate = _latest_usdkrw_rate(explicit_rate)
    results = []
    for account_id in account_ids:
        results.append(
            _reset_account_state(
                repository,
                account_id=str(account_id),
                reset_amount_krw=float(reset_amount_krw),
                usdkrw_rate=usdkrw_rate,
            )
        )
    repository.set_control_flag("runtime_profile_name", str(settings.profile_name or "active"), "reset_sim_accounts")
    repository.set_control_flag("runtime_profile_source", str(settings.profile_source or Path("config/runtime_settings.json").as_posix()), "reset_sim_accounts")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset SIM execution accounts to a KRW base capital.")
    parser.add_argument("--krw", type=float, default=DEFAULT_RESET_KRW, help="base capital per account in KRW")
    parser.add_argument("--usdkrw", type=float, default=0.0, help="override USD/KRW rate")
    parser.add_argument(
        "--accounts",
        nargs="+",
        default=[ACCOUNT_SIM_US_EQUITY, ACCOUNT_SIM_CRYPTO],
        help="execution account ids to reset",
    )
    args = parser.parse_args()

    results = reset_accounts(
        args.accounts,
        reset_amount_krw=float(args.krw),
        explicit_rate=float(args.usdkrw) if float(args.usdkrw) > 0 else None,
    )
    for row in results:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
