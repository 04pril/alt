from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

KR_EQUITY_ASSET_TYPE = "\ud55c\uad6d\uc8fc\uc2dd"
US_EQUITY_ASSET_TYPE = "\ubbf8\uad6d\uc8fc\uc2dd"
CRYPTO_ASSET_TYPE = "\ucf54\uc778"

ACCOUNT_KIS_KR_PAPER = "kis_kr_paper"
ACCOUNT_SIM_US_EQUITY = "sim_us_equity"
ACCOUNT_SIM_CRYPTO = "sim_crypto"
ACCOUNT_SIM_LEGACY_MIXED = "sim_legacy_mixed"

BROKER_MODE_KIS = "kis_mock"
BROKER_MODE_SIM = "sim"

KR_SYMBOL_SUFFIXES = (".KS", ".KQ")


@dataclass(frozen=True)
class ExecutionAccount:
    account_id: str
    broker_mode: str
    asset_scope: str
    currency: str
    display_name: str


def _normalize_symbol(symbol: str = "") -> str:
    return str(symbol or "").strip().upper()


def is_kr_symbol(symbol: str = "") -> bool:
    normalized_symbol = _normalize_symbol(symbol)
    return bool(
        normalized_symbol
        and (normalized_symbol.endswith(KR_SYMBOL_SUFFIXES) or (normalized_symbol.isdigit() and len(normalized_symbol) == 6))
    )


def is_crypto_symbol(symbol: str = "") -> bool:
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol or "." in normalized_symbol:
        return False
    return "-" in normalized_symbol and any(token in normalized_symbol for token in ("USD", "USDT", "KRW"))


def is_kis_routable_kr_equity(symbol: str = "", asset_type: str = "") -> bool:
    normalized_asset_type = str(asset_type or "").strip()
    if normalized_asset_type == US_EQUITY_ASSET_TYPE:
        return False
    if normalized_asset_type == CRYPTO_ASSET_TYPE:
        return False
    if normalized_asset_type == KR_EQUITY_ASSET_TYPE:
        return is_kr_symbol(symbol)
    return is_kr_symbol(symbol)


def default_broker_accounts() -> list[dict[str, Any]]:
    return [
        {
            "account_id": ACCOUNT_KIS_KR_PAPER,
            "broker_mode": BROKER_MODE_KIS,
            "asset_scope": KR_EQUITY_ASSET_TYPE,
            "currency": "KRW",
            "display_name": "KIS \ud55c\uad6d\uc8fc\uc2dd \ubaa8\uc758\uacc4\uc88c",
            "is_active": 1,
            "metadata_json": json.dumps(
                {
                    "bootstrap_cash_allocation": 0.0,
                    "external_sync_source": "kis_account_sync",
                },
                ensure_ascii=False,
            ),
        },
        {
            "account_id": ACCOUNT_SIM_US_EQUITY,
            "broker_mode": BROKER_MODE_SIM,
            "asset_scope": US_EQUITY_ASSET_TYPE,
            "currency": "USD",
            "display_name": "SIM \ubbf8\uad6d\uc8fc\uc2dd \uacc4\uc88c",
            "is_active": 1,
            "metadata_json": json.dumps({"bootstrap_cash_allocation": 0.5}, ensure_ascii=False),
        },
        {
            "account_id": ACCOUNT_SIM_CRYPTO,
            "broker_mode": BROKER_MODE_SIM,
            "asset_scope": CRYPTO_ASSET_TYPE,
            "currency": "USD",
            "display_name": "SIM \ucf54\uc778 \uacc4\uc88c",
            "is_active": 1,
            "metadata_json": json.dumps({"bootstrap_cash_allocation": 0.5}, ensure_ascii=False),
        },
        {
            "account_id": ACCOUNT_SIM_LEGACY_MIXED,
            "broker_mode": BROKER_MODE_SIM,
            "asset_scope": "legacy_mixed",
            "currency": "KRW",
            "display_name": "SIM Legacy Mixed",
            "is_active": 0,
            "metadata_json": json.dumps({"migration_only": True}, ensure_ascii=False),
        },
    ]


def get_account_metadata(account_id: str) -> Dict[str, Any]:
    for row in default_broker_accounts():
        if row["account_id"] == account_id:
            try:
                return json.loads(str(row.get("metadata_json") or "{}"))
            except Exception:
                return {}
    return {}


def account_scope_for_asset_type(asset_type: str = "") -> str:
    normalized_asset_type = str(asset_type or "").strip()
    if normalized_asset_type == KR_EQUITY_ASSET_TYPE:
        return ACCOUNT_KIS_KR_PAPER
    if normalized_asset_type == CRYPTO_ASSET_TYPE:
        return ACCOUNT_SIM_CRYPTO
    return ACCOUNT_SIM_US_EQUITY


def resolve_execution_account(symbol: str = "", asset_type: str = "", *, kis_enabled: bool) -> ExecutionAccount:
    normalized_asset_type = str(asset_type or "").strip()
    if not _normalize_symbol(symbol):
        if normalized_asset_type == CRYPTO_ASSET_TYPE:
            return ExecutionAccount(
                account_id=ACCOUNT_SIM_CRYPTO,
                broker_mode=BROKER_MODE_SIM,
                asset_scope=CRYPTO_ASSET_TYPE,
                currency="USD",
                display_name="SIM \ucf54\uc778 \uacc4\uc88c",
            )
        if normalized_asset_type == US_EQUITY_ASSET_TYPE:
            return ExecutionAccount(
                account_id=ACCOUNT_SIM_US_EQUITY,
                broker_mode=BROKER_MODE_SIM,
                asset_scope=US_EQUITY_ASSET_TYPE,
                currency="USD",
                display_name="SIM \ubbf8\uad6d\uc8fc\uc2dd \uacc4\uc88c",
            )
    if kis_enabled and is_kis_routable_kr_equity(symbol=symbol, asset_type=asset_type):
        return ExecutionAccount(
            account_id=ACCOUNT_KIS_KR_PAPER,
            broker_mode=BROKER_MODE_KIS,
            asset_scope=KR_EQUITY_ASSET_TYPE,
            currency="KRW",
            display_name="KIS \ud55c\uad6d\uc8fc\uc2dd \ubaa8\uc758\uacc4\uc88c",
        )
    if normalized_asset_type == CRYPTO_ASSET_TYPE or is_crypto_symbol(symbol):
        return ExecutionAccount(
            account_id=ACCOUNT_SIM_CRYPTO,
            broker_mode=BROKER_MODE_SIM,
            asset_scope=CRYPTO_ASSET_TYPE,
            currency="USD",
            display_name="SIM \ucf54\uc778 \uacc4\uc88c",
        )
    return ExecutionAccount(
        account_id=ACCOUNT_SIM_US_EQUITY,
        broker_mode=BROKER_MODE_SIM,
        asset_scope=US_EQUITY_ASSET_TYPE,
        currency="USD",
        display_name="SIM \ubbf8\uad6d\uc8fc\uc2dd \uacc4\uc88c",
    )


def infer_execution_account_id(
    *,
    symbol: str = "",
    asset_type: str = "",
    source: str = "",
    raw_payload: Dict[str, Any] | None = None,
    kis_enabled: bool = True,
    prefer_legacy_sim_snapshot: bool = False,
) -> str:
    payload = raw_payload or {}
    payload_account_id = str(payload.get("account_id") or payload.get("execution_account_id") or "").strip()
    if payload_account_id:
        return payload_account_id
    if str(source or "").strip() == "kis_account_sync":
        return ACCOUNT_KIS_KR_PAPER
    payload_broker = str(payload.get("broker") or "").strip().lower()
    if payload_broker == BROKER_MODE_KIS:
        return ACCOUNT_KIS_KR_PAPER
    if not str(symbol or "").strip() and not str(asset_type or "").strip() and payload_broker == "":
        return ""
    if prefer_legacy_sim_snapshot and not symbol and not asset_type and payload_broker in {"", BROKER_MODE_SIM}:
        return ACCOUNT_SIM_LEGACY_MIXED
    return resolve_execution_account(symbol=symbol, asset_type=asset_type, kis_enabled=kis_enabled).account_id
