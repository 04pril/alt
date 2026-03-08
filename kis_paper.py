from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml


SEOUL_TZ = timezone(timedelta(hours=9))
DATA_DIR = Path(".paper_trading")
TOKEN_CACHE_PATH = DATA_DIR / "kis_token_cache.json"
ORDER_LOG_PATH = DATA_DIR / "orders.jsonl"
EQUITY_LOG_PATH = DATA_DIR / "equity_curve.csv"
MIN_REQUEST_INTERVAL_SEC = 0.35
_LAST_REQUEST_MONOTONIC = 0.0


class KISPaperError(RuntimeError):
    pass


@dataclass(frozen=True)
class KISPaperConfig:
    app_key: str
    app_secret: str
    account_no: str
    product_code: str
    base_url: str
    user_agent: str

    @property
    def is_paper(self) -> bool:
        return "openapivts" in self.base_url.lower()

    @property
    def config_id(self) -> str:
        return "|".join(
            [
                self.base_url.rstrip("/"),
                self.account_no,
                self.product_code,
                self.app_key[-8:],
            ]
        )

    @property
    def account_masked(self) -> str:
        if len(self.account_no) <= 4:
            return self.account_no
        return f"{self.account_no[:3]}***{self.account_no[-2:]}"


@dataclass
class KISPaperSnapshot:
    summary: Dict[str, Any]
    holdings: pd.DataFrame
    raw_summary: Dict[str, Any]


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _pick_numeric(payload: Dict[str, Any], keys: List[str], default: float = float("nan")) -> float:
    for key in keys:
        value = _parse_float(payload.get(key))
        if math.isfinite(value):
            return value
    return default


def extract_kis_code(symbol_or_code: str) -> str:
    raw = str(symbol_or_code).strip().upper()
    if not raw:
        raise KISPaperError("종목코드가 비어 있습니다.")
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if re.fullmatch(r"Q?\d{6,7}", raw):
        return raw
    match = re.search(r"(Q?\d{6,7})", raw)
    if match:
        return match.group(1)
    raise KISPaperError(f"KIS용 종목코드를 해석할 수 없습니다: {symbol_or_code}")


def load_kis_paper_config(config_path: str | Path = "kis_devlp.yaml") -> KISPaperConfig:
    path = Path(config_path)
    if not path.exists():
        raise KISPaperError(f"KIS 설정 파일이 없습니다: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("my_app", "my_sec", "my_acct", "my_prod", "prod") if not raw.get(key)]
    if missing:
        raise KISPaperError(f"kis_devlp.yaml에 필수 키가 없습니다: {', '.join(missing)}")

    cfg = KISPaperConfig(
        app_key=str(raw["my_app"]).strip(),
        app_secret=str(raw["my_sec"]).strip(),
        account_no=str(raw["my_acct"]).strip(),
        product_code=str(raw["my_prod"]).strip(),
        base_url=str(raw["prod"]).strip().rstrip("/"),
        user_agent=str(raw.get("my_agent") or "codex-paper-trader"),
    )
    if not cfg.is_paper:
        raise KISPaperError("kis_devlp.yaml이 모의투자 도메인(openapivts)을 가리키지 않습니다.")
    return cfg


class KISPaperClient:
    def __init__(self, config_path: str | Path = "kis_devlp.yaml"):
        self.config = load_kis_paper_config(config_path=config_path)
        _ensure_data_dir()

    def _throttle(self) -> None:
        global _LAST_REQUEST_MONOTONIC
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_MONOTONIC
        if elapsed < MIN_REQUEST_INTERVAL_SEC:
            time.sleep(MIN_REQUEST_INTERVAL_SEC - elapsed)
        _LAST_REQUEST_MONOTONIC = time.monotonic()

    def _read_token_cache(self) -> Dict[str, Any] | None:
        if not TOKEN_CACHE_PATH.exists():
            return None
        try:
            payload = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
        if payload.get("config_id") != self.config.config_id:
            return None
        expires_at = payload.get("expires_at")
        if not expires_at:
            return None
        try:
            expires_dt = datetime.strptime(str(expires_at), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        if datetime.now() >= expires_dt - timedelta(minutes=5):
            return None
        token = str(payload.get("access_token") or "").strip()
        return payload if token else None

    def _write_token_cache(self, access_token: str, expires_at: str) -> None:
        TOKEN_CACHE_PATH.write_text(
            json.dumps(
                {
                    "config_id": self.config.config_id,
                    "access_token": access_token,
                    "expires_at": expires_at,
                    "updated_at": datetime.now(SEOUL_TZ).isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def get_access_token(self, force_refresh: bool = False) -> str:
        if not force_refresh:
            cached = self._read_token_cache()
            if cached:
                return str(cached["access_token"])

        payload = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }
        headers = {
            "content-type": "application/json; charset=utf-8",
            "accept": "application/json",
            "user-agent": self.config.user_agent,
        }
        response = requests.post(
            f"{self.config.base_url}/oauth2/tokenP",
            headers=headers,
            data=json.dumps(payload),
            timeout=20,
        )
        if response.status_code != 200:
            try:
                body = response.json()
            except Exception:
                body = {}
            raise KISPaperError(
                f"KIS 토큰 발급 실패: HTTP {response.status_code} {body.get('error_description') or response.text}"
            )

        body = response.json()
        token = str(body.get("access_token") or "").strip()
        expires_at = str(body.get("access_token_token_expired") or "").strip()
        if not token or not expires_at:
            raise KISPaperError(f"KIS 토큰 응답이 예상 형식이 아닙니다: {body}")
        self._write_token_cache(access_token=token, expires_at=expires_at)
        return token

    def _headers(self, tr_id: str | None = None, token: str | None = None) -> Dict[str, str]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "accept": "application/json",
            "user-agent": self.config.user_agent,
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }
        if token:
            headers["authorization"] = f"Bearer {token}"
        if tr_id:
            headers["tr_id"] = tr_id
            headers["custtype"] = "P"
        return headers

    def _get_hashkey(self, body: Dict[str, Any], token: str) -> str:
        self._throttle()
        response = requests.post(
            f"{self.config.base_url}/uapi/hashkey",
            headers=self._headers(token=token),
            data=json.dumps(body),
            timeout=20,
        )
        if response.status_code != 200:
            raise KISPaperError(f"KIS hashkey 발급 실패: HTTP {response.status_code} {response.text}")
        payload = response.json()
        hashkey = str(payload.get("HASH") or "").strip()
        if not hashkey:
            raise KISPaperError(f"KIS hashkey 응답이 비어 있습니다: {payload}")
        return hashkey

    def _request(
        self,
        method: str,
        api_path: str,
        tr_id: str,
        *,
        params: Dict[str, Any] | None = None,
        body: Dict[str, Any] | None = None,
        include_hashkey: bool = False,
        force_refresh: bool = False,
    ) -> tuple[Dict[str, Any], requests.structures.CaseInsensitiveDict]:
        token = self.get_access_token(force_refresh=force_refresh)
        headers = self._headers(tr_id=tr_id, token=token)
        if include_hashkey and body:
            headers["hashkey"] = self._get_hashkey(body=body, token=token)

        if method.upper() == "POST":
            self._throttle()
            response = requests.post(
                f"{self.config.base_url}{api_path}",
                headers=headers,
                data=json.dumps(body or {}),
                timeout=20,
            )
        else:
            self._throttle()
            response = requests.get(
                f"{self.config.base_url}{api_path}",
                headers=headers,
                params=params or {},
                timeout=20,
            )

        if response.status_code in {401, 403} and not force_refresh:
            return self._request(
                method,
                api_path,
                tr_id,
                params=params,
                body=body,
                include_hashkey=include_hashkey,
                force_refresh=True,
            )

        if response.status_code != 200:
            try:
                payload = response.json()
                detail = payload.get("msg1") or payload.get("error_description") or response.text
            except Exception:
                detail = response.text
            raise KISPaperError(f"KIS API 실패({api_path}): HTTP {response.status_code} {detail}")

        payload = response.json()
        rt_cd = payload.get("rt_cd")
        if rt_cd not in (None, "0"):
            raise KISPaperError(
                f"KIS API 실패({api_path}): {payload.get('msg_cd') or ''} {payload.get('msg1') or payload}"
            )
        return payload, response.headers

    def get_quote(self, symbol_or_code: str) -> Dict[str, Any]:
        code = extract_kis_code(symbol_or_code)
        payload, _ = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
            },
        )
        output = payload.get("output") or {}
        current_price = _pick_numeric(output, ["stck_prpr"])
        change_amount = _pick_numeric(output, ["prdy_vrss"], default=0.0)
        prev_close = current_price - change_amount if math.isfinite(current_price) else float("nan")
        return {
            "symbol_code": code,
            "market_name": str(output.get("rprs_mrkt_kor_name") or "KRX"),
            "name": str(output.get("bstp_kor_isnm") or ""),
            "current_price": current_price,
            "previous_close": prev_close,
            "change_amount": change_amount,
            "change_pct": _pick_numeric(output, ["prdy_ctrt"], default=0.0),
            "open_price": _pick_numeric(output, ["stck_oprc"]),
            "high_price": _pick_numeric(output, ["stck_hgpr"]),
            "low_price": _pick_numeric(output, ["stck_lwpr"]),
            "volume": _pick_numeric(output, ["acml_vol"]),
            "raw": output,
        }

    def get_daily_history(self, symbol_or_code: str, years: int = 5) -> pd.DataFrame:
        code = extract_kis_code(symbol_or_code)
        end_date = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
        start_date = (end_date - pd.DateOffset(years=max(int(years), 1), days=10)).normalize()
        frames: List[pd.DataFrame] = []
        current_end = end_date

        while current_end >= start_date:
            payload, _ = self._request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": current_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0",
                },
            )
            rows = payload.get("output2") or []
            if not rows:
                break
            frame = pd.DataFrame(rows)
            if frame.empty or "stck_bsop_date" not in frame.columns:
                break
            frame["Date"] = pd.to_datetime(frame["stck_bsop_date"], format="%Y%m%d", errors="coerce")
            frame["Open"] = pd.to_numeric(frame.get("stck_oprc"), errors="coerce")
            frame["High"] = pd.to_numeric(frame.get("stck_hgpr"), errors="coerce")
            frame["Low"] = pd.to_numeric(frame.get("stck_lwpr"), errors="coerce")
            frame["Close"] = pd.to_numeric(frame.get("stck_clpr"), errors="coerce")
            frame["Volume"] = pd.to_numeric(frame.get("acml_vol"), errors="coerce")
            frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
            if frame.empty:
                break
            frames.append(frame[["Date", "Open", "High", "Low", "Close", "Volume"]])
            oldest = frame["Date"].min()
            if pd.isna(oldest):
                break
            next_end = pd.Timestamp(oldest) - pd.Timedelta(days=1)
            if next_end >= current_end:
                break
            current_end = next_end.normalize()

        if not frames:
            raise KISPaperError("KIS 일봉 데이터를 불러오지 못했습니다.")

        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date"]).sort_values("Date")
        merged = merged[merged["Date"] >= start_date]
        merged = merged.set_index("Date")
        merged = merged[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if merged.empty:
            raise KISPaperError("KIS 일봉 데이터가 비어 있습니다.")
        return merged

    def get_account_snapshot(self) -> KISPaperSnapshot:
        all_rows: List[Dict[str, Any]] = []
        summary_raw: Dict[str, Any] = {}
        ctx_fk100 = ""
        ctx_nk100 = ""
        tr_cont = ""

        while True:
            payload, headers = self._request(
                "GET",
                "/uapi/domestic-stock/v1/trading/inquire-balance",
                "VTTC8434R",
                params={
                    "CANO": self.config.account_no,
                    "ACNT_PRDT_CD": self.config.product_code,
                    "AFHR_FLPR_YN": "N",
                    "OFL_YN": "",
                    "INQR_DVSN": "02",
                    "UNPR_DVSN": "01",
                    "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N",
                    "PRCS_DVSN": "00",
                    "CTX_AREA_FK100": ctx_fk100,
                    "CTX_AREA_NK100": ctx_nk100,
                },
            )

            rows = payload.get("output1") or []
            all_rows.extend(rows)
            if not summary_raw:
                summary_list = payload.get("output2") or []
                summary_raw = summary_list[0] if summary_list else {}

            tr_cont = str(headers.get("tr_cont") or "")
            ctx_fk100 = str(payload.get("ctx_area_fk100") or "")
            ctx_nk100 = str(payload.get("ctx_area_nk100") or "")
            if tr_cont not in {"M", "F"}:
                break

        holdings = pd.DataFrame(all_rows)
        if holdings.empty:
            holdings = pd.DataFrame(
                columns=[
                    "symbol_code",
                    "종목명",
                    "보유수량",
                    "매입평균가",
                    "현재가",
                    "평가손익",
                    "수익률(%)",
                    "평가금액",
                    "name",
                    "quantity",
                    "avg_price",
                    "current_price",
                    "unrealized_pnl",
                    "return_pct",
                    "market_value",
                ]
            )
        else:
            holdings["symbol_code"] = holdings.get("pdno", "")
            holdings["종목명"] = holdings.get("prdt_name", "")
            holdings["보유수량"] = pd.to_numeric(holdings.get("hldg_qty", 0), errors="coerce").fillna(0).astype(int)
            holdings["매입평균가"] = pd.to_numeric(holdings.get("pchs_avg_pric", 0), errors="coerce")
            holdings["현재가"] = pd.to_numeric(holdings.get("prpr", 0), errors="coerce")
            holdings["평가손익"] = pd.to_numeric(holdings.get("evlu_pfls_amt", 0), errors="coerce")
            holdings["수익률(%)"] = pd.to_numeric(holdings.get("evlu_pfls_rt", 0), errors="coerce")
            holdings["평가금액"] = pd.to_numeric(holdings.get("evlu_amt", 0), errors="coerce")
            holdings["name"] = holdings["종목명"]
            holdings["quantity"] = holdings["보유수량"]
            holdings["avg_price"] = holdings["매입평균가"]
            holdings["current_price"] = holdings["현재가"]
            holdings["unrealized_pnl"] = holdings["평가손익"]
            holdings["return_pct"] = holdings["수익률(%)"]
            holdings["market_value"] = holdings["평가금액"]
            holdings = holdings.loc[holdings["보유수량"] > 0].reset_index(drop=True)
            holdings = holdings[
                [
                    "symbol_code",
                    "종목명",
                    "보유수량",
                    "매입평균가",
                    "현재가",
                    "평가손익",
                    "수익률(%)",
                    "평가금액",
                    "name",
                    "quantity",
                    "avg_price",
                    "current_price",
                    "unrealized_pnl",
                    "return_pct",
                    "market_value",
                ]
            ]

        cash = _pick_numeric(summary_raw, ["dnca_tot_amt"])
        stock_eval = _pick_numeric(summary_raw, ["scts_evlu_amt", "evlu_amt_smtl_amt"])
        total_eval = _pick_numeric(summary_raw, ["tot_evlu_amt", "nass_amt"])
        pnl = _pick_numeric(summary_raw, ["evlu_pfls_smtl_amt"])
        buy_amount = _pick_numeric(summary_raw, ["pchs_amt_smtl_amt"])
        return_pct = _pick_numeric(summary_raw, ["tot_pftrt"])
        if not math.isfinite(return_pct) and math.isfinite(pnl) and math.isfinite(buy_amount) and buy_amount > 0:
            return_pct = pnl / buy_amount * 100.0

        summary = {
            "account_masked": self.config.account_masked,
            "product_code": self.config.product_code,
            "cash": cash,
            "stock_eval": stock_eval,
            "total_eval": total_eval,
            "pnl": pnl,
            "return_pct": return_pct,
            "buy_amount": buy_amount,
            "holding_count": int(len(holdings)),
            "is_paper": self.config.is_paper,
            "raw": summary_raw,
        }
        return KISPaperSnapshot(summary=summary, holdings=holdings, raw_summary=summary_raw)

    def get_buying_power(
        self,
        symbol_or_code: str,
        *,
        order_price: float,
        order_division: str = "01",
        include_cma: str = "N",
        include_overseas: str = "N",
    ) -> Dict[str, Any]:
        code = extract_kis_code(symbol_or_code)
        payload, _ = self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            "VTTC8908R" if self.config.is_paper else "TTTC8908R",
            params={
                "CANO": self.config.account_no,
                "ACNT_PRDT_CD": self.config.product_code,
                "PDNO": code,
                "ORD_UNPR": str(int(round(float(order_price)))),
                "ORD_DVSN": order_division,
                "CMA_EVLU_AMT_ICLD_YN": include_cma,
                "OVRS_ICLD_YN": include_overseas,
            },
        )
        output = payload.get("output") or {}
        return {
            "symbol_code": code,
            "max_buy_qty": int(_pick_numeric(output, ["max_buy_qty"], default=0.0)),
            "cash_buy_qty": int(_pick_numeric(output, ["nrcvb_buy_qty"], default=0.0)),
            "max_buy_amount": _pick_numeric(output, ["max_buy_amt"], default=0.0),
            "cash_buy_amount": _pick_numeric(output, ["nrcvb_buy_amt"], default=0.0),
            "order_price": float(order_price),
            "order_division": order_division,
            "raw": output,
        }

    def place_cash_order(
        self,
        symbol_or_code: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        price: float | None = None,
    ) -> Dict[str, Any]:
        code = extract_kis_code(symbol_or_code)
        if quantity <= 0:
            raise KISPaperError("주문 수량은 1주 이상이어야 합니다.")
        side = side.lower().strip()
        if side not in {"buy", "sell"}:
            raise KISPaperError("주문 방향은 buy 또는 sell 이어야 합니다.")
        order_type = order_type.lower().strip()
        if order_type not in {"market", "limit"}:
            raise KISPaperError("주문 유형은 market 또는 limit 이어야 합니다.")

        body = {
            "CANO": self.config.account_no,
            "ACNT_PRDT_CD": self.config.product_code,
            "PDNO": code,
            "ORD_DVSN": "01" if order_type == "market" else "00",
            "ORD_QTY": str(int(quantity)),
            "ORD_UNPR": "0" if order_type == "market" else str(int(round(float(price or 0.0)))),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "01" if side == "sell" else "",
            "CNDT_PRIC": "",
        }
        tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
        payload, _ = self._request(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            body=body,
            include_hashkey=True,
        )
        output = payload.get("output") or {}
        return {
            "requested_at": datetime.now(SEOUL_TZ).isoformat(timespec="seconds"),
            "symbol_code": code,
            "side": side,
            "quantity": int(quantity),
            "order_type": order_type,
            "price": None if order_type == "market" else float(price or 0.0),
            "order_no": str(output.get("ODNO") or output.get("odno") or ""),
            "parent_order_no": str(output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno") or ""),
            "message": str(payload.get("msg1") or ""),
            "raw_output": output,
        }


def append_order_log(record: Dict[str, Any]) -> None:
    from prediction_store import append_order_log as store_append_order_log

    store_append_order_log(record)


def load_order_log(limit: int = 200) -> pd.DataFrame:
    from prediction_store import load_order_log as store_load_order_log

    return store_load_order_log(limit=limit)


def append_equity_snapshot(summary: Dict[str, Any], holdings: pd.DataFrame | None = None) -> None:
    from prediction_store import append_equity_snapshot as store_append_equity_snapshot

    store_append_equity_snapshot(summary=summary, holdings=holdings, source="kis_paper")


def load_equity_curve() -> pd.DataFrame:
    from prediction_store import load_equity_curve as store_load_equity_curve

    return store_load_equity_curve()


def compute_equity_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    from prediction_store import compute_equity_metrics as store_compute_equity_metrics

    return store_compute_equity_metrics(equity_curve)
