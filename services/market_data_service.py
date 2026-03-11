from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config.settings import AssetScheduleConfig, RuntimeSettings
from kr_strategy import latest_completed_bar_close
from predictor import download_kr_price_data, extract_korean_stock_code

try:
    from kis_paper import KISPaperClient
except Exception:  # pragma: no cover
    KISPaperClient = None

try:  # optional but recommended
    import holidays
except Exception:  # pragma: no cover
    holidays = None


NAVER_HEADERS = {"User-Agent": "Mozilla/5.0"}
CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data/v2"
CRYPTOCOMPARE_TIMEOUT_SEC = (3.5, 5.0)
KIS_INTRADAY_TIMEOUT_SEC = 20
MAX_KIS_INTRADAY_DAYS = 25
MAJOR_CRYPTO_YFINANCE_FALLBACK = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "BNB",
    "DOGE",
    "ADA",
    "AVAX",
    "LINK",
    "DOT",
}


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    asset_type: str
    timeframe: str
    price: float
    high: float
    low: float
    open: float
    volume: float
    timestamp: pd.Timestamp


class MarketDataService:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self._kis_client = None
        self._http = requests.Session()

    def schedule(self, asset_type: str) -> AssetScheduleConfig:
        return self.settings.asset_schedules[asset_type]

    def now(self, asset_type: str) -> datetime:
        return datetime.now(ZoneInfo(self.schedule(asset_type).timezone))

    def current_time(self, asset_type: str) -> datetime:
        return self.now(asset_type)

    def _country_holidays(self, country: str, year: int):
        if not country or holidays is None:
            return set()
        try:
            return holidays.country_holidays(country, years=[year])
        except Exception:
            return set()

    def is_holiday(self, asset_type: str, when: datetime) -> bool:
        schedule = self.schedule(asset_type)
        if asset_type == "코인":
            return False
        return when.date() in self._country_holidays(schedule.holiday_country, when.year)

    def is_market_open(self, asset_type: str, when: datetime | None = None) -> bool:
        schedule = self.schedule(asset_type)
        if schedule.session_mode == "always":
            return True
        current = when or self.now(asset_type)
        if current.weekday() >= 5 or self.is_holiday(asset_type, current):
            return False
        start = datetime.combine(current.date(), time.fromisoformat(schedule.market_open), current.tzinfo)
        end = datetime.combine(current.date(), time.fromisoformat(schedule.market_close), current.tzinfo)
        return start <= current <= end

    def is_pre_close_window(self, asset_type: str, when: datetime | None = None) -> bool:
        schedule = self.schedule(asset_type)
        current = when or self.now(asset_type)
        if not self.is_market_open(asset_type, current):
            return False
        close_dt = datetime.combine(current.date(), time.fromisoformat(schedule.market_close), current.tzinfo)
        window_start = close_dt - timedelta(minutes=int(schedule.pre_close_buffer_minutes))
        return window_start <= current <= close_dt

    def market_phase(self, asset_type: str, when: datetime | None = None) -> str:
        schedule = self.schedule(asset_type)
        current = when or self.now(asset_type)
        if schedule.session_mode == "always":
            return "always_open"
        if current.weekday() >= 5 or self.is_holiday(asset_type, current):
            return "holiday"
        if not self.is_market_open(asset_type, current):
            return "closed"
        if self.is_pre_close_window(asset_type, current):
            return "pre_close"
        return "open"

    def _period_for(self, timeframe: str, lookback_bars: int) -> str:
        if timeframe == "1d":
            years = max(2, int(np.ceil(lookback_bars / 252.0)) + 1)
            return f"{years}y"
        if timeframe == "1h":
            days = max(90, int(np.ceil(lookback_bars / 24.0)) + 30)
            return f"{days}d"
        if timeframe == "15m":
            days = min(59, max(30, int(np.ceil(lookback_bars / 26.0)) + 10))
            return f"{days}d"
        days = max(30, lookback_bars)
        return f"{days}d"

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame.columns]
        out = frame[cols].copy()
        out.index = pd.to_datetime(out.index)
        return out.dropna().sort_index()

    def _trim_incomplete_intraday_bars(self, frame: pd.DataFrame, asset_type: str, timeframe: str) -> pd.DataFrame:
        if frame.empty or timeframe not in {"15m", "1h"}:
            return frame
        schedule = self.schedule(asset_type)
        bar_minutes = 60 if timeframe == "1h" else 15
        index = pd.to_datetime(frame.index, errors="coerce")
        localized = index.tz_localize(schedule.timezone) if getattr(index, "tz", None) is None else index.tz_convert(schedule.timezone)
        cutoff = latest_completed_bar_close(schedule, timeframe)
        if cutoff is None:
            current = self.now(asset_type)
            session_open = datetime.combine(current.date(), time.fromisoformat(schedule.market_open), current.tzinfo)
            keep_mask = localized < session_open
            trimmed = frame.loc[keep_mask].copy()
            trimmed.index = localized[keep_mask]
            return trimmed
        bar_end = localized + pd.Timedelta(minutes=bar_minutes)
        trimmed = frame.loc[bar_end <= cutoff].copy()
        trimmed.index = localized[bar_end <= cutoff]
        return trimmed

    def _yfinance_bars(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        period = self._period_for(timeframe, lookback_bars)
        frame = yf.download(
            symbol,
            period=period,
            interval=timeframe,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return self._normalize_frame(frame)

    def _get_kis_client(self):
        if self._kis_client is False:
            return None
        if self._kis_client is None:
            if KISPaperClient is None:
                self._kis_client = False
                return None
            try:
                self._kis_client = KISPaperClient()
            except Exception:
                self._kis_client = False
                return None
        return self._kis_client

    def _parse_kis_minute_rows(self, rows: Iterable[dict]) -> pd.DataFrame:
        parsed_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_text = str(row.get("stck_bsop_date") or "").strip()
            time_text = str(row.get("stck_cntg_hour") or "").strip().zfill(6)
            if len(date_text) != 8 or len(time_text) != 6:
                continue
            try:
                timestamp = pd.Timestamp(f"{date_text}{time_text}", tz="Asia/Seoul")
            except Exception:
                continue
            parsed_rows.append(
                {
                    "Date": timestamp,
                    "Open": float(pd.to_numeric(row.get("stck_oprc"), errors="coerce")),
                    "High": float(pd.to_numeric(row.get("stck_hgpr"), errors="coerce")),
                    "Low": float(pd.to_numeric(row.get("stck_lwpr"), errors="coerce")),
                    "Close": float(pd.to_numeric(row.get("stck_prpr"), errors="coerce")),
                    "Volume": float(pd.to_numeric(row.get("cntg_vol"), errors="coerce")),
                }
            )
        frame = pd.DataFrame(parsed_rows)
        if frame.empty:
            return frame
        frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        if frame.empty:
            return frame
        frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
        return frame[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_kis_intraday_day(self, symbol: str, trading_date: pd.Timestamp) -> pd.DataFrame:
        client = self._get_kis_client()
        if client is None:
            return pd.DataFrame()
        code = extract_korean_stock_code(symbol)
        current_hour = "153000"
        session_open = pd.Timestamp(f"{trading_date:%Y-%m-%d} 09:00:00", tz="Asia/Seoul")
        collected: list[pd.DataFrame] = []
        seen_hours: set[str] = set()

        for _ in range(5):
            payload, _ = client._request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
                "FHKST03010230",
                params={
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_HOUR_1": current_hour,
                    "FID_INPUT_DATE_1": f"{trading_date:%Y%m%d}",
                    "FID_PW_DATA_INCU_YN": "Y",
                    "FID_FAKE_TICK_INCU_YN": "N",
                },
            )
            rows = payload.get("output2") or []
            frame = self._parse_kis_minute_rows(rows)
            if frame.empty:
                break
            collected.append(frame)
            earliest = pd.Timestamp(frame.index.min())
            next_hour = (earliest - pd.Timedelta(minutes=1)).strftime("%H%M%S")
            if next_hour in seen_hours or earliest <= session_open:
                break
            seen_hours.add(next_hour)
            current_hour = next_hour

        if not collected:
            return pd.DataFrame()
        merged = pd.concat(collected).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

    def _resample_intraday_minutes(self, frame: pd.DataFrame, timeframe: str, asset_type: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        schedule = self.schedule(asset_type)
        if timeframe == "1h":
            rule = "60min"
        elif timeframe == "15m":
            rule = "15min"
        else:
            return frame
        session_open = time.fromisoformat(schedule.market_open)
        session_close = time.fromisoformat(schedule.market_close)
        session = frame.between_time(session_open.isoformat(), session_close.isoformat(), inclusive="left")
        if session.empty:
            return session
        aggregated = session.resample(
            rule,
            label="left",
            closed="left",
            origin="start_day",
            offset=f"{session_open.hour}h",
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        aggregated = aggregated.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        return aggregated

    def _get_kr_intraday_bars_from_kis(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        client = self._get_kis_client()
        if client is None:
            return pd.DataFrame()
        target_bars = lookback_bars + 5
        per_day = 26 if timeframe == "15m" else 7
        trading_days_needed = max(3, int(np.ceil(target_bars / float(per_day))) + 2)
        current_day = pd.Timestamp.now(tz="Asia/Seoul").normalize()
        minute_frames: list[pd.DataFrame] = []

        for _ in range(min(trading_days_needed + 6, MAX_KIS_INTRADAY_DAYS)):
            if current_day.weekday() >= 5:
                current_day -= pd.Timedelta(days=1)
                continue
            daily = self._fetch_kis_intraday_day(symbol=symbol, trading_date=current_day)
            if not daily.empty:
                minute_frames.append(daily)
                combined = pd.concat(reversed(minute_frames)).sort_index()
                aggregated = self._resample_intraday_minutes(combined, timeframe=timeframe, asset_type="한국주식")
                if len(aggregated) >= target_bars:
                    return aggregated.tail(target_bars)
            current_day -= pd.Timedelta(days=1)

        if not minute_frames:
            return pd.DataFrame()
        combined = pd.concat(reversed(minute_frames)).sort_index()
        return self._resample_intraday_minutes(combined, timeframe=timeframe, asset_type="한국주식")

    def _get_kr_intraday_bars_from_naver(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        code = extract_korean_stock_code(symbol)
        response = self._http.get(
            "https://fchart.stock.naver.com/siseJson.nhn",
            params={
                "symbol": code,
                "requestType": "0",
                "count": "5000",
                "timeframe": "minute",
            },
            headers=NAVER_HEADERS,
            timeout=KIS_INTRADAY_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = ast.literal_eval(response.text.replace("null", "None").strip())
        rows = []
        for item in payload[1:]:
            if not isinstance(item, list) or len(item) < 6:
                continue
            timestamp = pd.to_datetime(str(item[0]), format="%Y%m%d%H%M", errors="coerce")
            close_price = pd.to_numeric(item[4], errors="coerce")
            volume = pd.to_numeric(item[5], errors="coerce")
            if pd.isna(timestamp) or pd.isna(close_price) or pd.isna(volume):
                continue
            rows.append({"Date": timestamp.tz_localize("Asia/Seoul"), "Close": float(close_price), "CumulativeVolume": float(volume)})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date")
        frame["TradeDate"] = frame["Date"].dt.normalize()
        frame["Volume"] = frame.groupby("TradeDate")["CumulativeVolume"].diff().fillna(frame["CumulativeVolume"])
        frame["Volume"] = frame["Volume"].clip(lower=0.0)
        frame["Open"] = frame["Close"]
        frame["High"] = frame["Close"]
        frame["Low"] = frame["Close"]
        frame = frame.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        aggregated = self._resample_intraday_minutes(frame, timeframe=timeframe, asset_type="한국주식")
        return aggregated.tail(lookback_bars + 5)

    def _crypto_symbol_parts(self, symbol: str) -> tuple[str, str]:
        raw = str(symbol or "").strip().upper()
        if "-" in raw:
            base, quote = raw.split("-", 1)
        else:
            base, quote = raw, "USD"
        return base, quote

    def _get_crypto_bars_from_cryptocompare(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        base, quote = self._crypto_symbol_parts(symbol)
        if quote != "USD":
            return pd.DataFrame()
        if timeframe == "1h":
            endpoint = "histohour"
            params = {"fsym": base, "tsym": quote, "limit": min(max(lookback_bars + 10, 120), 2000)}
        elif timeframe == "15m":
            endpoint = "histominute"
            params = {"fsym": base, "tsym": quote, "aggregate": 15, "limit": min(max(lookback_bars + 10, 180), 2000)}
        else:
            return pd.DataFrame()
        response = self._http.get(
            f"{CRYPTOCOMPARE_BASE_URL}/{endpoint}",
            params=params,
            timeout=CRYPTOCOMPARE_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        rows = ((payload.get("Data") or {}).get("Data") or [])
        parsed_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            timestamp = pd.to_datetime(row.get("time"), unit="s", utc=True, errors="coerce")
            open_price = pd.to_numeric(row.get("open"), errors="coerce")
            high_price = pd.to_numeric(row.get("high"), errors="coerce")
            low_price = pd.to_numeric(row.get("low"), errors="coerce")
            close_price = pd.to_numeric(row.get("close"), errors="coerce")
            volume = pd.to_numeric(row.get("volumefrom"), errors="coerce")
            if pd.isna(timestamp) or pd.isna(close_price):
                continue
            parsed_rows.append(
                {
                    "Date": timestamp,
                    "Open": float(open_price),
                    "High": float(high_price),
                    "Low": float(low_price),
                    "Close": float(close_price),
                    "Volume": float(volume if not pd.isna(volume) else 0.0),
                }
            )
        frame = pd.DataFrame(parsed_rows)
        if frame.empty:
            return frame
        frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return frame.tail(lookback_bars + 5)

    def get_bars(self, symbol: str, asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        if asset_type == "한국주식" and timeframe == "1d":
            years = max(2, int(np.ceil(lookback_bars / 252.0)) + 1)
            return download_kr_price_data(symbol=symbol, years=years).tail(lookback_bars + 5)

        frame = pd.DataFrame()
        if asset_type == "한국주식" and timeframe in {"15m", "1h"}:
            try:
                frame = self._yfinance_bars(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)
            except Exception:
                frame = pd.DataFrame()
            if frame.empty:
                try:
                    frame = self._get_kr_intraday_bars_from_kis(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)
                except Exception:
                    frame = pd.DataFrame()
            if frame.empty:
                try:
                    frame = self._get_kr_intraday_bars_from_naver(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)
                except Exception:
                    frame = pd.DataFrame()
        elif asset_type == "코인" and timeframe in {"15m", "1h"}:
            base, _ = self._crypto_symbol_parts(symbol)
            try:
                frame = self._get_crypto_bars_from_cryptocompare(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)
            except Exception:
                frame = pd.DataFrame()
            if frame.empty and base in MAJOR_CRYPTO_YFINANCE_FALLBACK:
                frame = self._yfinance_bars(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)
        else:
            frame = self._yfinance_bars(symbol=symbol, timeframe=timeframe, lookback_bars=lookback_bars)

        frame = self._trim_incomplete_intraday_bars(frame, asset_type=asset_type, timeframe=timeframe)
        if frame.empty:
            raise ValueError(f"시세 데이터가 비어 있습니다: {symbol} {timeframe}")
        return frame.tail(lookback_bars + 5)

    def latest_quote(self, symbol: str, asset_type: str, timeframe: str) -> MarketQuote:
        frame = self.get_bars(symbol=symbol, asset_type=asset_type, timeframe=timeframe, lookback_bars=5)
        row = frame.iloc[-1]
        return MarketQuote(
            symbol=symbol,
            asset_type=asset_type,
            timeframe=timeframe,
            price=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            open=float(row["Open"]),
            volume=float(row.get("Volume", np.nan)),
            timestamp=pd.Timestamp(frame.index[-1]),
        )

    def validate_bars(self, bars: pd.DataFrame, min_history_bars: int) -> Tuple[bool, str, Dict[str, float]]:
        if bars.empty:
            return False, "empty", {}
        if len(bars) < min_history_bars:
            return False, "insufficient_history", {"bars": float(len(bars))}
        if bars[["Open", "High", "Low", "Close"]].isna().any().any():
            return False, "missing_ohlc", {}
        last_close = pd.to_numeric(bars["Close"], errors="coerce")
        returns = last_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return False, "no_returns", {}
        if abs(float(returns.iloc[-1])) > 0.35:
            return False, "last_bar_outlier", {"last_return": float(returns.iloc[-1])}
        volatility = float(returns.tail(20).std()) if len(returns) >= 20 else float(returns.std())
        median_volume = float(pd.to_numeric(bars.get("Volume"), errors="coerce").tail(20).median())
        liquidity_score = float(np.clip(np.log10(max(median_volume, 1.0)), 0.0, 10.0) / 10.0)
        return True, "ok", {"volatility": volatility, "liquidity_score": liquidity_score, "bars": float(len(bars))}

    def correlation_matrix(
        self,
        symbols: Iterable[str],
        asset_type: str,
        timeframe: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        series_map: Dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                bars = self.get_bars(symbol=symbol, asset_type=asset_type, timeframe=timeframe, lookback_bars=lookback_bars)
            except Exception:
                continue
            returns = pd.to_numeric(bars["Close"], errors="coerce").pct_change().dropna()
            if not returns.empty:
                series_map[symbol] = returns
        if not series_map:
            return pd.DataFrame()
        frame = pd.DataFrame(series_map).dropna(how="all")
        return frame.corr()
