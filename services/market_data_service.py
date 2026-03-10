from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import AssetScheduleConfig, RuntimeSettings
from kr_strategy import latest_completed_bar_close
from predictor import download_kr_price_data

try:  # optional but recommended
    import holidays
except Exception:  # pragma: no cover
    holidays = None


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
        cutoff = latest_completed_bar_close(schedule, timeframe)
        if cutoff is None:
            return frame.iloc[0:0].copy()
        index = pd.to_datetime(frame.index, errors="coerce")
        localized = index.tz_localize(schedule.timezone) if getattr(index, "tz", None) is None else index.tz_convert(schedule.timezone)
        bar_end = localized + pd.Timedelta(minutes=bar_minutes)
        trimmed = frame.loc[bar_end <= cutoff].copy()
        trimmed.index = localized[bar_end <= cutoff]
        return trimmed

    def get_bars(self, symbol: str, asset_type: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        if asset_type == "한국주식" and timeframe == "1d":
            years = max(2, int(np.ceil(lookback_bars / 252.0)) + 1)
            return download_kr_price_data(symbol=symbol, years=years).tail(lookback_bars + 5)

        period = self._period_for(timeframe, lookback_bars)
        frame = yf.download(
            symbol,
            period=period,
            interval=timeframe,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        frame = self._normalize_frame(frame)
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
