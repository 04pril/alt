from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MIN_TRAIN_DAYS = 120
MIN_VAL_DAYS = 20
MIN_TEST_DAYS = 20
MIN_HOLDOUT_DAYS = 20


@dataclass
class ForecastResult:
    symbol: str
    price_data: pd.DataFrame
    test_frame: pd.DataFrame
    future_frame: pd.DataFrame
    metrics: pd.DataFrame
    weights: Dict[str, float]
    latest_close: float
    trade_backtest: pd.DataFrame
    trade_metrics: pd.DataFrame
    validation_mode: str
    final_holdout_frame: pd.DataFrame
    final_holdout_metrics: pd.DataFrame
    final_holdout_trade_backtest: pd.DataFrame
    final_holdout_trade_metrics: pd.DataFrame
    regime_metrics: pd.DataFrame
    validation_summary: pd.DataFrame
    validation_frame: pd.DataFrame
    validation_metrics: pd.DataFrame


def normalize_symbol(asset_type: str, raw_symbol: str, korea_market: str = "KOSPI") -> str:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        raise ValueError("심볼을 입력해 주세요.")

    if asset_type == "한국주식":
        if symbol.isdigit() and len(symbol) == 6:
            suffix = ".KQ" if korea_market == "KOSDAQ" else ".KS"
            return f"{symbol}{suffix}"
        return symbol
    return symbol


def download_price_data(symbol: str, years: int = 5) -> pd.DataFrame:
    if years < 1:
        raise ValueError("조회 기간(years)은 1 이상이어야 합니다.")

    frame = yf.download(
        symbol,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError("가격 데이터를 불러오지 못했습니다. 심볼을 확인해 주세요.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
    frame = frame[cols].dropna()
    if frame.empty or "Close" not in frame.columns or "Open" not in frame.columns:
        raise ValueError("유효한 시가/종가 데이터가 없습니다.")

    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1 / window, adjust=False).mean()
    down = (-delta.clip(upper=0.0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / down.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _safe_series_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _prepare_ohlcv(price_data: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(index=price_data.index)
    raw["open"] = price_data["Open"].astype(float)
    raw["high"] = price_data["High"].astype(float)
    raw["low"] = price_data["Low"].astype(float)
    raw["close"] = price_data["Close"].astype(float)
    raw["volume"] = price_data["Volume"].astype(float)
    return raw


def _build_feature_frame(raw: pd.DataFrame, lags: int) -> pd.DataFrame:
    close = raw["close"]
    open_ = raw["open"]
    high = raw["high"]
    low = raw["low"]
    volume = raw["volume"]

    feat = pd.DataFrame(index=raw.index)
    feat["open"] = open_
    feat["high"] = high
    feat["low"] = low
    feat["close"] = close
    feat["volume"] = volume

    feat["ret_1"] = close.pct_change()
    feat["sma_5"] = close.rolling(5).mean()
    feat["sma_20"] = close.rolling(20).mean()
    feat["ema_12"] = close.ewm(span=12, adjust=False).mean()
    feat["ema_26"] = close.ewm(span=26, adjust=False).mean()
    feat["macd"] = feat["ema_12"] - feat["ema_26"]
    feat["vol_5"] = feat["ret_1"].rolling(5).std()
    feat["momentum_10"] = _safe_series_div(close, close.shift(10)) - 1.0
    feat["rsi_14"] = _compute_rsi(close, 14)

    feat["hl_range_pct"] = _safe_series_div(high - low, close)
    feat["oc_return"] = _safe_series_div(close - open_, open_)
    feat["gap_pct"] = _safe_series_div(open_, close.shift(1)) - 1.0

    feat["vol_sma_20"] = volume.rolling(20).mean()
    feat["vol_ratio"] = _safe_series_div(volume, feat["vol_sma_20"])
    feat["vol_chg"] = volume.pct_change()
    feat["atr_14"] = _safe_series_div(high - low, close).rolling(14).mean()

    dow = pd.Series(raw.index.dayofweek, index=raw.index, dtype=float)
    month = pd.Series(raw.index.month, index=raw.index, dtype=float)
    feat["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    feat["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    feat["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    for lag in range(1, lags + 1):
        feat[f"lag_{lag}"] = close.shift(lag)

    return feat


def _build_dataset(price_data: pd.DataFrame, lags: int, target_mode: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    raw = _prepare_ohlcv(price_data)
    feat = _build_feature_frame(raw=raw, lags=lags)

    next_close = raw["close"].shift(-1)
    if target_mode == "return":
        target = _safe_series_div(next_close, raw["close"]) - 1.0
    elif target_mode == "price":
        target = next_close
    else:
        raise ValueError("target_mode는 'return' 또는 'price'만 지원합니다.")

    data = feat.copy()
    data["next_close"] = next_close
    data["target"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in data.columns if c not in {"target", "next_close"}]
    return data, feature_cols, raw


def _safe_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.where(np.abs(actual) < 1e-12, np.nan, np.abs(actual))
    return float(np.nanmean(np.abs(actual - pred) / denom) * 100.0)


def _directional_accuracy(current_close: np.ndarray, pred_next: np.ndarray, actual_next: np.ndarray) -> float:
    actual_dir = np.sign(actual_next - current_close)
    pred_dir = np.sign(pred_next - current_close)
    return float((actual_dir == pred_dir).mean() * 100.0)


def _target_to_next_close(target: np.ndarray, current_close: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "return":
        out = current_close * (1.0 + target)
    else:
        out = target
    return np.maximum(out.astype(float), 1e-12)


def _infer_future_index(index: pd.DatetimeIndex, periods: int) -> pd.DatetimeIndex:
    weekend_ratio = float((index.weekday >= 5).mean()) if len(index) else 0.0
    freq = "D" if weekend_ratio > 0.15 else "B"
    return pd.date_range(start=index[-1] + pd.Timedelta(days=1), periods=periods, freq=freq)


def _build_base_models() -> Dict[str, object]:
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2.0, random_state=42))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }


def _fit_cloned_models(base_models: Dict[str, object], x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    fitted: Dict[str, object] = {}
    for name, base_model in base_models.items():
        model = clone(base_model)
        model.fit(x_train, y_train)
        fitted[name] = model
    return fitted


def _predict_targets(fitted_models: Dict[str, object], x: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {name: model.predict(x).astype(float) for name, model in fitted_models.items()}


def _sanitize_split_days(
    dataset_len: int,
    test_days: int,
    final_holdout_days: int,
    validation_days: int,
    gap_days: int,
) -> Tuple[int, int, int]:
    max_final = dataset_len - (MIN_TRAIN_DAYS + MIN_VAL_DAYS + MIN_TEST_DAYS + 2 * gap_days)
    if max_final < MIN_HOLDOUT_DAYS:
        raise ValueError("데이터가 부족해 train/val/test/final_holdout 분리를 만들 수 없습니다. 기간을 늘려 주세요.")

    final_holdout_days = int(max(MIN_HOLDOUT_DAYS, min(final_holdout_days, max_final)))
    research_len = dataset_len - final_holdout_days

    max_test = research_len - (MIN_TRAIN_DAYS + MIN_VAL_DAYS + 2 * gap_days)
    if max_test < MIN_TEST_DAYS:
        raise ValueError("연구구간 test 분리를 만들 수 없습니다. 기간을 늘리거나 홀드아웃을 줄여 주세요.")
    test_days = int(max(MIN_TEST_DAYS, min(test_days, max_test)))

    max_val = research_len - test_days - (MIN_TRAIN_DAYS + 2 * gap_days)
    if max_val < MIN_VAL_DAYS:
        raise ValueError("연구구간 validation 분리를 만들 수 없습니다. 기간을 늘리거나 test/홀드아웃을 줄여 주세요.")
    validation_days = int(max(MIN_VAL_DAYS, min(validation_days, max_val)))

    return test_days, final_holdout_days, validation_days


def _metrics_from_next_close(
    actual_next: np.ndarray,
    current_close: np.ndarray,
    model_pred_next: Dict[str, np.ndarray],
    ensemble_pred_next: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for name, pred in model_pred_next.items():
        rows.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(actual_next, pred)),
                "mape_pct": _safe_mape(actual_next, pred),
                "direction_acc_pct": _directional_accuracy(current_close, pred, actual_next),
            }
        )

    baseline_pred = current_close.copy()
    rows.append(
        {
            "model": "Baseline(RandomWalk)",
            "mae": float(mean_absolute_error(actual_next, baseline_pred)),
            "mape_pct": _safe_mape(actual_next, baseline_pred),
            "direction_acc_pct": _directional_accuracy(current_close, baseline_pred, actual_next),
        }
    )

    rows.append(
        {
            "model": "Ensemble",
            "mae": float(mean_absolute_error(actual_next, ensemble_pred_next)),
            "mape_pct": _safe_mape(actual_next, ensemble_pred_next),
            "direction_acc_pct": _directional_accuracy(current_close, ensemble_pred_next, actual_next),
        }
    )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def _weights_from_validation(
    actual_next: np.ndarray,
    model_pred_next: Dict[str, np.ndarray],
) -> Dict[str, float]:
    inv = {}
    for name, pred in model_pred_next.items():
        mae = float(mean_absolute_error(actual_next, pred))
        inv[name] = 1.0 / max(mae, 1e-12)
    total = float(sum(inv.values()))
    if total <= 1e-12:
        w = 1.0 / max(len(inv), 1)
        return {k: w for k in inv}
    return {k: v / total for k, v in inv.items()}


def _ensemble_from_targets(
    model_target_preds: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    names = list(model_target_preds.keys())
    if not names:
        return np.array([], dtype=float)
    total_weight = float(sum(weights.get(name, 0.0) for name in names))
    if total_weight <= 1e-12:
        w = 1.0 / len(names)
        weights = {name: w for name in names}
    else:
        weights = {name: weights.get(name, 0.0) / total_weight for name in names}

    out = np.zeros_like(model_target_preds[names[0]], dtype=float)
    for name, pred in model_target_preds.items():
        out += weights.get(name, 0.0) * pred
    return out


def _predict_walk_forward_targets(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    start_pos: int,
    base_models: Dict[str, object],
    retrain_every: int,
    gap_days: int,
) -> Dict[str, np.ndarray]:
    if retrain_every < 1:
        raise ValueError("워크포워드 재학습 주기는 1 이상이어야 합니다.")

    pred_buckets: Dict[str, List[float]] = {name: [] for name in base_models}
    fitted: Dict[str, object] = {}
    for step, pos in enumerate(range(start_pos, len(dataset))):
        if (step % retrain_every == 0) or (not fitted):
            train_end = pos - gap_days
            if train_end < MIN_TRAIN_DAYS:
                raise ValueError("워크포워드 학습 데이터가 부족합니다. 검증 기간/갭을 줄이거나 기간을 늘려 주세요.")
            train = dataset.iloc[:train_end]
            fitted = _fit_cloned_models(
                base_models=base_models,
                x_train=train[feature_cols],
                y_train=train["target"],
            )

        row = dataset.iloc[[pos]]
        x_row = row[feature_cols]
        for name, model in fitted.items():
            pred_buckets[name].append(float(model.predict(x_row)[0]))
    return {name: np.array(vals, dtype=float) for name, vals in pred_buckets.items()}


def _trade_metric_rows(
    signal: pd.Series,
    net_return: pd.Series,
    round_trip_cost_bps: float,
    signal_threshold_pct: float,
    allow_short: bool,
) -> List[Dict[str, float]]:
    trade_flag = (signal != 0.0).astype(float)
    trade_returns = net_return[trade_flag > 0].dropna()
    trades = int(len(trade_returns))

    win_returns = trade_returns[trade_returns > 0]
    loss_returns = trade_returns[trade_returns < 0]
    equity_curve = (1.0 + net_return.fillna(0.0)).cumprod()

    win_rate_pct = float((trade_returns > 0).mean() * 100.0) if trades else 0.0
    avg_win_pct = float(win_returns.mean() * 100.0) if len(win_returns) else 0.0
    avg_loss_pct = float(loss_returns.mean() * 100.0) if len(loss_returns) else 0.0
    expectancy_pct = float(trade_returns.mean() * 100.0) if trades else 0.0
    profit_factor = float(win_returns.sum() / abs(loss_returns.sum())) if len(loss_returns) else np.inf
    net_cum_return_pct = float((equity_curve.iloc[-1] - 1.0) * 100.0) if len(equity_curve) else 0.0
    max_drawdown_pct = float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0) if len(equity_curve) else 0.0
    exposure_pct = float(trade_flag.mean() * 100.0) if len(trade_flag) else 0.0

    return [
        {"metric": "trades", "value": float(trades)},
        {"metric": "win_rate_pct", "value": win_rate_pct},
        {"metric": "expectancy_pct", "value": expectancy_pct},
        {"metric": "net_cum_return_pct", "value": net_cum_return_pct},
        {"metric": "max_drawdown_pct", "value": max_drawdown_pct},
        {"metric": "profit_factor", "value": profit_factor},
        {"metric": "exposure_pct", "value": exposure_pct},
        {"metric": "avg_win_pct", "value": avg_win_pct},
        {"metric": "avg_loss_pct", "value": avg_loss_pct},
        {"metric": "round_trip_cost_bps_assumed", "value": float(round_trip_cost_bps)},
        {"metric": "signal_threshold_pct", "value": float(signal_threshold_pct)},
        {"metric": "allow_short", "value": float(1.0 if allow_short else 0.0)},
    ]


def _simulate_trade_backtest(
    price_data: pd.DataFrame,
    eval_frame: pd.DataFrame,
    round_trip_cost_bps: float,
    signal_threshold_pct: float,
    allow_short: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    next_open = price_data["Open"].shift(-1).reindex(eval_frame.index)
    next_close = price_data["Close"].shift(-1).reindex(eval_frame.index)

    predicted_move_pct = (eval_frame["ensemble_pred"] / eval_frame["current_close"] - 1.0) * 100.0
    if allow_short:
        raw_signal = np.where(
            predicted_move_pct > signal_threshold_pct,
            1.0,
            np.where(predicted_move_pct < -signal_threshold_pct, -1.0, 0.0),
        )
    else:
        raw_signal = np.where(predicted_move_pct > signal_threshold_pct, 1.0, 0.0)
    signal = pd.Series(raw_signal, index=eval_frame.index, dtype=float)

    valid = next_open.notna() & next_close.notna()
    signal = signal.where(valid, 0.0)

    day_move = (next_close - next_open) / next_open
    gross_return = signal * day_move
    trade_flag = (signal != 0.0).astype(float)
    net_cost = trade_flag * (round_trip_cost_bps / 10000.0)
    net_return = gross_return - net_cost
    equity_curve = (1.0 + net_return.fillna(0.0)).cumprod()

    backtest_frame = pd.DataFrame(
        index=eval_frame.index,
        data={
            "signal": signal,
            "signal_label": signal.map({1.0: "LONG", -1.0: "SHORT", 0.0: "FLAT"}),
            "predicted_move_pct": predicted_move_pct,
            "next_open": next_open,
            "next_close": next_close,
            "gross_return": gross_return,
            "net_return": net_return,
            "equity_curve": equity_curve,
        },
    )

    metrics = pd.DataFrame(
        _trade_metric_rows(
            signal=backtest_frame["signal"],
            net_return=backtest_frame["net_return"],
            round_trip_cost_bps=round_trip_cost_bps,
            signal_threshold_pct=signal_threshold_pct,
            allow_short=allow_short,
        )
    )
    return backtest_frame, metrics


def _metric_from_frame(metric_df: pd.DataFrame, key: str, default: float = 0.0) -> float:
    row = metric_df.loc[metric_df["metric"] == key, "value"]
    if row.empty:
        return default
    v = float(row.iloc[0])
    if np.isnan(v):
        return default
    return v


def _pick_signal_threshold(
    price_data: pd.DataFrame,
    validation_frame: pd.DataFrame,
    base_threshold_pct: float,
    round_trip_cost_bps: float,
    allow_short: bool,
) -> float:
    candidates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    if base_threshold_pct > 0:
        candidates += [
            base_threshold_pct * 0.5,
            base_threshold_pct,
            base_threshold_pct * 1.5,
            base_threshold_pct * 2.0,
        ]
    candidates = sorted({float(max(0.0, min(3.0, c))) for c in candidates})

    best_t = float(base_threshold_pct)
    best_score = (-np.inf, -np.inf, -np.inf, -np.inf)
    for t in candidates:
        _, m = _simulate_trade_backtest(
            price_data=price_data,
            eval_frame=validation_frame,
            round_trip_cost_bps=round_trip_cost_bps,
            signal_threshold_pct=t,
            allow_short=allow_short,
        )
        expectancy = _metric_from_frame(m, "expectancy_pct")
        net_ret = _metric_from_frame(m, "net_cum_return_pct")
        max_dd = _metric_from_frame(m, "max_drawdown_pct")
        trades = _metric_from_frame(m, "trades")
        trade_bonus = 1.0 if trades >= 5 else 0.0
        score = (trade_bonus, expectancy, net_ret, -abs(max_dd))
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def _volatility_series(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(252) * 100.0


def _regime_thresholds(volatility_pct: pd.Series, reference_index: pd.DatetimeIndex) -> Tuple[float, float]:
    reference = volatility_pct.reindex(reference_index).dropna()
    if len(reference) < 30:
        reference = volatility_pct.dropna()
    if reference.empty:
        return 0.0, 0.0
    low = float(reference.quantile(0.33))
    high = float(reference.quantile(0.67))
    if high < low:
        low, high = high, low
    return low, high


def _regime_breakdown_for_segment(
    segment_name: str,
    backtest_frame: pd.DataFrame,
    volatility_pct: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    part = backtest_frame.copy()
    part["volatility_pct"] = volatility_pct.reindex(part.index)
    part = part.dropna(subset=["volatility_pct"])
    if part.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "regime",
                "days",
                "coverage_pct",
                "trades",
                "win_rate_pct",
                "expectancy_pct",
                "net_cum_return_pct",
                "max_drawdown_pct",
                "avg_signal_abs",
                "avg_volatility_pct",
            ]
        )

    part["regime"] = np.where(
        part["volatility_pct"] <= low_threshold,
        "LowVol",
        np.where(part["volatility_pct"] >= high_threshold, "HighVol", "MidVol"),
    )

    rows = []
    total_days = len(part)
    for regime in ["LowVol", "MidVol", "HighVol"]:
        sl = part[part["regime"] == regime]
        if sl.empty:
            continue
        metric_rows = _trade_metric_rows(
            signal=sl["signal"],
            net_return=sl["net_return"],
            round_trip_cost_bps=0.0,
            signal_threshold_pct=0.0,
            allow_short=True,
        )
        mm = {r["metric"]: r["value"] for r in metric_rows}
        rows.append(
            {
                "segment": segment_name,
                "regime": regime,
                "days": float(len(sl)),
                "coverage_pct": float(len(sl) / total_days * 100.0),
                "trades": float(mm.get("trades", 0.0)),
                "win_rate_pct": float(mm.get("win_rate_pct", 0.0)),
                "expectancy_pct": float(mm.get("expectancy_pct", 0.0)),
                "net_cum_return_pct": float(mm.get("net_cum_return_pct", 0.0)),
                "max_drawdown_pct": float(mm.get("max_drawdown_pct", 0.0)),
                "avg_signal_abs": float(sl["signal"].abs().mean()),
                "avg_volatility_pct": float(sl["volatility_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _feature_row_from_history(history_raw: pd.DataFrame, lags: int, feature_cols: List[str]) -> pd.DataFrame:
    min_len = max(26, 20, lags + 1, 11) + 1
    if len(history_raw) < min_len:
        raise ValueError(f"history 길이가 부족합니다. 필요: {min_len}, 현재: {len(history_raw)}")

    feat = _build_feature_frame(raw=history_raw, lags=lags)
    row = feat.iloc[[-1]][feature_cols].replace([np.inf, -np.inf], np.nan)
    return row.fillna(0.0)


def run_forecast(
    symbol: str,
    years: int = 5,
    test_days: int = 60,
    forecast_days: int = 14,
    lags: int = 20,
    validation_mode: str = "holdout",
    retrain_every: int = 5,
    round_trip_cost_bps: float = 8.0,
    min_signal_strength_pct: float = 0.2,
    final_holdout_days: int = 40,
    purge_days: int = 2,
    embargo_days: int = 1,
    target_mode: str = "return",
    validation_days: int = 40,
    allow_short: bool = False,
) -> ForecastResult:
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days / embargo_days는 0 이상이어야 합니다.")
    if validation_mode not in {"holdout", "walk_forward"}:
        raise ValueError("validation_mode는 'holdout' 또는 'walk_forward'만 지원합니다.")
    if target_mode not in {"return", "price"}:
        raise ValueError("target_mode는 'return' 또는 'price'만 지원합니다.")

    gap_days = int(purge_days + embargo_days)
    price_data = download_price_data(symbol=symbol, years=years)
    dataset, feature_cols, raw_ohlcv = _build_dataset(price_data=price_data, lags=lags, target_mode=target_mode)
    if len(dataset) < 260:
        raise ValueError("데이터가 부족합니다. 조회 기간을 늘려 주세요.")

    test_days, final_holdout_days, validation_days = _sanitize_split_days(
        dataset_len=len(dataset),
        test_days=test_days,
        final_holdout_days=final_holdout_days,
        validation_days=validation_days,
        gap_days=gap_days,
    )

    final_start = len(dataset) - final_holdout_days
    research = dataset.iloc[:final_start]
    test_start = len(research) - test_days
    val_end = test_start - gap_days
    val_start = val_end - validation_days
    train_end = val_start - gap_days
    if train_end < MIN_TRAIN_DAYS:
        raise ValueError("train/validation 분할을 만들 수 없습니다. 기간을 늘려 주세요.")

    train = research.iloc[:train_end]
    val = research.iloc[val_start:val_end]
    test = research.iloc[test_start:]

    base_models = _build_base_models()

    # 1) validation: weights only
    val_models = _fit_cloned_models(base_models=base_models, x_train=train[feature_cols], y_train=train["target"])
    val_pred_target = _predict_targets(val_models, val[feature_cols])
    val_current_close = val["close"].to_numpy(dtype=float)
    val_actual_next = val["next_close"].to_numpy(dtype=float)
    val_pred_next_by_model = {
        name: _target_to_next_close(pred, val_current_close, target_mode) for name, pred in val_pred_target.items()
    }
    weights = _weights_from_validation(actual_next=val_actual_next, model_pred_next=val_pred_next_by_model)
    val_ensemble_target = _ensemble_from_targets(model_target_preds=val_pred_target, weights=weights)
    val_ensemble_next = _target_to_next_close(val_ensemble_target, val_current_close, target_mode)
    validation_metrics = _metrics_from_next_close(
        actual_next=val_actual_next,
        current_close=val_current_close,
        model_pred_next=val_pred_next_by_model,
        ensemble_pred_next=val_ensemble_next,
    )

    validation_frame = pd.DataFrame(
        index=val.index,
        data={
            "current_close": val_current_close,
            "actual_next_close": val_actual_next,
            "ensemble_pred": val_ensemble_next,
            **{f"{name}_pred": pred for name, pred in val_pred_next_by_model.items()},
        },
    )
    validation_frame["predicted_move_pct"] = (validation_frame["ensemble_pred"] / validation_frame["current_close"] - 1.0) * 100.0

    tuned_threshold = _pick_signal_threshold(
        price_data=price_data,
        validation_frame=validation_frame,
        base_threshold_pct=min_signal_strength_pct,
        round_trip_cost_bps=round_trip_cost_bps,
        allow_short=allow_short,
    )

    # 2) research test: no weight tuning here
    if validation_mode == "holdout":
        train_for_test_end = test_start - gap_days
        if train_for_test_end < MIN_TRAIN_DAYS:
            raise ValueError("test 학습 데이터가 부족합니다.")
        train_for_test = research.iloc[:train_for_test_end]
        test_models = _fit_cloned_models(
            base_models=base_models,
            x_train=train_for_test[feature_cols],
            y_train=train_for_test["target"],
        )
        test_pred_target = _predict_targets(test_models, test[feature_cols])
        test_index = test.index
    else:
        test_pred_target = _predict_walk_forward_targets(
            dataset=research,
            feature_cols=feature_cols,
            start_pos=test_start,
            base_models=base_models,
            retrain_every=retrain_every,
            gap_days=gap_days,
        )
        test_index = research.index[test_start:]
        test = research.loc[test_index]

    test_current_close = test["close"].to_numpy(dtype=float)
    test_actual_next = test["next_close"].to_numpy(dtype=float)
    test_pred_next_by_model = {
        name: _target_to_next_close(pred, test_current_close, target_mode) for name, pred in test_pred_target.items()
    }
    test_ensemble_target = _ensemble_from_targets(test_pred_target, weights=weights)
    test_ensemble_next = _target_to_next_close(test_ensemble_target, test_current_close, target_mode)
    metrics = _metrics_from_next_close(
        actual_next=test_actual_next,
        current_close=test_current_close,
        model_pred_next=test_pred_next_by_model,
        ensemble_pred_next=test_ensemble_next,
    )

    test_frame = pd.DataFrame(
        index=test_index,
        data={
            "current_close": test_current_close,
            "actual_next_close": test_actual_next,
            "ensemble_pred": test_ensemble_next,
            **{f"{name}_pred": pred for name, pred in test_pred_next_by_model.items()},
        },
    )
    test_frame["predicted_move_pct"] = (test_frame["ensemble_pred"] / test_frame["current_close"] - 1.0) * 100.0
    trade_backtest, trade_metrics = _simulate_trade_backtest(
        price_data=price_data,
        eval_frame=test_frame,
        round_trip_cost_bps=round_trip_cost_bps,
        signal_threshold_pct=tuned_threshold,
        allow_short=allow_short,
    )

    # 3) final holdout (strictly separated)
    holdout = dataset.iloc[final_start:]
    holdout_train_end = final_start - gap_days
    if holdout_train_end < MIN_TRAIN_DAYS:
        raise ValueError("최종 홀드아웃 학습 데이터가 부족합니다.")
    holdout_train = dataset.iloc[:holdout_train_end]
    holdout_models = _fit_cloned_models(
        base_models=base_models,
        x_train=holdout_train[feature_cols],
        y_train=holdout_train["target"],
    )
    holdout_pred_target = _predict_targets(holdout_models, holdout[feature_cols])
    holdout_current_close = holdout["close"].to_numpy(dtype=float)
    holdout_actual_next = holdout["next_close"].to_numpy(dtype=float)
    holdout_pred_next_by_model = {
        name: _target_to_next_close(pred, holdout_current_close, target_mode) for name, pred in holdout_pred_target.items()
    }
    holdout_ensemble_target = _ensemble_from_targets(holdout_pred_target, weights=weights)
    holdout_ensemble_next = _target_to_next_close(holdout_ensemble_target, holdout_current_close, target_mode)
    final_holdout_metrics = _metrics_from_next_close(
        actual_next=holdout_actual_next,
        current_close=holdout_current_close,
        model_pred_next=holdout_pred_next_by_model,
        ensemble_pred_next=holdout_ensemble_next,
    )
    final_holdout_frame = pd.DataFrame(
        index=holdout.index,
        data={
            "current_close": holdout_current_close,
            "actual_next_close": holdout_actual_next,
            "ensemble_pred": holdout_ensemble_next,
            **{f"{name}_pred": pred for name, pred in holdout_pred_next_by_model.items()},
        },
    )
    final_holdout_frame["predicted_move_pct"] = (
        final_holdout_frame["ensemble_pred"] / final_holdout_frame["current_close"] - 1.0
    ) * 100.0

    final_holdout_trade_backtest, final_holdout_trade_metrics = _simulate_trade_backtest(
        price_data=price_data,
        eval_frame=final_holdout_frame,
        round_trip_cost_bps=round_trip_cost_bps,
        signal_threshold_pct=tuned_threshold,
        allow_short=allow_short,
    )

    # regime decomposition
    vol_pct = _volatility_series(close=price_data["Close"].astype(float))
    vol_low, vol_high = _regime_thresholds(volatility_pct=vol_pct, reference_index=research.index)
    regime_metrics = pd.concat(
        [
            _regime_breakdown_for_segment(
                segment_name="Research",
                backtest_frame=trade_backtest,
                volatility_pct=vol_pct,
                low_threshold=vol_low,
                high_threshold=vol_high,
            ),
            _regime_breakdown_for_segment(
                segment_name="FinalHoldout",
                backtest_frame=final_holdout_trade_backtest,
                volatility_pct=vol_pct,
                low_threshold=vol_low,
                high_threshold=vol_high,
            ),
        ],
        ignore_index=True,
    )

    # future forecasting
    final_models = _fit_cloned_models(base_models=base_models, x_train=dataset[feature_cols], y_train=dataset["target"])
    future_index = _infer_future_index(price_data.index, forecast_days)
    raw_history = raw_ohlcv.copy()
    future_records = []

    val_actual_ret = (val_actual_next / val_current_close) - 1.0
    val_pred_ret = (val_ensemble_next / val_current_close) - 1.0
    residual_ret = val_actual_ret - val_pred_ret
    if len(residual_ret) > 5:
        q10, q90 = np.nanquantile(residual_ret, [0.10, 0.90])
        sigma = float(np.nanstd(residual_ret))
    else:
        q10, q90, sigma = -0.02, 0.02, 0.02

    for step, dt in enumerate(future_index, start=1):
        row = _feature_row_from_history(history_raw=raw_history, lags=lags, feature_cols=feature_cols)
        current_close = float(raw_history["close"].iloc[-1])

        model_pred_target: Dict[str, float] = {}
        model_pred_close: Dict[str, float] = {}
        for name, model in final_models.items():
            pred_t = float(model.predict(row)[0])
            model_pred_target[name] = pred_t
            model_pred_close[name] = float(_target_to_next_close(np.array([pred_t]), np.array([current_close]), target_mode)[0])

        ensemble_target = float(sum(weights.get(name, 0.0) * model_pred_target[name] for name in model_pred_target))
        ensemble_close = float(_target_to_next_close(np.array([ensemble_target]), np.array([current_close]), target_mode)[0])

        scale = float(np.sqrt(step))
        lower_q = max(1e-9, ensemble_close * (1.0 + float(q10) * scale))
        upper_q = max(lower_q, ensemble_close * (1.0 + float(q90) * scale))
        lower_s = max(1e-9, ensemble_close * (1.0 - sigma * scale))
        upper_s = max(lower_s, ensemble_close * (1.0 + sigma * scale))

        future_records.append(
            {
                "ensemble_pred": ensemble_close,
                "ensemble_pred_return_pct": (ensemble_close / current_close - 1.0) * 100.0,
                "lower_band_q10": lower_q,
                "upper_band_q90": upper_q,
                "lower_band_1sigma": lower_s,
                "upper_band_1sigma": upper_s,
                **{f"{name}_pred": value for name, value in model_pred_close.items()},
            }
        )

        synth_open = current_close
        synth_close = ensemble_close
        synth_high = max(synth_open, synth_close)
        synth_low = min(synth_open, synth_close)
        recent_vol = raw_history["volume"].tail(20)
        synth_vol = float(np.nanmedian(recent_vol)) if len(recent_vol) else float(raw_history["volume"].iloc[-1])
        raw_history.loc[dt] = {
            "open": synth_open,
            "high": synth_high,
            "low": synth_low,
            "close": synth_close,
            "volume": max(1.0, synth_vol),
        }

    future_frame = pd.DataFrame(future_records, index=future_index)

    validation_summary = pd.DataFrame(
        [
            {"item": "validation_mode", "value": validation_mode},
            {"item": "target_mode", "value": target_mode},
            {"item": "gap_days_total", "value": float(gap_days)},
            {"item": "purge_days", "value": float(purge_days)},
            {"item": "embargo_days", "value": float(embargo_days)},
            {"item": "validation_days", "value": float(validation_days)},
            {"item": "research_test_days", "value": float(test_days)},
            {"item": "final_holdout_days", "value": float(final_holdout_days)},
            {"item": "selected_signal_threshold_pct", "value": float(tuned_threshold)},
            {"item": "allow_short", "value": float(1.0 if allow_short else 0.0)},
            {"item": "vol_low_threshold_pct", "value": float(vol_low)},
            {"item": "vol_high_threshold_pct", "value": float(vol_high)},
        ]
    )

    return ForecastResult(
        symbol=symbol,
        price_data=price_data,
        test_frame=test_frame,
        future_frame=future_frame,
        metrics=metrics,
        weights=weights,
        latest_close=float(price_data["Close"].iloc[-1]),
        trade_backtest=trade_backtest,
        trade_metrics=trade_metrics,
        validation_mode=validation_mode,
        final_holdout_frame=final_holdout_frame,
        final_holdout_metrics=final_holdout_metrics,
        final_holdout_trade_backtest=final_holdout_trade_backtest,
        final_holdout_trade_metrics=final_holdout_trade_metrics,
        regime_metrics=regime_metrics,
        validation_summary=validation_summary,
        validation_frame=validation_frame,
        validation_metrics=validation_metrics,
    )
