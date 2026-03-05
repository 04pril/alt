from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import normalize_symbol, run_forecast


WATCHLIST_PRESETS: Dict[str, List[str]] = {
    "코인": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD"],
    "미국주식": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"],
    "한국주식": ["005930", "000660", "035420", "051910", "005380", "035720", "068270", "207940"],
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
    "cost_bps_assumed": "가정비용(bps)",
    "signal_threshold_pct": "최소신호강도(%)",
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


def build_scan_row(symbol: str, result, forecast_days: int) -> Dict[str, float | str]:
    ensemble = result.metrics[result.metrics["model"] == "Ensemble"]
    ensemble_row = ensemble.iloc[0] if not ensemble.empty else result.metrics.iloc[0]

    latest_close = float(result.latest_close)
    next_pred = float(result.future_frame["ensemble_pred"].iloc[0])
    last_pred = float(result.future_frame["ensemble_pred"].iloc[-1])
    expected_return = (last_pred / latest_close - 1.0) * 100.0
    next_day_return = (next_pred / latest_close - 1.0) * 100.0

    mae = float(ensemble_row["mae"])
    mape = float(ensemble_row["mape_pct"])
    direction_acc = float(ensemble_row["direction_acc_pct"])
    mae_pct = (mae / max(latest_close, 1e-9)) * 100.0

    win_rate_pct = metric_value(result.trade_metrics, "win_rate_pct")
    expectancy_pct = metric_value(result.trade_metrics, "expectancy_pct")
    max_dd_pct = metric_value(result.trade_metrics, "max_drawdown_pct")
    net_cum_return_pct = metric_value(result.trade_metrics, "net_cum_return_pct")
    trades = metric_value(result.trade_metrics, "trades")

    trend_score = float(np.clip((expected_return + 12.0) / 24.0, 0.0, 1.0) * 100.0)
    error_score = float(np.clip(100.0 - mae_pct * 12.0, 0.0, 100.0))
    drawdown_penalty = float(np.clip(abs(min(max_dd_pct, 0.0)) * 2.0, 0.0, 40.0))
    trade_score = float(np.clip(50.0 + expectancy_pct * 6.0 + (win_rate_pct - 50.0) * 0.8 - drawdown_penalty, 0, 100))
    score = 0.40 * direction_acc + 0.25 * trend_score + 0.20 * error_score + 0.15 * trade_score

    return {
        "심볼": symbol,
        "등급": grade_from_score(score),
        "유망도점수": score,
        "예상수익률(%)": expected_return,
        "내일예상수익률(%)": next_day_return,
        "거래기대값(%)": expectancy_pct,
        "누적수익률(%)": net_cum_return_pct,
        "방향정확도(%)": direction_acc,
        "승률(%)": win_rate_pct,
        "최대낙폭(%)": max_dd_pct,
        "거래횟수": trades,
        "MAE/가격(%)": mae_pct,
        "MAPE(%)": mape,
        "최근종가": latest_close,
        "내일예측종가": next_pred,
        f"{forecast_days}일예측종가": last_pred,
    }


def render_single_result(result, forecast_days: int) -> None:
    latest_close = result.latest_close
    last_pred = float(result.future_frame["ensemble_pred"].iloc[-1])
    next_pred = float(result.future_frame["ensemble_pred"].iloc[0])
    expected_return = (last_pred / latest_close - 1.0) * 100.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최근 종가", f"{latest_close:,.2f}")
    c2.metric("내일 예측 종가", f"{next_pred:,.2f}")
    c3.metric(f"{forecast_days}일 기대 수익률", f"{expected_return:+.2f}%")
    c4.metric("검증모드", "워크포워드" if result.validation_mode == "walk_forward" else "홀드아웃")

    st.subheader("모델 성능 (백테스트)")
    metrics_view = result.metrics.rename(
        columns={
            "model": "모델",
            "mae": "MAE",
            "mape_pct": "MAPE(%)",
            "direction_acc_pct": "방향정확도(%)",
        }
    )
    st.dataframe(metrics_view, use_container_width=True, hide_index=True)

    weight_table = result.metrics[result.metrics["model"].isin(result.weights.keys())][["model"]].copy().assign(
        weight_pct=lambda df: df["model"].map(result.weights).fillna(0.0) * 100.0
    )
    weight_table = weight_table.rename(columns={"model": "모델", "weight_pct": "가중치(%)"})
    st.dataframe(weight_table, use_container_width=True, hide_index=True)

    st.subheader("실전 가정 백테스트 (신호일 종가 계산 → 다음날 시가 진입)")
    trade_view = result.trade_metrics.copy()
    trade_view["지표"] = trade_view["metric"].map(TRADE_METRIC_LABELS).fillna(trade_view["metric"])
    trade_view = trade_view[["지표", "value"]].rename(columns={"value": "값"})
    st.dataframe(trade_view, use_container_width=True, hide_index=True)

    eq = result.trade_backtest[["equity_curve"]].copy()
    eq["net_cum_return_pct"] = (eq["equity_curve"] - 1.0) * 100.0
    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["net_cum_return_pct"],
            mode="lines",
            name="누적수익률(%)",
            line=dict(width=2),
        )
    )
    eq_fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis_title="누적수익률(%)",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    st.subheader("가격 추이 + 예측")
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
            y=result.test_frame["actual_next_close"],
            mode="lines",
            name="백테스트 실제",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.test_frame.index,
            y=result.test_frame["ensemble_pred"],
            mode="lines",
            name="백테스트 예측",
            line=dict(width=2),
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
        height=620,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=20, r=20, t=10, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("미래 예측 상세")
    future_view = result.future_frame.rename(
        columns={
            "ensemble_pred": "앙상블예측",
            "Ridge_pred": "Ridge예측",
            "RandomForest_pred": "RandomForest예측",
            "GradientBoosting_pred": "GradientBoosting예측",
            "lower_band_1sigma": "하단밴드(1σ)",
            "upper_band_1sigma": "상단밴드(1σ)",
        }
    )
    st.dataframe(future_view, use_container_width=True)


st.set_page_config(page_title="멀티마켓 가격 예측기", layout="wide")

st.title("코인 · 미국주식 · 한국주식 가격 예측기")
st.caption("Yahoo Finance 데이터를 기반으로 앙상블 모델 예측 결과를 시각화합니다.")
st.info("이 도구는 실험/학습 목적입니다. 어떤 모델도 미래 가격을 보장하지 않습니다.", icon="ℹ️")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def run_cached(
    symbol: str,
    years: int,
    test_days: int,
    forecast_days: int,
    validation_mode: str,
    retrain_every: int,
    total_cost_bps: float,
    min_signal_strength_pct: float,
):
    return run_forecast(
        symbol=symbol,
        years=years,
        test_days=test_days,
        forecast_days=forecast_days,
        validation_mode=validation_mode,
        retrain_every=retrain_every,
        total_cost_bps=total_cost_bps,
        min_signal_strength_pct=min_signal_strength_pct,
    )


with st.sidebar:
    st.subheader("공통 설정")
    asset_type = st.selectbox("자산 유형", ["코인", "미국주식", "한국주식"], index=0)

    if asset_type == "코인":
        default_symbol = "BTC-USD"
        korea_market = "KOSPI"
        st.caption("예: BTC-USD, ETH-USD, SOL-USD")
    elif asset_type == "미국주식":
        default_symbol = "AAPL"
        korea_market = "KOSPI"
        st.caption("예: AAPL, NVDA, MSFT")
    else:
        default_symbol = "005930"
        korea_market = st.radio("한국 시장", ["KOSPI", "KOSDAQ"], index=0, horizontal=True)
        st.caption("숫자 6자리 입력 시 자동으로 .KS / .KQ를 붙입니다.")

    years = st.slider("학습 데이터 기간(년)", min_value=2, max_value=10, value=5)
    test_days = st.slider("검증 구간(일)", min_value=30, max_value=220, value=60)
    forecast_days = st.slider("미래 예측(일)", min_value=7, max_value=60, value=14)

    validation_label = st.selectbox("검증 모드", list(VALIDATION_LABEL_TO_MODE.keys()), index=0)
    validation_mode = VALIDATION_LABEL_TO_MODE[validation_label]
    if validation_mode == "walk_forward":
        retrain_every = st.slider("워크포워드 재학습 주기(일)", min_value=1, max_value=20, value=5)
    else:
        retrain_every = 5

    total_cost_bps = st.slider("왕복 거래비용 가정(bps)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
    min_signal_strength_pct = st.slider(
        "최소 신호 강도(%)",
        min_value=0.0,
        max_value=3.0,
        value=0.2,
        step=0.05,
    )
    st.caption("신호 강도 이하 구간은 거래하지 않아 과도한 회전을 줄입니다.")

    st.divider()
    st.subheader("단일 종목")
    raw_symbol = st.text_input("심볼", value=default_symbol)


tab_single, tab_scan = st.tabs(["단일 종목 상세 예측", "유망 종목 빠른 보기"])

with tab_single:
    run_single = st.button("단일 종목 예측 실행", type="primary", key="single_run")
    if run_single:
        try:
            symbol = normalize_symbol(asset_type=asset_type, raw_symbol=raw_symbol, korea_market=korea_market)
            with st.spinner(f"{symbol} 데이터를 불러오고 모델을 학습하는 중..."):
                single_result = run_cached(
                    symbol=symbol,
                    years=years,
                    test_days=test_days,
                    forecast_days=forecast_days,
                    validation_mode=validation_mode,
                    retrain_every=retrain_every,
                    total_cost_bps=total_cost_bps,
                    min_signal_strength_pct=min_signal_strength_pct,
                )
        except Exception as exc:
            st.error(f"실행 중 오류: {exc}")
        else:
            render_single_result(result=single_result, forecast_days=forecast_days)
    else:
        st.write("사이드바에서 설정 후 **단일 종목 예측 실행**을 눌러 주세요.")

with tab_scan:
    st.caption("여러 종목을 한 번에 분석해서 유망도 순위로 보여줍니다.")
    preset_symbols = WATCHLIST_PRESETS[asset_type]
    selected_symbols = st.multiselect(
        "기본 후보",
        options=preset_symbols,
        default=preset_symbols[: min(6, len(preset_symbols))],
        key=f"preset_{asset_type}",
    )
    extra_symbols_raw = st.text_input(
        "추가 심볼 (쉼표/줄바꿈 구분, 선택사항)",
        value="",
        key=f"extra_{asset_type}",
    )
    top_n = st.slider("상위 카드 표시 개수", min_value=3, max_value=10, value=5, key="top_n")
    run_scan = st.button("유망 종목 스캔 실행", type="primary", key="scan_run")

    if run_scan:
        extra_symbols = parse_symbols(extra_symbols_raw)
        scan_input = dedupe_keep_order(selected_symbols + extra_symbols)
        if not scan_input:
            st.error("스캔할 심볼이 없습니다. 기본 후보를 선택하거나 추가 심볼을 입력해 주세요.")
        else:
            if validation_mode == "walk_forward" and len(scan_input) >= 8:
                st.warning("워크포워드 모드는 느릴 수 있습니다. 심볼 수를 줄이면 더 빠릅니다.")

            rows: List[Dict[str, float | str]] = []
            errors: List[Dict[str, str]] = []
            progress = st.progress(0.0)
            status = st.empty()

            for idx, raw in enumerate(scan_input, start=1):
                try:
                    symbol = normalize_symbol(asset_type=asset_type, raw_symbol=raw, korea_market=korea_market)
                    status.write(f"{idx}/{len(scan_input)} 분석 중: `{symbol}`")
                    scan_result = run_cached(
                        symbol=symbol,
                        years=years,
                        test_days=test_days,
                        forecast_days=forecast_days,
                        validation_mode=validation_mode,
                        retrain_every=retrain_every,
                        total_cost_bps=total_cost_bps,
                        min_signal_strength_pct=min_signal_strength_pct,
                    )
                    rows.append(build_scan_row(symbol=symbol, result=scan_result, forecast_days=forecast_days))
                except Exception as exc:
                    errors.append({"심볼": raw, "오류": str(exc)})
                progress.progress(idx / len(scan_input))

            status.empty()
            progress.empty()

            if rows:
                summary = (
                    pd.DataFrame(rows)
                    .sort_values(["유망도점수", "거래기대값(%)", "예상수익률(%)"], ascending=[False, False, False])
                    .reset_index(drop=True)
                )
                top_df = summary.head(min(top_n, len(summary)))

                st.subheader("유망 후보 카드")
                card_cols = st.columns(min(3, len(top_df)))
                for i in range(len(top_df)):
                    row = top_df.iloc[i]
                    with card_cols[i % len(card_cols)]:
                        st.metric(
                            f"{row['심볼']} · {row['등급']}",
                            f"{row['유망도점수']:.1f}점",
                            f"{row['예상수익률(%)']:+.2f}%",
                        )
                        st.caption(
                            f"기대값/거래 {row['거래기대값(%)']:+.3f}% · 승률 {row['승률(%)']:.1f}% · MDD {row['최대낙폭(%)']:.2f}%"
                        )
                        st.caption(f"방향정확도 {row['방향정확도(%)']:.1f}% · MAE/가격 {row['MAE/가격(%)']:.2f}%")

                score_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=top_df["심볼"],
                            y=top_df["유망도점수"],
                            text=[f"{value:.1f}" for value in top_df["유망도점수"]],
                            textposition="outside",
                        )
                    ]
                )
                score_fig.update_layout(
                    title="상위 후보 유망도 점수",
                    height=360,
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis_title="점수",
                    xaxis_title="심볼",
                )
                st.plotly_chart(score_fig, use_container_width=True)

                st.subheader("전체 스캔 결과")
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.warning("정상 분석된 종목이 없습니다. 심볼 형식이나 인터넷 연결 상태를 확인해 주세요.")

            if errors:
                st.subheader("실패한 심볼")
                st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else:
        st.write("기본 후보를 선택하고 **유망 종목 스캔 실행**을 눌러 주세요.")
