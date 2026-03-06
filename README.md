# 멀티마켓 예측 + 24/7 Paper Trading

이 저장소는 두 계층으로 나뉩니다.

- `predictor.py`: 연구/검증/백테스트용 signal engine
- `services/`, `storage/`, `jobs/`: 24/7 background paper trading 운영 계층

중요 원칙:

- `predictor.py`의 검증 로직은 유지하고, 운영 계층이 감싸서 사용합니다.
- Streamlit은 거래 루프를 돌리지 않습니다.
- 예측 레코드는 append-only로 저장하고, outcome/evaluation은 만기 후 별도 확정합니다.
- 예측 정확도와 매매 성과는 분리 저장/표시합니다.

## 현재 코드 기준 누락 요소와 이번 구현 범위

기존 누락:

- worker/service/process 계층 부재
- 시장시간/휴장일/자산군별 실행 주기 분리 부재
- candidate scan, job_runs, system_events, control_flags 부재
- 내부 paper broker 부재
- 재시작 복구 가능한 주문/포지션 라이프사이클 부재
- 배치 재학습 파이프라인 부재

이번 구현:

- `predictor.py` 재사용형 `run_forecast_on_price_data(...)`
- `storage/repository.py` SQLite 운영 저장소
- `services/*` 운영 모듈
- `jobs/scheduler.py` 별도 worker 루프
- `monitoring/dashboard_hooks.py` 읽기 전용 대시보드 reader
- `scripts/init_runtime.py` 초기화 코드
- `tests/*` 기본 테스트

## 최종 아키텍처

```text
Market Data -> Universe Scanner -> Signal Engine(predictor wrapper) -> Candidate Scans
                                                             |
                                                             v
                                                    Prediction Ledger

Candidate Scans -> Risk Engine -> Paper Broker -> Orders -> Fills -> Positions
                                                \-> Account Snapshots

Unresolved Predictions -> Outcome Resolver -> Outcomes -> Evaluations

Evaluations + Account Snapshots + Job Runs + System Events
    -> Monitoring Dashboard Hooks
    -> Daily Reports
    -> Retrainer(weekly/monthly)
```

## 디렉터리 구조

```text
config/
  settings.py
  runtime_settings.example.json
jobs/
  scheduler.py
  tasks.py
monitoring/
  dashboard_hooks.py
scripts/
  init_runtime.py
services/
  market_data_service.py
  universe_scanner.py
  signal_engine.py
  risk_engine.py
  portfolio_manager.py
  paper_broker.py
  outcome_resolver.py
  evaluator.py
  retrainer.py
storage/
  models.py
  repository.py
tests/
  test_repository.py
  test_risk_engine.py
  test_dashboard_hooks.py
predictor.py
app.py
```

## DB 스키마

운영 DB는 `config/settings.py`의 `storage.db_path`를 사용합니다. 기본값은 `.runtime/paper_trading.sqlite3` 입니다.

필수 테이블:

- `predictions`
- `outcomes`
- `evaluations`
- `candidate_scans`
- `orders`
- `fills`
- `positions`
- `account_snapshots`
- `model_registry`
- `retrain_runs`
- `job_runs`
- `system_events`

추가 테이블:

- `control_flags`

각 테이블은 최소한 아래를 저장합니다.

- timestamps: `created_at`, `updated_at`, `resolved_at`, `finished_at`
- symbol / asset_type / timeframe
- model_version / feature_version / strategy_version
- signal / score / confidence / threshold
- expected_return / expected_risk / position_size
- order status / fill price / fees / slippage
- pnl / drawdown / exposure
- error message / retry count / job status

## 설정 파일 예시

기본 설정은 코드에 내장되어 있고, 선택적으로 `config/runtime_settings.json`으로 override 할 수 있습니다.

생성:

```bash
python scripts/init_runtime.py
```

예시 파일:

- [config/runtime_settings.example.json](C:/Users/admin/Desktop/develop/codex/config/runtime_settings.example.json)

핵심 가정:

- 코인: `1h` bar, 24/7, bar close 기준 진입/청산
- 미국주식/한국주식: `1d` predictor 유지, `pre-close` 윈도우에서 paper MOC proxy로 진입
- 휴장일: `holidays` 패키지 기반 국가 휴일 + 주말
- 반일장/특수 휴장일은 완전 정밀하게 모델링하지 않음

## 운영 로직

### 1) 후보 스캔

- `watchlist + top_universe`를 주기적으로 스캔
- 최소 히스토리 길이, 결측, 이상치, 유동성 검사
- 기대수익, confidence, 비용, 변동성, 최근 성능으로 score 계산
- 보유 중 / cooldown 종목은 scan row에 표시

### 2) 진입 규칙

- `min_expected_return_pct`
- `min_confidence`
- `max_expected_risk_pct`
- `max_cost_bps`
- 시장 open 필요
- `max_open_positions`
- `max_daily_new_entries`
- `symbol_max_weight`
- `asset_type_max_weight`
- `max_same_direction_correlation`
- `daily_loss_limit_pct`
- `max_drawdown_limit_pct`

### 3) 청산 규칙

- stop loss
- take profit
- trailing stop
- time stop
- opposite signal exit
- score decay exit

### 4) 리스크 관리

- 종목당 risk budget
- 자산군별 risk budget
- 계좌 총 risk budget
- 일일 손실 한도
- 최대 드로우다운 도달 시 신규 진입 중단

### 5) 스케줄링

- `scan job`
- `entry decision job`
- `exit management job`
- `outcome resolution job`
- `daily report job`
- `retrain check job`

각 job은:

- `job_runs` 기반 idempotent run key 사용
- 중복 실행 방지
- 실패 시 error log 기록

## 로컬 실행 방법

설치:

```bash
pip install -r requirements.txt
```

초기화:

```bash
python scripts/init_runtime.py
```

DB만 초기화:

```bash
python -m jobs.scheduler --init-db
```

worker 1회 실행:

```bash
python -m jobs.scheduler --once
```

worker 루프 실행:

```bash
python -m jobs.scheduler
```

기존 Streamlit 앱:

```bash
python -m streamlit run app.py
```

주의:

- worker는 Streamlit과 분리된 별도 프로세스로 띄워야 합니다.
- app은 read-only 모니터링 용도로 같은 DB를 읽는 구조를 전제로 합니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

## 운영 섹션

### pause / resume

`control_flags.trading_paused`를 사용합니다.

초기값:

- `0`: 신규 진입 허용
- `1`: 신규 진입 중단

현재 구현에서는 DB flag를 기준으로 entry job이 차단됩니다.

### 재시작 복구

- 주문, 포지션, 계좌 스냅샷은 모두 DB에 저장됩니다.
- worker 재시작 후 `open_orders`, `open_positions`, `account_snapshots`를 읽어 복구합니다.

### 모델 재평가 / 재학습

- 즉시 온라인 학습은 하지 않습니다.
- `services/retrainer.py`는 benchmark universe에 대해 batch 평가를 수행합니다.
- promotion 조건을 통과하면 `model_registry.is_champion=1`로 승격합니다.
- 현재 최소 버전은 “새 모델 생성”보다 “현행 predictor 버전 재평가 및 승격 체크”에 초점을 둡니다.

## systemd 예시

```ini
[Unit]
Description=Paper Trading Worker
After=network.target

[Service]
WorkingDirectory=C:/Users/admin/Desktop/develop/codex
ExecStart=python -m jobs.scheduler
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 장시간 운영 시 주의사항

- `yfinance`는 무료 데이터이므로 지연/누락/제한 가능성이 있습니다.
- 국가 공휴일 기반 휴장일 처리라서 거래소 특수 휴장일/반일장은 완전 정밀하지 않습니다.
- 미국/한국주식은 `pre-close proxy fill` 가정이라 실제 MOC 체결과 다를 수 있습니다.
- 코인 `1h` 데이터는 bar close 기준이며, 초단타 타이밍 시스템이 아닙니다.
- worker는 단일 프로세스 기준입니다. 다중 worker를 돌리면 DB lock 충돌이 생길 수 있습니다.

## 향후 바로 확장 가능한 것

- `fills`에 부분 체결 history를 더 정교하게 확장
- app에서 `monitoring/dashboard_hooks.py`를 직접 읽는 전용 관제 탭 추가
- retrainer에 challenger artifact 저장/rollback 추가
- 거래소 캘린더를 국가 휴일 근사치에서 더 정밀한 exchange calendar로 교체
