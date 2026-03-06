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

### 저장소 단일화

- 운영 데이터의 단일 저장소는 `storage/repository.py`가 관리하는 SQLite DB입니다.
- `prediction_store.py`와 `prediction_memory.py`는 기존 호출부를 유지하기 위한 compatibility layer입니다.
- 예측, outcome, evaluation, candidate scan, order, fill, position, account snapshot, job run, system event는 모두 같은 DB를 사용합니다.

### 브로커 구조

- `services/broker_base.py`
  - `BrokerProtocol`
  - `BrokerRouter`
- `services/paper_broker.py`
  - 내부 시뮬레이션용 `SimBroker`
- `services/kis_paper_broker.py`
  - 한국투자증권 모의투자 계좌를 쓰는 `KISPaperBroker`
  - 기존 `kis_paper.py`의 `KISPaperClient`를 재사용합니다.

기본 broker mode:

- 코인: `sim`
- 미국주식: `sim`
- 한국주식: `kis_paper`

관련 설정:

- `broker.asset_broker_mode`
- `broker.broker_short_support`
- `broker.kis_config_path`

### KIS 모의계좌 vs 내부 Sim Broker

- `SimBroker`
  - 주문/체결/포지션/계좌 변화를 내부 DB에서 시뮬레이션합니다.
  - short는 설정상 허용할 수 있습니다.
- `KISPaperBroker`
  - KIS mock 계좌의 실시간 quote, 일봉 history, 주문, 계좌/보유종목 동기화를 사용합니다.
  - 한국주식 short는 명시적으로 금지합니다.
  - 최근 sync 결과는 control flag와 system event에 남깁니다.
  - KIS rate limiter는 기본적으로 process-wide lock 정책을 사용합니다.

### predictor timeframe vs execution quote source

- predictor 연구/검증은 기존 `run_forecast_on_price_data(...)` 기반 semantics를 유지합니다.
- 실행 가격 소스는 별도로 분리됩니다.
  - 한국주식 + `kis_paper`: KIS 실시간 quote 우선
  - 그 외: timeframe bar 기반 quote
- 이 구조는 연구용 일봉 predictor와 실행용 실시간 quote를 명시적으로 분리한 것입니다.
- 따라서 한국주식 live paper execution 가격은 연구용 daily close와 완전히 같지 않을 수 있습니다.

### pause / resume / control flags

주요 control flag:

- `entry_paused`
  - 신규 진입만 중단
- `worker_paused`
  - worker loop 전체 중단
- `exit_only_mode`
  - 신규 진입은 막고 청산만 허용
- `trading_paused`
  - legacy alias
  - 내부적으로 `entry_paused`와 동기화되며 신규 주 설명으로는 사용하지 않습니다.

### 재시작 복구

- 주문, 포지션, 계좌 스냅샷은 모두 DB에 저장됩니다.
- worker 재시작 후 `open_orders`, `open_positions`, `account_snapshots`, `control_flags`를 읽어 복구합니다.
- KIS broker는 재시작 후 `sync_state()`를 다시 수행해 계좌 상태를 맞춥니다.

### Scheduler 하드닝

- 각 job은 `job_name + run_key`로 idempotent 하게 실행됩니다.
- `retry_backoff_seconds`, `max_retry_count`, `job_lease_timeout_seconds`를 실제로 사용합니다.
- stale running job은 lease timeout 경과 시 회수해서 재실행합니다.
- worker heartbeat는 `control_flags.worker_heartbeat_at`에 기록되고, 운영 모니터 상태 표시는 이 heartbeat를 우선 사용합니다.

### 모델 재평가 / 재학습

- 즉시 온라인 학습은 하지 않습니다.
- `services/retrainer.py`는 benchmark universe에 대해 batch 평가를 수행합니다.
- promotion 조건을 통과하면 `model_registry.is_champion=1`로 승격합니다.
- 현재 최소 버전은 “새 모델 생성”보다 “현행 predictor 버전 재평가 및 승격 체크”에 초점을 둡니다.

### pyc / runtime 산출물 정리

- `__pycache__/`, `*.pyc`, `.runtime/`, `.paper_trading/`, `.prediction_tracking/`는 git ignore 대상입니다.
- 이미 추적된 pyc가 남아 있다면 아래처럼 index에서 제거합니다.

```bash
git rm --cached __pycache__/*.pyc
```

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
- KIS mock 계좌는 한국주식 현금 주문 중심입니다. 부분 체결/정정/취소/공매도는 실거래 수준으로 완전하게 모델링하지 않습니다.
- 한국주식 `kis_paper` 경로는 실시간 quote를 execution source로 쓰므로, 연구용 daily close 기반 백테스트와 체결 가격이 완전히 같지 않을 수 있습니다.

## 향후 바로 확장 가능한 것

- `fills`에 부분 체결 history를 더 정교하게 확장
- app에서 `monitoring/dashboard_hooks.py`를 직접 읽는 전용 관제 탭 추가
- retrainer에 challenger artifact 저장/rollback 추가
- 거래소 캘린더를 국가 휴일 근사치에서 더 정밀한 exchange calendar로 교체
