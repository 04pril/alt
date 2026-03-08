# Streamlit Monitor + 24/7 Paper Trading Runtime

이 저장소는 두 축으로 나뉩니다.

- `app.py`: 관제, 수동 개입, 읽기 중심 UI
- `jobs/`, `services/`, `storage/`: Streamlit 비의존 worker/runtime

이번 브랜치 기준 원칙:

- predictor 검증/워크포워드 semantics는 유지합니다.
- `run_forecast_on_price_data(...)` 기반 재사용 구조를 유지합니다.
- 한국주식은 KIS mock broker 경로를 우선 사용합니다.
- 미국주식/코인은 SQLite 기반 sim broker를 유지합니다.
- 운영 상태의 single source of truth는 `storage/repository.py` 입니다.

## Runtime Architecture

```text
Market Data -> Universe Scanner -> Signal Engine -> Candidate Scans
                                           |
                                           v
                                   Prediction Ledger

Candidate Scans -> Risk Engine -> Broker Router
                                   |          |
                                   |          +-> KISPaperBroker (KR mock)
                                   |
                                   +-> PaperBroker (sim)

Broker -> Orders -> Fills -> Positions -> Account Snapshots

Predictions -> Outcome Resolver -> Outcomes -> Evaluations

Scheduler
  -> scan
  -> entry
  -> broker_position_sync
  -> exit_management
  -> broker_order_sync
  -> broker_account_sync
  -> outcome_resolution
  -> daily_report
  -> retrain_check

Job Runs + System Events + Account Snapshots -> Monitoring / app read model
```

## Repository-Centered Read Model

운영 UI와 worker는 같은 `storage/repository.py`를 사용합니다.

- worker는 orders / fills / positions / account snapshots / job_runs / system_events 를 기록합니다.
- monitoring reader는 `monitoring/dashboard_hooks.py`를 통해 repository를 읽습니다.
- app의 KIS 수동 주문/조회는 `services/manual_kis_service.py`로 감쌌습니다.
- `prediction_store.py`, `prediction_memory.py`는 예전 prediction ledger와의 compatibility wrapper 역할만 유지합니다.

즉, 운영 상태를 볼 때 기준은 repository이고, legacy wrapper는 prediction history 호환용으로만 남겨 둡니다.

## Broker Routing

- 한국주식(`.KS`, `.KQ`)이고 KIS mock 설정이 살아 있으면 `KISPaperBroker`
- 그 외는 `PaperBroker`

`BrokerRouter`는 이 분기를 감추고 worker/app에서 동일한 진입점을 제공합니다.

## Explicit Broker Sync Jobs

이번 라운드부터 broker sync는 명시적 job으로 분리됐습니다.

- `broker_position_sync_job`
  - open position mark-to-market
  - repository position 업데이트
  - unified account snapshot 갱신
- `broker_order_sync_job`
  - sim open order 체결 처리
  - KIS mock order reconcile
  - fills / order status 반영
- `broker_account_sync_job`
  - unified account snapshot 보강
  - KIS mock account summary 조회 결과를 운영 메트릭으로 기록

각 job은 다음을 만족합니다.

- `job_runs`에 개별 기록
- `system_events`에 sync 결과 기록
- scheduler retry/backoff 대상
- app에서 수동 강제 실행 가능

## Accounting Definitions

런타임 전체에서 아래 정의를 씁니다.

- `gross_exposure`: 열린 포지션의 절대 노출 합
- `net_exposure`: 열린 포지션의 signed 노출 합
- `equity = cash + net_exposure`
- `unrealized_pnl`: 포지션별 미실현 손익 합
- `drawdown_pct`: peak equity 대비 현재 equity 하락률

중요:

- `market value` 위에 `unrealized_pnl`을 다시 더하지 않습니다.
- short 포지션은 `exposure_value < 0` 을 유지합니다.
- `PortfolioManager`, `PaperBroker`, `RiskEngine`이 같은 회계식 위에서 동작합니다.

## Risk / Entry Guardrails

진입 판단에는 다음이 반영됩니다.

- `max_daily_new_entries`
- `max_open_positions`
- `symbol_max_weight`
- `asset_type_max_weight`
- `per_trade_risk_budget_pct`
- `total_risk_budget_pct`
- `daily_loss_limit_pct`
- `max_drawdown_limit_pct`
- `max_same_direction_correlation`
- `cooldown_bars_after_exit`

추가로 pending entry 주문도 다음에 포함됩니다.

- 동일 심볼 중복 진입 차단
- reserved cash
- reserved exposure
- reserved risk budget

## Cooldown Semantics

- `1h`: `cooldown_bars_after_exit = N` 이면 정확히 `N시간`
- `1d`: 현재 단계에서는 해당 자산 timezone 기준 영업일 근사

완전한 exchange calendar 계산은 아직 아닙니다.

## Partial Fill Policy

`BrokerSettings.allow_partial_fills`

- `True`: 부분 체결 허용, 주문 상태는 `partially_filled`
- `False`: 전량 체결 가능할 때만 체결, 아니면 열린 상태 유지

이 정책은 sim broker에 강제됩니다.

## KIS Mock Order Lifecycle

한국주식 KIS mock 주문은 즉시 `filled`로 확정하지 않습니다.

상태 전이:

- `submitted`
- `acknowledged`
- `pending_fill`
- `partially_filled`
- `filled`
- `rejected`
- `cancelled`

현재 reconcile은 holdings delta 기반입니다.

- `broker_order_id`와 내부 `order_id`를 함께 저장
- `FillRecord`는 reconcile 시점에만 생성
- 같은 sync를 여러 번 돌려도 이미 반영한 fill을 중복 생성하지 않음

## Scheduler / Retry / Lease / Heartbeat

`jobs/scheduler.py`는 job별로 독립적으로 retry/backoff를 적용합니다.

- 실패한 job만 `next_retry_at`까지 backoff
- 다른 job은 같은 loop에서 계속 수행
- `retry_count`는 실제 재시도 횟수 누적
- `job_runs`에는 `next_retry_at`, `lease_expires_at` 저장

Long-running job 대응:

- `_run_guarded()`가 job 시작 시 lease를 획득
- job 진행 중 `context.touch_runtime(...)` 호출로 lease refresh
- 같은 touch에서 `worker_heartbeat_at`, `worker_heartbeat_job`도 갱신
- scan/sync loop는 심볼/주문/포지션 단위로 progress touch를 호출

기본 cadence:

- `loop_sleep_seconds`
- `retry_backoff_seconds`
- `max_retry_count`
- `job_lease_seconds`
- `broker_position_sync_interval_minutes`
- `broker_order_sync_interval_minutes`
- `broker_account_sync_interval_minutes`
- `exit_management_interval_minutes`
- `outcome_resolution_interval_minutes`

## Monitoring

`monitoring/dashboard_hooks.py`는 repository read model을 통해 다음을 제공합니다.

- auto trading 상태
- recent job health
- broker sync job별 마지막 상태/시각
- 최근 broker sync 오류
- open positions / open orders / candidates / prediction report

`app.py`의 운영 모니터에서 다음 수동 실행이 가능합니다.

- `broker_account_sync`
- `broker_order_sync`
- `broker_position_sync`

거래 loop 자체는 계속 worker에서만 실행됩니다.

## Runtime Initialization

```bash
python scripts/init_runtime.py
```

초기화 시 기본 runtime flags:

- `trading_paused`
- `entry_paused`
- `exit_only_mode`
- `worker_paused`

## Local Run

의존성 설치:

```bash
pip install -r requirements.txt
```

DB 초기화만:

```bash
python -m jobs.scheduler --init-db
```

worker 1회 실행:

```bash
python -m jobs.scheduler --once
```

worker loop:

```bash
python -m jobs.scheduler
```

Streamlit monitor:

```bash
python -m streamlit run app.py
```

## Tests

전체 테스트:

```bash
python -m unittest discover -s tests -v
```

회귀 테스트는 아래를 고정합니다.

- equity / drawdown / signed exposure
- exit order가 daily entry count에 섞이지 않는지
- pending entry / partial fill이 duplicate entry를 막는지
- same-second `latest_*` ordering 안정성
- scheduler retry/backoff non-blocking
- broker sync job idempotency
- broker sync retry/backoff
- long-running job lease refresh / worker heartbeat
- monitoring broker sync status reader
- prediction memory wrapper forwarding
- KIS order lifecycle / reconcile idempotency

## Known Limitations

- 한국주식 KIS mock 체결은 holdings delta 기반 reconcile이라 실시간 체결 원장과 1:1은 아닙니다.
- `1d` cooldown은 영업일 근사이며 완전한 exchange calendar는 아닙니다.
- background worker는 단일 프로세스 운영을 기본 가정으로 합니다.
- `app.py`의 prediction history는 아직 legacy compatibility wrapper를 경유합니다.
- UI routing / floating nav / 상단 네비는 이번 라운드 범위 밖입니다.
