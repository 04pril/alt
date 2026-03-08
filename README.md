# Streamlit Monitor + 24/7 Paper Trading Runtime

이 저장소는 두 축으로 나뉩니다.

- `app.py`: 관제, 수동 개입, 대시보드
- `jobs/`, `services/`, `storage/`: Streamlit 비의존 worker/runtime

이번 라운드 기준 핵심 원칙은 다음과 같습니다.

- predictor 검증/워크포워드 semantics는 유지합니다.
- `run_forecast_on_price_data(...)` 기반 재사용 구조를 유지합니다.
- 한국주식은 KIS mock broker 경로를 우선 사용합니다.
- 미국주식/코인은 SQLite 기반 sim broker를 유지합니다.
- app / worker / broker sync / prediction / evaluation / orders / fills / positions / account snapshot은 모두 `storage/repository.py`를 공용 저장소로 사용합니다.

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

Job Runs + System Events + Account Snapshots -> Monitoring
```

## Broker Routing

- 한국주식(`.KS`, `.KQ`)은 KIS mock 설정이 활성화되어 있으면 `KISPaperBroker`로 보냅니다.
- 그 외 자산은 `PaperBroker`로 처리합니다.
- `BrokerRouter`는 수동 UI와 background worker 사이에서 공통 진입점 역할만 합니다.

## Accounting Definitions

런타임 전체에서 아래 정의를 사용합니다.

- `gross_exposure`: 열린 포지션의 절대 노출 합계
- `net_exposure`: 열린 포지션의 signed 노출 합계
- `equity`: `cash + net_exposure`
- `unrealized_pnl`: 포지션별 미실현 손익 합계
- `drawdown_pct`: 최신 equity와 과거 peak equity 기준 하락률

중요한 점:

- `market value` 위에 `unrealized_pnl`를 다시 더하지 않습니다.
- short 포지션은 `exposure_value < 0` 으로 유지합니다.
- `PortfolioManager` mark-to-market, `PaperBroker` snapshot, `RiskEngine` 입력이 같은 회계식을 사용합니다.

## Risk / Entry Guardrails

진입 판단에는 다음이 모두 반영됩니다.

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

추가로, 아래 항목도 진입 차단에 포함됩니다.

- 동일 심볼의 미체결 엔트리 주문
- `partially_filled` 엔트리 주문
- pending entry 주문이 예약한 cash / exposure / risk budget

### Daily Entry Counting

`count_daily_entries()`는 이제 `reason='entry'` 이고 `cancelled/rejected`가 아닌 주문만 집계합니다.

즉 아래 주문은 일일 신규 진입 수에 포함되지 않습니다.

- `stop_loss`
- `take_profit`
- `time_stop`
- `opposite_signal`
- `manual_exit`

## Cooldown Semantics

청산 후 cooldown은 실제 진입 차단에 연결되어 있습니다.

- `1h`: `cooldown_bars_after_exit = N` 이면 정확히 `N시간` 후 만료
- `1d`: 현재 단계에서는 거래소 캘린더 전부를 쓰지 않고, 해당 자산 timezone 기준으로 영업일(주말 제외) 근사로 계산

완전한 exchange calendar 계산은 아직 아닙니다.

## Partial Fill Policy

`BrokerSettings.allow_partial_fills`가 실제 체결 로직에 반영됩니다.

- `True`: 유동성 한도 내 부분 체결 허용, 주문 상태는 `partially_filled`
- `False`: 전량 체결 가능할 때만 체결, 아니면 주문은 열린 상태 유지

이 정책은 sim broker에 강제됩니다.

## KIS Mock Order Lifecycle

한국주식 KIS mock 주문은 더 이상 `place_cash_order()` 직후 즉시 `filled`로 확정하지 않습니다.

현재 상태 전이는 아래와 같습니다.

- `submitted`
- `acknowledged`
- `pending_fill`
- `partially_filled`
- `filled`
- `rejected`
- `cancelled`

현재 구현은 KIS mock 보유수량 스냅샷을 기준으로 fill을 reconcile 합니다.

- `broker_order_id`와 내부 `order_id`를 함께 저장합니다.
- `FillRecord`는 실제 reconcile 시점에만 생성됩니다.
- 같은 sync를 여러 번 돌려도 이미 반영한 체결은 중복 반영하지 않습니다.

제한사항:

- 아직 별도 KIS 주문체결 조회 endpoint 기반이 아니라 holdings delta 기반입니다.
- 따라서 체결 시점/체결가 추정은 mock 환경에서 보수적 근사입니다.

## Scheduler / Retry / Lease

`jobs/scheduler.py`는 job별로 독립적으로 retry/backoff를 적용합니다.

- 실패한 job만 `next_retry_at`까지 backoff 됩니다.
- 다른 job은 같은 loop에서 계속 수행됩니다.
- `retry_count`는 실제 재시도 횟수를 누적합니다.
- `job_runs`에는 `next_retry_at`, `lease_expires_at`가 저장됩니다.

기본 설정:

- `loop_sleep_seconds`
- `retry_backoff_seconds`
- `max_retry_count`
- `job_lease_seconds`
- `exit_management_interval_minutes`
- `outcome_resolution_interval_minutes`

`begin_job_run()`은 `(job_name, run_key)` 기준으로 canonical row를 다시 조회해서 실제 DB의 `job_run_id`를 반환합니다.

## Repository Ordering Guarantees

`latest_*` helper는 같은 초에 여러 행이 생겨도 안정적으로 최신 상태를 고르도록 정렬을 강화했습니다.

- timestamp는 microseconds까지 저장
- `ORDER BY created_at DESC, rowid DESC`
- 또는 `ORDER BY updated_at DESC, rowid DESC`

적용 대상 예:

- `latest_account_snapshot()`
- `latest_position_by_symbol()`
- `latest_cooldown_until()`
- `latest_job_heartbeat()`
- `recent_job_health()`

## Runtime Initialization

초기화:

```bash
python scripts/init_runtime.py
```

초기화 시 기본 runtime flag는 batch `INSERT OR IGNORE`로 생성됩니다.

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

현재 회귀 테스트는 아래 항목을 고정합니다.

- equity / drawdown / signed exposure
- exit order가 daily entry count에 섞이지 않는지
- pending entry / partial fill이 duplicate entry를 막는지
- same-second `latest_*` ordering 안정성
- scheduler retry/backoff non-blocking
- `begin_job_run()` canonical idempotency
- partial fill on/off
- cooldown 진입 차단
- retraining cadence
- KIS order lifecycle / reconcile idempotency

## Known Limitations

- 한국주식 KIS mock 체결은 holdings delta 기반 reconcile이라 실시간 체결 원장과 1:1은 아닙니다.
- `1d` cooldown은 영업일 근사이며 완전한 exchange calendar는 아닙니다.
- background worker는 단일 프로세스 운영을 기본 가정으로 합니다.
- UI routing/floating nav는 이번 라운드 범위 밖입니다.
