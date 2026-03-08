# Streamlit Monitor + 24/7 Paper Trading Runtime

This repository separates the operator UI from the runtime worker.

- `app.py`: monitoring and manual intervention UI
- `jobs/`, `services/`, `storage/`: Streamlit-independent worker/runtime
- `storage/repository.py`: single source of truth for predictions, orders, fills, positions, account snapshots, job runs, and system events

The current routing model stays the same:

- KR equities: `KISPaperBroker` first when KIS mock config is available
- US equities and crypto: SQLite-backed `PaperBroker`
- predictor semantics and `run_forecast_on_price_data(...)` reuse are unchanged

## Runtime Architecture

```text
Market Data -> Universe Scanner -> Signal Engine -> Candidate Scans
                                           |
                                           v
                                   Prediction Ledger

Candidate -> Risk Gate -> Broker Preflight -> Submit -> Ack/Pending/Fill/Reject
                                              |
                                              +-> PaperBroker (sim)
                                              +-> KISPaperBroker (KR mock)

Orders -> Fills -> Positions -> Account Snapshots
Predictions -> Outcome Resolver -> Outcomes -> Evaluations
Job Runs + System Events + Account Snapshots -> Monitoring
```

## Execution Assurance Pipeline

The worker now treats execution as an explicit state machine instead of a best-effort side effect.

Entry pipeline:

1. `candidate`
2. `entry_allowed` or `entry_rejected`
3. `submit_requested`
4. `submitted`
5. `acknowledged`
6. `pending_fill`
7. `partially_filled`
8. `filled`
9. `rejected`
10. `cancelled`
11. `expired`

Silent skips are not allowed. If the worker does not trade, it records a reason in `system_events`.

Typical reason codes:

- `market_closed`
- `outside_preclose_window`
- `duplicate_pending_entry`
- `cooldown_active`
- `insufficient_buying_power`
- `no_sellable_qty`
- `no_quote`
- `broker_rejected`
- `missing_prediction`
- `no_candidate`
- `no_submit`

## Explicit Broker Sync Jobs

The scheduler now runs broker-facing sync work as explicit jobs.

- `broker_market_status`
- `broker_position_sync`
- `broker_order_sync`
- `broker_account_sync`

These jobs are:

- idempotent
- visible in `job_runs`
- visible in `system_events`
- eligible for retry/backoff
- manually runnable from the monitor UI

## KR Execution Path

KR daily entries keep pre-close gating. The system does not submit KR daily entry orders outside the allowed pre-close window unless a separate reservation policy is added later.

Current KR execution order:

1. schedule gate and market phase check
2. pre-close gate
3. KIS quote and expected/ask-bid price lookup
4. KIS buying power or sellable quantity lookup
5. repository checks for pending entry, holding state, cooldown, and risk budget
6. broker submit
7. broker order sync via:
   - websocket execution event if available
   - REST daily order/fill query
   - holdings/account fallback only as last resort

Source-of-truth priorities for KR mock execution:

1. websocket execution event
2. REST order/fill reconcile
3. account/holdings delta fallback

`broker_order_id` is preserved alongside internal `order_id`, and `FillRecord` is only written after execution is actually confirmed.

## KIS API Surface Used

The runtime now has code paths for these KIS mock capabilities:

- OAuth access token
- hashkey issuance
- websocket approval key issuance
- cash stock order
- order cancel
- domestic balance / account snapshot
- buying power lookup
- sellable quantity lookup
- daily order/fill lookup
- quote lookup
- expected execution / orderbook lookup

HTS ID:

- `kis_devlp.yaml` may include `my_htsid`
- websocket-style execution handling depends on it being configured
- the runtime records whether HTS ID and websocket approval key are available during account sync

## Scheduler / Retry / Lease

Scheduler behavior:

- failed jobs get per-job backoff through `next_retry_at`
- one failing job does not block unrelated jobs
- long-running jobs refresh lease and worker heartbeat via runtime `touch()` callbacks
- broker sync jobs use their own cadence from settings

Relevant settings:

- `scheduler.loop_sleep_seconds`
- `scheduler.retry_backoff_seconds`
- `scheduler.max_retry_count`
- `scheduler.job_lease_seconds`
- `scheduler.broker_market_status_interval_minutes`
- `scheduler.broker_order_sync_interval_minutes`
- `scheduler.broker_position_sync_interval_minutes`
- `scheduler.broker_account_sync_interval_minutes`
- `broker.websocket_reconnect_interval_seconds`
- `broker.stale_submitted_order_timeout_minutes`

## Monitoring Read Model

`monitoring/dashboard_hooks.py` builds the monitor view from repository data.

It exposes:

- today candidate / allowed / rejected / submit requested / submitted / acknowledged / filled / rejected / cancelled / noop counts
- noop reason breakdown
- latest broker sync statuses
- pending submitted KR orders
- broker rejects today
- last websocket execution timestamp

## Accounting Definitions

Runtime accounting uses these definitions consistently:

- `gross_exposure`: sum of absolute open position exposure
- `net_exposure`: sum of signed open position exposure
- `equity`: `cash + net_exposure`
- `unrealized_pnl`: sum of mark-to-market PnL across open positions
- `drawdown_pct`: drawdown versus peak account equity

`unrealized_pnl` is not added on top of market value a second time.

## Local Run

Install:

```bash
pip install -r requirements.txt
```

Initialize DB only:

```bash
python -m jobs.scheduler --init-db
```

Run one worker cycle:

```bash
python -m jobs.scheduler --once
```

Run worker loop:

```bash
python -m jobs.scheduler
```

Run monitor:

```bash
python -m streamlit run app.py
```

## Tests

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Execution assurance coverage includes:

- pending entry duplicate blocking
- KR market closed / outside pre-close / insufficient buying power rejection
- KR submit -> acknowledged -> pending_fill -> filled / rejected / cancelled
- websocket execution event state transition
- REST reconcile fallback when websocket is absent
- scheduler retry/backoff and lease refresh
- dashboard execution summary and broker sync state

## Known Limitations

- KR websocket execution handling is implemented as an ingest path, but a persistent websocket manager is not auto-started by the worker yet.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- Reservation orders are not wired into the worker yet.
- KR daily pre-close gating is still schedule-based, not exchange-calendar-perfect.
- US equities and crypto still use the sim broker path.
