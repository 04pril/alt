# Streamlit Monitor + 24/7 Paper Trading Runtime

This repository separates the operator UI from the runtime worker.

- `app.py`: monitoring and manual intervention UI
- `jobs/`, `services/`, `storage/`: Streamlit-independent worker/runtime
- `storage/repository.py`: single source of truth for predictions, orders, fills, positions, account snapshots, job runs, and system events

The current routing model stays the same:

- KR equities: `KISPaperBroker` first when KIS mock config is available
- US equities and crypto: SQLite-backed `PaperBroker`
- predictor semantics and `run_forecast_on_price_data(...)` reuse are unchanged

KR-only routing rule:

- `한국주식` + KR symbol (`.KS` / `.KQ`) or bare 6-digit KR stock code only: `kis_mock`
- `미국주식` and `코인`: always `sim`
- blank symbol never routes to KIS; asset-only runtime/monitoring events use an explicit asset-scope helper instead of execution routing
- if `asset_type` is blank or ambiguous, KR suffix / bare 6-digit KR code is the fallback safety check
- non-KR symbols never route to KIS, even if a wrong `한국주식` label is passed in

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

Orders -> Fills -> Positions -> Account Snapshots(account_id scoped)
Predictions -> Outcome Resolver -> Outcomes -> Evaluations
Job Runs + System Events + Account Snapshots -> Monitoring
```

`account_snapshots` in `storage/repository.py` remain the canonical account-state ledger, but the canonical key is now the execution account, not the event source.

Execution accounts:

- `kis_kr_paper`
- `sim_us_equity`
- `sim_crypto`
- `sim_legacy_mixed`
  - migration/backfill-only legacy bucket for very old mixed sim rows without asset hints
  - not used by new runtime writes

Account policy:

- KR equities + KR symbols -> `broker_mode=kis_mock`, `account_id=kis_kr_paper`
- US equities -> `broker_mode=sim`, `account_id=sim_us_equity`
- crypto -> `broker_mode=sim`, `account_id=sim_crypto`

Design rules:

- `orders`, `fills`, `positions`, `account_snapshots`, and execution events are written with `account_id`
- `predictions` and `candidate_scans` keep `execution_account_id`
- `source` remains provenance metadata only
- `RiskEngine` always reads account-scoped snapshots, pending orders, positions, cooldowns, and drawdown peaks
- total portfolio aggregation is read-only reference data and is never used for buying power or risk gating

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

`broker_account_sync` also propagates runtime `touch()` callbacks through the router and broker layers, so a slow account sync refreshes lease and worker heartbeat mid-run instead of only at job start.

## KR Execution Path

KR execution now has an explicit strategy layer instead of treating all KR trades as one schedule.

KR strategies:

- `kr_daily_preclose_v1`
  - legacy daily strategy
  - pre-close gated
  - disabled by default
- `kr_intraday_1h_v1`
  - current recommended default KR strategy
  - `1h` bars
  - KIS mock only
- `kr_intraday_15m_v1`
  - separate experimental KR intraday strategy
  - `15m` bars
  - disabled by default
  - complete-bar only, opening bar blocked, no overnight, session flatten near the close

The `15m` path is intentionally a separate strategy instead of a timeframe toggle on the `1h` strategy. It has its own target horizon, feature profile, gating window, conflict rules, and monitoring rows.

Current default KR execution policy:

- recommended default: `kr_intraday_1h_v1`
- experimental opt-in only: `kr_intraday_15m_v1`
- legacy only: `kr_daily_preclose_v1`

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

KR strategy rules:

- only `kis_kr_paper` is allowed as the execution account
- `kr_intraday_15m_v1` uses completed `15m` bars only
- `09:00~09:15` opening bar is blocked for the `15m` strategy
- new `15m` entries are allowed only from `09:15` to `14:45`
- intraday KR strategies flatten between `15:15` and `15:20`
- duplicate pending entry, active strategy conflict, and cross-strategy same-symbol overlap are blocked inside the same KIS account

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

## Runtime Tuning Profiles

Gate tuning is tracked as explicit runtime profiles instead of editing the embedded defaults in place.

- `baseline`: current production-equivalent gate values
- `balanced`: recommended default for live paper trading; relaxes stock entry gates enough to reduce `outside_preclose_window`, `expected_return_too_low`, and `confidence_too_low` pressure without changing the core loss-budget controls
- `active`: higher-submission experimental profile; keeps the same drawdown and daily-loss ceilings but further lowers stock entry thresholds, expands daily entry capacity, and runs US equities on a `15m` model / `15분` scan cadence

KR strategy policy inside all profiles:

- default KR strategy id: `kr_intraday_1h_v1`
- `kr_intraday_15m_v1`: experimental, off-by-default
- `kr_daily_preclose_v1`: legacy fallback, disabled unless explicitly opted in

This round keeps the score formula unchanged. Only entry gates and pre-close cadence windows are tuned.

When trades are too sparse, inspect these breakdowns first:

- `today_noop_reason_breakdown`
- `today_entry_rejected_reason_breakdown`
- `outside_preclose_window`
- `expected_return_too_low`
- `confidence_too_low`
- `cooldown_active`
- `max_daily_entries`

The worker writes the loaded runtime profile name/source into control flags, and the operations monitor reads that value back so operators can verify which profile is active. The recommended default profile is `balanced`.
This workspace may still choose to run `active` intentionally, but that should be treated as an opt-in aggressive profile rather than the recommended default.

## Monitoring Read Model

`monitoring/dashboard_hooks.py` builds the monitor view from repository data.

It exposes:

- `accounts_overview`
  - `kis_kr_paper`
  - `sim_us_equity`
  - `sim_crypto`
- per account:
  - `cash`
  - `equity`
  - `drawdown_pct`
  - `open_positions`
  - `pending_orders`
  - `broker_mode`
  - `last_sync_time`
  - `last_sync_status`
- `total_portfolio_overview`
  - read-only aggregate summary
  - currency buckets stay separate unless the UI converts them explicitly
  - must be labeled as "not orderable buying power"
- today candidate / allowed / rejected / submit requested / submitted / acknowledged / filled / rejected / cancelled / noop counts
- noop reason breakdown
- latest broker sync statuses
- pending submitted KR orders
- broker rejects today
- last websocket execution timestamp
- `kr_strategy_overview`
  - strategy id / label / timeframe / enabled / experimental
  - today candidate / allowed / rejected / submitted / filled / noop counts
  - per-strategy open positions / pending orders
  - top noop / reject reason
  - recent strategy event timestamps

## Accounting Definitions

Runtime accounting uses these definitions consistently:

- `gross_exposure`: sum of absolute open position exposure
- `net_exposure`: sum of signed open position exposure
- `equity`: `cash + net_exposure`
- `unrealized_pnl`: sum of mark-to-market PnL across open positions
- `drawdown_pct`: drawdown versus peak account equity

`unrealized_pnl` is not added on top of market value a second time.

Account-level rules:

- drawdown peaks are tracked per `account_id`
- KR KIS cash/equity/positions never mix with US equity sim or crypto sim
- USD accounts remain USD in the ledger
- dashboards may convert USD to KRW for display only, and must keep the original currency visible

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
- KR market closed / outside intraday entry window / insufficient buying power rejection
- KR submit -> acknowledged -> pending_fill -> filled / rejected / cancelled
- KR strategy conflict blocking across `kr_daily_preclose_v1`, `kr_intraday_1h_v1`, `kr_intraday_15m_v1`
- KR `15m` complete-bar gate / opening bar block / session flatten
- websocket execution event state transition
- REST reconcile fallback when websocket is absent
- scheduler retry/backoff and lease refresh
- dashboard execution summary and broker sync state

## Known Limitations

- KR websocket execution handling is implemented as an ingest path, but a persistent websocket manager is not auto-started by the worker yet.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- Reservation orders are not wired into the worker yet.
- KR daily pre-close and intraday flatten windows are still schedule-based, not exchange-calendar-perfect.
- KR intraday history currently depends on the configured market data provider for `1h` / `15m` bars and trims incomplete bars locally before signal generation.
- US equities and crypto still use the sim broker path.
- legacy rows without `account_id` are backfilled during migration, but very old mixed sim snapshots can only be mapped conservatively to `sim_legacy_mixed` when asset type is unavailable.
- `sim_legacy_mixed` is migration/backfill-only and must not appear in new runtime writes.
- beta monitor depends on Streamlit DOM selectors for header/toolbar suppression and still needs visual regression checks after Streamlit upgrades.

## Final Merge Note

- Merge recommendation: `go`
- Recommended default profile: `balanced`
- Broker policy:
  - `한국주식 + KR 심볼 -> kis_mock -> kis_kr_paper`
  - `미국주식 -> sim -> sim_us_equity`
  - `코인 -> sim -> sim_crypto`
- Canonical ledger: `account_snapshots`
  - canonical scope is `account_id`
  - `source` is metadata, not routing or risk identity
  - total portfolio view is reference-only

Manual pre-merge checks:

1. Start the worker with the balanced profile and confirm `runtime_profile_name=balanced` and the same value in the monitoring read model.
2. Verify one KR candidate flows through `broker_account_sync -> candidate -> entry_allowed -> submit_requested -> submitted/acknowledged -> filled` with matching `account_id=kis_kr_paper` order/fill/position rows.
3. Verify US equity rows land in `sim_us_equity` and crypto rows land in `sim_crypto` with independent cash/equity/drawdown.
4. Confirm operations monitoring shows per-account cards first and labels the total portfolio view as reference-only.
