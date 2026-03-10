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
- `kr_intraday_15m` experimental family
  - `kr_intraday_15m_v1`
    - regular-session `15m`
    - complete-bar only, opening bar blocked, no overnight, session flatten near the close
    - disabled by default
  - `kr_intraday_15m_v1_after_close_close`
    - after-close close-price session
    - `15:40~16:00`
    - fixed same-day close price
    - disabled by default
    - first recommended after-hours mode when testing KR 15m after-hours execution
  - `kr_intraday_15m_v1_after_close_single`
    - after-close single-price session
    - `16:00~18:00`
    - 10-minute auction cadence
    - disabled by default
    - more experimental than the close-price mode

The `15m` path is intentionally a separate strategy family instead of a timeframe toggle on the `1h` strategy. Regular and after-hours modes share the completed-bar signal base, but keep separate session windows, price policies, order intents, no-op reasons, conflict handling, and monitoring rows.

### KR strategy family matrix

| strategy_id | timeframe | session_mode | enabled_by_default | experimental | execution_account | intended use |
| --- | --- | --- | --- | --- | --- | --- |
| `kr_daily_preclose_v1` | `1d` | `pre_close` | `false` | `false` | `kis_kr_paper` | legacy |
| `kr_intraday_1h_v1` | `1h` | `regular` | `true` | `false` | `kis_kr_paper` | recommended default |
| `kr_intraday_15m_v1` | `15m` | `regular` | `false` | `true` | `kis_kr_paper` | experimental |
| `kr_intraday_15m_v1_after_close_close` | `15m family` | `after_close_close_price` | `false` | `true` | `kis_kr_paper` | post-close fixed close |
| `kr_intraday_15m_v1_after_close_single` | `15m family` | `after_close_single_price` | `false` | `true` | `kis_kr_paper` | 10-minute single-price auction |

Current default KR execution policy:

- recommended default: `kr_intraday_1h_v1`
- experimental opt-in only:
  - `kr_intraday_15m_v1`
  - `kr_intraday_15m_v1_after_close_close`
  - `kr_intraday_15m_v1_after_close_single`
- legacy only: `kr_daily_preclose_v1`

Current KR execution order:

1. choose active KR strategy and session mode
2. apply the matching session gate
   - `kr_daily_preclose_v1` -> daily pre-close gate
   - `kr_intraday_1h_v1` / `kr_intraday_15m_v1` -> regular intraday session gate
   - `kr_intraday_15m_v1_after_close_close` -> `15:40~16:00`
   - `kr_intraday_15m_v1_after_close_single` -> `16:00~18:00`
3. fetch the matching KIS quote / orderbook context for that session
4. check KIS buying power or sellable quantity
5. run repository checks for pending entry, holding state, cooldown, and risk budget
6. submit with the session-specific order intent
7. broker order sync via:
   - websocket execution event if available
   - REST daily order/fill query
   - holdings/account fallback only as last resort

### Session-mode differences

- `kr_intraday_1h_v1`
  - allowed time: regular intraday window
  - price policy: market best-effort / orderbook-based
  - execution policy: regular-session submit + regular broker sync
  - typical no-op / reject: `market_closed`, `outside_intraday_entry_window`, `insufficient_buying_power`
  - why recommended: operationally simplest KR strategy and least fragile runtime behavior
- `kr_intraday_15m_v1`
  - allowed time: `09:15~14:45`
  - price policy: market best-effort / orderbook-based
  - execution policy: completed `15m` bar only, opening bar blocked, flatten near close
  - typical no-op / reject: `opening_bar_blocked`, `outside_intraday_entry_window`, `kr_strategy_conflict_pending`
  - why experimental: more frequent intraday entries and tighter cadence than the default `1h`
- `kr_intraday_15m_v1_after_close_close`
  - allowed time: `15:40~16:00`
  - price policy: same-day close price
  - execution policy: session-specific `after_close_close_price` submit on KIS mock
  - typical no-op / reject: `outside_after_close_close_session`, `after_close_price_unavailable`, `after_close_buying_power_insufficient`
  - why recommended first among after-hours modes: simpler than auction mode and easier to reason about operationally
- `kr_intraday_15m_v1_after_close_single`
  - allowed time: `16:00~18:00`
  - price policy: after-hours single-price expected / auction quote
  - execution policy: `10분` 단일가 경매 cadence, while signals still come from completed `15m` bars
  - typical no-op / reject: `outside_after_close_single_session`, `after_close_single_waiting_auction`, `after_close_single_quote_unavailable`
  - why experimental: signal cadence (`15m`) and execution cadence (`10m`) differ, so it is materially harder to reason about than regular or after-close close mode

KR strategy rules:

- only `kis_kr_paper` is allowed as the execution account
- the full KR `15m` family uses completed `15m` bars only; no partial-bar look-ahead
- `kr_intraday_15m_v1`
  - `09:00~09:15` opening bar blocked
  - new entries allowed only from `09:15` to `14:45`
  - flatten between `15:15` and `15:20`
- `kr_intraday_15m_v1_after_close_close`
  - session mode: `after_close_close_price`
  - entries allowed only from `15:40` to `16:00`
  - price policy: same-day close price
- `kr_intraday_15m_v1_after_close_single`
  - session mode: `after_close_single_price`
  - entries allowed only from `16:00` to `18:00`
  - price policy: after-hours single-price expected/auction quote
  - 10-minute auction cadence gate
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
- `kr_intraday_15m` family: experimental, off-by-default
  - `kr_intraday_15m_v1`
  - `kr_intraday_15m_v1_after_close_close`
  - `kr_intraday_15m_v1_after_close_single`
- `kr_daily_preclose_v1`: legacy fallback, disabled unless explicitly opted in

For KR operations, the default recommendation remains `kr_intraday_1h_v1`. The `15m` family exists for opt-in experimentation, and `kr_intraday_15m_v1_after_close_single` should be treated as the highest-risk KR mode in this repository.

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
  - strategy id / label / timeframe / session mode / enabled / experimental
  - broker mode / execution account / session window / price policy / execution cadence / intended use
  - candidate / allowed / submit requested / submitted / acknowledged / pending fill / filled / rejected / cancelled
  - no-op reason breakdown / reject reason breakdown
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
- KR strategy conflict blocking across `kr_daily_preclose_v1`, `kr_intraday_1h_v1`, and the full `kr_intraday_15m` family
- KR `15m` complete-bar gate / opening bar block / session flatten / after-hours session-mode gating
- websocket execution event state transition
- REST reconcile fallback when websocket is absent
- scheduler retry/backoff and lease refresh
- dashboard execution summary and broker sync state

## Known Limitations

- KR websocket execution handling is implemented as an ingest path.
- KR quote websocket streaming is now auto-started by the worker for domestic stock symbols that matter to runtime monitoring (open KR positions, open KR orders, recent KR candidates, watchlist).
- realtime KR quote updates are used for monitoring-only mark-to-market display; strategy signal generation still uses completed bars only.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- Reservation orders are not wired into the worker yet.
- KR daily pre-close and intraday flatten windows are still schedule-based, not exchange-calendar-perfect.
- KR intraday history currently depends on the configured market data provider for `1h` / `15m` bars and trims incomplete bars locally before signal generation.
- KR after-close close-price execution uses explicit session-mode submission and REST/websocket reconcile fallback; it is not backed by a dedicated after-hours execution manager yet.
- KR after-close single-price execution is more fragile than regular or after-close close-price because quote availability and 10-minute auction timing can delay or reject orders.
- US equities and crypto still use the sim broker path.
- legacy rows without `account_id` are backfilled during migration, but very old mixed sim snapshots can only be mapped conservatively to `sim_legacy_mixed` when asset type is unavailable.
- `sim_legacy_mixed` is migration/backfill-only and must not appear in new runtime writes.
- beta monitor depends on Streamlit DOM selectors for header/toolbar suppression and still needs visual regression checks after Streamlit upgrades.
- beta monitor now auto-refreshes on a short cadence for realtime quote display, so visual checks should also verify scroll/anchor behavior after Streamlit upgrades.

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
