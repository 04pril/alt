# Final Runtime Merge Note

- Merge recommendation: `go`
- Recommended default profile: `balanced`
- Broker policy:
  - `한국주식` + KR symbol only -> `kis_mock` -> `kis_kr_paper`
  - `미국주식` -> `sim` -> `sim_us_equity`
  - `코인` -> `sim` -> `sim_crypto`
- Canonical account ledger: `account_snapshots`
  - canonical scope -> `account_id`
  - `source` -> provenance metadata only
  - total portfolio view -> read-only reference only

Manual pre-merge checks:

1. Start the worker with the balanced profile and confirm `runtime_profile_name=balanced` in control flags and operations monitoring.
2. Verify one KR candidate passes through `broker_account_sync -> candidate -> entry_allowed -> submit_requested -> submitted/acknowledged -> filled` with matching `account_id=kis_kr_paper` order/fill/position rows.
3. Verify US equity and crypto keep separate `account_id`, cash, equity, and drawdown state.
4. Confirm operations monitoring exposes per-account cards and marks the total portfolio view as non-orderable reference data.

Known limitations:

- KR websocket execution handling exists as an ingest path, but a persistent websocket manager is not auto-started by the worker yet.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- KR daily pre-close gating remains schedule-based rather than exchange-calendar-perfect.
- Reservation orders are not wired into the worker yet.
- legacy rows without `account_id` are backfilled conservatively; very old mixed sim rows can fall back to `sim_legacy_mixed` when no asset hint exists.
