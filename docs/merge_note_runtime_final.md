# Final Runtime Merge Note

- Merge recommendation: `go`
- Recommended default profile: `balanced`
- Broker policy:
  - `한국주식` + KR symbol only -> `kis_mock`
  - `미국주식`, `코인` -> `sim`
- Canonical account ledger: `account_snapshots`
  - KR risk path -> `source="kis_account_sync"` first
  - US/crypto risk path -> sim-only snapshots

Manual pre-merge checks:

1. Start the worker with the balanced profile and confirm `runtime_profile_name=balanced` in control flags and operations monitoring.
2. Verify one KR candidate passes through `broker_account_sync -> candidate -> entry_allowed -> submit_requested -> submitted/acknowledged -> filled` with matching order/fill/position rows.
3. Confirm operations monitoring still shows `한국주식=kis_mock`, `미국주식=sim`, `코인=sim` and `broker_sync_errors=0` on a healthy path.

Known limitations:

- KR websocket execution handling exists as an ingest path, but a persistent websocket manager is not auto-started by the worker yet.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- KR daily pre-close gating remains schedule-based rather than exchange-calendar-perfect.
- Reservation orders are not wired into the worker yet.
