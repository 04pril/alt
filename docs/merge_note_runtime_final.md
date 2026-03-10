# Final Runtime Merge Note

- Merge recommendation: `go`
- Recommended default profile: `balanced`
- `active` remains an opt-in aggressive / experimental profile; this branch may still run it intentionally for US `15m` verification
- KR recommended default strategy: `kr_intraday_1h_v1`
- KR experimental 15m family:
  - `kr_intraday_15m_v1` -> regular session
  - `kr_intraday_15m_v1_after_close_close` -> after-close close-price session
  - `kr_intraday_15m_v1_after_close_single` -> after-close single-price session
  - all off-by-default
- KR legacy strategy: `kr_daily_preclose_v1` (disabled by default)
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
2. Verify one KR `kr_intraday_1h_v1` candidate passes through `broker_account_sync -> candidate -> entry_allowed -> submit_requested -> submitted/acknowledged -> filled` with matching `account_id=kis_kr_paper` order/fill/position rows.
3. Verify US equity and crypto keep separate `account_id`, cash, equity, and drawdown state.
4. Confirm operations monitoring exposes per-account cards, KR strategy summary rows, and marks the total portfolio view as non-orderable reference data.
5. If KR 15m after-hours is being tested, enable only one session mode at a time and confirm monitoring shows the expected `session_mode` (`regular`, `after_close_close`, `after_close_single`) for submit / fill / noop rows.

Known limitations:

- KR websocket execution ingest remains separate from the KR quote websocket stream.
- KR quote websocket is auto-started by the worker for monitoring, but it only updates display mark-to-market paths; strategy decisions still stay on completed bars.
- If websocket events are unavailable, KR execution falls back to REST daily order/fill reconcile.
- KR daily pre-close and intraday flatten windows remain schedule-based rather than exchange-calendar-perfect.
- KR intraday `1h` / `15m` history still depends on provider availability and local incomplete-bar trimming.
- KR after-close execution is session-mode aware, but it still relies on explicit KIS quote/orderbook lookups and REST reconcile fallback rather than a dedicated after-hours websocket execution manager.
- `kr_intraday_15m_v1_after_close_single` is the highest-risk KR mode in this branch: 10-minute auction cadence, quote availability, and fill timing are materially more fragile than regular or after-close close-price mode.
- Reservation orders are not wired into the worker yet.
- legacy rows without `account_id` are backfilled conservatively; very old mixed sim rows can fall back to `sim_legacy_mixed` when no asset hint exists.
- `sim_legacy_mixed` is migration/backfill-only and should not appear in new runtime writes.
- beta monitor depends on Streamlit DOM selectors for header/toolbar suppression, so visual regression checks are still required after Streamlit upgrades.
- beta monitor now auto-refreshes on a short cadence; visual checks should include scroll position and anchor behavior during refresh.
