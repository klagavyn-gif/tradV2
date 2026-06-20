# Debug Session: local-binance-runtime

- Status: OPEN
- Scope: debug local Binance wrapper runtime issue for cache refresh and phase1 refresh scripts
- Started: 2026-06-13

## Symptoms

- `tools/daily_cache_refresh_binance.ps1` reaches the underlying script but the Python work fails at runtime.
- `tools/daily_phase1_refresh_binance.ps1` also reaches the underlying script but the Python work fails at runtime.

## Expected

- Local Binance wrapper scripts should run through the existing refresh pipeline successfully when Binance access works on the local machine.

## Hypotheses

1. The wrapper scripts are correct, but the underlying Python dataset/cache scripts fail because they assume Yahoo-specific cache state or file layout in one or more code paths.
2. The local Binance fetch path returns data in a shape or time index that breaks downstream refresh logic during cache load/coverage summarization.
3. The runtime failure is caused by an unhandled provider-specific path resolution issue, such as reading from a provider-aware store in one step and a Yahoo-specific location in another.
4. The failure is not Binance fetch itself, but a CLI/runtime argument handling issue inside the helper scripts that only appears when the wrapper forwards empty or default values.
5. The local environment can reach Binance, but one of the downstream scripts crashes before logs are persisted, which is why the expected `.log` files were not created.

## Evidence Log

- Confirmed hypothesis 4 with a PowerShell native-command repro:
  - Passing an empty trailing argument to `python` drops that argument entirely.
  - Repro output in `.dbg/argv_probe.out` showed `['.\\\\.dbg\\\\argv_probe.py', 'alpha', 'beta']`, not a fourth empty argument.
- This matches `tools/daily_cache_refresh.ps1`, where the temp Python script expected `sys.argv[3]` even when `Watchlist` was blank.
- Post-fix runtime evidence from `.dbg/trae-debug-log-local-binance-runtime.ndjson` shows:
  - `refresh_cache success` with `provider=binance`, `items=22`
  - `load_cache success` with `cache_keys=22`
  - Provider-aware cache paths exist under `.data\\market_history\\binance`

## Findings

- Confirmed: hypothesis 4
- Rejected for this failure: hypotheses 1, 2, 3, and 5 as primary root cause for the `cache refresh` runtime failure

## Fix

- `tools/daily_cache_refresh.ps1`
  - Make the temp Python script tolerate missing `sys.argv[3]`
  - Build the Python argument array dynamically and omit `Watchlist` when blank

## Verification

- `daily_cache_refresh_binance.ps1 -Days 1` no longer fails immediately on the temp-script argument read path.
- Debug events confirm Binance refresh and provider-aware cache loading succeed after the fix.

## Next Step

- Keep instrumentation in place until user confirms the local Binance wrappers behave as expected.
