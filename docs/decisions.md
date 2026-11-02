# Decisions

## ADR-001 — Storage: append-only Parquet partitions
- Status: accepted
- Date: 2026-11-01
- Context: A nightly GitHub Action must append half-hourly carbon intensity rows and commit them to a public repository so reviewers can inspect the data in git diffs.
- Decision: Store data as append-only Parquet partitions, one file per day: `data/curated/date=<YYYY-MM-DD>.parquet` (superseding the initial `data/raw/` path proposal as codified in ADR-003), deduplicated on the period start timestamp in UTC. No DuckDB, no SQLite, no database file committed to git.
- Consequences: Each daily append creates an isolated partition file easily versioned and inspected; avoids binary database corruption or unreviewable blob diffs.
- Alternatives rejected: DuckDB database file (rejected: a binary file the nightly Action rewrites and commits whole, which no reviewer can diff), SQLite (same), Parquet plus a DuckDB query layer (an extra dependency and a second system to explain, for no benefit at ~35k rows).

## ADR-002 — Makefile: POSIX-first common subset
- Status: accepted
- Date: 2026-11-01
- Context: Local development runs on Windows 11 where Make recipes execute under cmd.exe, while CI executes under /bin/sh on ubuntu-latest. The build system must operate reliably across both without platform divergence.
- Decision: A single POSIX-first Makefile with an `ifeq ($(OS),Windows_NT)` switch for the interpreter path, and recipe bodies strictly restricted to the cmd/sh common syntax subset. Conditional or destructive logic is handled exclusively in Python.
- Consequences: Commands run identically on local Windows workstations and Linux CI runners without PowerShell recipes, wrapper scripts, Docker overhead, or task-runner indirection.
- Alternatives rejected: PowerShell-based build scripts (non-portable to Linux CI), Docker-wrapping every target (slow local development feedback loop, Docker deferred to P5), Python task runners like Invoke/Taskfile (extra indirection layer when Make is standard).

### Amendment: Quoting and forward-slash path convention
- Date: 2026-11-01
- Context: On Windows workstations, Make may invoke recipes under `cmd.exe` or Git Bash / MSYS2 `sh.exe`, while paths may contain spaces (e.g. `C:\Program Files\Python310\python.exe`). Testing verified that:
  - `cmd /c ".venv/Scripts/python.exe" -V` -> `Python 3.10.11` (quoted forward-slash path succeeds in cmd.exe)
  - `cmd /c .venv/Scripts/python.exe -V` -> `'.venv' is not recognized` (unquoted forward slash fails in cmd.exe)
  - `sh -c 'echo .venv\Scripts\python.exe'` -> `.venvScriptspython.exe` (POSIX shells strip unquoted backslashes)
- Decision: Use forward slashes in interpreter paths (`PY ?= .venv/Scripts/python.exe`) and double-quote all expansions of `$(PY)` and `$(PYTHON)` across all recipes (e.g., `"$(PY)" -m pytest`). Shell detection via `$(SHELL)` is avoided because `$(SHELL)` reports `sh.exe` even when Make is actually driving `cmd.exe`.


## ADR-003 — Two-tier storage and local-day partitioning
- Status: accepted
- Date: 2026-11-01
- Context: Live measurements against the GB Carbon Intensity API on 2026-11-01 established the following facts:
  - The endpoint (`/intensity/date/<YYYY-MM-DD>`) is addressed by Europe/London local day, not UTC day:
    - `2026-10-25` (autumn clocks go back): 50 periods, first period `2026-10-24T23:00Z`, last period ending `2026-10-26T00:00Z`.
    - `2026-03-29` (spring clocks go forward): 46 periods, first period `2026-03-29T00:00Z`, last period ending `2026-03-29T23:00Z`.
    - Ordinary day: 48 periods.
    - No duplicate `from` timestamps exist because the API emits UTC timestamps. Consecutive local days are contiguous and non-overlapping.
  - Archive floor is `2017-09-12` (local day). `2017-09-11` and earlier return `data: []`. The operator's baseline `forecast` is non-null from the very first period on `2017-09-12`, confirming the comparison window is limited solely by the archive, not by forecast retention. The committed dataset covers the 2-year backfill window `[2024-11-01, present]`; probe partitions from 2017 are excluded from `data/curated/` so lag features (e.g. `shift(48)`) do not bridge a 7-year unrecorded gap.
  - Storage must support fast replay without API hammering, whilst keeping git diffs clean and reliable.
- Decision:
  1. Implement two-tier storage with separate retentions:
     | Tier | Path | Committed | Retention |
     |---|---|---|---|
     | Raw HTTP cache (replay buffer) | `data/cache/http/<date>.json.gz` | **no** (gitignored) | 90 days, pruned automatically |
     | Curated half-hourly table | `data/curated/date=<YYYY-MM-DD>.parquet` | **yes** | never deleted; append-only |
  2. Partition by requested local day (`local_date`), matching the API request boundary. One partition file equals one successful request for one local day, so rerunning a date rewrites exactly that file. The uniqueness key is `period_start_utc`.
  3. Store both `utc_date` and `local_date` columns so downstream features or models never need a re-partition.
- Consequences:
  Re-ingesting any date is idempotent and atomic. Raw cache allows re-runs without hitting the external API, while curated parquet files remain compact, diffable, and permanent.
- Alternatives rejected:
  - Single-tier raw storage (rejected: committing raw JSON blobs bloats git history with ephemeral responses).
  - UTC day partitioning (rejected: requires multi-request stitching per API call due to Europe/London local day addressing and DST offsets).

### Amendment: Upstream malformed JSON and range endpoint fallback
- Date: 2026-11-01
- Context: During historical backfills, certain dates on `/intensity/date/<YYYY-MM-DD>` (e.g. `2025-01-12`, `2025-01-13`, `2025-08-11`) return HTTP 200 with unparseable JSON where absent forecasts are serialized as `"forecast": ,`. The identical periods are available with valid JSON via `/intensity/{start_utc}/{end_utc}` with actuals intact and null forecasts.
- Decision: Treat `JSONDecodeError` on `/intensity/date` as a transient failure with exponential backoff retries. If parsing still fails after `max_attempts`, fall back to `/intensity/{start_utc}/{end_utc}` for the local Europe/London day (local midnight to next local midnight in UTC). Filter records strictly where `local_date` matches the requested date, preserving null forecasts without imputation. Guard `data: []` post archive floor (`2017-09-12`) with a single range check before reporting `no_data`.

### Amendment: Cache usability predicate and coalescing upsert
- Date: 2026-11-01
- Context: When ingesting intraday or unsettled days, raw cache payloads may capture transient null actual values. If subsequent runs unconditionally hit the cache, newly published upstream observations are never received, and once periods exceed 2 hours they are erroneously recorded to `data/audit/gaps.csv` as permanent data gaps. Furthermore, last-write-wins upsert logic can destroy existing valid historical observations if a subsequent request returns a transient null actual for that period.
- Decision:
  1. Enforce a strict cache-usability and completion predicate (`is_cache_usable`, `is_day_complete`): a past date is complete when `rows + recorded_gaps == expected_rows` for its DST type and every null `actual` in the partition is already recorded in `data/audit/gaps.csv`. Such a date is skipped on subsequent runs and its cache is usable without refetching. Newly discovered holes are processed once. Today must always be fetched live.
  2. Implement atomic observation coalescing (`upsert_day`): `actual` and `index` are treated as an atomic observation updated together or not at all. If incoming `actual` is null, stored `actual` and `index` remain untouched. `forecast` updates independently. When incoming payload offers a null where a stored value exists, keep the stored value and audit the disagreement in `data/audit/conflicts.csv` (`local_date,period_start_utc,field,stored_value`). `rows_refreshed` counts only rows whose stored values actually changed.
  3. Validate period invariants strictly in `payload_to_frame`: reject inverted periods (`period_end_utc <= period_start_utc`), periods not exactly 30 minutes in duration, or payloads whose `local_date` does not match the requested date with `ValueError`.
  4. Record upstream-withheld periods in settled dates as `missing_period` gaps in `data/audit/gaps.csv` without synthesising artificial rows. Prune cache files older than 90 days automatically at the end of successful ingest runs.

## ADR-004 — Audit file location: separate data/audit directory from curated Parquet dataset
- Status: accepted
- Date: 2026-11-01
- Context: Standard dataset loaders and analysis tools (e.g. `pd.read_parquet("data/curated")`, PyArrow dataset scanner, DuckDB) expect directory targets to contain only valid Parquet files. When audit CSV files (`gaps.csv`, `conflicts.csv`) were located in `data/curated/`, loading the directory failed immediately upon the appearance of the first audit row:
  ```
  pd.read_parquet("data/curated") -> ArrowInvalid: Could not open Parquet input source 'data/curated/gaps.csv': Parquet magic bytes not found
  ```
  Documenting a file glob workaround (`pd.read_parquet(glob("data/curated/date=*.parquet"))`) would shift the burden of an unnatural directory layout onto downstream consumers and models.
- Decision:
  1. Separate audit CSV files from the curated Parquet partition directory:
     - Move `gaps.csv` to `data/audit/gaps.csv`.
     - Move `conflicts.csv` to `data/audit/conflicts.csv`.
  2. Keep `data/curated/` containing exclusively daily Parquet partition files (`date=<YYYY-MM-DD>.parquet`).
  3. Create `data/audit/` lazily only when genuine gaps or retractions are recorded.
  4. Automatically reconcile `missing_actual` entries in `data/audit/gaps.csv` when late-published actual observations arrive during subsequent force re-ingestion, and prune `gaps.csv` when empty.
- Consequences:
  `pd.read_parquet("data/curated")` functions as the canonical, intuitive dataset loader without error. The curated table remains a pure, high-performance partitioned Parquet dataset, while audit records stay isolated and human-diffable.
- Alternatives rejected:
  - Documenting explicit globbing (`data/curated/date=*.parquet`) for all readers (rejected: brittle, pushes an avoidable directory layout defect onto consumers).
  - Storing audit trails inside Parquet metadata or separate Parquet tables (rejected: CSV is transparent, easily inspected in git diffs, and lightweight for small anomaly logs).

## ADR-005 — Review scope: repository state is out of scope for a pre-commit gate
- Status: accepted
- Date: 2026-11-01
- Context: Phase P1 ran three review rounds. In rounds 1 and 3 the cold-reader review returned SHIP WITH FIXES with one blocking item being that the work was not committed or pushed. A gate that runs before a commit cannot require the commit it authorises. Treated as blocking, it either deadlocks the phase or forces the operator to override a review verdict on their own authority, which is the thing a review gate exists to prevent. Shipping is downstream of review by definition.
- Decision:
  1. Reviewers of any phase gate assess the working tree only.
  2. Repository state - committed, pushed, CI green, deployed, published - is explicitly out of scope for every pre-commit gate and can never be a blocking finding there. It may be noted as an observation.
  3. Shipping is verified separately after the gate closes: against a fresh clone of the published repository, running the documented commands. That result belongs in the phase report, not in the reviewer's verdict.
- Consequences: The gate measures the artefact and publication is a separate, independently verified act. Applied retrospectively to Phase P1: R1's round-3 verdict carried three blocking items. Two were substantive - the dataset loader and leaked defect identifiers - and both are fixed and verified. The third, that the work was not committed, is out of scope under this ADR. Re-scored, Phase P1 carries zero in-scope blocking findings from either reviewer.
- Alternatives rejected:
  - Overriding the reviewer's verdict on the operator's authority (rejected: it makes the verdict meaningless and hides the circularity instead of removing it).
  - Letting reviewers block on publication (rejected: it creates a deadlock in which a phase can never ship because it is not yet shipped).

## ADR-006 — Weather source: previous-runs forecast issued 48 hours earlier
- Status: accepted
- Date: 2026-11-01
- Context:
  Weather is a primary driver of space heating and electricity demand across Great Britain. However, historical weather providers present a critical train/serve trap:
  - Open-Meteo reanalysis (`archive-api.open-meteo.com`) reconstructs what weather *was* with full hindsight. It was not available to a forecaster at issue time.
  - Open-Meteo historical-forecast (`historical-forecast-api.open-meteo.com`) emits a blended series that overwrites prior predictions and conceals forecast issue time.
  - Open-Meteo previous-runs (`previous-runs-api.open-meteo.com/v1/forecast`) provides explicit run-lagged outputs (`temperature_2m_previous_day1`, `temperature_2m_previous_day2`, `temperature_2m_previous_day3`), exposing what was actually predicted one, two, and three days prior.
  Reanalysis and day-ahead forecast differ by up to 5 °C on heating-critical winter days. Substituting reanalysis creates severe target leakage and train/serve skew.
- Decision:
  1. The weather feature for target period `t` is strictly `temperature_2m_previous_day2` evaluated at `t`. This matches the operational 48-hour day-ahead forecast horizon honestly.
  2. Never use `archive-api` reanalysis anywhere in gridcast.
  3. Never mix reanalysis and forecast series in one feature set.
  4. If previous-runs emits null for a period within the window, the feature remains null; do not silently impute or fall back to reanalysis. Tests fail loudly on any violation.
- Consequences:
  The feature layer reproduces the exact information set available to an operational model at issue time `t-48h`. Reanalysis leakage is prevented by contract.
- Alternatives rejected:
  - ERA5/Open-Meteo reanalysis (rejected: severe retrospective leakage; what was observed is not what was forecasted).
  - Blended historical forecast API (rejected: hides issue time and mutates historical forecasts retroactively).

### Amendment: P2 Horizon Redesign and Anchoring to Issue Time
- Date: 2026-11-02
- Context:
  The original ADR-006 claimed the feature layer "reproduces the exact information set available to an operational model at issue time `t-48h`". However, the initial feature frame was anchored to target period `t` with actual lags (`actual_lag_1h` .. `actual_lag_24h`) and rolling windows evaluated relative to target `t` (e.g. `t-1h`). At an issue time of `t-48h`, `t-1h` was 47 hours in the future. A model trained on that frame depended on observations that could not possibly exist at serving time (operational leakage).
- Decision:
  Rebuild the feature frame around `(issue_time, horizon)` where `DEFAULT_HORIZONS = range(1, 97)`.
  1. All lags of actual are anchored to `issue_time - N hours` (1h, 2h, 3h, 24h, 48h, 168h).
  2. All rolling statistics of actual end strictly at `issue_time - 30min`.
  3. Generation mix fuel shares are lagged to `issue_time - 48h`.
  4. Weather features evaluate `temperature_2m_previous_day2` at `target_time`, which was issued 48 hours earlier at `target_time - 48h <= issue_time`.
  With this redesign, the claim becomes genuinely true: every feature column is computable strictly from data available at or before `issue_time`.

## ADR-007 — Evaluation gate policy: validation-fixed accuracy threshold and single-score test split
- Status: accepted
- Date: 2026-11-01
- Context:
  `make eval` executes the backtesting benchmark against the National Energy System Operator (NESO) official baseline forecast. To avoid data snooping and threshold p-hacking, the accuracy gate policy must be codified before model training (P3) and containerised CI (P5) are implemented. `make eval` is deliberately not wired into CI at P2 or P5 until P3 establishes a validated backtest.
- Decision:
  1. The evaluation accuracy gate threshold is fixed exclusively from the **validation** split, prior to scoring the test split.
  2. The held-out test split is scored exactly once per model candidate and is never used to tune hyperparameters, select features, or adjust the accuracy gate threshold.
  3. The evaluation threshold, split partition dates, and NESO operator baseline comparison metrics must be permanently documented in the model card upon determination.
- Consequences:
  Prevents overfitting to the test split and guarantees unbiased model verification when CI gating is activated. Binding on Phases P3 and P5.
- Alternatives rejected:
  - Wiring an uncalibrated evaluation gate into CI at P2 (rejected: violates honest phase gating; backtest arrives in P3).
  - Iteratively tuning model thresholds against the test split (rejected: classic test-set contamination and data leakage).

## ADR-008 — Weather joining rule: containing hour for hourly-to-half-hourly alignment
- Status: accepted
- Date: 2026-11-01
- Context:
  Open-Meteo previous-runs day-2 forecasts (`temperature_2m_previous_day2`) are emitted hourly, whereas Great Britain electricity settlement periods and the target feature frame are half-hourly (every 30 minutes). When joining the hourly forecast series to half-hourly settlement periods, an alignment rule is required to map period `t` to a forecast value without temporal leakage or boundary artifacts.
- Decision:
  1. For each half-hourly period starting at UTC timestamp `t`, map to the discrete hourly forecast value of the containing hour `t.floor("h")`.
  2. Both half-hours within an hour (e.g. `12:00:00` and `12:30:00`) receive the identical forecast value issued at `t-48h`. Because the entire previous-runs issue run was published 48 hours prior to `t`, neither half-hour utilizes any future information.
  3. 24-hour rolling mean (`temperature_24h_mean`) and 24-hour rolling min (`temperature_24h_min`) are computed strictly over the hourly forecast history ending at the containing hour.
- Consequences:
  - Preserves exact Numerical Weather Prediction (NWP) model values without synthesizing artificial intermediate temperatures.
  - Guarantees future-perturbation invariance across all cutoff boundaries (including half-hour boundaries).
  - Eliminates end-of-partition boundary edge cases (such as period `23:30` requiring an uningested `00:00` hour from the next day).
- Alternatives rejected:
  - Linear interpolation between bracketing hours `t.floor("h")` and `t.ceil("h")`: rejected because it synthesizes artificial intermediate temperatures not emitted by the NWP model, introduces an artificial dependency across day boundaries at `23:30`, and risks leaking post-cutoff values when a perturbation cutoff falls on a half-hour.
  - Forward-looking or nearest-hour rounding: rejected because rounding up looks forward into the subsequent hour.

## ADR-009 — Store coverage is audited the same way the archive is
- Status: accepted
- Date: 2026-11-01
- Context:
  The curated carbon intensity archive enforces a strict completeness invariant: every settlement period is either present in daily Parquet partitions or recorded in `data/audit/gaps.csv`. Auxiliary feature stores (`data/weather/` and `data/mix/`) initially lacked post-fetch coverage checking, which permitted 27 half-hourly generation mix periods to be silently omitted across the two-year window (resulting in 123 null values across all mix feature columns without an audit trail).
  Investigation revealed two distinct root causes:
  1. Chunk-boundary straddling behaviour (2 periods): The NESO generation endpoint (`/generation/{from}/{to}`) emits the half-hour period straddling before `{from}`. When fetching in monthly chunks bounded by midnight, `2024-12-31T23:30Z` and `2025-12-31T23:30Z` were excluded from the December partition and dropped during January ingestion because January's date filter only retained January dates. Re-requesting with `period_start_utc + 30min` recovers both periods.
  2. Upstream absent data (25 periods): Across 25 generation mix periods grouped by `local_date` (2 on `2025-01-12`, 15 on `2025-01-13`, and 8 on `2025-08-11`; the partition for `2025-08-10` is complete at 48/48 periods) — identical to the dates where upstream intensity payloads returned malformed JSON — the generation mix endpoint returns `data: []`. These periods are genuinely absent upstream and unrecoverable.
- Decision:
  1. Auxiliary stores must be audited under the same standards as the curated intensity archive: no period may be silently absent.
  2. Implement a post-fetch coverage check with targeted retries (`check_mix_coverage_and_retry`, `check_weather_coverage_and_retry`):
     - After fetching, compare ingested period starts against the curated settlement grid.
     - For missing periods, re-request targeted windows (e.g. `period_start_utc + 30min` for mix) to recover straddling boundary periods.
  3. Record unrecoverable missing periods in `data/audit/source-gaps.csv` with columns `store,local_date,period_start_utc,reason` (`upstream_empty`).
  4. Manage `source-gaps.csv` lazily (created only when gaps exist, unlinked when empty) and automatically reconcile recorded gaps if subsequent ingest runs acquire the data.
  5. Add strict coverage verification in `make data-check` (`tests/test_data_integrity.py`):
     - Every curated period must have a store row or a corresponding `source-gaps.csv` row.
     - `source-gaps.csv` must contain no duplicate `(store, period_start_utc)` pairs.
     - No recorded gap may overlap with an existing row in the store.
  6. Feature generation must report source gaps and null counts transparently at the point of use (`load_feature_sources()`) without synthesising or imputing artificial values.
- Consequences:
  Auxiliary stores provide full auditability and coverage guarantees identical to the primary curated dataset. Boundary periods are recovered automatically, and upstream source gaps are explicitly recorded and tracked in git rather than producing silent null features.
- Alternatives rejected:
  - Silently imputing missing mix periods via forward-fill or mean fuel shares: rejected because it fabricates generation fuel mix observations and masks upstream outages.
  - Ignoring auxiliary store holes because intensity actuals are complete: rejected because downstream models depend on auxiliary feature stores, and unrecorded holes violate data integrity standards.

## ADR-010 — A verification must assert worth, not only shape
- Status: accepted
- Date: 2026-11-01
- Context: Three defects escaped a green suite during phases P1 and P2, each in the same form. A feature frame with nine of twenty-five columns entirely null passed, because the "no constant column" check filtered out null values before counting. A column that was 100% null is constant by any useful definition, and the suite asserted only that the frame had the right shape. A partition missing a period passed a row-count check nobody had written. An audit file that asserted the weather features were null made the gap look intended. In every case the check tested the presence of a thing rather than whether the thing carried information.
- Decision:
  1. A check on a dataset or a feature frame must assert a content property, not only row counts, column names or dtypes.
  2. Specifically: no feature column may be entirely null; every period must be accounted for either by a stored value or by a recorded audit row; every reported metric must be reported beside the baseline it is compared with.
  3. Shape assertions remain necessary and are never sufficient on their own.
  4. A test that would pass equally well if the feature, store or metric were empty is a bug in the test, and is treated as a defect of the same severity as the code it was meant to guard.
- Consequences: A green suite means the data carries information and the gaps are enumerated, rather than that the columns exist. It is slower to write such checks and they fail more often, which is the point.
- Alternatives rejected:
  - Relying on coverage percentages to indicate thoroughness (rejected: both escaped defects were found under 89% and 96% coverage).
  - Relying on the generator's own test output as evidence (rejected: in each case the output was green and described the defect as normal behaviour).

## ADR-011 — Leakage taxonomy: distinguishing temporal and operational leakage
- Status: accepted
- Date: 2026-11-02
- Context:
  During review of Phase P2, two independent reviews returned a blocking architectural defect: the initial feature frame was constructed with one row per target period `t`, anchoring actual lags (`actual_lag_1h` .. `_24h`) and rolling windows to target `t` instead of forecast issue time `T`. For a 48-hour ahead forecast (`h=96`), `actual_lag_1h` at target `t` was 47 hours in the future at issue time. Yet the test suite passed all tests green.
  The defect escaped because two fundamentally distinct forms of leakage were conflated:
  1. **Temporal leakage**: A feature reads a source value stamped after the target period (`t_source > target_time`).
  2. **Operational leakage**: A feature reads a source value that exists in the historical dataset at the target period, but did not exist at the forecast issue time (`t_source > issue_time`).
  A future-perturbation invariance test checking cutoffs against target time `t` alone proves only the absence of temporal leakage: because `t - 1h < t`, a feature reading `t - 1h` genuinely precedes target `t` and is invariant to perturbations after `t`. However, operationally the forecast is served at `issue_time = t - 48h`, where `t - 1h` has not occurred.
- Decision:
  1. Formally distinguish temporal leakage and operational leakage in architecture, documentation, and tests.
  2. The canonical example frame is keyed by `(issue_time, horizon)` where `target_time = issue_time + horizon * 30min` for `horizon in 1..96` (48-hour half-hourly curve).
  3. The horizon contract is enforced directly in code: each feature family declares the maximum allowed source timestamp, and the builder asserts that for every row and feature, `max_source_timestamp <= issue_time`. Any build attempting to read source data timestamped after `issue_time` must raise `ValueError`.
  4. Operational leakage is guarded via sampled per-row perturbation testing across at least 25 `(issue_time, horizon)` pairs spanning dataset boundaries, DST transition days, and horizon endpoints: for each pair, all source rows timestamped after `issue_time` are replaced with sentinels, and the rebuilt feature row is asserted to be identical.
- Consequences:
  Guarantees train/serve integrity: models trained on the example frame depend exclusively on observations available at serving time.
- Alternatives rejected:
  - Relying on target-based invariance tests alone (rejected: blind to operational leakage when horizon > 1).
  - Documenting horizon constraints solely in comments (rejected: permitted operational leakage through two review rounds).

## ADR-012 — Issue anchor: 00:00 Europe/London for local-day alignment
- Status: accepted
- Date: 2026-11-02
- Context:
  The curated carbon intensity dataset and public scoreboard are partitioned by Europe/London local calendar day (`data/curated/date=<YYYY-MM-DD>.parquet`). Previously, issue times defaulted to 00:00 UTC. In British Summer Time (BST, UTC+1), 00:00 UTC corresponds to 01:00 local time. This left the first local hour of each summer day (00:00 to 01:00 local) uncovered by that day's own issue. Local midnight is the only anchor at which a day's issue and a day's partition begin at the same instant.
- Decision:
  Anchor daily forecast issue times `issue_time` to 00:00 Europe/London for each local day rather than 00:00 UTC.
  1. In BST (UTC+1), the issue instant is 23:00Z of the previous UTC calendar date.
  2. In GMT (UTC+0), the issue instant is 00:00Z of the same calendar date.
  3. Consecutive issue instants are 23, 24, or 25 hours apart in absolute time across DST boundaries.
  4. The forecast horizon remains an absolute half-hour offset from `issue_time` (`target_time = issue_time + horizon * 30min`), ensuring DST clock changes never alter the 96-horizon sequence.
- Consequences:
  Each day's issue begins at the exact instant of that local day's partition, fully covering all settlement periods in summer without unforecasted leading hours.
- Alternatives rejected:
  - 00:00 UTC anchor (rejected: sits at 01:00 local for half the year, leaving the first hour of every summer day uncovered by that day's forecast).
