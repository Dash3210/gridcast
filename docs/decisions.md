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

## ADR-013 — Split embargo: dropping boundary issue dates to prevent target overlap
- Status: accepted
- Date: 2026-11-02
- Context:
  The 48-hour forecasting horizon means targets generated from an issue date on day D extend through day D+2 (`issue_time + 48h`). Splitting datasets by issue date without a temporal buffer allows targets in the test period to overlap with targets in the training or validation splits, introducing subtle target leakage into model evaluation.
- Decision:
  Implement `split_by_issue_date(frame, train_end, val_end, embargo_days=4)`:
  1. Splits are cut strictly on `issue_time` (Europe/London local issue date), never on `target_time`.
  2. At least `embargo_days` (default: 4) issue dates are dropped at the train/validation boundary and at the validation/test boundary. These dates are discarded rather than assigned to either split.
  3. Rows with null `y` are excluded from all splits so models cannot train or score on unsettled targets.
  4. Guarantees `max(target_time) of split_i < min(target_time) of split_{i+1} - 48h`, strictly preventing target overlap.
- Consequences:
  Eliminates evaluation leakage between train, validation, and test splits while preserving contiguous issue blocks.
- Alternatives rejected:
  - Splitting on target_time (rejected: scrambles multi-horizon forecasts from a single issue across train and test).
  - Contiguous issue-date splits without an embargo (rejected: 48-hour target horizon creates 2-day target overlap across boundaries).

## ADR-014 — Positive model input selection: enforcing explicit feature matrix extraction
- Status: accepted
- Date: 2026-11-02
- Context:
  Machine learning workflows frequently extract feature matrix `X` via negative column dropping, e.g. `df.drop(columns=["y"])` or `df.drop(columns=["target_time"])`. This practice is dangerous: any auxiliary metadata, intermediate computation, leaked label column, or upstream schema expansion added to `df` is silently passed into model training and inference.
- Decision:
  1. Define `select_model_inputs(frame) -> pd.DataFrame` as the only documented and supported way to construct `X`.
  2. `select_model_inputs` strictly enforces a positive selection: it returns exactly `frame[FEATURE_COLUMNS]` in declared order.
  3. It validates schema integrity by raising `ValueError` if any declared feature column is missing or if `frame` carries any undeclared column outside `ALLOWED_FRAME_COLUMNS`.
  4. Negative column dropping (`df.drop(columns=["y"])`) is strictly forbidden across the codebase and user documentation.
- Consequences:
  Guarantees that downstream model inputs remain strictly bounded, predictable, and immune to undeclared column leakage.
- Alternatives rejected:
  - Negative selection via `df.drop(columns=["y"])` (rejected: silently leaks unexpected or undeclared columns directly into model features).

## ADR-015 — Injection tests: asserting mutation preconditions before verifying guards
- Status: accepted
- Date: 2026-11-02
- Context:
  In defensive testing, guard assertions check that an injection, perturbation, or malformed payload is detected and rejected (or that causal invariance holds under future perturbations). However, if an injection silently fails to apply—such as modifying a non-existent column, targeting a filtered slice, or injecting values identical to the original—the test passes trivially (false green). In this project's history, two injections silently failed to alter anything, and the resulting green test suite was briefly misread as a vulnerability in the guard rather than a defect in the test.
- Decision:
  An injection test must first assert that the injection took effect—the injected column exists, or the altered value actually differs from the original—and only then assert that the guard fires or invariance holds.
- Consequences:
  Eliminates silent injection failures. Any test whose mutation does not alter state will fail immediately on the precondition assertion before reaching the guard verdict.
- Alternatives rejected:
  - Trusting guard verdicts without verifying mutation application (rejected: permits vacuous green tests).

## ADR-016 — Availability allowance: three-hour lag on actual-derived features
- Status: accepted
- Date: 2026-11-02
- Context:
  Live observation on 2026-11-01 demonstrated that NESO publication of actual carbon intensity is subject to significant operational latency:
  - Settlement periods ending at 20:00Z were observed still null at 20:11Z.
  - Earlier in the evening, settlement periods were observed still null 40 minutes after they ended.
  A feature that is mathematically correct by timestamp (such as reading the actual for the half-hour ending 30 minutes prior to issue) can still be unavailable in production at issue time. Retaining `actual_lag_1h`, `actual_lag_2h`, or `actual_lag_3h` conceals operational leakage where serving code attempts to read data that has not yet arrived.
- Decision:
  1. Define a parameterized module-level constant `ACTUAL_AVAILABILITY_LAG = 3 hours` (`pd.Timedelta(hours=3)`).
  2. Apply this allowance once for the entire frame: because a single forecast issue produces all 96 horizons simultaneously from one information set, no horizon may observe fresher actuals than another by construction.
  3. Rename `actual_lag_1h` to `actual_recent_3h`—the most recent actual available at issue time, defined as "the actual for the half hour ending immediately before `issue_time - ACTUAL_AVAILABILITY_LAG`".
  4. Drop `actual_lag_2h` and `actual_lag_3h`: under a 3-hour availability allowance, they are unavailable by definition, and preserving them would either be false or duplicate `actual_recent_3h`.
  5. Retain `actual_lag_24h`, `actual_lag_48h`, and `actual_lag_168h` under their existing names because their source timestamps remain valid and precede the availability cutoff.
  6. Align all actual rolling windows (`actual_rolling_*`) to terminate strictly at `issue_time - ACTUAL_AVAILABILITY_LAG` rather than `issue_time - 30min`.
  7. The 3-hour allowance chosen on 2026-11-02 is conservative pending empirical arrival measurement from `first_seen_at` once nightly runs have accumulated.
- Consequences:
  All actual-derived features read at or before `issue_time - ACTUAL_AVAILABILITY_LAG`, ensuring train/serve parity without operational leakage.
- Alternatives rejected:
  - Allowing horizons to observe different freshness of actuals (rejected: one issue produces all 96 horizons from a single information set).
  - Forward-imputing stale values under the old lag names (rejected: dishonest naming conceals data drift).

## ADR-017 — Independent test oracles: separate naive computation paths for contract guards
- Status: accepted
- Date: 2026-11-02
- Context:
  A guard whose expected value is computed by the same code it guards cannot catch a bug that lies about its own bound. In this system, contract test assertions previously validated declarations by comparing them against the output of the declaration function itself (`max_source_timestamp_allowed`). When a faulty declaration (such as `issue - 23h` on autumn-back dates) was tested, the entire test suite remained green because the guard validated the declaration against itself. An effective test oracle must be completely decoupled from the production builder.
- Decision:
  1. No guard may take its expected value from the code it guards.
  2. Test oracles must recompute expected feature values directly from raw stores through an independent, separate, deliberately naive path (such as explicit, slow pandas lookups: `store.loc[store["period_start_utc"] == expected_ts, ...]`).
  3. Feature builders must be verified against this independent oracle across dense value grids covering all operational boundaries (including all DST transition days and source-gap adjacent dates across all 96 horizons).
- Review Rubric:
  This structural rule and the rule that an injection test must prove it landed (ADR-015) form mandatory review rubric criteria for any defensive guard or invariant in the repository:
  - *Rule 1 (Proof of Landing, ADR-015)*: Every injection or perturbation test must assert its mutation preconditions (that source values were genuinely modified) before asserting the guard verdict or invariance.
  - *Rule 2 (Independent Oracle, ADR-017)*: No guard or contract test may take its expected value from the code it guards; the oracle must recompute from the raw store via a separate, deliberately naive path.
- Consequences:
  Guarantees that subtle algorithmic errors, position-dependent indexing defects, or self-fulfilling contract declarations are detected immediately.
- Alternatives rejected:
  - Validating contract functions against their own declared return values (rejected: permits false green test suites when declarations lie about their bounds).

## ADR-018 — Serving risk: train/serve lead distribution mismatch on weather features
- Status: implemented
- Date: 2026-11-02 (opened 2026-11-02, review updated 2026-11-07, ablation concluded 2026-11-08, implemented 2026-11-08)
- Context:
  At inference the targets are in the future, and the previous-runs series has no future values. The weather available for `T+h` at serve time is the current model run with lead `h`, not the day-1/day-2 series the frame trains on with lead `>= 24h`.
  Training and serving therefore use different lead distributions, so "the model has weather features" is true in training and not true as stated for the service.
- Evidence & Ablations:
  1. Lead Misalignment Ablation (stale vs current):
     Measured on the frozen 15-origin split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins), same frame builder, same origins, same parameters; only the weather timestamps moved by +24h:
     ```
     baseline (current weather join)       39.032
     weather values lagged a further 24h   38.279
     delta                                 -0.753 gCO2/kWh   (-1.93%)
     materiality threshold                 5.00%
     ```
     This bounded lead sensitivity and demonstrated that weather features are weakly utilized at these horizons.
  2. Weather Removal Ablation (Phase P4-c, does weather earn its place?):
     Pre-registered decision rule (amended before results were read):
     Let `W` = GATE_VALUE with weather (Control arm) and `N` = GATE_VALUE without weather (Treatment arm), scored on frozen fixture v1.
     The decision rule as applied has two conditions, where the second condition was added before the result was read:
     - Condition 1 (aggregate): `N <= 1.02W` (no-weather GATE_VALUE is within 2.0% of weather)
     - Condition 2 (tail / long horizons): `N_long <= 1.05W_long` over `h >= 48` (no-weather mean MAE over `h >= 48` is within 5.0% of weather)
     - Decision criteria: DROP weather only if both conditions hold; otherwise KEEP weather.
     Rationale: ADR-021 demands closing 30% of a 48.127 gap (14.438 gCO2/kWh). 2.0% of 39.03 is ~0.78 gCO2/kWh, ~5% of what the gate demands. A feature buying less than a twentieth of gate demand does not justify an external serving dependency that cannot reproduce its training distribution. The secondary condition over `h >= 48` bounds long-horizon degradation to 5.0% to guard against tail collapse.
     Precondition verified: Control arm reproduces frozen fixture v1 bitwise (max candidate delta: 0.00000000, GATE_VALUE: 39.032478 gCO2/kWh).
     Measured ablation results (15 folds, 100 boost rounds, seed=42, deterministic):
     ```
     Control arm (W, with weather)          39.032478 gCO2/kWh
     Treatment arm (N, without weather)     39.118459 gCO2/kWh
     Delta (N - W)                          +0.085981 gCO2/kWh   (+0.22%)
     2.0% aggregate limit (1.02 * W)        39.813127 gCO2/kWh   (Condition 1: PASS, 39.118459 <= 39.813127)
     h >= 48  W 38.0234  N 38.3512  delta   +0.328 gCO2/kWh      (+0.86%)
     5.0% tail limit (1.05 * W_long)        39.9246 gCO2/kWh     (Condition 2: PASS, 38.3512 <= 39.9246)
     Decision Rule Applied                  DROP (both conditions satisfied)
     ```
     Key horizons:
     - `h = 1`:  W = 13.2536, N = 12.9836, delta = -0.2700 gCO2/kWh (-2.04%)
     - `h = 2`:  W = 13.9344, N = 13.9959, delta = +0.0615 gCO2/kWh (+0.44%)
     - `h = 48`: W = 53.9933, N = 54.0323, delta = +0.0390 gCO2/kWh (+0.07%)
     - `h = 96`: W = 39.1453, N = 36.6272, delta = -2.5181 gCO2/kWh (-6.43%)
  3. Horizon-band tension:
     The per-horizon data exhibits a clear structural tension across horizon bands:
     - No-weather is better at `h = 96` (-6.43%, -2.518 gCO2/kWh) and across roughly `h = 49..65` (e.g., h=52 is -4.31%, h=60 is -5.56%, h=62 is -5.87%, h=63 is -6.31%).
     - Weather is better across roughly `h = 82..93` (+11% to +15% degradation without weather, e.g., h=82 is +12.44%, h=88 is +13.73%, h=89 is +14.56%).
     The pre-registered rule is an unweighted mean test across horizons and it says DROP. A horizon-weighted rule that placed high premium on day-2 evening peak hours (h=82..93) would have decided differently. However, we do not re-decide after observing the data: the rule was fixed in advance as an unweighted mean test, both aggregate and tail conditions hold, and DROP stands.
- Decision:
  DROP weather features from the production feature set.
  Implementation details (Phase P4-d):
  - Dropped `temperature`, `temperature_24h_mean`, and `temperature_24h_min` from `FEATURE_COLUMNS`, reducing feature count from 29 to 26.
  - `build_example_frame` updated with `include_weather=False` as default, constructing the canonical feature frame without requiring weather inputs or store (`weather_df=None` supported).
  - Serving servability check `check_startup_servability` and `resolve_features_for_issue` in `src/gridcast/serve.py` no longer declare or require weather dependencies, enabling the service to start with the weather store absent.
  - Retrained production model artifact `data/models/lgbm_model.txt` with 26 features across full window as `lgbm-v2` (content SHA-256 `1cd13471d793735748f81d17fb195e34f6f784e8f642ac60a16f016280c45310`, git object `fe838f87dac4cb5412f5118600353b5293d52cd4`).
  - Refrozen evaluation fixture as `data/eval/frozen_split_v2.parquet` + sidecar JSON with candidate GATE_VALUE `39.118459` (SHA-256 `87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51`, archive commit `3617143`, window 2026-09-05 to 2026-11-01, 15 origins).
- Alternatives rejected:
  - Keeping weather features (rejected: buys only 0.086 gCO2/kWh or 0.22% overall and +0.86% over h >= 48, satisfying both DROP conditions; creates serving complexity and irreconcilable train/serve lead distribution mismatch).
  - Deleting ADR-018 (rejected: weather does buy something in a band; ADR-018 is retained and closed by rule).
  - Re-deciding based on horizon-band weighting (rejected: violates pre-registration integrity; rule is a mean test).
  - Restricting the product to horizons $\ge 24$h (rejected: forfeits 48 half-hours essential for system balancing).

## ADR-019 — Evaluation gate: single scalar threshold fixed from validation split
- Status: accepted
- Date: 2026-11-02
- Context:
  The evaluation gate requires a single scalar metric so 96 separate numbers do not gate a build. Each target half-hour appears in up to three issues in a rolling evaluation, so a pooled average weights the middle of the window three times more than the edges. Furthermore, under evaluation integrity principles, the accuracy gate threshold must be determined strictly from the validation split before the test split is scored once, avoiding tuning leakage.
- Decision:
  1. Define the single gate scalar:
     `MAE_h = mean absolute error over scored periods at horizon h`
     `GATE_VALUE = (1 / 96) * sum_{h=1}^{96} MAE_h`
     Every horizon is weighted equally.
  2. Fix the accuracy gate threshold to `GATE_VALUE <= 35.0 gCO2/kWh` on 2026-11-02 from the validation split (`2026-06-05` to `2026-08-30`, 15 rolling origins). On this split, the LightGBM candidate achieved `GATE_VALUE = 33.815 gCO2/kWh`, beating the persistence baseline (`57.776 gCO2/kWh`, 41.5% improvement) and seasonal-naive baseline (`46.732 gCO2/kWh`, 27.6% improvement).
  3. Score the test split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) against this threshold (initial evaluation declared void and superseded by ADR-020).
  4. The gate must evaluate all candidates and baselines strictly on the intersection of non-null periods.
  5. The evaluation gate is executed locally via `make eval` (`python -m gridcast.cli eval`). Wiring into CI is deferred to Phase P5, where CI will execute `make eval` alongside `make check`.
- Consequences:
  Deterministic evaluation machinery. Superseded by ADR-020 (threshold invalid; void verdict) and ADR-021 (gap-closure gate).
- Alternatives rejected:
  - Multi-metric or per-horizon gating (rejected: complex and conflicting pass/fail conditions).
  - Pooled sample mean absolute error (rejected: triple-weights overlapping middle horizons).
  - Tuning or tightening the threshold on the test set (rejected: corrupts test split isolation).

### Amendment: Frozen test split, evaluation integrity defects, and void verdict
- Date: 2026-11-02
- Context:
  An evaluation review identified three integrity defects in the evaluation pipeline:
  1. Moving origin window: `run_rolling_origin_backtest` defaulted to `test_issues[-60:][::4]`, walking one day forward every night and intruding into the 4-day embargo (`2026-09-04`). On the sliding default origins, the model scored `33.045`.
     On the committed documented split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) it scored `39.032`.
     (The pre-freeze archive value at commit `0bbb497` was `39.043`, which no longer reproduces).
  2. Tautological provenance guard: the assertion compared the assigned timestamp against the formula used to assign it without verifying prediction values against the raw curated store.
  3. Dead-code period-set guard: `compute_skill_score` enforced period-set equality at division time, but `run_rolling_origin_backtest` computed skill inline.
  4. Silent denominator: `Series.mean(skipna=True)` silently dropped missing horizons, dividing by 95 instead of 96.
- Decision:
  1. Fix the backtest split to a committed, immutable range in constants:
     - `TEST_SPLIT_START_DATE = "2026-09-05"`
     - `TEST_SPLIT_END_DATE = "2026-11-01"`
     - `TEST_SPLIT_STEP_DAYS = 4`
     Origins are derived in one place (`derive_test_split_origins()`), producing 15 fixed rolling origins (`2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`). The first origin is asserted to be at or after `2026-09-05`, never inside the 4-day embargo (`2026-08-31` to `2026-09-04`). New ingested data appends to the archive but does not extend the split.
  2. The provenance guard (`assert_baseline_provenance`) reads the raw curated store directly via plain lookup at `issue_time - ACTUAL_AVAILABILITY_LAG`, asserting that the store row exists, is non-null, and exactly equals the prediction value.
  3. All skill figures (per-horizon and overall) must execute through `compute_skill_score` on the live path.
  4. The gate scalar is strictly defined as `(1 / 96) * sum(MAE_h)`; all 96 horizons are required to contribute non-null MAEs, raising `ValueError` otherwise.
  5. The 35.0 threshold is recognized as invalid under ADR-020 (derived from candidate's own validation score). On the frozen test split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins), the candidate scores `GATE_VALUE = 39.032 gCO2/kWh`. The first scoring is void, P3 closes with no gate verdict (neither pass nor fail), and `make eval` prints `GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed` and exits 0.
- Consequences:
  Evaluation verdicts are completely deterministic and independent of execution date.

## ADR-020 — Invalid threshold and void evaluation verdict
- Status: accepted
- Date: 2026-11-02
- Context:
  The evaluation gate threshold of 35.0 gCO2/kWh used in Phase P3 was derived from the candidate model's own validation score (33.815 gCO2/kWh) plus 3.5%. It was therefore never an external standard: a model cannot be judged against a bar computed from itself. A threshold computed from the thing it judges cannot gate that thing.
- Decision:
  1. The first scoring is void, and Phase P3 closes with **neither a pass nor a fail** (GATE VERDICT: NONE; no verdict).
  2. No pass or fail may be cited from Phase P3. All wording implying the candidate passed, failed, met, or missed a bar is removed across the repository (README, MODEL_CARD, and ADRs).
  3. `make eval` prints the metrics on the frozen split and then prints, exactly and prominently:
     `GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed`
     It exits 0: nothing failed, because nothing was judged.
  4. What is retained: the per-horizon metrics, the four evaluated candidates/baselines (LightGBM candidate, issue-anchored persistence, seasonal-naive, and NESO operator), the frozen evaluation window (15 origins from `2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`), and the backtest machinery.
- Consequences:
  P3 closes with no claimed verdict. Evaluation machinery is preserved intact for Phase P4.
- Alternatives rejected:
  - Overriding the threshold retroactively to claim a pass (rejected: dishonest and circular).
  - Recording a gate failure against an illegitimate bar (rejected: a threshold derived from the candidate is mathematically invalid to judge it).

## ADR-021 — External evaluation gate referencing persistence and operator baselines
- Status: accepted
- Date: 2026-11-02
- Context:
  An evaluation gate must be specified before a model is trained and must reference quantities entirely outside the model itself. Furthermore, the benchmark is a data source, not an oracle: it needs the same alignment discipline as any feature. The vintage of an archived forecast is a property that must be recorded rather than assumed, and the only fair comparison is between two forecasts made at the same instant.
- Decision:
  1. State the gate rule now, defined only in terms of quantities outside the model:
     > On the frozen evaluation split, with candidate, persistence and operator measured on identical periods:
     > the candidate must close at least 30 percent of the MAE gap between issue-anchored persistence and the
     > operator.
     >
     > gap = MAE_persistence - MAE_operator
     > required = MAE_persistence - 0.30 * gap
     > passes only if MAE_candidate <= required
  2. Derive the number from the VALIDATION split only, with its dates (`2026-06-05` to `2026-08-30`, 15 rolling origins):
     - `MAE_persistence = 57.776 gCO2/kWh`
     - `MAE_operator = 9.649 gCO2/kWh`
     - `gap = MAE_persistence - MAE_operator = 57.776 - 9.649 = 48.127 gCO2/kWh`
     - `required = MAE_persistence - 0.30 * gap = 57.776 - 0.30 * 48.127 = 43.3379 gCO2/kWh` (43.338 gCO2/kWh)
     Do not compute or quote the frozen-split threshold in this ADR: that split is scored once, later, and a threshold printed beside a result looks tuned.
  3. Why 30 percent: a floor demanding much of the distance between doing nothing and the operator, without requiring a solo model to beat an operator with information it does not have. State that the fraction is a policy choice fixed now, changeable only before a model is trained, never after a score is seen.
  4. The gate is not wired into CI until P5.
  5. The benchmark is a data source, not an oracle, and needs the same alignment discipline as any feature; the vintage of an archived forecast is a property to be recorded, not assumed; the only fair comparison is between two forecasts made at the same instant.
- Consequences:
  Pre-specified, objective gate rule for Phase P4 referencing external reality rather than model-derived numbers.
- Alternatives rejected:
  - Absolute error threshold like 35.0 gCO2/kWh (rejected: arbitrary and unanchored to difficulty of the period).
  - Requiring the model to beat the operator (rejected: operator possesses private dispatch and transmission information).
  - Computing the frozen-split threshold prior to P4 model training (rejected: risks appearing post-hoc tuned).

## ADR-022 — Operator baseline information set mismatch
- Status: open
- Date: 2026-11-02
- Context:
  The operator forecast is read from the curated `forecast` column at `target_time`. That column is NESO's published forecast for that settlement period, and it is published relative to the **target**, not to our issue instant. For a long horizon (e.g. 48 hours ahead), the operator had information closer to the target than the candidate did; for a short horizon (e.g. 30 minutes ahead), it may have had less (or was published earlier). The published upstream data does not expose the publication timestamp, so alignment to what genuinely existed at issue instant `T` is not possible from this archive source.
- Decision:
  1. Record the information set mismatch in both directions as an explicit, open limitation of the comparison, not a bug to be patched silently.
  2. No claim of "we beat" or "we lost to" the operator may appear anywhere in repository documentation without this caveat.
  3. In all tables and scoreboards comparing against historical archive forecasts, describe the baseline as `Published NESO forecast for the same periods, vintage unverified`.
  4. The option to seek an upstream source with explicit publication timestamps is deferred to a later phase.
- Consequences:
  All comparisons with the operator are caveated and honest about differing information sets.
- Alternatives rejected:
  - Silently assuming archived forecasts are day-ahead (rejected: false and flatters the operator).
  - Imputing publication timestamps (rejected: unverified fiction).






## ADR-023 — Frozen evaluation fixture, dual data planes, and archive commit tracking
- Status: accepted
- Date: 2026-11-07
- Context:
  The evaluation split was previously frozen in origins, not in targets. Target values were a snapshot of the archive at run time.
  Nightly ingest (`--reconcile-recent --window-days 3`) re-fetches recent partitions, so the final ~48 hours of every evaluation window were revised after the fact.
  Measured on `data/curated/date=2026-11-02.parquet`, the partition the last origin depends on, between P3 commit `0bbb497` and commit `3617143`:
  ```
  actual     11 of 48 rows changed, 14 filled in from null, max delta 3.0
  forecast   13 of 48 rows changed, max delta 28.0
  ```
  Consequence: `GATE_VALUE` previously moved with the archive. The same code and same origins produced `39.032 gCO2/kWh` on archive commit `3617143`, where commit `0bbb497` published `39.043 gCO2/kWh`; persistence (`53.741`), seasonal-naive (`51.495`), and operator (`11.666`) remained unchanged. The origins were frozen, but ground truth was not.
- Decision:
  1. The split is frozen in origins AND targets: materialised in `data/eval/frozen_split_v1.parquet` (1,440 rows: 15 origins x 96 horizons) and accompanied by metadata sidecar `data/eval/frozen_split_v1.json`.
  2. Two data planes:
     - The evaluation fixture (`data/eval/`) is immutable and strictly decoupled from live ingest. `make eval` reads only the fixture and never touches `data/curated`, hard-failing on SHA-256 mismatch or missing sidecar.
     - The live scoreboard continues reading the reconciled store (`data/curated/` and `data/scoreboard/`) so late-arriving actuals are captured for production evaluation.
  3. Window-selection rule: an evaluation window is eligible to be frozen only if the window's last target is at least 3 days behind the newest archive period (`newest_archive_period - window_last_target >= 3 days`). Because the nightly reconcile window is 3 days (`--window-days 3`), partitions older than 3 days are settled and final. At freeze time (2026-11-07 23:30 UTC), the split's last target (2026-11-02 00:00 UTC) was 5.98 days behind and had been final since 2026-11-03, satisfying the eligibility requirement.
  4. What the fixture pins and what it cannot:
     The candidate and baseline predictions are frozen alongside the realised actuals (`candidate_pred`, `neso_forecast`, `persistence_pred`, `seasonal_naive_pred`). The fixture pins the metric and the ground truth, not the training run; only `make eval-refresh` can regenerate predictions, and the sidecar's environment versions (python, lightgbm, pandas, numpy) and hyperparameters make that reproducible. If the model or feature set changes, a new version file (`v2`) must be written; `v1` is never edited.
  5. Published number authority and guard blind spot: `data/eval/` is the single authority for all published figures. `tests/test_model_card_consistency.py` computes numbers from `data/eval/` at test time. The fixture file `tests/fixtures/frozen_split_metrics.json` is reduced to recorded hashes only. Previously, the consistency guard had a blind spot: comparing against a static metrics JSON could not detect when live backtests drifted against an updated archive; pinning the frozen fixture directly eliminates the blind spot against silent live run drift.
  6. All published numbers are restated with full provenance:
     `frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins`.
     Fixture v2 was refrozen after dropping weather features per ADR-018. Fixture v1 is retired but retained on disk. Ground truth is shared and asserted identical byte-for-byte (`actual`, `neso_forecast`, `persistence_pred`, `seasonal_naive_pred`); only `candidate_pred` was updated for the 26-feature retrain (GATE_VALUE 39.118459).
  7. Model reproduction check (`make eval-reproduce`):
     Retraining the 15 folds on the frozen origins from the canonical feature frame reproduces the fixture bitwise identically:
     - Per-row `candidate_pred` max absolute difference: 0.0 (tolerance pinned to 0.0)
     - `GATE_VALUE` delta: 0.0 (tolerance pinned to 0.0)
     - Library versions match sidecar: python 3.10.11, lightgbm 4.7.0, pandas 2.3.3, numpy 2.2.6
     - Measured runtime: 19.3s (Run 1: 19.88s, Run 2: 18.79s). This number provides the empirical basis for deciding whether to wire reproduction into nightly CI in Phase P5.
- Consequences:
  Evaluation is fast, hermetic, and completely decoupled from archive mutability.
- Alternatives rejected:
  - Re-reading `data/curated` during `make eval` (rejected: subject to upstream backfill revisions; breaks metric stability).
  - Silently mutating numbers without commit and hash references (rejected: breaks reproducibility).
