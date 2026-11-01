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
