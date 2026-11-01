# gridcast

**Target.** gridcast is a containerised forecasting service for Great Britain's half-hourly grid carbon intensity, forecasting 48 hours ahead and gated in CI on forecast accuracy. It aims to answer whether a solo open-source model can beat the National Energy System Operator's (NESO) official day-ahead forecast on a permanent, public scoreboard.

> Phase P2 complete: feature extraction layer and temporal leakage test suite are operational; downstream modeling and service phases arrive in subsequent milestones. The status table below is the honest state.

> Note: Upon completion of Phase P7, this README will open with the live forecast-vs-actual scoreboard table comparing gridcast models against the NESO day-ahead baseline, positioned directly above the Quickstart.

## Status

| Phase | Description | Status |
|---|---|---|
| P0 | Setup: repository skeleton, environment, CLI gating, and smoke tests | done |
| P1 | Ingest: half-hourly API fetch to append-only Parquet daily partitions | done |
| P2 | Features: feature extraction and temporal leakage test suite | done |
| P3 | Models: baseline models, backtesting engine, and accuracy gate vs NESO | next |
| P4 | Service: FastAPI prediction API and Prometheus metrics endpoint | not started |
| P5 | Packaging: Docker containerisation and GitHub Actions CI workflow | not started |
| P6 | Verification: deliberately failing pull request demonstrating accuracy gate | not started |
| P7 | Deployment: automated schedule, documentation, and scoreboard publication | not started |

## Quickstart

### Prerequisites

- Python 3.10+
- GNU Make
  - Windows: `winget install ezwinports.make`
  - Debian / Ubuntu: `sudo apt-get install make python3-venv`

### Clone and setup

```bash
git clone https://github.com/Dash3210/gridcast.git
cd gridcast
make install
```

If your Python 3.10+ interpreter is not the default `python` on PATH, pass `PYTHON` to override it:

```bash
make install PYTHON=/path/to/python3.10
```

### Verification targets

Run the test suite, linter, and type checker:

```bash
make test
make lint
make typecheck
```

`make test` runs pytest with coverage reporting and strictly enforces the 85% code coverage gate.

### Running the CLI

Run subcommands using the virtual environment interpreter without needing manual shell activation:

- Windows:
  ```cmd
  .venv\Scripts\python.exe -m gridcast version
  ```
- Linux / macOS:
  ```bash
  .venv/bin/python -m gridcast version
  ```

Alternatively, if the virtual environment is activated in your shell:

```bash
python -m gridcast version
```

## What works today

Today (Phase P2), the working automation targets are:
- `make install` — creates the virtual environment and installs dependencies in editable mode
- `make test` — runs the test suite with coverage enforcement (>= 85%) and temporal leakage validation
- `make lint` — checks codebase formatting and linting rules with ruff
- `make typecheck` — validates static typing with mypy across `src` and `tests`
- `make clean` — removes virtual environment, caches, and build artefacts
- `make ingest` — runs the nightly half-hourly ingest CLI (`gridcast ingest --days 2` followed by `gridcast ingest --reconcile-recent`)
- `make data-check` — asserts integrity and completeness of the committed curated dataset

### Features

Phase P2 provides a bounded, leakage-free feature construction pipeline in `gridcast.features`:
- **Temporal Integrity**: Enforces that no feature at target period `t` uses any value stamped after `t`. Verified via future-perturbation invariance tests across multiple cutoffs (mid-window, DST transitions, and dataset boundary).
- **Time-based Keying**: All lags and rolling stats are keyed and sorted by `period_start_utc` timestamp lookup rather than row positions, ensuring correctness across 46-period (spring forward) and 50-period (autumn back) clock changes.
- **Bounded Feature Set (ADR-006 / D3)**:
  - *Actual Lags*: `actual_lag_1h`, `actual_lag_2h`, `actual_lag_3h`, `actual_lag_24h`, `actual_lag_48h`, `actual_lag_168h`.
  - *Actual Rolling Stats*: 3h, 24h, and 168h rolling mean and std, each window strictly ending at `t-1h`.
  - *Calendar*: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_uk_bank_holiday` (from static 2024-2027 register), plus sin/cos cyclical encodings of hour and day-of-year.
  - *Weather*: Open-Meteo previous-runs day-2 forecast temperature at `t` (matching the 48-hour issue horizon; reanalysis is forbidden), plus 24h mean and 24h min ending at `t`.
  - *Generation Mix*: Observed fuel shares for `wind`, `solar`, `gas`, `coal`, `nuclear`, and `biomass` lagged to `t-48h` to prevent contemporaneous leakage.

### Ingest

`gridcast` ingests half-hourly GB grid carbon intensity data from the official National Energy System Operator (NESO) Carbon Intensity API (`api.carbonintensity.org.uk`).

- **Column Contract**:
  - `period_start_utc` (`datetime64[ns, UTC]`): Half-hour start timestamp in UTC (inclusive). Primary key.
  - `period_end_utc` (`datetime64[ns, UTC]`): Half-hour end timestamp in UTC (`to` exclusive).
  - `utc_date` (`object`, `datetime.date`): Calendar date of `period_start_utc` in UTC.
  - `local_date` (`object`, `datetime.date`): Calendar date of `period_start_utc` in `Europe/London`.
  - `forecast` (`Int32`, nullable): Day-ahead intensity forecast in gCO2/kWh.
  - `actual` (`Int32`, nullable): Realised half-hourly intensity in gCO2/kWh.
  - `index` (`string`, nullable): Intensity category descriptor (`very low`, `low`, `moderate`, `high`, `very high`).
  - `ingested_at` (`datetime64[ns, UTC]`): Timestamp of ingestion run.

- **Local-Day Partitioning & DST**:
  Partitions are addressed and partitioned by the `Europe/London` local calendar day (`date=<YYYY-MM-DD>.parquet`). Due to UK Daylight Saving Time (DST):
  - Normal day: exactly 48 rows (24 hours).
  - Autumn clock change (clocks go back in October): exactly 50 rows (25 hours).
  - Spring clock change (clocks go forward in March): exactly 46 rows (23 hours).

- **Coverage & CLI Usage**:
  The repository maintains historical curated partitions from `2024-11-01` to the present day. Ingest coverage is managed and extended via the CLI:
  - `gridcast ingest --days N`: Ingest the last N local days, inclusive of today (default: 2 days).
  - `gridcast ingest --recent N`: Alias for `--days N` used by scheduled automation jobs.
  - `gridcast ingest --start YYYY-MM-DD --end YYYY-MM-DD`: Ingest a specific inclusive historical range.
  - `gridcast ingest --force`: Re-fetch and re-evaluate partitions that look complete.
  - `gridcast ingest --prune-cache`: Manually prune raw HTTP cache files older than 90 days.

- **Pending Observations, Missing Gaps, and Null Forecasts**:
  - A `null` actual value inside a partition means the observation was **not settled at ingest time** (intraday periods awaiting settlement). These values are naturally updated when downstream actuals publish.
  - A genuinely missing observation (an actual that has remained null for a period that ended more than 2 hours ago) or an omitted period withheld by upstream is audited and recorded in `data/audit/gaps.csv` (`local_date,period_start_utc,reason`) with reasons `missing_actual` or `missing_period`. Missing observations are never interpolated or synthesised.
  - A `null` forecast also occurs in the dataset (43 rows across three historical dates: `2025-01-12`, `2025-01-13`, and `2025-08-11`). On these dates, the upstream date endpoint (`/intensity/date`) emitted malformed/invalid JSON, and the `/intensity/{from}/{to}` range fallback endpoint had genuinely absent forecast values. These absent values are preserved as null rather than synthesised.
  - `data/audit/gaps.csv` is created only when a gap exists, which is why it may be absent in clean runs. When actuals publish downstream and a date is re-ingested with `--force`, stale `missing_actual` records are reconciled and removed, and the file is deleted if empty.

- **Upstream Retractions and Conflict Audit**:
  - When an incoming payload offers a null where a non-null observation already exists in storage, this indicates an upstream disagreement or retraction. Gridcast treats `actual` and `index` as an atomic observation, never overwriting existing non-null observations with incoming nulls.
  - Disagreements between upstream responses are logged once in `data/audit/conflicts.csv` (`local_date,period_start_utc,field,stored_value`) to keep retractions visible rather than silent.

- **Two-tier storage architecture & Cache Usability**:
  - **Raw HTTP cache buffer**: `data/cache/http/<YYYY-MM-DD>.json.gz` (gitignored, 90-day retention pruned automatically at the end of successful ingest runs).
  - **Curated partition table**: `data/curated/date=<YYYY-MM-DD>.parquet` (committed, append-only, deduplicated on primary key `period_start_utc`).
  - The raw HTTP cache is used only when the cached payload is *settled* (date is before today, has the expected row count for the DST type, and contains no null actuals for periods ended >2 hours ago). Today is always fetched live.

### How to Load the Dataset

`pd.read_parquet("data/curated")` is the supported loader and works out-of-the-box because `data/curated/` holds only pure Parquet partition files (`date=*.parquet`); audit CSV files live separately in `data/audit/`. If you load via filesystem globbing instead of directory loading, match `data/curated/date=*.parquet`.

```python
import pandas as pd

# Load the entire curated dataset directly from the partition directory
df = pd.read_parquet("data/curated")

# Sanity check: 731 daily partitions (2024-11-01 to 2026-11-01) = 35,088 half-hourly rows
assert len(df) == 35088, f"Unexpected row count: {len(df)}"
assert df["period_start_utc"].is_monotonic_increasing, "Dataset must be sorted by period_start_utc"
print(f"Loaded {len(df):,} rows successfully across {df['local_date'].nunique()} daily partitions.")
```

**Temporal Indexing and Daylight Saving Time (DST)**:
- Always sort and filter using `period_start_utc`, which is UTC-aware and strictly monotonic.
- When shifting or lagging by 24 hours (e.g. for seasonal day-ahead features), **use a time offset** (e.g. `df.set_index("period_start_utc").shift(freq="24h")` or join on `period_start_utc - pd.Timedelta(hours=24)`), **not a fixed 48-row shift** (`shift(48)`). Because UK clock changes occur in March and October, local days contain 46 rows (spring clock change) or 50 rows (autumn clock change).

**Audit Files**:
Audit logs are stored in `data/audit/` and created lazily only when anomalies occur:
- `data/audit/gaps.csv`: Records genuinely missing observations (`missing_actual` when an actual remains unpublished >2 hours post period end) or omitted periods withheld by upstream (`missing_period`). When downstream actuals publish and a date is re-ingested with `--force`, stale `missing_actual` entries are automatically reconciled and dropped, and the file is deleted if no gaps remain.
- `data/audit/conflicts.csv`: Records upstream retractions where an incoming payload presents a null value for a period where a non-null observation is already stored in curated partitions.

Everything else fails on purpose:
- `make train`, `make eval`, `make serve`, and `make docker` exit non-zero with explicit messages indicating the respective future phase in which they arrive.
- Note: `make eval` maps to the `gridcast backtest` CLI subcommand (scheduled for P3).

## Layout

```
gridcast/
|-- Makefile              # Automation recipes restricted to cmd/sh common subset
|-- pyproject.toml        # PEP 621 package metadata and tool configurations
|-- README.md             # Project overview, status tracking, and module guide
|-- docs/
|   `-- decisions.md      # Architecture Decision Records (ADRs)
|-- src/
|   `-- gridcast/
|       |-- py.typed      # PEP 561 marker for inline type annotations
|       |-- __init__.py   # Package version resolution from metadata
|       |-- _phase.py     # Loud exit helper for unimplemented phase milestones
|       |-- cli.py        # CLI entry point routing subcommands to modules or phase gates
|       |-- ingest.py     # Half-hourly API data ingest to Parquet partitions (P1)
|       |-- features.py   # Feature engineering and temporal leakage prevention tests (P2)
|       |-- models.py     # Machine learning forecast models and baseline comparison (P3)
|       |-- backtest.py   # Backtesting engine and historical evaluation (P3)
|       `-- serve.py      # FastAPI service and prediction endpoints (P4)
|-- tests/
|   |-- fixtures/         # Static API payload response fixtures
|   |-- test_smoke.py     # Smoke tests verifying package metadata, CLI routing, and gating
|   |-- test_ingest.py    # Unit and integration tests for ingest pipeline and schema
|   `-- test_data_integrity.py # Invariant assertions for committed curated partitions
`-- data/
    |-- audit/            # Audited anomalies: gaps.csv and conflicts.csv (created lazily)
    |-- cache/
    |   `-- http/         # Raw gzipped HTTP cache replay buffer (gitignored, 90-day retention)
    `-- curated/          # Committed append-only daily Parquet partitions (pure Parquet dataset)
```
