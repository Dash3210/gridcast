# gridcast

**Target.** gridcast is a containerised forecasting service for Great Britain's half-hourly grid carbon intensity, forecasting 48 hours ahead and will be gated in CI on forecast accuracy. It aims to answer whether a solo open-source model can beat the National Energy System Operator's (NESO) published forecast on a permanent, public scoreboard.

> Phase P3 closes with no gate verdict (neither a pass nor a fail; ADR-020). The previously proposed 35.0 gCO2/kWh threshold was derived from the candidate's own validation score and was mathematically invalid to judge it. On the frozen evaluation split (frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins; 2026-09-04 23:00 UTC to 2026-10-31 00:00 UTC), the candidate model scores GATE_VALUE = 39.118 gCO2/kWh against persistence (53.741 gCO2/kWh), seasonal-naive (51.495 gCO2/kWh), and NESO operator (11.666 gCO2/kWh) (archive commit 3617143, restated from 39.032 gCO2/kWh at commit 3617143; frozen fixture v2). An external gap-closure gate referencing persistence and operator baselines is defined in ADR-021 for Phase P4.

> Note: Upon completion of Phase P7, this README will open with the live forecast-vs-actual scoreboard table comparing gridcast models against the NESO baseline, positioned directly above the Quickstart.

## Status

| Phase | Description | Status |
|---|---|---|
| P0 | Setup: repository skeleton, environment, CLI gating, and smoke tests | done |
| P1 | Ingest: half-hourly API fetch to append-only Parquet daily partitions | done |
| P2 | Features: feature extraction and temporal leakage test suite | done |
| P3 | Models: baseline models, backtesting engine, and evaluation gate | closed (no gate verdict; ADR-020) |
| P4 | Service: FastAPI prediction API and Prometheus metrics endpoint | next |
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

### Quality gates

`make check` runs lint, typecheck, test, and data-check; nothing is committed without it.

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

Today (Phase P3), the working automation targets are:
- `make install` — creates the virtual environment and installs dependencies in editable mode
- `make test` — runs the test suite with coverage enforcement (>= 85%) and temporal leakage validation
- `make lint` — checks codebase formatting and linting rules with ruff
- `make typecheck` — validates static typing with mypy across `src` and `tests`
- `make clean` — removes virtual environment, caches, and build artefacts
- `make ingest` — runs the nightly half-hourly ingest CLI (`gridcast ingest --days 2` followed by `gridcast ingest --reconcile-recent`)
- `make features` — builds and inspects the canonical example feature frame keyed by `(issue_time, horizon)`
- `make train` — trains the native LightGBM model on settled features and writes artifact hash
- `make eval` — runs the rolling-origin backtest across 15 origins and evaluates the single scalar accuracy gate
- `make data-check` — asserts integrity and completeness of the committed curated dataset

### Features

Phase P2 provides a bounded feature construction pipeline in `gridcast.features`, with two leakage properties checked separately: **temporal** (no value stamped after the target period) and **operational** (no value that did not exist at the issue instant). The evidence is a horizon contract enforced in code, invariance tests on multi-issue frames, and a value grid compared against an independent naive oracle:
- **Issue-Time Framing (ADR-011, ADR-012)**: Forecasts Great Britain's carbon intensity 48 hours ahead from a single issue time. The canonical frame is keyed by `(issue_time, horizon)` where `issue_time` is anchored to `00:00 Europe/London` for each local day (`target_time = issue_time + horizon * 30min` for `horizon in range(1, 97)`). In BST, the issue instant is 23:00Z of the previous UTC date; in GMT, it is 00:00Z. Consecutive issue instants are 23, 24, or 25 hours apart across clock changes, and the horizon is an absolute half-hour offset so DST never changes the horizon count. The issue schedule is configurable via the `issue_times` parameter.
- **Horizon Contract Enforced in Code (ADR-011)**: Every feature declares the maximum source timestamp it is permitted to read, asserting directly in code that all values are computable strictly from information available at `<= issue_time`. Any feature build attempting to read source data timestamped after `issue_time` raises `ValueError`.
- **Temporal and Operational Leakage Guard**: Guarded by a sampled per-row perturbation invariance test across 25+ `(issue_time, horizon)` pairs covering dataset endpoints, DST transition days, and horizon limits (h=1 and h=96), asserting strict invariance when source rows stamped after `issue_time` are altered.
- **Time-based Keying**: All lags and rolling stats are resolved by UTC timestamp lookup rather than row positions, ensuring correctness across 46-period (spring forward) and 50-period (autumn back) clock changes.
- **Auxiliary Store Requirement**: `load_feature_sources()` loads source datasets from `data/curated/`, `data/weather/`, and `data/mix/`. `build_example_frame(..., require_aux=True)` enforces that auxiliary stores are present and populated.
- **Split Embargo (ADR-013)**: Because targets extend 48 hours ahead (`issue_time + 48h`), `split_by_issue_date(frame, train_end, val_end, embargo_days=4)` cuts splits strictly on `issue_time` (local issue date) and drops at least `embargo_days` issue dates at both split boundaries, guaranteeing that training targets never overlap validation or test targets (`max(target_time) of train < min(target_time) of val - 48h`). Rows with null `y` are excluded from all splits.
- **Bounded Feature Set (ADR-006, ADR-008, ADR-011)**:
  - *Keys*: `issue_time`, `horizon`, `target_time` (`issue_time + horizon * 30min`).
  - *Target (`y`)*: Realised carbon intensity `actual` at `target_time` (nullable for unsettled targets).
  - *Actual Features (ADR-016)*: Parameterized availability allowance `ACTUAL_AVAILABILITY_LAG = 3 hours` (`pd.Timedelta(hours=3)`), chosen on 2026-11-02 based on observed NESO publication delays, applied uniformly across all 96 horizons. Features comprise `actual_recent_3h` (the most recent actual available at issue time, for the half hour ending immediately before `issue_time - ACTUAL_AVAILABILITY_LAG`, renamed from `actual_lag_1h`), and historical lags `actual_lag_24h`, `actual_lag_48h`, `actual_lag_168h` relative to `issue_time`. `actual_lag_2h` and `actual_lag_3h` are dropped as unavailable under the 3-hour allowance.
  - *Actual Rolling Stats*: 3h, 24h, and 168h rolling mean and std, each window strictly ending at `issue_time - ACTUAL_AVAILABILITY_LAG`.
  - *Calendar*: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_uk_bank_holiday` (from static 2024-2027 register), plus sin/cos cyclical encodings of hour and day-of-year evaluated at `target_time`.
  - *Weather*: Dropped per pre-registered ablation in ADR-018. Former features `temperature`, `temperature_24h_mean`, and `temperature_24h_min` are removed, reducing the feature count from 29 to 26 and eliminating the train/serve lead distribution mismatch without requiring weather dependencies at serve time.
  - *Generation Mix*: Observed fuel shares for `wind`, `solar`, `gas`, `coal`, `nuclear`, and `biomass` lagged to `issue_time - 48h`.

#### How to Build and Inspect Features

You can build and inspect the feature frame directly from the CLI or in Python without reading module internals:

**Via Make / CLI**:
```bash
make features
# or: .venv/bin/python -m gridcast.cli features
```

**In Python**:
```python
from gridcast.features import (
    FEATURE_COLUMNS,
    build_example_frame,
    load_feature_sources,
    select_training_arrays,
)

# 1. Load source datasets
intensity_df, weather_df, mix_df = load_feature_sources()

# 2. Build example frame keyed by (issue_time, horizon)
df = build_example_frame(intensity_df, weather_df, mix_df, require_aux=True)

# 3. Extract training feature matrix X and target y (safely dropping unsettled targets)
X, y = select_training_arrays(df)

print(f"Dataset: {len(df):,} examples across {df['issue_time'].nunique()} issue times.")
print(f"Training arrays: {X.shape[0]:,} settled rows ({len(FEATURE_COLUMNS)} features)")
print(f"y shape: {y.shape} (strictly non-null training labels)")
```

`select_training_arrays(df)` drops unsettled target rows and verifies `y` contains no nulls before returning `(X, y)`. Model inputs `X` is a positive selection constructed via `select_model_inputs(df)` for inference, and `df.drop(columns=["y"])` is strictly forbidden.

### Models and Evaluation (Phase P3)

Phase P3 builds the forecast model, three comparison baselines, and the evaluation backtesting engine:
- **Models**:
  - *Candidate*: LightGBM regressor (`train_lgbm`) using the native `lgb.train` API (no scikit-learn dependency), trained deterministically (`seed=42`, `num_threads=1`) on the 26 canonical features (weather features dropped per ADR-018).
  - *Persistence Baseline*: Reads the most recent actual available at issue instant (`actual` at `issue_time - ACTUAL_AVAILABILITY_LAG`), held flat across all 96 horizons. Guarded by a strict non-tautological provenance assertion against the raw curated store (`source_period_start_utc == issue_time - ACTUAL_AVAILABILITY_LAG` and forecast equals raw store actual).
  - *Seasonal-Naive Baseline*: Reads the realised actual from 24 hours before the target time (`target_time - 24h`) from history strictly available at `<= issue_time - ACTUAL_AVAILABILITY_LAG`.
  - *Operator Baseline*: Published forecast from the National Energy System Operator (NESO). Vintage is unverified for historical archive rows (ADR-022); live snapshots at issue instant are recorded going forward.
- **Evaluation Integrity Principles**:
  - *Identical Period Scoring*: Models and baselines are scored exclusively on the exact intersection where model forecast, realised actual, and operator forecast are all present. Exclusion counts are audited per reason.
  - *Division-Time Period Set Assertion*: All skill scores are evaluated strictly through `compute_skill_score`, asserting that numerator and denominator period sets are identical at the point of division on the live path.
  - *Temporal Fold Isolation*: On every fold, the training set is asserted strictly to terminate before the fold issue origin: $\max(\text{train}[\text{target\_time}]) < T$.
  - *Single Scalar Metric*: Avoids pool-weighting bias where middle horizons are evaluated three times more frequently than edge horizons by defining:
    $$\text{MAE}_h = \text{mean absolute error at horizon } h$$
    $$\text{GATE\_VALUE} = \frac{1}{96} \sum_{h=1}^{96} \text{MAE}_h$$
    All 96 horizons are strictly required to contribute non-null MAEs (preventing silent denominator division by 95).
- **Evaluation Verdict Status (ADR-020)**:
  - The previously considered threshold of `35.000 gCO2/kWh` was derived from the candidate's own validation score (33.815 gCO2/kWh plus 3.5%) and was therefore invalid as an external gate. A model cannot be judged against a bar computed from itself; the first scoring is void, and Phase P3 closes with **no gate verdict (neither a pass nor a fail)**.
  - On the frozen test split (frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins; `2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`), the candidate scores `GATE_VALUE = 39.118 gCO2/kWh` vs persistence `53.741 gCO2/kWh`, seasonal-naive `51.495 gCO2/kWh`, and NESO operator `11.666 gCO2/kWh` (archive commit 3617143, restated from 39.032 gCO2/kWh at commit 3617143; frozen fixture v2). The candidate beats its own baselines by roughly a quarter and loses to the operator by about three times, on a fixed window, so any future number must be quoted beside its window. Do not describe this as accuracy, as a pass, or as meeting a standard.
  - *Benchmark Target Mutability & Frozen Fixtures (ADR-023)*: The split was previously frozen in origins, not in targets; targets were a snapshot of the archive at run time. Revisions in recent partitions caused GATE_VALUE to move from 39.043 to 39.032. ADR-023 froze both origins and targets into immutable evaluation fixtures (`data/eval/frozen_split_v1.parquet` and `data/eval/frozen_split_v2.parquet`), separating evaluation from live archive reconciliations into dual data planes.
  - `make eval` (`python -m gridcast.cli eval`) reports metrics on the frozen split, prints `GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed`, and exits 0. Nothing failed, because nothing was judged.
- **Phase P4 Gate Specification (ADR-021)**:
  - Defined prior to Phase P4 training in terms of external quantities:
    > On the frozen evaluation split, with candidate, persistence and operator measured on identical periods:
    > the candidate must close at least 30 percent of the MAE gap between issue-anchored persistence and the
    > operator.
    >
    > gap = MAE_persistence - MAE_operator
    > required = MAE_persistence - 0.30 * gap
    > passes only if MAE_candidate <= required
  - Derived on the validation split only (`2026-06-05` to `2026-08-30`, 15 rolling origins: persistence 57.776, operator 9.649, gap 48.127, required 43.338). Gating in CI arrives in Phase P5.

### Ingest

`gridcast` ingests half-hourly GB grid carbon intensity data, generation fuel mix, and previous-runs weather forecasts from official public APIs:
- **Carbon Intensity**: Official NESO Carbon Intensity API (`api.carbonintensity.org.uk/intensity`)
- **Generation Mix**: Official NESO Generation Mix API (`api.carbonintensity.org.uk/generation`)
- **Weather Forecasts**: Open-Meteo previous-runs API (`previous-runs-api.open-meteo.com/v1/forecast`), strictly previous-runs forecasts `temperature_2m_previous_day1` (day 1, issued 24h prior) and `temperature_2m_previous_day2` (day 2, issued 48h prior; reanalysis is rejected)

#### Data Layout

```
data/
  curated/     # Committed daily carbon intensity partitions (half-hourly)
  weather/     # Committed daily weather partitions (hourly previous-runs day 2)
  mix/         # Committed daily generation mix partitions (half-hourly fuel shares)
  cache/       # Ephemeral raw HTTP caches (http, weather, mix)
  audit/       # Gaps, source-gaps, and conflicts audit logs
```

- **Curated Intensity Schema Contract**:
  - `period_start_utc` (`datetime64[ns, UTC]`): Half-hour start timestamp in UTC (inclusive). Primary key.
  - `period_end_utc` (`datetime64[ns, UTC]`): Half-hour end timestamp in UTC (`to` exclusive).
  - `utc_date` (`object`, `datetime.date`): Calendar date of `period_start_utc` in UTC.
  - `local_date` (`object`, `datetime.date`): Calendar date of `period_start_utc` in `Europe/London`.
  - `forecast` (`Int32`, nullable): Operator's published forecast for the period, vintage unverified, in gCO2/kWh.
  - `actual` (`Int32`, nullable): Realised half-hourly intensity in gCO2/kWh.
  - `index` (`string`, nullable): Intensity category descriptor (`very low`, `low`, `moderate`, `high`, `very high`).
  - `ingested_at` (`datetime64[ns, UTC]`): Timestamp of ingestion run.
  - `first_seen_at` (`datetime64[ns, UTC]`, nullable): Timestamp when a realised actual was first observed for this period.
  - `forecast_first_seen_at` (`datetime64[ns, UTC]`, nullable): Timestamp when a forecast was first observed for this period (stamped on first non-null forecast, never overwritten).

- **Weather Schema Contract (`data/weather/date=YYYY-MM-DD.parquet`)**:
  - `period_start_utc` (`datetime64[ns, UTC]`): Hourly start timestamp in UTC.
  - `local_date` (`object`, `datetime.date`): Calendar date in `Europe/London`.
  - `temperature_2m_previous_day1` (`float64`): NWP 2-meter air temperature (°C) forecasted 24h prior (consumed by horizons h <= 48).
  - `temperature_2m_previous_day2` (`float64`): NWP 2-meter air temperature (°C) forecasted 48h prior (consumed by horizons h > 48).

- **Generation Mix Schema Contract (`data/mix/date=YYYY-MM-DD.parquet`)**:
  - `period_start_utc` (`datetime64[ns, UTC]`): Half-hourly start timestamp in UTC.
  - `local_date` (`object`, `datetime.date`): Calendar date in `Europe/London`.
  - Fuel share columns (`float64`): `wind`, `solar`, `gas`, `coal`, `nuclear`, `biomass`, `hydro`, `imports`, `other` (percentages summing to 100%).

- **Local-Day Partitioning & DST**:
  Partitions are addressed and partitioned by the `Europe/London` local calendar day (`date=<YYYY-MM-DD>.parquet`). Due to UK Daylight Saving Time (DST):
  - Normal day: exactly 48 half-hourly rows / 24 hourly weather rows.
  - Autumn clock change (clocks go back in October): exactly 50 half-hourly rows / 25 hourly weather rows.
  - Spring clock change (clocks go forward in March): exactly 46 half-hourly rows / 23 hourly weather rows.

- **Coverage & CLI Usage**:
  The repository maintains historical curated partitions from `2024-11-01` to the present day. Ingest coverage is managed and extended via the CLI:
  - `gridcast ingest [--days N | --start YYYY-MM-DD --end YYYY-MM-DD]`: Ingest carbon intensity.
  - `gridcast ingest --weather [--days N | --start YYYY-MM-DD --end YYYY-MM-DD]`: Ingest hourly weather forecasts.
  - `gridcast ingest --mix [--days N | --start YYYY-MM-DD --end YYYY-MM-DD]`: Ingest half-hourly generation mix.
  - `gridcast ingest --force`: Re-fetch and re-evaluate partitions that look complete.
  - `gridcast ingest --prune-cache`: Manually prune raw HTTP cache files older than 90 days.

- **Pending Observations, Missing Gaps, and Null Forecasts**:
  - A `null` actual value inside a partition means the observation was **not settled at ingest time** (intraday periods awaiting settlement). These values are naturally updated when downstream actuals publish.
  - A genuinely missing observation (an actual that has remained null for a period that ended more than 2 hours ago) or an omitted period withheld by upstream is audited and recorded in `data/audit/gaps.csv` (`local_date,period_start_utc,reason`) with reasons `missing_actual` or `missing_period`. Missing observations are never interpolated or synthesised.
  - A `null` forecast also occurs in the dataset (43 rows across three historical dates: `2025-01-12`, `2025-01-13`, and `2025-08-11`). On these dates, the upstream date endpoint (`/intensity/date`) emitted malformed/invalid JSON, and the `/intensity/{from}/{to}` range fallback endpoint had genuinely absent forecast values. These absent values are preserved as null rather than synthesised.
  - `data/audit/gaps.csv` is created only when a gap exists, which is why it may be absent in clean runs. When actuals publish downstream and a date is re-ingested with `--force`, stale `missing_actual` records are reconciled and removed, and the file is deleted if empty.

- **Auxiliary Store Coverage and Source Gaps (ADR-009)**:
  - Both auxiliary stores (`data/mix/` and `data/weather/`) are subject to the same strict completeness invariant as the curated intensity archive: every period is either present in storage or recorded in `data/audit/source-gaps.csv`.
  - Ingestion performs an automated post-fetch coverage check against the curated grid (`check_mix_coverage_and_retry`, `check_weather_coverage_and_retry`). For missing periods, targeted retries recover boundary straddling periods (e.g. `2024-12-31T23:30Z` and `2025-12-31T23:30Z` on month boundaries).
  - Periods that remain missing after retries because upstream returns empty payloads (`data: []`) are audited in `data/audit/source-gaps.csv` (`store,local_date,period_start_utc,reason`). In the current window, 25 generation mix periods grouped by `local_date` (2 on `2025-01-12`, 15 on `2025-01-13`, and 8 on `2025-08-11`; the partition for `2025-08-10` is complete at 48/48 periods) are documented source gaps (`upstream_empty`), while weather coverage is 100% complete.
  - At the point of use in feature extraction, mix feature columns reflect these 25 unrecoverable periods as nulls alongside the initial 96 warmup periods (at `t-48h`). These are reported transparently in `load_feature_sources()` and are never artificially imputed.

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
- `data/audit/source-gaps.csv`: Records settlement periods genuinely absent from auxiliary stores (`mix` or `weather`) where the upstream API returned empty responses (`upstream_empty`) after targeted boundary retries. Created lazily, automatically reconciled if data appears on subsequent re-ingestion, and strictly verified by `make data-check`.
- `data/audit/conflicts.csv`: Records upstream retractions where an incoming payload presents a null value for a period where a non-null observation is already stored in curated partitions.

Phase milestones not yet arrived fail on purpose:
- `make serve` and `make docker` exit non-zero with explicit messages indicating the future phase in which they arrive (P4 and P5).
- `make train` and `make eval` are active automation targets for Phase P3.

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
    |-- audit/            # Audited anomalies: gaps.csv, source-gaps.csv, conflicts.csv (created lazily)
    |-- cache/
    |   `-- http/         # Raw gzipped HTTP cache replay buffer (gitignored, 90-day retention)
    |-- curated/          # Committed append-only daily Parquet partitions (pure Parquet dataset)
    |-- mix/              # Committed daily generation mix Parquet partitions
    `-- weather/          # Committed daily weather forecast Parquet partitions
```
