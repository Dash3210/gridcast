# Model card

**Status: Phase P3 candidate trained and evaluated (no gate verdict; ADR-020).** Phase P3 closes with neither a pass nor a fail. LightGBM regressor (`lgb.train`) trained natively on 26 features from the canonical production frame (weather features dropped per ADR-018), evaluated against three baselines (issue-anchored persistence, seasonal-naive, and published NESO forecast) across 15 fixed rolling origins (`2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`, covering local issue dates `2026-09-05` to `2026-10-31`).

## What is serving today

`persistence-v0`. Every issued point is the most recent actual intensity available at the issue instant,
held flat across the 48-hour horizon:

```
forecast(issue, h) = actual at (issue_time - ACTUAL_AVAILABILITY_LAG)     for h in 1..96
```

It is deliberately weak. It exists so the public record starts accumulating before a trained model is
ready, and every score row records `model_version`, so a better model appends rather than rewrites.
Phase P4 packages the LightGBM candidate into the prediction service.

## Inputs

- **Target**: realised half-hourly carbon intensity for GB, in gCO2/kWh.
- **Issue cadence**: one issue per day at 00:00 Europe/London (23:00Z in BST, 00:00Z in GMT).
- **Horizon**: 1–96 half-hours ahead, as an absolute offset from the issue instant, so DST changes the
  calendar day's length but never the horizon count.

| feature group | definition | availability |
|---|---|---|
| recent level | actual at `issue − 3h` | the 3-hour allowance is conservative, see below |
| lags | actual at `issue − 24h`, `−48h`, `−168h` | always available |
| rolling | mean and std over 3h, 24h, 168h ending at `issue − 3h` | always available |
| calendar | hour, weekday, month, weekend, UK bank holiday, cyclical encodings of the target period | deterministic |
| generation mix | fuel shares at `issue − 48h` | published before the issue |

**Availability allowance: three hours.** Chosen 2026-11-02. Observed on 2026-11-01 that settlement
periods ending at 20:00Z were still null at 20:11Z, and earlier that evening periods were still null
40 minutes after they ended. NESO does not publish actuals instantly, so a feature that reads the last
half hour before issue is correct by timestamp but unavailable in production. The value is conservative
pending measurement: `first_seen_at` now records when each actual first appeared, and the allowance is
tightened from that distribution once enough nights have accumulated.

**Weather features dropped (ADR-018):**
Weather features (`temperature`, `temperature_24h_mean`, `temperature_24h_min`) were dropped by a pre-registered decision rule in ADR-018. At inference the targets are in the future, and previous-runs has no future values, producing an irreconcilable train/serve lead distribution mismatch. An empirical ablation on the frozen 15-origin split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) demonstrated that removing weather features changed aggregate GATE_VALUE by only +0.086 gCO2/kWh (+0.22%) and long-horizon (h >= 48) error by +0.328 gCO2/kWh (+0.86%), satisfying both pre-registered drop criteria. Across horizon bands, each horizon has 15 origins; the +11-15% and -6.4% differences are consistent with sampling noise at that sample size rather than evidence of horizon dependence, and answering that question requires more origins, not more analysis of these 15. The decision was a pre-registered mean test.

## Service Metadata (GET /model contract)

The prediction service exposes `GET /model`, returning the following computed metadata describing the loaded model artifact:

- **Model Version**: `lgbm-v2`
- **Artifact SHA-256**: `1cd13471d793735748f81d17fb195e34f6f784e8f642ac60a16f016280c45310`
- **Interval Artifact SHA-256**: `ab94aa7135bc797e4c81993a65c96702088dc02d845d0c4262aaf133e46b8965`
- **Training Window**: `2024-11-08` to `2026-11-07` (730 issue dates, 00:00 Europe/London daily)
- **Feature Count**: 26 declared features

### Prediction Intervals (Empirical Residual Quantiles)

The model has no native interval. Prediction intervals are generated using per-horizon empirical residual quantiles from the **validation split only** (`2026-06-05` to `2026-08-30`), 10th and 90th percentiles, written to `data/models/interval_v2.json`.

- **Calibration method**: For each horizon $h \in [1, 96]$, calculate empirical residual $r = y - \hat{y}$ on the validation split. Let $q_{10}(h) = \text{quantile}_{0.10}(r)$ and $q_{90}(h) = \text{quantile}_{0.90}(r)$.
- **Interval construction**: `lower = pred + q10(h)`, `upper = pred + q90(h)`.
- **Bracketing invariant**: By construction, $q_{10}(h) \le 0 \le q_{90}(h)$ guarantees that `lower <= value <= upper` for every horizon. All 96 horizons satisfy this natively without widening.
- **Empirical validation coverage**: The empirical coverage of these bands on the validation split is **79.31%** (6,624 of 8,352 points covered).
- **Caveat & Calibration Notice**: 87 summer samples per horizon understate winter volatility; calibration is descriptive, not a guarantee.
- **Fixture Isolation**: Quantiles are calibrated strictly on the validation split (`2026-06-05` to `2026-08-30`) and never on `data/eval`, preventing evaluation test data leakage into served predictions.

### Serving Endpoints (Phase P4 Contracts)

| Endpoint | Method | Contract & Invariants |
|---|---|---|
| `/health` | `GET` | Liveness probe: returns HTTP 200 while process is up and model booster is loaded. Reports `status="ok"`, `model_version`, `training_date`, `training_window`, `content_sha256`, `feature_count`, `interval_sha256`, `data_freshness_at`, `served_issue_time`, and `is_current_issue`. |
| `/ready` | `GET` | Readiness probe: returns HTTP 200 when served issue is fresh (<= 27 hours old; 24h expected publish + 3h measured margin), returns HTTP 503 Service Unavailable when served issue is older than 27 hours. Accompanied by `gridcast_served_issue_age_seconds` gauge in `/metrics`. |
| `/forecast` | `GET` | Parameter `hours` is anchor-relative, counted strictly from the anchor (`target_time = issue_time + horizon * 30min`), validated to `1..48` integer (invalid values return HTTP 422 with accepted range `[1, 48]`). Anchors (`date` or `issue_time`) are strictly enforced at the boundary to 00:00 Europe/London across timezones/offsets; non-midnight anchors or invalid dates return HTTP 422 with structured errors without reflecting inputs or host paths. Returns response wrapper with `issue_time`, `generated_at`, `is_current_issue`, `contains_future_points`, `retrospective_horizons`, `hours`, `model_version`, `content_sha256`, `interval_sha256`, `model_hash`, `interval_hash`, `generation`, `state_version`, and `points`. When `is_current_issue` is false, includes `fallback_reason` naming the null feature and date. For `hours` in 1..48, `len(points) == hours * 2` with timestamps strictly increasing by 30 minutes; each point contains `horizon`, `timestamp`, `target_time`, `value`, `lower`, and `upper` where `lower <= value <= upper`. |
| `/metrics` | `GET` | Exposes Prometheus text exposition including request counter (`gridcast_requests_total`), latency histogram (`gridcast_request_duration_seconds`), served issue age (`gridcast_served_issue_age_seconds`), last measured MAE macro mean (`gridcast_last_measured_mae` with `plane="live_scoreboard"` and `horizons` labels) computed as the unweighted mean of per-horizon MAEs matching project scalar evaluation, and companion sample size gauge (`gridcast_last_measured_mae_samples{series,plane}`). The operator series is labeled `status="partial"`, `horizons="1-33"` (n=33), never "settled". Unsettled series (candidate) are exposed as absent with status label and NaN. Exact endpoint label set is strictly bounded to `{"/forecast", "/health", "/metrics", "unmatched"}`. |
| `/model` | `GET` | Returns JSON matching model card metadata: `version`, `content_sha256`, `training_window`, `feature_count`, `interval_sha256`, and monotonic `generation`. |
| `/reload` | `POST` | Reloads model and interval artifacts. Authenticated exclusively via `Authorization: Bearer <token>` header (or `X-Reload-Token`) using constant-time `secrets.compare_digest`. Status codes: 401 for a missing or wrong token when configured, 503 when `GRIDCAST_RELOAD_TOKEN` is unset, 200 on success, 500 when artifact cannot be loaded (returning generic error detail without host paths or filenames while keeping active in-memory snapshot intact). |

### Feature Path & Source Gap Handling

- **Single feature path**: In-service feature resolution calls `resolve_features_for_issue()`, which drives `build_example_frame()`, asserting row-for-row equality with the backtest frame.
- **No nulls reach the model**: When an upstream source gap causes null values in any required model feature, the service refuses the request with named structured error `FeatureSourceGapError` (HTTP 503), reporting the corrupted features and timestamp without exposing internal directory paths. Silent NaNs are strictly forbidden.

### Operational Characteristics and Serving Contract (ADR-025, ADR-026)

- **Issue availability timing**: The current-day issue becomes buildable only after the nightly ingest, because `actual_recent_3h` needs the previous day's 21:00 actual and the ingest runs around 09:00Z.
- **Explicit fallback labeling**: Before the nightly ingest lands, the service serves the newest buildable issue and says so via `issue_time` and `is_current_issue` (with `fallback_reason` naming the null feature and date when `is_current_issue` is false). Refusing to build with a null feature is deliberate: imputing the missing actual would be fabrication.
- **Retrospective horizons**: Horizons earlier than the publication time are retrospective (`retrospective_horizons`), because the 00:00 issue is published when the nightly ingest makes it buildable.
- **Fixed 00:00 Europe/London anchor**: Anchors are fixed at 00:00 Europe/London. Non-00:00 Europe/London timestamps return HTTP 422 at the serving boundary. Arbitrary anchors would require a v3 model trained across a spread of anchor hours, with its own fixture - not a serving-time change.
- **Reload status codes**: 401 for a missing or wrong token when a token is configured (header only), 503 when `GRIDCAST_RELOAD_TOKEN` is unset (no auth configured, so the endpoint is closed), 200 on success, 500 when the artifact cannot be loaded and the old booster is kept.
- **Snapshot Reload Mechanism**: One immutable snapshot reference (`ModelSnapshot`), read once per request; a mutex serialises reload writers during snapshot swap so concurrent forecast requests observe an internally consistent snapshot without reader locks.

### Lineage and Historical Artifacts
Model version `lgbm-v1` existed previously with 29 features:
- **Model Version**: `lgbm-v1`
- **Artifact SHA-256**: `5ddfe624a05dee49fa2ec339f6d22ba4ad6783dfcb1ea30eb7bca4a501e19e15`
- **Git Object ID**: `8f841a6f048242085a36c33ed8b92b7acdf6d8e7`
- **Training Window**: `2024-11-08` to `2026-11-01` (724 issue dates, 00:00 Europe/London daily)
- **Feature Count**: 29 declared features
`lgbm-v1` was superseded by `lgbm-v2` when weather features were dropped per ADR-018.

### Model Artifact Hashing and Verification Commands

Two distinct hash values exist for the model artifact `data/models/lgbm_model.txt`. `git hash-object` is shell-proof but yields the **object ID** (git object database hash), a different function from the content **SHA-256** pinned by this card:

1. **Content SHA-256 (64 hex characters)**:
   Exact command:
   ```bash
   python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('data/models/lgbm_model.txt').read_text(encoding='utf-8').encode('utf-8')).hexdigest())"
   ```
   Output: `1cd13471d793735748f81d17fb195e34f6f784e8f642ac60a16f016280c45310`
   This is the value returned by `GET /model` (`artifact_sha256`). Marked `binary` in `.gitattributes` beside the parquet rule so a CRLF checkout cannot change the bytes.

2. **Git Object ID (40 hex characters, SHA-1)**:
   Exact command:
   ```bash
   git hash-object data/models/lgbm_model.txt
   ```
   Output: `fe838f87dac4cb5412f5118600353b5293d52cd4`
   This is git's internal object ID in the git object database (computed over `blob <size>\0<content>`).


## Evaluation

Leakage and value correctness are checked on a grid covering DST transitions, gap-adjacent dates, and
the first and last week of the archive.

### Evaluation Metrics on the Frozen Split (frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins)

The scalar evaluation metric is defined as the unweighted mean of per-horizon MAEs:

```
MAE_h      = mean absolute error over scored periods at horizon h
GATE_VALUE = (1 / 96) * sum_{h=1..96} MAE_h
```

Evaluated on the frozen split (frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins; 2026-09-04 23:00 UTC to 2026-10-31 00:00 UTC, 15 rolling origins), each number carrying this origin range:

| candidate | GATE_VALUE (gCO2/kWh) | Notes |
|---|---|---|
| LightGBM candidate | 39.118 | Archive commit 3617143 (restated from 39.032 at commit 3617143; frozen fixture v2) |
| persistence (frozen backtest, 15 origins) | 53.741 | Unchanged across commits |
| seasonal-naive | 51.495 | Unchanged across commits |
| NESO operator (frozen backtest, 15 origins) | 11.666 | Published NESO forecast, vintage unverified (ADR-022) |

The candidate beats its own baselines by roughly a quarter and loses to the operator by about three times, on a fixed window, so any future number must be quoted beside its window. Do not describe this as accuracy, as a pass, or as meeting a standard.

All candidates and baselines are scored strictly on the identical intersection of valid periods where model predictions, actuals, and operator forecasts are present (1,440 periods scored, 0 excluded periods). All 96 horizons contribute (`96 / 96`).

### Operator Baseline Information Set Caveat (ADR-022)

The operator forecast is read from the curated `forecast` column at `target_time`. That column is NESO's published forecast for that settlement period, and it is published relative to the **target**, not to our issue instant. For a long horizon the operator therefore had information closer to the target than the candidate did; for a short horizon it may have had less. The published upstream data does not expose the publication timestamp, so alignment to what genuinely existed at issue instant $T$ is not possible from this source.

The introduction of `forecast_first_seen_at` to the curated contract records the timestamp when a forecast is first observed going forward, but this does not recover the true day-ahead value for anything already backfilled. Comparisons against this archived baseline carry the statement: *Published NESO forecast for the same periods, vintage unverified*. No claim of beating or losing to the operator is made without this caveat.

### Gate Verdict Status (ADR-020)

Phase P3 closes with **no gate verdict (neither a pass nor a fail)**. The previously considered threshold of 35.0 gCO2/kWh was derived from the candidate's own validation score (33.815 gCO2/kWh) plus 3.5%. It was therefore never an external standard: a model cannot be judged against a bar computed from itself. The first scoring is void, no pass or fail is claimed, and `make eval` prints:

```
GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed
```

and exits 0. Nothing failed, because nothing was judged.

### The Phase P4 Evaluation Gate (ADR-021)

The evaluation gate rule for Phase P4 is defined outside the model:

> On the frozen evaluation split, with candidate, persistence and operator measured on identical periods:
> the candidate must close at least 30 percent of the MAE gap between issue-anchored persistence and the
> operator.
>
> gap = MAE_persistence - MAE_operator
> required = MAE_persistence - 0.30 * gap
> passes only if MAE_candidate <= required

Derived from the VALIDATION split only (`2026-06-05` to `2026-08-30`, 15 rolling origins):
- `MAE_persistence = 57.776 gCO2/kWh`
- `MAE_operator = 9.649 gCO2/kWh`
- `gap = 48.127 gCO2/kWh`
- `required = 43.338 gCO2/kWh`

The frozen-split threshold is deliberately not computed or quoted in this ADR: that split is scored once, later, and a threshold printed beside a result looks tuned.

Why 30 percent: a floor demanding much of the distance between doing nothing and the operator, without requiring a solo model to beat an operator with information it does not have. State that the fraction is a policy choice fixed now, changeable only before a model is trained, never after a score is seen.

The gate is not wired into CI until Phase P5.

The benchmark is a data source, not an oracle, and needs the same alignment discipline as any feature; the vintage of an archived forecast is a property to be recorded, not assumed; the only fair comparison is between two forecasts made at the same instant.

### Key Horizons Detail (frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins)

| Horizon | Lead Time | Model MAE (gCO2/kWh) | Operator MAE (gCO2/kWh)* | Persistence MAE (gCO2/kWh) | Seasonal-Naive MAE (gCO2/kWh) | Scored Periods |
|---|---|---|---|---|---|---|
| h = 2 | 1.0 hr | 14.00 | 7.67 | 34.67 | 30.13 | 15 |
| h = 48 | 24.0 hr | 54.03 | 14.47 | 63.60 | 47.00 | 15 |
| h = 96 | 48.0 hr | 36.63 | 8.47 | 53.67 | 54.93 | 15 |

*\* Published NESO forecast for the same periods, vintage unverified (ADR-022).*

Dropping weather costs roughly 11-15% at h=82-93 and gains roughly 5-6% at h=49-65 and h=96; the decision was a pre-registered mean test.

All skill metrics are computed inside `compute_skill_score` asserting identical period sets at the point of division. Gate evaluation is executed locally via `make eval` (`python -m gridcast.cli eval`).

### Benchmark Target Mutability and Guard Blind Spot (ADR-023)

- **Target mutability & Frozen Fixture Resolution**: The split was previously frozen in origins, not in targets; its target values were a snapshot of the archive at run time. Nightly `--reconcile-recent --window-days 3` re-fetches recent partitions, so the final ~48 hours of every evaluation window were revised after the fact. Between commit `0bbb497` and commit `3617143`, revisions caused `GATE_VALUE` to move from 39.043 to 39.032. ADR-023 resolved this by materialising immutable evaluation fixtures (`data/eval/frozen_split_v1.parquet` and `data/eval/frozen_split_v2.parquet`), decoupling evaluation from live archive reconciliations.
- **Fixture v2 and v1 Lineage**: Evaluation fixture v2 is the active pinned fixture. Ground truth is shared byte-for-byte with v1 (`actual`, `neso_forecast`, `persistence_pred`, `seasonal_naive_pred` are asserted identical); only `candidate_pred` was updated for the 26-feature retrain (GATE_VALUE 39.118459). The v1 fixture is retired but retained on disk for provenance.
- **Two Data Planes**: The evaluation fixture is immutable, read exclusively by `make eval`; the live scoreboard reads the reconciled store so it captures late actuals.
- **Guard blind spot**: `test_model_card_consistency.py` previously compared documents to a static JSON fixture. Under ADR-023, it computes published metrics directly from `data/eval/` at test time, making `data/eval/` the single source of truth. Every published number is qualified with: `frozen fixture v2, sha256 87d322235bdc9de95c32a9622ca75b226ba9e30c3d1ce4e5126a93ac77e9eb51, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins`.

### Comparison Across the Two Evaluation Planes

```
frozen fixture v2 (Sep 5 - Nov 1, 15 origins)   persistence 53.741   operator 11.666
live scoreboard   (Nov 2 - Nov 7, settled)      persistence 29.308   operator  9.152
```

Card lines for both planes:
- `persistence (frozen backtest, 15 origins)`: **53.741** gCO2/kWh (n=1440)
- `operator (frozen backtest, 15 origins)`: **11.666** gCO2/kWh (n=1440)
- `persistence (live scoreboard, settled horizons)`: **29.308** gCO2/kWh (n=464)
- `operator (live scoreboard, partial horizons 1-33)`: **9.152** gCO2/kWh (n=33)

Persistence is the same definition in both planes: the project scalar macro mean (unweighted mean of per-horizon MAEs). On the live window, persistence macro mean is 29.308 gCO2/kWh (row mean is 28.625 gCO2/kWh, n=464). The live record is also not short-horizon weighted: 464 rows across horizons 1-96, median horizon 44. The difference is the sample - six days were far more persistence-friendly (24h band 22.4 live vs 55.6 frozen) and calmer (mean 30-min change 4.00 vs 5.50, std 42.56 vs 51.51).
The operator series on the live scoreboard is partial: 33 rows from one issue across horizons 1-33 (n=33, MAE 9.152), compared to 15 origins x 96 (n=1440, MAE 11.666) in the frozen backtest. It is never labeled "settled". When it settles across all horizons, the sample size will expand and the score will reflect longer horizons.

## Known failure modes (to be quantified in P3)

- **Wind droughts**: multi-day low-wind periods where intensity stays high and persistence is poor.
- **Public holidays**: demand and generation patterns break the weekly rhythm; the UK bank holiday table
  covers 2024–2027.
- **Interconnector trips**: sudden imports or exports that move intensity by tens of gCO2/kWh within a
  settlement period, invisible to every input in this table.

## The public record

The scoreboard starts at the first issued forecast and appends as targets settle. A forecast settles at
`issue + 48h`, so the calendar claim lags the work:

> 21 consecutive **settled** days from a 2026-11-02 start requires 21 issues (2026-11-02 … 2026-11-22),
> the last of which settles at 2026-11-24. The earliest date the claim can honestly be made is therefore
> **2026-11-24**, before allowing for any missed night.

No history is backfilled. The date range in `docs/scoreboard.md` is the proof, and it can only be as
long as the record has actually run.
