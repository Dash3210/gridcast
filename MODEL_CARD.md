# Model card

**Status: Phase P3 candidate trained and evaluated (no gate verdict; ADR-020).** Phase P3 closes with neither a pass nor a fail. LightGBM regressor (`lgb.train`) trained natively on 29 features from the canonical production frame, evaluated against three baselines (issue-anchored persistence, seasonal-naive, and published NESO forecast) across 15 fixed rolling origins (`2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`, covering local issue dates `2026-09-05` to `2026-10-31`).

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
| weather | Open-Meteo previous-runs temperature for the target hour: day-1 forecast for `h ≤ 48`, day-2 for `h > 48` | issued at or before the issue instant |
| generation mix | fuel shares at `issue − 48h` | published before the issue |

**Availability allowance: three hours.** Chosen 2026-11-02. Observed on 2026-11-01 that settlement
periods ending at 20:00Z were still null at 20:11Z, and earlier that evening periods were still null
40 minutes after they ended. NESO does not publish actuals instantly, so a feature that reads the last
half hour before issue is correct by timestamp but unavailable in production. The value is conservative
pending measurement: `first_seen_at` now records when each actual first appeared, and the allowance is
tightened from that distribution once enough nights have accumulated.

**Short horizons use a 24-hour-old weather forecast, long horizons a 48-hour-old one.** That is the
freshest forecast that genuinely existed at the issue instant, and it caps short-horizon weather skill
by design.

**Serving decision on weather features (ADR-018 open, pending weather-removal ablation):**
At inference the targets are in the future, and the previous-runs series has no future values. The weather available for `T+h` at serve time is the current model run with lead `h`, not the day-1/day-2 series the frame trains on with lead `>= 24h`. Training and serving therefore use different lead distributions.
ADR-018 evaluated this empirical ablation on the frozen 15-origin split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) using identical frame builder, origins, and hyperparameters, shifting only weather timestamps by +24h:

```
baseline (current weather join)       39.032
weather values lagged a further 24h   38.279
delta                                 -0.753 gCO2/kWh   (-1.93%)
materiality threshold                 5.00%
```

Two principles keep this honest:
- The baseline reproduces today's `make eval` exactly (39.032 gCO2/kWh), so the ablation is the same evaluation with one input degraded, not a different experiment.
- The sign is negative: staler weather scored better. The feature is weakly used at these leads, and the result bounds the train/serve MISALIGNMENT, not the feature's informativeness. Do not write that weather is important; this measurement does not support that claim.

Decision: serve the current-run forecast for `T+h`, and record the lead actually used per horizon on every forecast row.

## Service Metadata (GET /model contract)

The prediction service exposes `GET /model`, returning the following computed metadata describing the loaded model artifact:

- **Model Version**: `lgbm-v1`
- **Artifact SHA-256**: `5ddfe624a05dee49fa2ec339f6d22ba4ad6783dfcb1ea30eb7bca4a501e19e15`
- **Training Window**: `2024-11-08` to `2026-11-01` (724 issue dates, 00:00 Europe/London daily)
- **Feature Count**: 29 declared features

### Model Artifact Hashing and Verification Commands

Two distinct hash values exist for the model artifact `data/models/lgbm_model.txt`. `git hash-object` is shell-proof but yields the **object ID** (git object database hash), a different function from the content **SHA-256** pinned by this card:

1. **Content SHA-256 (64 hex characters)**:
   Exact command:
   ```bash
   python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('data/models/lgbm_model.txt').read_text(encoding='utf-8').encode('utf-8')).hexdigest())"
   ```
   Output: `5ddfe624a05dee49fa2ec339f6d22ba4ad6783dfcb1ea30eb7bca4a501e19e15`
   This is the value returned by `GET /model` (`artifact_sha256`). Marked `binary` in `.gitattributes` beside the parquet rule so a CRLF checkout cannot change the bytes.

2. **Git Object ID (40 hex characters, SHA-1)**:
   Exact command:
   ```bash
   git hash-object data/models/lgbm_model.txt
   ```
   Output: `8f841a6f048242085a36c33ed8b92b7acdf6d8e7`
   This is git's internal object ID in the git object database (computed over `blob <size>\0<content>`).


## Evaluation

Leakage and value correctness are checked on a grid covering DST transitions, gap-adjacent dates, and
the first and last week of the archive.

### Evaluation Metrics on the Frozen Split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins)

The scalar evaluation metric is defined as the unweighted mean of per-horizon MAEs:

```
MAE_h      = mean absolute error over scored periods at horizon h
GATE_VALUE = (1 / 96) * sum_{h=1..96} MAE_h
```

Evaluated on the frozen split (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins; 2026-09-04 23:00 UTC to 2026-10-31 00:00 UTC, 15 rolling origins), each number carrying this origin range:

| candidate | GATE_VALUE (gCO2/kWh) | Notes |
|---|---|---|
| LightGBM candidate | 39.032 | Archive commit 3617143 (restated from 39.043 at 0bbb497; frozen fixture v1) |
| persistence (issue-anchored) | 53.741 | Unchanged across commits |
| seasonal-naive | 51.495 | Unchanged across commits |
| NESO operator | 11.666 | Published NESO forecast, vintage unverified (ADR-022) |

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

### Key Horizons Detail (frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins)

| Horizon | Lead Time | Model MAE (gCO2/kWh) | Operator MAE (gCO2/kWh)* | Persistence MAE (gCO2/kWh) | Seasonal-Naive MAE (gCO2/kWh) | Scored Periods |
|---|---|---|---|---|---|---|
| h = 2 | 1.0 hr | 13.93 | 7.67 | 34.67 | 30.13 | 15 |
| h = 48 | 24.0 hr | 53.99 | 14.47 | 63.60 | 47.00 | 15 |
| h = 96 | 48.0 hr | 39.15 | 8.47 | 53.67 | 54.93 | 15 |

*\* Published NESO forecast for the same periods, vintage unverified (ADR-022).*

All skill metrics are computed inside `compute_skill_score` asserting identical period sets at the point of division. Gate evaluation is executed locally via `make eval` (`python -m gridcast.cli eval`).

### Benchmark Target Mutability and Guard Blind Spot (ADR-023)

- **Target mutability & Frozen Fixture Resolution**: The split was previously frozen in origins, not in targets; its target values were a snapshot of the archive at run time. Nightly `--reconcile-recent --window-days 3` re-fetches recent partitions, so the final ~48 hours of every evaluation window were revised after the fact. Between commit `0bbb497` and commit `3617143`, revisions caused `GATE_VALUE` to move from 39.043 to 39.032. ADR-023 resolved this by materialising the immutable evaluation fixture `data/eval/frozen_split_v1.parquet` (and sidecar `data/eval/frozen_split_v1.json`), decoupling evaluation from live archive reconciliations.
- **Two Data Planes**: The evaluation fixture is immutable, read exclusively by `make eval`; the live scoreboard reads the reconciled store so it captures late actuals.
- **Guard blind spot**: `test_model_card_consistency.py` previously compared documents to a static JSON fixture. Under ADR-023, it computes published metrics directly from `data/eval/` at test time, making `data/eval/` the single source of truth. Every published number is qualified with: `frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins`.

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
