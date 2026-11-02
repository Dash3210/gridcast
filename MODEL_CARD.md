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

**Serving caveat on weather features (ADR-018 resolution):** At inference the targets are in the future,
and the previous-runs series has no future values. The weather available for `T+h` at serve time is the
current model run with lead `h`, not the day-1/day-2 series the frame trains on with lead `>= 24h`.
Training and serving therefore use different lead distributions. ADR-018 decided Option (b): retain all
horizons, document this caveat explicitly, and bound the damage empirically. On the validation split ($h \le 48$),
mean temperature divergence $|Day1 - Day2|$ is $1.008$ °C, and validation MAE differs by only $0.0875$ gCO2/kWh
($28.6310$ vs $28.5435$ gCO2/kWh, a $0.31\%$ divergence). The candidate uses weather features with this
caveat: short horizons receive forecast lead $h$ at inference while trained on lead $\ge 24$h. Whether the
served model keeps them is decided in Phase P4, not here.

## Evaluation

Leakage and value correctness are checked on a grid covering DST transitions, gap-adjacent dates, and
the first and last week of the archive.

### Evaluation Metrics on the Frozen Split (frozen split, 2026-09-05 to 2026-11-01, 15 origins, 2026-11-02 run)

The scalar evaluation metric is defined as the unweighted mean of per-horizon MAEs:

```
MAE_h      = mean absolute error over scored periods at horizon h
GATE_VALUE = (1 / 96) * sum_{h=1..96} MAE_h
```

Evaluated on the frozen split (frozen split, 2026-09-05 to 2026-11-01, 15 origins, 2026-11-02 run; 2026-09-04 23:00 UTC to 2026-10-31 00:00 UTC, 15 rolling origins), each number carrying this origin range:

| candidate | GATE_VALUE (gCO2/kWh) |
|---|---|
| LightGBM candidate | 39.043 |
| persistence (issue-anchored) | 53.741 |
| seasonal-naive | 51.495 |
| NESO operator | 11.666 |

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

### Key Horizons Detail (frozen split, 2026-09-05 to 2026-11-01, 15 origins, 2026-11-02 run)

| Horizon | Lead Time | Model MAE (gCO2/kWh) | Operator MAE (gCO2/kWh)* | Persistence MAE (gCO2/kWh) | Seasonal-Naive MAE (gCO2/kWh) | Scored Periods |
|---|---|---|---|---|---|---|
| h = 2 | 1.0 hr | 13.93 | 7.67 | 34.67 | 30.13 | 15 |
| h = 48 | 24.0 hr | 53.99 | 14.47 | 63.60 | 47.00 | 15 |
| h = 96 | 48.0 hr | 40.17 | 8.47 | 53.67 | 54.93 | 15 |

*\* Published NESO forecast for the same periods, vintage unverified (ADR-022).*

All skill metrics are computed inside `compute_skill_score` asserting identical period sets at the point of division. Gate evaluation is executed locally via `make eval` (`python -m gridcast.cli backtest`).

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
