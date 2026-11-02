# Model card

**Status: no trained model yet.** Phase P2 (features) is implemented and under review. Phase P3 builds
the models and the accuracy gate. This card records what is known now and is extended with per-horizon
metrics when they exist; nothing here is a placeholder number.

## What is serving today

`persistence-v0`. Every issued point is the most recent actual intensity available at the issue instant,
held flat across the 48-hour horizon:

```
forecast(issue, h) = actual at (issue_time - ACTUAL_AVAILABILITY_LAG)     for h in 1..96
```

It is deliberately weak. It exists so the public record starts accumulating before a trained model is
ready, and every score row records `model_version`, so a better model appends rather than rewrites.

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

**Serving caveat on weather features (open risk, ADR-018):** At inference the targets are in the future,
and the previous-runs series has no future values. The weather available for `T+h` at serve time is the
current model run with lead `h`, not the day-1/day-2 series the frame trains on with lead `>= 24h`.
Training and serving therefore use different lead distributions, so "the model has weather features" is
true in training and not true as stated for the service. Whether weather ships in the service will be
decided at Phase P3.

## Evaluation

Leakage and value correctness are checked on a grid covering DST transitions, gap-adjacent dates, and
the first and last week of the archive. Dates outside that grid are sampled, not exhaustive.

- Gate policy: the accuracy threshold is fixed from the validation split before the test split is
  scored; the test split is scored once per candidate and never used to tune.
- Splits are cut on issue date with a four-day embargo at each boundary, so no training target overlaps
  a validation or test target.
- Baseline to beat: the National Energy System Operator's own day-ahead forecast, scored on exactly the
  same periods and the same metric.

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
