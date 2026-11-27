# Scoreboard

scoreboard started 2026-11-02; consecutive days recorded: 26

## Live Model Performance (All Settled Points)

### lgbm-v2

- **Date range covered:** 2026-11-16 to 2026-11-27
- **Scored points:** 1046
- **Model MAE (macro mean, h=1..96):** 27.95
- **Model MAE (row mean):** 28.00
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 0, none: 1046

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 283 | 25.26 |
| 12–24h | 264 | 25.47 |
| 24–36h | 259 | 38.20 |
| 36–48h | 240 | 22.99 |

### persistence-v0

- **Date range covered:** 2026-11-02 to 2026-11-27
- **Scored points:** 2390
- **Model MAE (macro mean, h=1..96):** 36.94
- **Model MAE (row mean):** 36.84
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 33, none: 2357

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 619 | 30.82 |
| 12–24h | 600 | 35.55 |
| 24–36h | 595 | 40.16 |
| 36–48h | 576 | 41.20 |

## Snapshot at Issue (Verified Vintage)

> No settled periods have a verified issue-time NESO snapshot yet (`snapshot_at_issue: 0`).
> Refusing to aggregate across sources; awaiting verified snapshots for skill comparison.

- **Date range covered:** None
- **Scored points:** 0
- **Model MAE:** N/A
- **Operator MAE (at issue):** N/A
- **Skill vs Operator:** N/A

### Horizon Performance

| Horizon Bucket | Scored Points | Model MAE | Operator MAE | Skill vs Operator |
|---|---|---|---|---|
| 0–12h | 0 | N/A | N/A | N/A |
| 12–24h | 0 | N/A | N/A | N/A |
| 24–36h | 0 | N/A | N/A | N/A |
| 36–48h | 0 | N/A | N/A | N/A |

## Archive at Scoring (Vintage Unverified)

> Published NESO forecast for the same periods, vintage unverified.
> Refusing to aggregate across sources; not eligible for skill comparison.

### persistence-v0

- **Date range covered:** 2026-11-02 to 2026-11-02
- **Scored points:** 33
- **Model MAE:** 24.67
- **Operator MAE (vintage unverified):** 9.15

#### Horizon Performance

| Horizon Bucket | Scored Points | Model MAE | Operator MAE |
|---|---|---|---|
| 0–12h | 24 | 27.33 | 10.21 |
| 12–24h | 9 | 17.56 | 6.33 |
| 24–36h | 0 | N/A | N/A |
| 36–48h | 0 | N/A | N/A |
