# Scoreboard

scoreboard started 2026-11-02; consecutive days recorded: 16

## Live Model Performance (All Settled Points)

### lgbm-v2

- **Date range covered:** 2026-11-16 to 2026-11-17
- **Scored points:** 88
- **Model MAE (macro mean, h=1..96):** 34.94
- **Model MAE (row mean):** 36.15
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 0, none: 88

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 44 | 39.82 |
| 12–24h | 24 | 38.20 |
| 24–36h | 20 | 25.60 |
| 36–48h | 0 | N/A |

### persistence-v0

- **Date range covered:** 2026-11-02 to 2026-11-17
- **Scored points:** 1432
- **Model MAE (macro mean, h=1..96):** 39.26
- **Model MAE (row mean):** 39.01
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 33, none: 1399

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 380 | 31.18 |
| 12–24h | 360 | 39.14 |
| 24–36h | 356 | 40.59 |
| 36–48h | 336 | 46.07 |

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
