# Scoreboard

scoreboard started 2026-11-02; consecutive days recorded: 23

## Live Model Performance (All Settled Points)

### lgbm-v2

- **Date range covered:** 2026-11-16 to 2026-11-24
- **Scored points:** 764
- **Model MAE (macro mean, h=1..96):** 34.27
- **Model MAE (row mean):** 34.29
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 0, none: 764

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 214 | 29.68 |
| 12–24h | 192 | 32.19 |
| 24–36h | 190 | 46.45 |
| 36–48h | 168 | 28.82 |

### persistence-v0

- **Date range covered:** 2026-11-02 to 2026-11-24
- **Scored points:** 2108
- **Model MAE (macro mean, h=1..96):** 39.42
- **Model MAE (row mean):** 39.28
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 33, none: 2075

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 550 | 32.19 |
| 12–24h | 528 | 38.07 |
| 24–36h | 526 | 42.62 |
| 36–48h | 504 | 44.79 |

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
