# Scoreboard

scoreboard started 2026-11-02; consecutive days recorded: 21

## Live Model Performance (All Settled Points)

### lgbm-v2

- **Date range covered:** 2026-11-16 to 2026-11-22
- **Scored points:** 564
- **Model MAE (macro mean, h=1..96):** 32.50
- **Model MAE (row mean):** 32.61
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 0, none: 564

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 162 | 26.92 |
| 12–24h | 144 | 33.26 |
| 24–36h | 138 | 43.67 |
| 36–48h | 120 | 26.78 |

### persistence-v0

- **Date range covered:** 2026-11-02 to 2026-11-22
- **Scored points:** 1908
- **Model MAE (macro mean, h=1..96):** 39.04
- **Model MAE (row mean):** 38.90
- **Operator vintage breakdown:** snapshot_at_issue: 0, archive_at_scoring: 33, none: 1875

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
| 0–12h | 498 | 30.97 |
| 12–24h | 480 | 39.11 |
| 24–36h | 474 | 42.50 |
| 36–48h | 456 | 43.60 |

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
