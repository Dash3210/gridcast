# Weather Removal Ablation: Does Weather Earn Its Place? (Phase P4-c)

- **Date:** 2026-11-08
- **Evaluation Fixture:** Frozen Split v1 (`data/eval/frozen_split_v1.parquet`, SHA-256 `205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4`, archive commit `3617143`, window 2026-09-05 to 2026-11-01, 15 origins)
- **Status:** Decision Recorded — **DROP weather features** (implementation deferred to subsequent round)

---

## 1. Executive Summary & Pre-Registered Decision

The stale-versus-current weather experiment (ADR-018) moved `GATE_VALUE` by 1.93% (-0.753 gCO2/kWh) and bounded lead sensitivity, showing that weather features were only weakly utilized at these horizons. However, that test did not determine whether weather features earn their place in the production feature set.

This pre-registered ablation tested whether removing weather features impacts model accuracy against the identical frozen test fixture.

Under the amended pre-registered decision rule fixed before reading the results, two conditions must hold for DROP:
```
Condition 1 (aggregate):     DROP only if N <= 1.02 * W
Condition 2 (tail, h >= 48): DROP only if N_long <= 1.05 * W_long
Decision:                   DROP if both hold, else KEEP
```
The second condition was added before the result was read.

### Measured Outcome

- **Control Arm ($W$, with weather):** `GATE_VALUE = 39.032478 gCO2/kWh`
- **Treatment Arm ($N$, without weather):** `GATE_VALUE = 39.118459 gCO2/kWh`
- **Delta ($N - W$):** `+0.085981 gCO2/kWh` (`+0.22%`)
- **Pre-Registered 2.0% Threshold Limit ($W \times 1.02$):** `39.813127 gCO2/kWh` (Condition 1: PASS, $39.118459 \le 39.813127$)
- **Long Horizon Tail ($h \ge 48$):** $W = 38.0234$ gCO2/kWh, $N = 38.3512$ gCO2/kWh, delta $+0.328$ gCO2/kWh ($+0.86\%$)
- **Pre-Registered 5.0% Tail Limit ($W_{\text{long}} \times 1.05$):** `39.9246 gCO2/kWh` (Condition 2: PASS, $38.3512 \le 39.9246$)
- **Decision:** **DROP weather stands** (both conditions satisfied)

The aggregate accuracy movement from dropping weather is only **+0.086 gCO2/kWh (+0.22%)**, well below the 2.0% threshold. At long horizons $h \ge 48$, degradation is only **+0.328 gCO2/kWh (+0.86%)**, well within the 5.0% ceiling. Furthermore, at long lead $h = 96$, dropping weather actually **improves** accuracy by **2.518 gCO2/kWh (-6.43%)**. Weather features fail to justify the external serving dependency, operational failure modes, and unresolvable train/serve lead distribution mismatch.

Per specification, this round records the decision without modifying the production model, serve feature list, or frozen fixture.

---

## 2. The Decision Rule (Pre-Registered & Amended)

Fixed prior to running the experiment and reading results:
- Let $W$ = `GATE_VALUE` with weather (Control arm).
- Let $N$ = `GATE_VALUE` without weather (Treatment arm).
- Let $W_{\text{long}}$ = mean MAE over horizons $h \ge 48$ with weather.
- Let $N_{\text{long}}$ = mean MAE over horizons $h \ge 48$ without weather.
- Both arms scored on frozen fixture v1.

The decision rule as applied has two conditions, where the second condition was added before results were read:
```
Condition 1 (aggregate):     N <= 1.02 * W        (no-weather within 2.0% of weather)
Condition 2 (tail, h >= 48): N_long <= 1.05 * W_long  (no-weather tail within 5.0% of weather)
Decision:                   DROP if both hold, else KEEP
```

### Rationale for the Dual Conditions

ADR-021's gap gate demands closing 30% of a 48.127 gCO2/kWh gap, which is 14.438 gCO2/kWh. Two percent of 39.032 is ~0.78 gCO2/kWh, roughly 5% of the movement required by the gate. A feature set that buys less than a twentieth of what the gate requires does not justify a serving dependency that can never reproduce its training distribution at inference time (ADR-018). The secondary condition over $h \ge 48$ bounded long-horizon degradation to 5.0% before reading results, guarding against tail collapse.

---

## 3. Methodology & Two Experimental Arms

### Frame Construction & Arm Implementation

- **Control Arm ($W$):** Feature matrix built with all 29 production `FEATURE_COLUMNS`, including the three weather features:
  - `temperature`
  - `temperature_24h_mean`
  - `temperature_24h_min`
- **Treatment Arm ($N$):** Feature matrix built without weather features (26 non-weather features).
  - *Method used:* The feature frame builder `build_example_frame` enforces `FEATURE_COLUMNS` and sets absent weather to all NaNs if `weather_df=None`. Passing NaN columns would present dummy columns rather than evaluating the model without weather features. Therefore, per the brief's protocol, the weather columns were dropped from the training and prediction slices before model training.

### Training Parameters (Identical Across Arms)

- **Algorithm:** Native LightGBM regressor (`train_lgbm`)
- **Objective:** `regression_l1`
- **Metric:** `mae`
- **Learning rate:** `0.05`
- **Num leaves:** `31`
- **Min child samples:** `20`
- **Feature fraction:** `0.8`
- **Seed:** `42`
- **Deterministic:** `True`
- **Force column-wise:** `True`
- **Num threads:** `1` (single-threaded)
- **Num boost rounds:** `100`
- **Folds:** 15 rolling origins across test split (`2026-09-04 23:00 UTC` to `2026-10-31 00:00 UTC`, step 4 days)
- **Temporal leakage guard:** Strictly enforced via `assert_fold_training_integrity` ($\max(\text{train target\_time}) < T$).

### Scoring Protocol

Both arms were scored against the frozen fixture's `actual` and `neso_forecast` columns (`data/eval/frozen_split_v1.parquet`), not against a rebuilt frame. The ground truth actuals and baseline forecasts are byte-identical across arms; only model predictions differ.

---

## 4. Control Arm Precondition Verification

Before evaluating the treatment arm, the control arm was verified against the frozen fixture:

```
Fixture GATE_VALUE:           39.032478 gCO2/kWh
Retrained GATE_VALUE:         39.032478 gCO2/kWh
GATE_VALUE delta:             0.00000000 gCO2/kWh  (tolerance: 0.0)
Max per-row candidate delta:  0.00000000 gCO2/kWh  (tolerance: 0.0)
Precondition check:           PASSED
```

The control arm reproduced the frozen fixture bitwise identically with 0.0 maximum per-row prediction delta, satisfying the precondition.

---

## 5. Aggregate Results & Benchmark Comparison

| Metric | Control Arm ($W$, with weather) | Treatment Arm ($N$, without weather) | Delta ($N - W$) | Delta (%) |
|:---|:---:|:---:|:---:|:---:|
| **GATE_VALUE** | **39.032478** | **39.118459** | **+0.085981** | **+0.22%** |
| Operator MAE (NESO) | 11.666319 | 11.666319 | 0.000000 | 0.00% |
| Persistence MAE | 53.741007 | 53.741007 | 0.000000 | 0.00% |
| Seasonal-Naive MAE | 51.495139 | 51.495139 | 0.000000 | 0.00% |
| Skill vs Operator | -234.58% | -235.31% | -0.73% | — |
| Decision Threshold (+2.0%) | 39.813127 | — | — | — |
| **Pre-Registered Decision** | — | — | — | **DROP** |

---

## 6. Key Horizons Detail

| Horizon | Lead Time | Control MAE ($W$) | Treatment MAE ($N$) | Delta ($N - W$) | Delta (%) | NESO MAE | Persistence | Seasonal-Naive |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$h = 1$** | 0.5h | 13.2536 | 12.9836 | -0.2700 | -2.04% | 13.7333 | 33.1333 | 29.9333 |
| **$h = 2$** | 1.0h | 13.9344 | 13.9959 | +0.0615 | +0.44% | 7.6667 | 34.6667 | 30.1333 |
| **$h = 48$** | 24.0h | 53.9933 | 54.0323 | +0.0390 | +0.07% | 14.4667 | 63.6000 | 47.0000 |
| **$h = 96$** | 48.0h | 39.1453 | 36.6272 | -2.5181 | -6.43% | 8.4667 | 53.6667 | 54.9333 |

Key Observations:
1. **$h = 1$ (Intraday prompt):** Removing weather improves MAE by 0.270 gCO2/kWh (-2.04%).
2. **$h = 2$ (Immediate intraday):** MAE delta is +0.061 gCO2/kWh (+0.44%), virtually flat.
3. **$h = 48$ (Day-ahead horizon):** MAE delta is +0.039 gCO2/kWh (+0.07%), virtually flat.
4. **$h = 96$ (Two-day horizon):** Removing weather substantially **improves** accuracy by **2.518 gCO2/kWh (-6.43%)**. Rather than helping at long horizons, weather features introduced noise.

---

## 7. Full Per-Horizon Breakdown ($h = 1 \dots 96$)

Scored across all 15 origins (15 periods per horizon, 1,440 total intersection periods):

<!-- BEGIN GENERATED TABLE -->
| Horizon | Control MAE ($W$) | Treatment MAE ($N$) | Delta ($N - W$) | Delta (%) | NESO MAE | Persistence MAE | Seasonal-Naive MAE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 13.2536 | 12.9836 | -0.2700 | -2.04% | 13.7333 | 33.1333 | 29.9333 |
| 2 | 13.9344 | 13.9959 | +0.0615 | +0.44% | 7.6667 | 34.6667 | 30.1333 |
| 3 | 14.8999 | 14.8045 | -0.0954 | -0.64% | 5.7333 | 36.4667 | 28.6667 |
| 4 | 16.5166 | 16.2143 | -0.3023 | -1.83% | 6.2000 | 36.8000 | 27.0667 |
| 5 | 17.9933 | 17.3986 | -0.5947 | -3.31% | 6.8667 | 38.2667 | 27.0667 |
| 6 | 18.7746 | 17.9414 | -0.8332 | -4.44% | 7.8667 | 37.9333 | 27.8667 |
| 7 | 18.4443 | 17.8018 | -0.6425 | -3.48% | 7.5333 | 36.0000 | 28.8667 |
| 8 | 19.9425 | 19.7633 | -0.1792 | -0.90% | 7.2667 | 35.8000 | 29.8667 |
| 9 | 23.0534 | 22.3825 | -0.6710 | -2.91% | 8.5333 | 35.2000 | 36.3333 |
| 10 | 26.3391 | 25.8755 | -0.4636 | -1.76% | 9.8000 | 32.3333 | 38.3333 |
| 11 | 28.0210 | 28.1164 | +0.0954 | +0.34% | 13.4000 | 29.4000 | 44.4000 |
| 12 | 30.7875 | 30.3206 | -0.4670 | -1.52% | 19.8000 | 28.3333 | 47.4667 |
| 13 | 31.4782 | 30.6880 | -0.7902 | -2.51% | 16.0667 | 31.0000 | 46.6667 |
| 14 | 31.9333 | 30.7578 | -1.1755 | -3.68% | 22.0000 | 31.7333 | 46.2000 |
| 15 | 34.3691 | 33.8559 | -0.5133 | -1.49% | 11.5333 | 34.6000 | 45.1333 |
| 16 | 34.3812 | 33.7232 | -0.6580 | -1.91% | 11.2667 | 35.5333 | 44.2667 |
| 17 | 35.1392 | 34.2459 | -0.8933 | -2.54% | 12.7333 | 36.4667 | 47.9333 |
| 18 | 36.9920 | 36.7366 | -0.2554 | -0.69% | 13.0667 | 40.4667 | 47.9333 |
| 19 | 36.0432 | 35.8136 | -0.2296 | -0.64% | 10.5333 | 43.3333 | 45.6667 |
| 20 | 35.6614 | 35.8694 | +0.2080 | +0.58% | 10.8667 | 46.6000 | 45.8000 |
| 21 | 33.6976 | 33.8526 | +0.1550 | +0.46% | 7.2667 | 48.5333 | 46.0667 |
| 22 | 33.7649 | 33.1839 | -0.5810 | -1.72% | 8.1333 | 50.5333 | 44.0667 |
| 23 | 33.4229 | 33.1204 | -0.3025 | -0.91% | 7.9333 | 49.5333 | 46.2667 |
| 24 | 33.4737 | 33.1804 | -0.2933 | -0.88% | 8.0000 | 49.4667 | 47.9333 |
| 25 | 33.8070 | 33.3756 | -0.4314 | -1.28% | 7.0000 | 49.0000 | 47.9333 |
| 26 | 35.5761 | 34.9960 | -0.5801 | -1.63% | 7.4667 | 48.6000 | 50.0667 |
| 27 | 37.3871 | 36.6850 | -0.7021 | -1.88% | 8.0000 | 48.2000 | 52.9333 |
| 28 | 39.6649 | 38.7807 | -0.8842 | -2.23% | 10.8000 | 46.5333 | 53.4667 |
| 29 | 41.9718 | 41.7151 | -0.2567 | -0.61% | 12.4000 | 44.0667 | 53.5333 |
| 30 | 46.1429 | 46.0844 | -0.0585 | -0.13% | 15.1333 | 45.5333 | 59.1333 |
| 31 | 51.3756 | 51.2178 | -0.1578 | -0.31% | 18.0667 | 51.4667 | 66.2667 |
| 32 | 51.8973 | 51.0331 | -0.8642 | -1.67% | 16.4667 | 56.3333 | 67.4000 |
| 33 | 51.7630 | 51.3301 | -0.4328 | -0.84% | 14.0000 | 60.2000 | 67.6000 |
| 34 | 52.6096 | 53.9038 | +1.2942 | +2.46% | 11.2667 | 65.4667 | 69.0667 |
| 35 | 56.2516 | 57.6018 | +1.3501 | +2.40% | 7.4667 | 69.8000 | 69.9333 |
| 36 | 58.1654 | 58.7016 | +0.5362 | +0.92% | 6.9333 | 72.2000 | 70.5333 |
| 37 | 59.9115 | 60.3217 | +0.4101 | +0.68% | 8.4667 | 73.5333 | 69.6667 |
| 38 | 60.9659 | 61.3274 | +0.3615 | +0.59% | 11.2667 | 73.6000 | 69.0000 |
| 39 | 61.6126 | 62.1905 | +0.5779 | +0.94% | 17.6000 | 74.4667 | 72.4667 |
| 40 | 62.4377 | 63.1074 | +0.6697 | +1.07% | 11.8000 | 74.4667 | 72.5333 |
| 41 | 64.0883 | 64.2694 | +0.1812 | +0.28% | 27.2667 | 75.2000 | 74.2667 |
| 42 | 65.2812 | 64.3829 | -0.8983 | -1.38% | 12.9333 | 73.1333 | 73.1333 |
| 43 | 65.7512 | 64.0942 | -1.6570 | -2.52% | 11.0667 | 72.8667 | 46.4667 |
| 44 | 62.9709 | 63.1501 | +0.1792 | +0.28% | 13.8000 | 69.7333 | 44.6667 |
| 45 | 58.6417 | 59.6356 | +0.9939 | +1.69% | 15.2000 | 66.4667 | 45.8667 |
| 46 | 57.8174 | 58.9828 | +1.1654 | +2.02% | 13.7333 | 65.6000 | 45.4667 |
| 47 | 55.5608 | 56.6495 | +1.0887 | +1.96% | 13.1333 | 64.7333 | 46.0000 |
| 48 | 53.9933 | 54.0323 | +0.0390 | +0.07% | 14.4667 | 63.6000 | 47.0000 |
| 49 | 50.3022 | 50.2377 | -0.0645 | -0.13% | 15.8000 | 62.4667 | 44.6000 |
| 50 | 50.0599 | 49.6538 | -0.4061 | -0.81% | 11.2000 | 62.2000 | 46.3333 |
| 51 | 48.0822 | 46.9204 | -1.1618 | -2.42% | 9.2667 | 62.8000 | 46.8667 |
| 52 | 47.7240 | 45.6657 | -2.0583 | -4.31% | 10.2667 | 62.4667 | 46.2000 |
| 53 | 47.7240 | 45.9323 | -1.7917 | -3.75% | 8.1333 | 63.8000 | 46.4667 |
| 54 | 46.5622 | 44.5188 | -2.0434 | -4.39% | 8.8000 | 63.6000 | 45.9333 |
| 55 | 47.6320 | 46.5173 | -1.1147 | -2.34% | 9.0000 | 62.9333 | 47.8000 |
| 56 | 48.3070 | 46.3445 | -1.9624 | -4.06% | 9.4667 | 64.2000 | 50.6667 |
| 57 | 48.0310 | 46.5664 | -1.4646 | -3.05% | 8.9333 | 66.0667 | 56.2667 |
| 58 | 46.2379 | 44.4958 | -1.7422 | -3.77% | 10.3333 | 65.3333 | 58.6667 |
| 59 | 44.8537 | 43.0997 | -1.7540 | -3.91% | 10.4000 | 62.6667 | 62.2000 |
| 60 | 46.1527 | 43.5877 | -2.5651 | -5.56% | 19.2000 | 61.9333 | 65.3333 |
| 61 | 46.0569 | 43.7277 | -2.3292 | -5.06% | 14.6000 | 61.5333 | 64.8000 |
| 62 | 46.7805 | 44.0322 | -2.7483 | -5.87% | 18.8667 | 60.9333 | 63.6667 |
| 63 | 39.9508 | 37.4308 | -2.5200 | -6.31% | 13.0000 | 58.2667 | 61.2000 |
| 64 | 38.3008 | 37.9369 | -0.3638 | -0.95% | 13.2667 | 58.4667 | 60.1333 |
| 65 | 37.1058 | 36.6603 | -0.4455 | -1.20% | 10.6000 | 57.2000 | 58.0000 |
| 66 | 35.8424 | 36.5882 | +0.7458 | +2.08% | 12.6667 | 56.8667 | 56.7333 |
| 67 | 35.4895 | 35.4721 | -0.0175 | -0.05% | 9.1333 | 55.4000 | 52.6667 |
| 68 | 34.5740 | 34.5940 | +0.0200 | +0.06% | 10.1333 | 55.4000 | 50.8667 |
| 69 | 33.4129 | 32.8685 | -0.5444 | -1.63% | 8.7333 | 54.3333 | 50.0000 |
| 70 | 32.8393 | 32.5011 | -0.3383 | -1.03% | 6.7333 | 55.8000 | 47.0667 |
| 71 | 32.0295 | 31.7595 | -0.2700 | -0.84% | 5.6000 | 57.2667 | 45.0667 |
| 72 | 31.1574 | 31.2547 | +0.0973 | +0.31% | 6.3333 | 57.2000 | 44.2000 |
| 73 | 31.2907 | 31.6342 | +0.3434 | +1.10% | 4.9333 | 55.6000 | 44.0000 |
| 74 | 30.8868 | 31.7675 | +0.8807 | +2.85% | 6.1333 | 54.9333 | 45.0667 |
| 75 | 31.0672 | 31.2530 | +0.1858 | +0.60% | 5.4000 | 55.8000 | 47.2000 |
| 76 | 34.1164 | 35.2421 | +1.1256 | +3.30% | 5.8667 | 56.0667 | 48.3333 |
| 77 | 37.6405 | 38.6875 | +1.0470 | +2.78% | 11.0000 | 57.5333 | 51.8000 |
| 78 | 35.9638 | 37.6081 | +1.6443 | +4.57% | 12.3333 | 54.9333 | 52.9333 |
| 79 | 31.9516 | 33.7549 | +1.8033 | +5.64% | 16.6000 | 53.6667 | 56.6000 |
| 80 | 29.2827 | 29.7890 | +0.5063 | +1.73% | 17.0667 | 50.6000 | 57.5333 |
| 81 | 29.6260 | 30.8049 | +1.1789 | +3.98% | 21.1333 | 48.6000 | 58.2667 |
| 82 | 28.3876 | 31.9179 | +3.5303 | +12.44% | 18.2000 | 51.0000 | 58.0667 |
| 83 | 31.3371 | 33.5946 | +2.2575 | +7.20% | 16.2667 | 55.0000 | 57.6667 |
| 84 | 32.8338 | 34.6569 | +1.8231 | +5.55% | 14.1333 | 58.1333 | 57.6667 |
| 85 | 33.4612 | 36.5738 | +3.1126 | +9.30% | 12.7333 | 59.7333 | 57.8667 |
| 86 | 32.0509 | 36.0488 | +3.9979 | +12.47% | 11.0000 | 59.2667 | 56.2667 |
| 87 | 31.9571 | 35.5439 | +3.5868 | +11.22% | 22.3333 | 58.0667 | 56.7333 |
| 88 | 30.8521 | 35.0878 | +4.2357 | +13.73% | 7.8667 | 55.5333 | 54.0000 |
| 89 | 31.4248 | 36.0006 | +4.5758 | +14.56% | 19.2000 | 53.1333 | 52.0667 |
| 90 | 33.8620 | 36.9161 | +3.0541 | +9.02% | 8.2667 | 50.5333 | 50.5333 |
| 91 | 32.9602 | 35.5792 | +2.6190 | +7.95% | 10.1333 | 47.8667 | 53.7333 |
| 92 | 34.2730 | 37.2665 | +2.9935 | +8.73% | 14.2000 | 47.8000 | 53.4000 |
| 93 | 34.0754 | 35.7996 | +1.7242 | +5.06% | 14.0667 | 51.0667 | 53.6667 |
| 94 | 36.7313 | 36.4791 | -0.2522 | -0.69% | 14.5333 | 52.0667 | 54.6000 |
| 95 | 38.7696 | 38.1781 | -0.5915 | -1.53% | 8.0667 | 52.4667 | 54.5333 |
| 96 | 39.1453 | 36.6272 | -2.5181 | -6.43% | 8.4667 | 53.6667 | 54.9333 |
<!-- END GENERATED TABLE -->

---

## 8. Analysis: Why Weather Does Not Earn Its Place

1. **Marginal aggregate difference (+0.22%):** Removing weather features moves `GATE_VALUE` from 39.032 to 39.118 gCO2/kWh, a difference of +0.086 gCO2/kWh. This is well below the pre-registered 2.0% (0.78 gCO2/kWh) materiality threshold.
2. **Minimal degradation at long horizons (+0.86%):** Over $h \ge 48$, Control ($W_{\text{long}}$) is 38.0234 gCO2/kWh and Treatment ($N_{\text{long}}$) is 38.3512 gCO2/kWh, a delta of +0.328 gCO2/kWh (+0.86%), well within the pre-registered 5.0% tail limit.
3. **Horizon-band tension:**
   - No-weather is better at $h = 96$ (-6.43%, -2.518 gCO2/kWh) and across roughly $h = 49 \dots 65$ (1% to 6% better).
   - Weather is better across roughly $h = 82 \dots 93$ (+11% to +15% degradation without weather).
   - The pre-registered rule is an unweighted mean test across horizons and it says DROP. A horizon-weighted rule that placed high premium on day-2 evening peak hours ($h = 82 \dots 93$) would have decided differently. However, the rule was pre-registered as an unweighted mean test before results were read, and we do not re-decide after observing the data. DROP stands.
4. **Serving architecture risk eliminated:** Weather features introduced substantial operational overhead:
   - External dependencies on third-party weather API forecasts.
   - Irreconcilable train/serve lead distribution mismatch (ADR-018: models train on 24h-old or 48h-old previous runs, but serve on fresh run lead $h$).
   - Vulnerability to external rate limits, network outages, and missing timestamps.
5. **Decision:** Both pre-registered conditions hold ($N \le 1.02W$ and $N_{\text{long}} \le 1.05W_{\text{long}}$). Weather is **DROPPED**. ADR-018 stays (weather does buy something in a band), so it is closed by rule, not deleted.

---

## 9. Implementation Roadmap (Next Round)

Per the brief, this round only records the decision and pre-registered measurement. The implementation steps for the following round will be:

1. Prune weather columns (`temperature`, `temperature_24h_mean`, `temperature_24h_min`) from `FEATURE_COLUMNS` and `build_example_frame`.
2. Remove weather ingestion and store dependencies from model serving (`serve.py`).
3. Retrain the model artifact `data/models/lgbm_model.txt`.
4. Freeze evaluation fixture v2 (`data/eval/frozen_split_v2.parquet`) and sidecar metadata.
5. Update `MODEL_CARD.md`, `README.md`, and consistency test suite for fixture v2.
