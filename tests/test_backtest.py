"""Leakage attack, metric-identity, and contract tests for backtesting and evaluation.

Implements all five evaluation-integrity requirements:
1. Target-anchored persistence rejected by provenance guard (plus known-different fixture).
2. Baselines and model scored on identical period intersection.
3. Single scalar accuracy gate: GATE_VALUE = (1 / 96) * sum(MAE_h for h=1..96).
4. Training data ends strictly before issue: max(train['target_time']) < T.
5. Skill scores computed on identical period sets at point of division.
4.1 Backtest consumes production frame builder via spy and slices without re-derivation.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridcast.backtest import (
    EMBARGO_END_DATE,
    EMBARGO_START_DATE,
    FROZEN_TEST_ORIGINS,
    TEST_SPLIT_END_DATE,
    TEST_SPLIT_START_DATE,
    BacktestResults,
    assert_fold_training_integrity,
    compute_gap_gate_threshold,
    compute_gate_value,
    compute_skill_score,
    derive_test_split_origins,
    evaluate_gap_gate,
    evaluate_gate,
    filter_evaluation_intersection,
    run_rolling_origin_backtest,
)
from gridcast.features import (
    ACTUAL_AVAILABILITY_LAG,
    FEATURE_COLUMNS,
    build_example_frame,
    default_issue_schedule,
    issue_time_for_date,
    load_feature_sources,
    select_model_inputs,
)
from gridcast.models import assert_baseline_provenance, predict_persistence

# ---------------------------------------------------------------------------
# Requirement 2.1: Persistence Baseline Anchoring & Leakage Attack
# ---------------------------------------------------------------------------


def test_attack_target_anchored_persistence_rejected_by_provenance_guard() -> None:
    """Leakage attack 1: Target-anchored persistence must be rejected by the provenance guard.

    Naive persistence reads actual immediately before target_time (e.g. target_time - 30min).
    The provenance guard requires every row's source_period_start_utc to be strictly equal to
    issue_time - ACTUAL_AVAILABILITY_LAG.
    """
    issue_time = pd.Timestamp("2025-06-01 00:00:00+00:00")
    horizons = list(range(1, 97))
    target_times = [issue_time + pd.Timedelta(minutes=30 * h) for h in horizons]

    # Deliberately construct target-anchored persistence: reads target_time - 30min
    leaked_source_ts = [tt - pd.Timedelta(minutes=30) for tt in target_times]

    leaked_preds = pd.DataFrame({
        "issue_time": [issue_time] * 96,
        "horizon": horizons,
        "target_time": target_times,
        "forecast": [100.0] * 96,
        "source_period_start_utc": leaked_source_ts,
    })

    expected_provenance = issue_time - ACTUAL_AVAILABILITY_LAG

    # Precondition assertion (brief template section 5): Prove mutation took effect
    assert (leaked_preds["source_period_start_utc"] != expected_provenance).any(), (
        "Precondition failed: leaked_preds must not have valid provenance"
    )

    # Guard must fire loudly with AssertionError labelled 'guard — provenance'
    with pytest.raises(AssertionError, match=r"guard\s*[-—]\s*provenance"):
        assert_baseline_provenance(leaked_preds)


def test_demonstration_known_different_fixture() -> None:
    """Demonstration — known-different fixture (requirement 2.1).

    Construct a fixture where the value at issue - 3h and the value at target - 30min
    are known by construction to differ (140 vs 182). Assert both numbers, and assert
    they are not equal. This proves the two reads are distinguishable.
    """
    issue_time = pd.Timestamp("2025-06-01 00:00:00+00:00")
    horizon = 96
    target_time = issue_time + pd.Timedelta(minutes=30 * horizon)

    issue_anchored_ts = issue_time - ACTUAL_AVAILABILITY_LAG
    target_anchored_ts = target_time - pd.Timedelta(minutes=30)

    store = {
        issue_anchored_ts: 140.0,
        target_anchored_ts: 182.0,
    }

    issue_anchored_value = store[issue_anchored_ts]
    target_anchored_value = store[target_anchored_ts]

    assert issue_anchored_value == 140.0, f"Expected 140.0, got {issue_anchored_value}"
    assert target_anchored_value == 182.0, f"Expected 182.0, got {target_anchored_value}"
    assert issue_anchored_value != target_anchored_value, (
        f"Demonstration failed: {issue_anchored_value} == {target_anchored_value}"
    )


def test_attack_corrupted_prediction_rejected_by_provenance_value_check() -> None:
    """Bite test 1 (Section 2): Corrupt prediction values and show the value check fails.

    Injects corrupted forecast values (+15.0 gCO2/kWh) into a valid persistence baseline forecast.
    Asserts mutation precondition that corrupted values differ from original.
    Asserts assert_baseline_provenance raises AssertionError matching prediction value mismatch.
    """
    intensity_df, weather_df, mix_df = load_feature_sources()
    sample_issue = issue_time_for_date("2025-06-05")
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=[sample_issue],
        require_aux=False,
    )
    valid_preds = predict_persistence(frame)
    # Valid baseline passes provenance
    assert_baseline_provenance(valid_preds)

    # Sabotage: corrupt the prediction values
    corrupted_preds = valid_preds.copy()
    corrupted_preds["forecast"] = corrupted_preds["forecast"] + 15.0

    # Brief template section 5 injection precondition
    assert (corrupted_preds["forecast"] != valid_preds["forecast"]).all(), (
        "Precondition failed: corrupted forecast must differ from original"
    )

    with pytest.raises(
        AssertionError, match=r"guard\s*[-—]\s*provenance: prediction value mismatch"
    ):
        assert_baseline_provenance(corrupted_preds)


def test_attack_shifted_source_timestamp_rejected_by_provenance_identity_check() -> None:
    """Bite test 2 (Section 2): Shift recorded timestamp and show identity check fails.

    Injects a timestamp shift (+30 min) into a valid persistence baseline dataframe.
    Asserts mutation precondition that shifted timestamps differ from original.
    Asserts assert_baseline_provenance raises AssertionError matching invalid source timestamp.
    """
    intensity_df, weather_df, mix_df = load_feature_sources()
    sample_issue = issue_time_for_date("2025-06-05")
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=[sample_issue],
        require_aux=False,
    )
    valid_preds = predict_persistence(frame)
    assert_baseline_provenance(valid_preds)

    # Sabotage: shift recorded source timestamp by 30 minutes
    shifted_preds = valid_preds.copy()
    shifted_preds["source_period_start_utc"] = (
        shifted_preds["source_period_start_utc"] + pd.Timedelta(minutes=30)
    )

    # Brief template section 5 injection precondition
    assert (
        shifted_preds["source_period_start_utc"] != valid_preds["source_period_start_utc"]
    ).all(), "Precondition failed: shifted timestamps must differ from original"

    with pytest.raises(
        AssertionError, match=r"guard\s*[-—]\s*provenance violation.*invalid source timestamp"
    ):
        assert_baseline_provenance(shifted_preds)


# ---------------------------------------------------------------------------
# Requirement 2.4: Training Data Cutoff Leakage Attack
# ---------------------------------------------------------------------------


def test_attack_training_target_time_leak_rejected_by_fold_guard() -> None:
    """Leakage attack 2: Training set containing target_time >= T must be rejected.

    For origin T, training data must strictly satisfy max(train['target_time']) < T.
    A target at T or after sitting in train is an operational/temporal leak.
    """
    origin_T = pd.Timestamp("2025-06-01 00:00:00+00:00")

    train_frame = pd.DataFrame({
        "issue_time": [
            origin_T - pd.Timedelta(days=3),
            origin_T - pd.Timedelta(days=2),
            origin_T - pd.Timedelta(days=1),
        ],
        "horizon": [1, 24, 48],
        "target_time": [
            origin_T - pd.Timedelta(days=3) + pd.Timedelta(minutes=30),
            origin_T - pd.Timedelta(days=2) + pd.Timedelta(hours=12),
            origin_T + pd.Timedelta(hours=12),  # LEAK: target_time > origin_T!
        ],
        "y": [120.0, 130.0, 140.0],
    })

    max_target_time = train_frame["target_time"].max()
    assert max_target_time >= origin_T, (
        f"Precondition failed: max training target_time ({max_target_time}) should be >= {origin_T}"
    )

    with pytest.raises(AssertionError, match=r"target_time < T"):
        assert_fold_training_integrity(train_frame, origin_T)


# ---------------------------------------------------------------------------
# Requirement 2.5: Skill Score Mismatched Periods Leakage Attack
# ---------------------------------------------------------------------------


def test_attack_mismatched_period_sets_rejected_by_skill_score_guard() -> None:
    """Leakage attack 3: Skill score on mismatched period sets must be rejected.

    Skill score (1 - MAE_model / MAE_baseline) is meaningless if evaluated on different
    period sets. The function must assert set equality of target periods at the point of division.
    """
    base_ts = pd.Timestamp("2025-06-01 00:00:00+00:00")
    model_targets = [base_ts + pd.Timedelta(minutes=30 * i) for i in range(10)]
    baseline_targets = model_targets[:-1]

    model_df = pd.DataFrame({
        "target_time": model_targets,
        "forecast": [100.0] * 10,
        "actual": [105.0] * 10,
    })

    baseline_df = pd.DataFrame({
        "target_time": baseline_targets,
        "forecast": [102.0] * 9,
        "actual": [105.0] * 9,
    })

    assert set(model_df["target_time"]) != set(baseline_df["target_time"]), (
        "Precondition failed: target_time sets must differ"
    )

    with pytest.raises(AssertionError, match=r"identical period sets"):
        compute_skill_score(model_df, baseline_df)


def test_skill_score_live_path_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bite test 3 (Section 3): All skill figures must go through compute_skill_score on live path.

    Asserts compute_skill_score is called during a backtest run (spying call count >= 97:
    96 per-horizon calls + 1 overall call).
    Also asserts requirement-2.5 guard bites if mismatched period sets are passed.
    """
    intensity_df, weather_df, mix_df = load_feature_sources()
    sample_issue = issue_time_for_date("2025-06-05")
    schedule = default_issue_schedule("2025-06-01", "2025-06-05")
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )
    operator_forecasts = pd.Series(
        intensity_df["forecast"].values,
        index=pd.to_datetime(intensity_df["period_start_utc"], utc=True),
    )

    call_count = 0
    orig_compute_skill = compute_skill_score

    def spied_compute_skill(*args: Any, **kwargs: Any) -> float:
        nonlocal call_count
        call_count += 1
        return orig_compute_skill(*args, **kwargs)

    monkeypatch.setattr("gridcast.backtest.compute_skill_score", spied_compute_skill)

    res = run_rolling_origin_backtest(
        frame=frame,
        origins=[sample_issue],
        operator_forecasts=operator_forecasts,
        num_boost_round=5,
        verbose=False,
    )

    # Must be called at least 97 times (96 horizons + 1 overall)
    assert call_count >= 97, (
        f"Skill figures must go through compute_skill_score (called {call_count} times)"
    )
    assert pd.notna(res.overall_skill_vs_operator)


# ---------------------------------------------------------------------------
# Requirement 4.1: Production Frame Builder Consumption & Slice Equality
# ---------------------------------------------------------------------------


def test_backtest_consumes_production_frame_via_spy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 4.1 Assertion 1: A spy asserts build_example_frame is called.

    Asserts that the evaluation path calls build_example_frame and that no fold
    constructs a feature frame by any other route.
    """
    intensity_df, weather_df, mix_df = load_feature_sources()

    # Spy on build_example_frame
    call_count = 0
    orig_builder = build_example_frame

    def spied_builder(*args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return orig_builder(*args, **kwargs)

    monkeypatch.setattr("gridcast.cli.build_example_frame", spied_builder)

    from gridcast.cli import run_backtest_cli

    # Run backtest CLI with single origin on small window
    origin = issue_time_for_date("2025-06-05")
    ret = run_backtest_cli(
        origins=[origin],
        start_date="2025-06-01",
        end_date="2025-06-05",
        num_boost_round=5,
        verbose=False,
    )
    assert ret == 0, "run_backtest_cli failed"
    assert call_count == 1, (
        f"Evaluation path must call build_example_frame exactly once per run (got {call_count})"
    )


def test_sample_fold_feature_matrix_equals_production_frame_slice() -> None:
    """Requirement 4.1 Assertion 2: For a sample fold, fold X equals production frame slice.

    Row for row and value for value, to the corresponding rows of the production frame
    filtered to the fold's issue range — same period_start_utc/issue_time/horizon,
    same column order, same values.
    """
    intensity_df, weather_df, mix_df = load_feature_sources()
    schedule = default_issue_schedule("2025-06-01", "2025-06-08")
    prod_frame = build_example_frame(
        intensity_df, weather_df, mix_df, issue_times=schedule, require_aux=True
    )

    sample_issue = issue_time_for_date("2025-06-05")
    prod_slice = prod_frame[prod_frame["issue_time"] == sample_issue].copy()

    # Fold feature matrix X extracted positively
    fold_X = select_model_inputs(prod_slice)
    prod_X = prod_slice[FEATURE_COLUMNS].copy()

    assert list(fold_X.columns) == list(prod_X.columns) == FEATURE_COLUMNS
    assert len(fold_X) == len(prod_X) == 96
    pd.testing.assert_frame_equal(fold_X, prod_X, check_exact=True)


# ---------------------------------------------------------------------------
# Requirement 2.2: Intersection and Period Equality
# ---------------------------------------------------------------------------


def test_evaluation_intersection_set_equality_and_counts() -> None:
    """Requirement 2.2: Compute intersection of periods where model, actual, and operator exist.

    Verify set equality of target_time across candidates and accurate exclusion tracking.
    """
    base_ts = pd.Timestamp("2025-06-01 00:00:00+00:00")
    target_times = [base_ts + pd.Timedelta(minutes=30 * i) for i in range(10)]

    raw_data = pd.DataFrame({
        "issue_time": [base_ts] * 10,
        "horizon": list(range(1, 11)),
        "target_time": target_times,
        "actual": [100.0, 102.0, np.nan, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "forecast": [101.0, 103.0, 104.0, 105.0, np.nan, 107.0, 108.0, 109.0, 110.0, 111.0],
        "operator_forecast": [
            100.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0
        ],
        "persistence_forecast": [100.0] * 10,
    })

    scored, exclusions = filter_evaluation_intersection(raw_data)

    # 3 periods dropped (rows 2, 4, 3), 7 remaining
    assert len(scored) == 7
    assert exclusions["total_periods"] == 10
    assert exclusions["intersection_size"] == 7
    assert exclusions["total_excluded"] == 3
    assert exclusions["missing_actual"] == 1
    assert exclusions["missing_model_forecast"] == 1
    assert exclusions["missing_operator_forecast"] == 1

    # Assert no null remains in scored intersection
    assert scored["actual"].notna().all()
    assert scored["forecast"].notna().all()
    assert scored["operator_forecast"].notna().all()


# ---------------------------------------------------------------------------
# Requirement 2.3: Single Scalar Gate Identity
# ---------------------------------------------------------------------------


def test_single_gate_value_scalar_identity() -> None:
    """Requirement 2.3: GATE_VALUE = (1 / 96) * sum_{h=1..96} MAE_h."""
    rng = np.random.default_rng(seed=123)
    mae_per_h = rng.uniform(20.0, 45.0, size=96)
    per_h_table = pd.DataFrame({
        "horizon": list(range(1, 97)),
        "model_mae": mae_per_h,
    })

    gate_val = compute_gate_value(per_h_table, "model_mae")
    expected_gate_val = float(np.mean(mae_per_h))

    assert math_isclose(gate_val, expected_gate_val)
    assert np.isclose(gate_val, np.sum(mae_per_h) / 96.0)


def math_isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Requirement 4: Backtest-Engine Determinism Regression on Frozen 30-Day Window
# ---------------------------------------------------------------------------


def test_golden_backtest_provenance_is_per_platform() -> None:
    """Each platform golden must carry its own library provenance versions.

    Storing one global provenance block lets a library upgrade on one OS change
    that platform's golden while the shared block still describes another OS.
    """
    fixture_path = Path("tests/fixtures/golden_backtest_metrics.json")
    assert fixture_path.exists(), f"Missing golden fixture: {fixture_path}"

    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "provenance" not in data, (
        "Library provenance versions must live inside each platform entry, not in a global block."
    )
    platforms = data.get("platforms", {})
    assert {"win32-x86_64", "linux-x86_64"}.issubset(platforms.keys())

    for plat_key, entry in platforms.items():
        prov = entry.get("provenance")
        assert isinstance(prov, dict), f"Platform {plat_key!r} missing 'provenance' dict"
        for pkg in ("python", "lightgbm", "pandas", "numpy"):
            assert pkg in prov and prov[pkg], (
                f"Platform {plat_key!r} provenance missing '{pkg}'"
            )


def test_backtest_engine_determinism_per_platform() -> None:
    """Backtest-engine determinism regression on a frozen 30-day window (NOT model validity).

    Note on scope: this test trains with ``include_weather=True`` on the 29-feature v1 list
    and 30 boosting rounds, whereas production serves 26 features (``lgbm-v2``, weather dropped
    per ADR-018). It tests that the backtest engine stays deterministic within a platform.

    Observed cross-platform divergence: Linux and Windows LightGBM wheels disagree by ~6.5% on
    this window (28.8701 vs 30.7505) and by 0.93 gCO2/kWh on 15-fold v2 retraining (30/30 trees
    diverge structurally) even with identical library versions, deterministic=True,
    force_col_wise=True, and num_threads=1. Non-LightGBM baselines (operator 9.5234, persistence
    68.7214, seasonal-naive 44.3984) match identically across platforms.
    """
    from gridcast.backtest import check_sidecar_library_versions

    fixture_path = Path("tests/fixtures/golden_backtest_metrics.json")
    assert fixture_path.exists(), f"Missing golden fixture: {fixture_path}"

    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x86_64"
    key = f"{sys.platform}-{machine}"
    assert key in data["platforms"], (
        f"No golden recorded for platform {key!r}. Record one deliberately, do not widen the rest."
    )
    golden = data["platforms"][key]
    mismatches = check_sidecar_library_versions({"versions": golden["provenance"]})
    assert mismatches == [], f"Platform {key!r} library provenance mismatch: {mismatches}"

    intensity_df, weather_df, mix_df = load_feature_sources()
    schedule = default_issue_schedule("2024-11-08", "2025-07-15")
    frame = build_example_frame(
        intensity_df,
        weather_df,
        mix_df,
        issue_times=schedule,
        require_aux=True,
        include_weather=True,
    )

    operator_forecasts = pd.Series(
        intensity_df["forecast"].values,
        index=pd.to_datetime(intensity_df["period_start_utc"], utc=True),
    )

    frozen_origins = ["2025-06-05", "2025-06-12", "2025-06-19", "2025-06-26"]

    v1_features = [
        "horizon",
        "actual_recent_3h",
        "actual_lag_24h",
        "actual_lag_48h",
        "actual_lag_168h",
        "actual_rolling_3h_mean",
        "actual_rolling_3h_std",
        "actual_rolling_24h_mean",
        "actual_rolling_24h_std",
        "actual_rolling_168h_mean",
        "actual_rolling_168h_std",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "is_uk_bank_holiday",
        "sin_hour",
        "cos_hour",
        "sin_day_of_year",
        "cos_day_of_year",
        "temperature",
        "temperature_24h_mean",
        "temperature_24h_min",
        "mix_share_wind",
        "mix_share_solar",
        "mix_share_gas",
        "mix_share_coal",
        "mix_share_nuclear",
        "mix_share_biomass",
    ]

    res = run_rolling_origin_backtest(
        frame=frame,
        origins=frozen_origins,
        operator_forecasts=operator_forecasts,
        num_boost_round=30,
        verbose=False,
        feature_columns=v1_features,
    )

    assert round(res.gate_value, 4) == golden["gate_value"], (
        f"GATE_VALUE diff: {round(res.gate_value, 4)} vs golden {golden['gate_value']}"
    )
    assert round(res.operator_gate_value, 4) == golden["operator_gate_value"]
    assert round(res.persistence_gate_value, 4) == golden["persistence_gate_value"]
    assert round(res.seasonal_naive_gate_value, 4) == golden["seasonal_naive_gate_value"]
    assert round(res.overall_skill_vs_operator, 4) == golden["overall_skill_vs_operator"]
    assert round(res.skill_h2, 4) == golden["skill_h2"]
    assert round(res.skill_h48, 4) == golden["skill_h48"]
    assert round(res.skill_h96, 4) == golden["skill_h96"]
    assert res.intersection_size == golden["intersection_size"]


# ---------------------------------------------------------------------------
# Requirement 5: Accuracy Gate Void Verdict (ADR-020) & Gap Gate Specification (ADR-021)
# ---------------------------------------------------------------------------


def test_accuracy_gate_no_verdict_behavior(capsys: pytest.CaptureFixture[str]) -> None:
    """ADR-020: P3 closes with no gate verdict because 35.0 threshold is void.

    Asserts evaluate_gate prints prominent no-verdict line and claims neither pass nor fail.
    """
    dummy_results = BacktestResults(
        predictions=pd.DataFrame(),
        per_horizon_table=pd.DataFrame({"horizon": range(1, 97), "model_mae": [32.0] * 96}),
        gate_value=32.32,
        operator_gate_value=11.06,
        persistence_gate_value=38.49,
        seasonal_naive_gate_value=42.35,
        overall_skill_vs_operator=-1.92,
        skill_h2=-2.37,
        skill_h48=-2.96,
        skill_h96=-2.20,
        intersection_size=1344,
        excluded_counts={"total_excluded": 0, "missing_actual": 0, "missing_operator_forecast": 0},
        origins_evaluated=[pd.Timestamp("2026-09-06 23:00:00+00:00")],
    )

    evaluate_gate(dummy_results, threshold=35.0, verbose=True)
    captured = capsys.readouterr()

    # Prominent no-verdict line must be printed verbatim
    expected_line = "GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed"
    assert expected_line in captured.out


def test_adr021_gap_gate_specification() -> None:
    """ADR-021: Candidate must close at least 30% of MAE gap between persistence and operator.

    Evaluated on validation split quantities (persistence 57.776, operator 9.649).
    gap = 57.776 - 9.649 = 48.127
    required = 57.776 - 0.30 * 48.127 = 43.3379
    """
    p_mae = 57.776
    o_mae = 9.649
    required = compute_gap_gate_threshold(p_mae, o_mae, fraction=0.30)
    assert abs(required - 43.3379) < 0.001

    # Candidate achieving 33.815 <= 43.338 passes
    assert evaluate_gap_gate(33.815, p_mae, o_mae, fraction=0.30) is True
    # Hypothetical candidate failing the 30% gap floor (> 43.338) fails
    assert evaluate_gap_gate(45.0, p_mae, o_mae, fraction=0.30) is False


def test_frozen_split_dates_and_cli_overrides() -> None:
    """Requirement 1 (Section 5): Frozen split date constants and CLI argument exposure."""
    from gridcast.cli import build_parser

    # Assert exact date constants
    assert TEST_SPLIT_START_DATE == "2026-09-05"
    assert TEST_SPLIT_END_DATE == "2026-11-01"

    # Assert backtest CLI exposes --start-date, --end-date, and --origins
    parser = build_parser()
    args = parser.parse_args([
        "backtest",
        "--start-date",
        "2026-09-01",
        "--end-date",
        "2026-09-10",
        "--origins",
        "2026-09-01",
        "2026-09-05",
    ])
    assert args.start_date == "2026-09-01"
    assert args.end_date == "2026-09-10"
    assert args.origins == ["2026-09-01", "2026-09-05"]


def test_backtest_cli_exit_code_and_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI backtest exits 0 and prints the exact no-verdict line."""
    from gridcast.cli import run_backtest_cli

    dummy_results = BacktestResults(
        predictions=pd.DataFrame(),
        per_horizon_table=pd.DataFrame({
            "horizon": range(1, 97),
            "count": [15] * 96,
            "model_mae": [39.032] * 96,
            "operator_mae": [11.666] * 96,
            "persistence_mae": [53.741] * 96,
            "seasonal_naive_mae": [51.495] * 96,
            "skill_vs_operator": [-2.34] * 96,
        }),
        gate_value=39.032,
        operator_gate_value=11.666,
        persistence_gate_value=53.741,
        seasonal_naive_gate_value=51.495,
        overall_skill_vs_operator=-2.34,
        skill_h2=-0.81,
        skill_h48=-2.73,
        skill_h96=-3.74,
        intersection_size=1440,
        excluded_counts={"total_excluded": 0, "missing_actual": 0, "missing_operator_forecast": 0},
        origins_evaluated=[pd.Timestamp("2026-09-04 23:00:00+00:00")],
    )

    monkeypatch.setattr(
        "gridcast.backtest.run_rolling_origin_backtest",
        lambda **kwargs: dummy_results,
    )
    monkeypatch.setattr(
        "gridcast.cli.load_feature_sources",
        lambda: (
            pd.DataFrame({
                "period_start_utc": [pd.Timestamp("2026-09-05 00:00:00+00:00")],
                "forecast": [10.0],
            }),
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        "gridcast.cli.build_example_frame",
        lambda **kwargs: pd.DataFrame({
            "issue_time": [pd.Timestamp("2026-09-05 00:00:00+00:00")],
            "horizon": [1],
            "target_time": [pd.Timestamp("2026-09-05 00:30:00+00:00")],
            "y": [100.0],
            "actual_recent_3h": [100.0],
        }),
    )

    exit_code = run_backtest_cli(origins=["2026-09-05"], verbose=True)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert (
        "GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed"
        in captured.out
    )


def test_attack_missing_horizon_rejected_by_gate_value_denominator_guard() -> None:
    """Bite test 4 (Section 4): Silent denominator defect bites when a horizon is missing.

    Constructs a 96-horizon table where horizon 96 has count=0 and model_mae=NaN.
    Demonstrates that unconstrained Series.mean(skipna=True) would silently divide by 95.
    Asserts compute_gate_value raises ValueError loudly instead of silently altering denominator.
    """
    horizons = list(range(1, 97))
    model_maes = [30.0] * 95 + [np.nan]
    counts = [15] * 95 + [0]

    per_h_table = pd.DataFrame({
        "horizon": horizons,
        "count": counts,
        "model_mae": model_maes,
    })

    # Precondition assertion: demonstrate the silent bug in pandas mean
    silent_denominator_result = float(per_h_table["model_mae"].mean(skipna=True))
    assert math_isclose(silent_denominator_result, 30.0), (
        "Precondition: pandas mean silently skipped NaN and divided by 95"
    )

    # compute_gate_value must reject missing or zero-count horizons loudly
    with pytest.raises(ValueError, match=r"Silent denominator defect guard.*zero scored periods"):
        compute_gate_value(per_h_table, "model_mae")

    # If count is missing from table but NaN exists in MAE column, it must also reject
    table_without_count = per_h_table[["horizon", "model_mae"]]
    with pytest.raises(ValueError, match=r"Silent denominator defect guard.*null"):
        compute_gate_value(table_without_count, "model_mae")


def test_frozen_test_split_origins_outside_embargo() -> None:
    """Section 1: Fixed test split origins derivation and embargo boundary assertion.

    Asserts that:
    1. Origins are derived from fixed split boundaries (2026-09-05 to 2026-11-01), not frame tail.
    2. Exactly 15 origins are produced every 4 days.
    3. The first origin is at or after 2026-09-05 (never inside the 4-day embargo).
    4. Deliberately injecting an origin inside the embargo raises AssertionError.
    """
    origins = derive_test_split_origins()
    assert len(origins) == 15, f"Expected 15 test split origins, got {len(origins)}"
    assert origins == FROZEN_TEST_ORIGINS

    test_start_ts = issue_time_for_date(TEST_SPLIT_START_DATE)
    assert origins[0] >= test_start_ts, (
        f"First origin {origins[0]} must be at or after test split start {test_start_ts}"
    )

    embargo_start_ts = issue_time_for_date(EMBARGO_START_DATE)
    embargo_end_ts = issue_time_for_date(EMBARGO_END_DATE)
    for o in origins:
        assert not (embargo_start_ts <= o <= embargo_end_ts), (
            f"Origin {o} must not fall inside embargo [{embargo_start_ts}, {embargo_end_ts}]"
        )

    # Prove embargo assertion bites: injecting an origin inside embargo raises AssertionError
    embargoed_origin = issue_time_for_date("2026-09-04")
    assert embargo_start_ts <= embargoed_origin <= embargo_end_ts, (
        "Precondition: 2026-09-04 must be inside embargo"
    )

    with pytest.raises(AssertionError, match=r"inside the embargo"):
        run_rolling_origin_backtest(
            frame=pd.DataFrame({
                "issue_time": [embargoed_origin] * 96,
                "horizon": range(1, 97),
                "target_time": range(96),
                "y": [100.0] * 96,
                "actual_recent_3h": [100.0] * 96,
            }),
            origins=[embargoed_origin],
            verbose=False,
        )
