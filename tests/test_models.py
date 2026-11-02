"""Tests for models, determinism, and baseline anchoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.features import (
    ACTUAL_AVAILABILITY_LAG,
    FEATURE_COLUMNS,
    build_example_frame,
    issue_time_for_date,
    load_feature_sources,
)
from gridcast.models import (
    DEFAULT_LGBM_PARAMS,
    assert_baseline_provenance,
    model_artifact_hash,
    predict_lgbm,
    predict_persistence,
    predict_seasonal_naive,
    train_lgbm,
)


def _synthetic_training_data(n_rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate deterministic synthetic feature matrix and target series."""
    rng = np.random.default_rng(seed=42)
    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        if col == "horizon":
            data[col] = rng.integers(1, 97, size=n_rows)
        elif col in ("hour_of_day", "day_of_week", "month"):
            data[col] = rng.integers(0, 24, size=n_rows)
        elif col in ("is_weekend", "is_uk_bank_holiday"):
            data[col] = rng.choice([0.0, 1.0], size=n_rows)
        else:
            data[col] = rng.normal(100.0, 20.0, size=n_rows)

    X = pd.DataFrame(data)
    # Generate realistic carbon intensity target
    y = pd.Series(
        0.5 * X["actual_recent_3h"] + 0.2 * X["mix_share_gas"] + rng.normal(0, 5, size=n_rows),
        name="y",
    )
    return X, y


def test_model_training_is_deterministic() -> None:
    """Requirement 3: Deterministic training produces identical model artifact hash twice."""
    X, y = _synthetic_training_data(n_rows=250)

    # Train twice with identical parameters
    model1 = train_lgbm(X, y, params=DEFAULT_LGBM_PARAMS, num_boost_round=20)
    hash1 = model_artifact_hash(model1)

    model2 = train_lgbm(X, y, params=DEFAULT_LGBM_PARAMS, num_boost_round=20)
    hash2 = model_artifact_hash(model2)

    assert hash1 == hash2, f"Model artifact hash diverged: {hash1} != {hash2}"
    assert len(hash1) == 64

    # Predictions must also be bitwise identical
    preds1 = predict_lgbm(model1, X)
    preds2 = predict_lgbm(model2, X)
    np.testing.assert_array_equal(preds1, preds2)


def test_persistence_baseline_provenance_and_flatness() -> None:
    """Requirement 2.1: Persistence baseline must be issue-anchored and flat across horizons."""
    issue_time = pd.Timestamp("2025-06-01 00:00:00+00:00")
    horizons = list(range(1, 97))
    target_times = [issue_time + pd.Timedelta(minutes=30 * h) for h in horizons]
    expected_source_ts = issue_time - ACTUAL_AVAILABILITY_LAG

    frame = pd.DataFrame({
        "issue_time": [issue_time] * 96,
        "horizon": horizons,
        "target_time": target_times,
        "actual_recent_3h": [140.0] * 96,
    })

    mock_store = pd.DataFrame({
        "period_start_utc": [expected_source_ts],
        "actual": [140.0],
    })

    result = predict_persistence(frame, raw_store=mock_store)

    # Assert flatness across all horizons
    assert (result["forecast"] == 140.0).all()

    # Guard — provenance assertion
    assert_baseline_provenance(result, raw_store=mock_store)
    assert (result["source_period_start_utc"] == expected_source_ts).all()


def test_persistence_provenance_value_check_against_raw_store() -> None:
    """Requirement 2.1 / Section 2: Provenance value check against raw curated store.

    For a sample of rows — including both ends of the frame and DST days — read the
    curated store directly with a plain lookup at issue_time - ACTUAL_AVAILABILITY_LAG
    and assert the prediction equals that value.
    Assert store row exists and is non-null at that instant.
    """
    curated_intensity = pd.read_parquet("data/curated")
    curated_intensity["period_start_utc"] = pd.to_datetime(
        curated_intensity["period_start_utc"], utc=True
    )

    intensity_df, weather_df, mix_df = load_feature_sources()

    # Sample dates: head of frame, tail of frame, and DST transition days
    sample_dates = [
        "2024-11-08",  # First issue date after warmup (head)
        "2025-03-30",  # Spring clocks forward (DST transition)
        "2025-10-26",  # Autumn clocks back (DST transition)
        "2026-10-25",  # Autumn clocks back 2026 (DST transition)
        "2026-11-01",  # Test split tail (tail)
    ]

    for sample_date in sample_dates:
        it = issue_time_for_date(sample_date)
        frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            issue_times=[it],
            require_aux=False,
        )
        assert len(frame) == 96

        # Plain lookup in curated store at issue_time - ACTUAL_AVAILABILITY_LAG
        src_ts = it - ACTUAL_AVAILABILITY_LAG
        store_match = curated_intensity[curated_intensity["period_start_utc"] == src_ts]

        # Assert store row exists and is non-null at that instant
        assert not store_match.empty, f"Store row does not exist at instant {src_ts}"
        raw_actual = store_match["actual"].iloc[0]
        assert pd.notna(raw_actual), f"Store row at {src_ts} has null actual"

        expected_val = float(raw_actual)

        preds = predict_persistence(frame)
        assert (preds["forecast"] == expected_val).all()
        assert_baseline_provenance(preds)

    # Negative test 1: Missing store row is a failure, not a null
    bogus_issue = pd.Timestamp("2099-01-01 00:00:00+00:00")
    bogus_preds = pd.DataFrame({
        "issue_time": [bogus_issue] * 96,
        "horizon": list(range(1, 97)),
        "target_time": [bogus_issue + pd.Timedelta(minutes=30 * h) for h in range(1, 97)],
        "forecast": [100.0] * 96,
        "source_period_start_utc": [bogus_issue - ACTUAL_AVAILABILITY_LAG] * 96,
    })
    with pytest.raises(
        AssertionError, match=r"guard\s*[-—]\s*provenance: store row does not exist"
    ):
        assert_baseline_provenance(bogus_preds)

    # Negative test 2: Store row with null actual is a failure
    null_store = pd.DataFrame({
        "period_start_utc": [pd.Timestamp("2025-06-01 00:00:00+00:00") - ACTUAL_AVAILABILITY_LAG],
        "actual": [np.nan],
    })
    preds_for_null = pd.DataFrame({
        "issue_time": [pd.Timestamp("2025-06-01 00:00:00+00:00")] * 96,
        "horizon": list(range(1, 97)),
        "target_time": [
            pd.Timestamp("2025-06-01 00:00:00+00:00") + pd.Timedelta(minutes=30 * h)
            for h in range(1, 97)
        ],
        "forecast": [100.0] * 96,
        "source_period_start_utc": [
            pd.Timestamp("2025-06-01 00:00:00+00:00") - ACTUAL_AVAILABILITY_LAG
        ]
        * 96,
    })
    with pytest.raises(
        AssertionError, match=r"guard\s*[-—]\s*provenance: store row.*has null actual"
    ):
        assert_baseline_provenance(preds_for_null, raw_store=null_store)


def test_provenance_guard_bites_on_corrupted_prediction_value() -> None:
    """Bite demonstration 2a (Section 5): Corrupting a prediction value fails provenance."""
    issue_time = pd.Timestamp("2025-06-01 00:00:00+00:00")
    expected_src = issue_time - ACTUAL_AVAILABILITY_LAG
    mock_store = pd.DataFrame({
        "period_start_utc": [expected_src],
        "actual": [140.0],
    })

    frame = pd.DataFrame({
        "issue_time": [issue_time] * 96,
        "horizon": list(range(1, 97)),
        "target_time": [issue_time + pd.Timedelta(minutes=30 * h) for h in range(1, 97)],
        "actual_recent_3h": [140.0] * 96,
    })

    valid_preds = predict_persistence(frame, raw_store=mock_store)
    assert_baseline_provenance(valid_preds, raw_store=mock_store)

    # Corrupt a prediction value
    corrupted = valid_preds.copy()
    corrupted.loc[5, "forecast"] = 999.0

    # Section 5 precondition: assert mutation took effect
    assert corrupted.loc[5, "forecast"] != 140.0

    # Must fail provenance value check against store
    with pytest.raises(
        AssertionError, match=r"guard\s*[-—]\s*provenance:\s*prediction value mismatch"
    ):
        assert_baseline_provenance(corrupted, raw_store=mock_store)


def test_provenance_guard_bites_on_target_anchored_timestamp() -> None:
    """Bite demonstration 2b (Section 5): Target-anchored timestamp fails provenance."""
    issue_time = pd.Timestamp("2025-06-01 00:00:00+00:00")
    expected_src = issue_time - ACTUAL_AVAILABILITY_LAG
    mock_store = pd.DataFrame({
        "period_start_utc": [expected_src],
        "actual": [140.0],
    })

    frame = pd.DataFrame({
        "issue_time": [issue_time] * 96,
        "horizon": list(range(1, 97)),
        "target_time": [issue_time + pd.Timedelta(minutes=30 * h) for h in range(1, 97)],
        "actual_recent_3h": [140.0] * 96,
    })

    valid_preds = predict_persistence(frame, raw_store=mock_store)

    # Stamp target-anchored timestamp (target_time - 30min)
    target_anchored = valid_preds.copy()
    target_anchored["source_period_start_utc"] = [
        tt - pd.Timedelta(minutes=30) for tt in target_anchored["target_time"]
    ]

    # Section 5 precondition: assert mutation took effect
    assert (target_anchored["source_period_start_utc"] != expected_src).any()

    # Must fail provenance timestamp check
    with pytest.raises(AssertionError, match=r"guard\s*[-—]\s*provenance violation"):
        assert_baseline_provenance(target_anchored, raw_store=mock_store)


def test_seasonal_naive_provenance_and_diurnal_matching() -> None:
    """Requirement 3: Seasonal-naive must strictly read history available at issue time."""
    issue_time = pd.Timestamp("2025-06-03 00:00:00+00:00")
    horizons = list(range(1, 97))
    target_times = [issue_time + pd.Timedelta(minutes=30 * h) for h in horizons]

    frame = pd.DataFrame({
        "issue_time": [issue_time] * 96,
        "horizon": horizons,
        "target_time": target_times,
    })

    # Build history of actuals covering the past week in half-hourly periods
    hist_times = [
        issue_time - pd.Timedelta(minutes=30 * p) for p in range(1, 336)
    ]
    history = {ts: float(100 + (ts.hour * 5)) for ts in hist_times}

    result = predict_seasonal_naive(frame, history_actuals=history)

    assert len(result) == 96
    cutoff = issue_time - ACTUAL_AVAILABILITY_LAG

    # All source timestamps must be <= cutoff
    assert (result["source_period_start_utc"] <= cutoff).all()

    # For h=1 (target = 00:30), target - 24h = issue - 23.5h <= cutoff, so source is target - 24h
    row_h1 = result[result["horizon"] == 1].iloc[0]
    expected_h1_source = pd.Timestamp("2025-06-02 00:30:00+00:00")
    assert row_h1["source_period_start_utc"] == expected_h1_source
    assert row_h1["forecast"] == history[expected_h1_source]
