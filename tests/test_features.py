"""Temporal leakage tests and feature engineering invariants.

Validates that no feature at issue_time uses any value stamped after issue_time.
Tests future-perturbation invariance across 25+ sampled (issue_time, horizon) pairs
including boundaries (first/last issue times, DST transitions, horizon extremes h=1/h=96,
and dates adjacent to recorded source gaps). Enforces the horizon contract in code.
"""

from __future__ import annotations

import math
from datetime import date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from helpers import assert_injection_changed

from gridcast.features import (
    ACTUAL_AVAILABILITY_LAG,
    FEATURE_COLUMNS,
    MIX_FUELS,
    build_example_frame,
    default_issue_schedule,
    issue_time_for_date,
    load_feature_sources,
    max_source_timestamp_allowed,
    select_model_inputs,
    select_training_arrays,
    split_by_issue_date,
)
from gridcast.holidays import UK_BANK_HOLIDAYS, is_uk_bank_holiday

LONDON_TZ = ZoneInfo("Europe/London")
UTC_TZ = timezone.utc

CURATED_PATH = Path("data/curated")


_CANONICAL_SOURCES = load_feature_sources()
_CANONICAL_FRAME = build_example_frame(
    intensity_df=_CANONICAL_SOURCES[0],
    weather_df=_CANONICAL_SOURCES[1],
    mix_df=_CANONICAL_SOURCES[2],
    require_aux=True,
)


@pytest.fixture(scope="module")
def curated_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load curated carbon intensity and real auxiliary test sources."""
    return _CANONICAL_SOURCES


# Explicit special dates: all four DST transition days and gap-adjacent dates
DST_TRANSITION_DATES: list[str] = [
    "2025-03-30",
    "2025-10-26",
    "2026-03-29",
    "2026-10-25",
]

GAP_ADJACENT_DATES: list[str] = [
    "2025-01-12",
    "2025-01-13",
    "2025-08-10",
    "2025-08-11",
    "2025-08-12",
    "2025-08-13",
]

EXPLICIT_SPECIAL_DATES: list[str] = DST_TRANSITION_DATES + GAP_ADJACENT_DATES


def derive_required_grid_dates(frame: pd.DataFrame) -> set[str]:
    """Derive required test grid dates as a computed property from the frame itself.

    Rule: first seven and last seven issue dates present in the frame, plus the four
    DST transition days, plus the gap-adjacent dates already covered.
    """
    issue_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in frame["issue_time"]})
    if len(issue_dates) < 14:
        raise ValueError(f"Frame has fewer than 14 issue dates ({len(issue_dates)})")
    first_seven = {d.isoformat() for d in issue_dates[:7]}
    last_seven = {d.isoformat() for d in issue_dates[-7:]}
    return first_seven | last_seven | set(EXPLICIT_SPECIAL_DATES)


def derive_invariance_windows(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Derive realistic multi-issue frame test windows including first 7 and last 7 issue dates."""
    issue_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in frame["issue_time"]})
    first_7_start = issue_dates[0].isoformat()
    first_7_end = issue_dates[6].isoformat()
    last_7_start = issue_dates[-7].isoformat()
    last_7_end = issue_dates[-1].isoformat()
    return [
        (first_7_start, first_7_end, "first_seven_issue_dates_week"),
        # Standard summer week
        ("2025-06-01", "2025-06-07", "summer_week"),
        # DST spring-forward week (contains 2025-03-30)
        ("2025-03-27", "2025-04-02", "dst_spring_forward_week"),
        # DST autumn-back week (contains 2025-10-26)
        ("2025-10-23", "2025-10-29", "dst_autumn_back_week"),
        # Source-gap adjacent week
        ("2025-08-08", "2025-08-14", "gap_adjacent_week"),
        (last_7_start, last_7_end, "last_seven_issue_dates_week"),
    ]


DENSE_GRID_TEST_DATES: list[str] = sorted(derive_required_grid_dates(_CANONICAL_FRAME))
REALISTIC_INVARIANCE_WINDOWS: list[tuple[str, str, str]] = (
    derive_invariance_windows(_CANONICAL_FRAME)
)


class TestTemporalLeakageInvariance:
    """Core future-perturbation invariance property across realistic multi-issue frames."""

    @pytest.mark.parametrize(
        ("start_date", "end_date", "window_name"),
        REALISTIC_INVARIANCE_WINDOWS,
        ids=[w[2] for w in REALISTIC_INVARIANCE_WINDOWS],
    )
    def test_future_perturbation_invariance_on_realistic_frame(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        start_date: str,
        end_date: str,
        window_name: str,
    ) -> None:
        """Multi-issue frame of at least 7 consecutive dates x 96 horizons must be invariant."""
        intensity_df, weather_df, mix_df = curated_sample_data
        schedule = default_issue_schedule(start_date, end_date)
        assert len(schedule) == 7

        # 1. Build original unperturbed multi-issue frame (7 dates x 96 horizons)
        orig_frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            issue_times=schedule,
            horizons=range(1, 97),
            require_aux=False,
        )
        assert orig_frame["issue_time"].nunique() == 7 and len(orig_frame) == 7 * 96

        # 2. Test one perturbation per issue inside the frame
        for issue_time in schedule:
            perturbed_intensity = intensity_df.copy()
            perturbed_weather = weather_df.copy()
            perturbed_mix = mix_df.copy()

            # Intensity actuals and forecasts after issue_time
            mask_int = (
                pd.to_datetime(perturbed_intensity["period_start_utc"], utc=True) > issue_time
            )
            perturbed_intensity.loc[mask_int, "actual"] = (
                perturbed_intensity.loc[mask_int, "actual"].fillna(100) + 1000
            )
            if "forecast" in perturbed_intensity.columns:
                perturbed_intensity.loc[mask_int, "forecast"] = (
                    perturbed_intensity.loc[mask_int, "forecast"].fillna(100) * 2
                )

            # Mix fuel shares after issue_time
            mask_mix = pd.to_datetime(perturbed_mix["period_start_utc"], utc=True) > issue_time
            for fuel in MIX_FUELS:
                if fuel in perturbed_mix.columns:
                    arr = perturbed_mix.loc[mask_mix, fuel].to_numpy()
                    rolled = np.roll(arr, shift=1)
                    if np.array_equal(arr, rolled):
                        rolled = arr + 10.0
                    perturbed_mix.loc[mask_mix, fuel] = rolled

            # Weather previous-runs forecasts:
            # Day 1 forecast was issued at period_start_utc - 24h.
            # Day 2 forecast was issued at period_start_utc - 48h.
            w_ts = pd.to_datetime(perturbed_weather["period_start_utc"], utc=True)
            mask_wth_d1 = (w_ts - pd.Timedelta(hours=24)) > issue_time
            if "temperature_2m_previous_day1" in perturbed_weather.columns:
                perturbed_weather.loc[mask_wth_d1, "temperature_2m_previous_day1"] = (
                    perturbed_weather.loc[mask_wth_d1, "temperature_2m_previous_day1"] + 40.0
                )

            mask_wth_d2 = (w_ts - pd.Timedelta(hours=48)) > issue_time
            if "temperature_2m_previous_day2" in perturbed_weather.columns:
                perturbed_weather.loc[mask_wth_d2, "temperature_2m_previous_day2"] = (
                    perturbed_weather.loc[mask_wth_d2, "temperature_2m_previous_day2"] + 40.0
                )

            for col in perturbed_weather.columns:
                if col not in (
                    "period_start_utc",
                    "local_date",
                    "temperature_2m_previous_day1",
                    "temperature_2m_previous_day2",
                ):
                    perturbed_weather.loc[w_ts > issue_time, col] = 999.0

            assert mask_int.any(), (
                "Precondition failed: intensity mutation mask must not be empty"
            )
            assert_injection_changed(
                intensity_df.loc[mask_int, "actual"],
                perturbed_intensity.loc[mask_int, "actual"],
                label="intensity actuals future perturbation",
            )
            assert mask_mix.any(), "Precondition failed: mix mutation mask must not be empty"
            for fuel in MIX_FUELS:
                if fuel in perturbed_mix.columns:
                    assert_injection_changed(
                        mix_df.loc[mask_mix, fuel],
                        perturbed_mix.loc[mask_mix, fuel],
                        label=f"mix fuel {fuel} future perturbation",
                    )
            if "temperature_2m_previous_day1" in perturbed_weather.columns and mask_wth_d1.any():
                assert_injection_changed(
                    weather_df.loc[mask_wth_d1, "temperature_2m_previous_day1"],
                    perturbed_weather.loc[mask_wth_d1, "temperature_2m_previous_day1"],
                    label="weather previous_day1 future perturbation",
                )
            if "temperature_2m_previous_day2" in perturbed_weather.columns and mask_wth_d2.any():
                assert_injection_changed(
                    weather_df.loc[mask_wth_d2, "temperature_2m_previous_day2"],
                    perturbed_weather.loc[mask_wth_d2, "temperature_2m_previous_day2"],
                    label="weather previous_day2 future perturbation",
                )

            # 3. Rebuild full multi-issue frame (7 dates x 96 horizons) from perturbed sources
            rebuilt_frame = build_example_frame(
                intensity_df=perturbed_intensity,
                weather_df=perturbed_weather,
                mix_df=perturbed_mix,
                issue_times=schedule,
                horizons=range(1, 97),
                require_aux=False,
            )
            assert rebuilt_frame["issue_time"].nunique() == 7 and len(rebuilt_frame) == 7 * 96

            # 4. Assert all 96 horizons for the perturbed issue are strictly invariant
            orig_slice = orig_frame[orig_frame["issue_time"] == issue_time]
            rebuilt_slice = rebuilt_frame[rebuilt_frame["issue_time"] == issue_time]
            assert len(orig_slice) == 96
            assert len(rebuilt_slice) == 96

            for h in range(1, 97):
                orig_row = orig_slice[orig_slice["horizon"] == h].iloc[0]
                rebuilt_row = rebuilt_slice[rebuilt_slice["horizon"] == h].iloc[0]

                for col in FEATURE_COLUMNS:
                    orig_val = orig_row[col]
                    rebuilt_val = rebuilt_row[col]

                    if pd.isna(orig_val) and pd.isna(rebuilt_val):
                        continue

                    matches = (orig_val == rebuilt_val) or (
                        isinstance(orig_val, (int, float, np.floating))
                        and orig_val == pytest.approx(rebuilt_val, rel=1e-5)
                    )
                    assert matches, (
                        f"Temporal leakage in '{col}' at issue_time {issue_time}, horizon {h}!\n"
                        f"orig={orig_val}, rebuilt={rebuilt_val}"
                    )


class TestFeatureContractAndInvariants:
    """Validate structure, column names, coverage, dead-feature avoidance, and contracts."""

    def test_example_frame_shape_and_columns(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Every declared feature must be present, alongside keys and y."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issues = [
            pd.Timestamp("2025-01-01 00:00:00+00:00"),
            pd.Timestamp("2025-01-02 00:00:00+00:00"),
        ]
        sample_horizons = [1, 2, 48, 96]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=sample_horizons,
            require_aux=True,
        )

        assert "issue_time" in features.columns
        assert "horizon" in features.columns
        assert "target_time" in features.columns
        assert "y" in features.columns
        assert len(features) == len(sample_issues) * len(sample_horizons)

        for expected_col in FEATURE_COLUMNS:
            assert expected_col in features.columns, f"Missing feature column: {expected_col}"

        # The frame must carry exactly the declared contract, no more. An undeclared
        # column can smuggle a value past a check that only inspects FEATURE_COLUMNS,
        # and downstream slicing would then carry it into a model.
        allowed = {"issue_time", "horizon", "target_time", "y", *FEATURE_COLUMNS}
        unexpected = set(features.columns) - allowed
        assert not unexpected, (
            f"Undeclared columns emitted by the frame builder: {sorted(unexpected)}"
        )

    def test_no_column_named_or_derived_from_target(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Assert no feature column is named or derived from target."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issues = [pd.Timestamp("2025-06-01 00:00:00+00:00")]
        sample_horizons = [1, 48, 96]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=sample_horizons,
            require_aux=True,
        )

        forbidden_names = [
            "target_value",
            "observed_temperature_at_t",
            "actual",
            "target",
            "actual_at_t",
        ]
        for name in forbidden_names:
            assert (
                name not in FEATURE_COLUMNS
            ), f"Forbidden target-derived name in FEATURE_COLUMNS: {name}"
            assert name not in features.columns, f"Forbidden target-derived column in frame: {name}"

        # Target column is exclusively 'y'
        assert "y" in features.columns
        assert "y" not in FEATURE_COLUMNS

    def test_horizon_is_a_feature(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Horizon must be explicitly present in FEATURE_COLUMNS."""
        assert "horizon" in FEATURE_COLUMNS
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issues = [pd.Timestamp("2025-06-01 00:00:00+00:00")]
        sample_horizons = [1, 10, 48, 96]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=sample_horizons,
            require_aux=True,
        )
        assert list(features["horizon"].to_numpy()) == sample_horizons

    def test_every_fuel_share_is_lagged_48h(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Every one of the six fuel shares is lagged to issue_time - 48h."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-08-20 00:00:00+00:00")
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[sample_issue],
            horizons=[1, 48, 96],
            require_aux=True,
        )

        m_ts = pd.to_datetime(mix_df["period_start_utc"], utc=True)
        lagged_mix_ts = sample_issue - pd.Timedelta(hours=48)
        mix_at_lag = mix_df[m_ts == lagged_mix_ts].iloc[0]
        mix_at_issue = mix_df[m_ts == sample_issue].iloc[0]

        for fuel in MIX_FUELS:
            col_name = f"mix_share_{fuel}"
            assert col_name in FEATURE_COLUMNS
            feat_val = features[col_name].iloc[0]
            stored_lag_val = mix_at_lag[fuel]
            stored_issue_val = mix_at_issue[fuel]

            assert feat_val == pytest.approx(stored_lag_val, rel=1e-5), (
                f"Mix feature '{col_name}' at {sample_issue} did not match mix at {lagged_mix_ts}"
            )
            if stored_lag_val != stored_issue_val:
                assert feat_val != pytest.approx(stored_issue_val, rel=1e-5), (
                    f"Mix feature '{col_name}' matched contemporaneous mix rather than lagged!"
                )

    def test_lags_resolved_by_timestamp_never_by_row_shift(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Lags must be resolved by timestamp, never by row position/shift."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-06-05 00:00:00+00:00")
        target_lag_ts = sample_issue - pd.Timedelta(hours=48)

        # Drop 4 half-hour periods between target_lag_ts and sample_issue
        gap_ts = sample_issue - pd.Timedelta(hours=24)
        drop_range = pd.date_range(gap_ts, periods=4, freq="30min", tz="UTC")
        gapped_intensity = intensity_df[
            ~pd.to_datetime(intensity_df["period_start_utc"], utc=True).isin(drop_range)
        ].copy()

        assert_injection_changed(
            intensity_df,
            gapped_intensity,
            label="gapped intensity dropped 4 periods",
        )

        features = build_example_frame(
            gapped_intensity,
            weather_df,
            mix_df,
            issue_times=[sample_issue],
            horizons=[1],
            require_aux=False,
        )

        int_ts = pd.to_datetime(intensity_df["period_start_utc"], utc=True)
        expected_val = intensity_df.loc[int_ts == target_lag_ts, "actual"].iloc[0]
        actual_feat = features["actual_lag_48h"].iloc[0]
        assert actual_feat == pytest.approx(expected_val, rel=1e-5), (
            f"actual_lag_48h ({actual_feat}) did not match timestamp "
            f"{target_lag_ts} ({expected_val}); "
            "lag was likely computed by row position rather than timestamp!"
        )

    def test_weather_feature_matches_previous_runs_differs_from_reanalysis(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Weather feature matches stored previous-runs and differs from reanalysis."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-01-20 00:00:00+00:00")
        sample_horizons = [1, 16, 48, 72]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[sample_issue],
            horizons=sample_horizons,
            require_aux=True,
        )

        w_ts = pd.to_datetime(weather_df["period_start_utc"], utc=True)
        w_s_d1 = pd.Series(
            weather_df["temperature_2m_previous_day1"].to_numpy(dtype=float),
            index=w_ts,
        )
        w_s_d2 = pd.Series(
            weather_df["temperature_2m_previous_day2"].to_numpy(dtype=float),
            index=w_ts,
        )

        for h in sample_horizons:
            target_time = sample_issue + pd.Timedelta(minutes=30 * h)
            containing_hour = target_time.floor("h")
            row = features[features["horizon"] == h].iloc[0]
            feat_val = row["temperature"]
            expected_series = w_s_d1 if h <= 48 else w_s_d2
            stored_val = expected_series.loc[containing_hour]

            assert feat_val == pytest.approx(stored_val, rel=1e-5)
            # Reanalysis divergence test
            reanalysis_divergent = stored_val + 5.0
            assert feat_val != pytest.approx(reanalysis_divergent, rel=1e-5)

    def test_horizon_contract_enforced_in_code(self) -> None:
        """Validate max_source_timestamp_allowed helper and enforcement."""
        issue_time = pd.Timestamp("2026-06-01 00:00:00+00:00")
        target_time = issue_time + pd.Timedelta(hours=48)  # h=96

        # Lags of actual
        t_recent_3h = issue_time - ACTUAL_AVAILABILITY_LAG
        assert (
            max_source_timestamp_allowed("actual_recent_3h", issue_time, target_time)
            == t_recent_3h
        )
        t_48h = issue_time - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("actual_lag_48h", issue_time, target_time) == t_48h
        t_168h = issue_time - pd.Timedelta(hours=168)
        assert max_source_timestamp_allowed("actual_lag_168h", issue_time, target_time) == t_168h

        # Rolling of actual: window ends at issue_time - ACTUAL_AVAILABILITY_LAG
        t_roll = issue_time - ACTUAL_AVAILABILITY_LAG
        roll_3h = max_source_timestamp_allowed("actual_rolling_3h_mean", issue_time, target_time)
        roll_24h = max_source_timestamp_allowed("actual_rolling_24h_mean", issue_time, target_time)
        assert roll_3h == t_roll
        assert roll_24h == t_roll

        # Generation mix: at issue_time - 48h
        assert max_source_timestamp_allowed("mix_share_wind", issue_time, target_time) == t_48h

        # Weather: previous_day2 forecast issued at target_time - 48h <= issue_time
        w_issue = target_time - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("temperature", issue_time, target_time) == w_issue
        assert max_source_timestamp_allowed("temperature", issue_time, target_time) <= issue_time

    def test_actual_availability_allowance(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Assert actual-derived features observe ACTUAL_AVAILABILITY_LAG and match store."""
        intensity_df, weather_df, mix_df = curated_sample_data
        issue_time = pd.Timestamp("2026-06-01 00:00:00+00:00")
        target_time = issue_time + pd.Timedelta(hours=24)

        # 1. Assert every actual-derived feature's source timestamp is at or before issue_time - 3h
        actual_features = [
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
        ]
        three_hours = pd.Timedelta(hours=3)
        assert three_hours == ACTUAL_AVAILABILITY_LAG

        for feat in actual_features:
            max_ts = max_source_timestamp_allowed(feat, issue_time, target_time)
            assert max_ts <= issue_time - three_hours, (
                f"Feature {feat} source ts {max_ts} > cutoff {issue_time - three_hours}"
            )
            assert max_ts <= issue_time - ACTUAL_AVAILABILITY_LAG

        # 2. Assert actual_recent_3h equals the store value at that instant
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[issue_time],
            horizons=[1, 48],
            require_aux=True,
        )
        expected_instant = issue_time - ACTUAL_AVAILABILITY_LAG
        sorted_i = intensity_df.sort_values("period_start_utc").reset_index(drop=True)
        expected_val = float(
            sorted_i.loc[sorted_i["period_start_utc"] == expected_instant, "actual"].iloc[0]
        )
        assert frame["actual_recent_3h"].iloc[0] == expected_val

        # 3. Assert constant is parameterised and used rather than hardcoded literal 3
        custom_lag = pd.Timedelta(hours=5)
        custom_frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[issue_time],
            horizons=[1],
            require_aux=True,
            actual_availability_lag=custom_lag,
        )
        custom_expected_instant = issue_time - custom_lag
        custom_expected_val = float(
            sorted_i.loc[sorted_i["period_start_utc"] == custom_expected_instant, "actual"].iloc[0]
        )
        assert custom_frame["actual_recent_3h"].iloc[0] == custom_expected_val

        # For all declared features and h <= 96, latest allowed timestamp must be <= issue_time
        for feat in FEATURE_COLUMNS:
            cutoff = max_source_timestamp_allowed(feat, issue_time, target_time)
            assert cutoff <= issue_time, (
                f"Feature '{feat}' allowed cutoff {cutoff} > issue_time {issue_time}"
            )

    def test_horizon_contract_rejects_horizon_greater_than_48h(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """A build with horizon > 96 (target_time - 48h > issue_time) must raise ValueError."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-06-01 00:00:00+00:00")

        # Horizon 97 corresponds to 48.5 hours ahead.
        # Its forecast was issued at target_time - 48h = issue_time + 30min > issue_time.
        bad_horizons = [97]
        assert_injection_changed(
            96,
            bad_horizons[0],
            label="horizon contract invalid horizon > 96",
        )
        with pytest.raises(ValueError, match="(?i)(horizon contract|source timestamp|issue_time)"):
            build_example_frame(
                intensity_df,
                weather_df,
                mix_df,
                issue_times=[sample_issue],
                horizons=bad_horizons,
                require_aux=False,
            )

    def test_real_frame_has_no_dead_columns(self) -> None:
        """Build a sample from real data and assert no column is entirely null or constant."""
        intensity_df, weather_df, mix_df = load_feature_sources()
        sample_issues = [
            pd.Timestamp("2024-12-25 00:00:00+00:00"),
            pd.Timestamp("2025-05-01 00:00:00+00:00"),
            pd.Timestamp("2026-05-22 08:30:00+00:00"),
        ]
        sample_horizons = [1, 48]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=sample_horizons,
            require_aux=True,
        )
        assert len(features) == len(sample_issues) * len(sample_horizons)
        for col in FEATURE_COLUMNS:
            assert not features[col].isna().all(), f"Feature column '{col}' is entirely null"
            assert features[col].dropna().nunique() > 1, f"Feature column '{col}' is constant"

    def test_feature_frame_requires_auxiliary_sources(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Building features without auxiliary sources with require_aux=True raises."""
        intensity_df, _, _ = curated_sample_data
        with pytest.raises(ValueError, match="(?i)(auxiliary|weather|mix|null)"):
            build_example_frame(intensity_df, require_aux=True)

    def test_weather_source_must_be_previous_runs_not_reanalysis(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Archive-api reanalysis weather must be rejected."""
        intensity_df, weather_df, mix_df = curated_sample_data
        bad_weather = pd.DataFrame({
            "period_start_utc": intensity_df["period_start_utc"],
            "temperature_reanalysis": [15.0] * len(intensity_df),
        })

        assert_injection_changed(
            weather_df,
            bad_weather,
            label="weather previous_runs replaced by reanalysis",
        )

        with pytest.raises(
            (ValueError, KeyError),
            match="(?i)(previous_day2|previous-runs|reanalysis)",
        ):
            build_example_frame(intensity_df, weather_df=bad_weather, mix_df=mix_df)

    def test_dst_spring_forward_lag_lookup(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """A 24h lag across DST must look up exactly issue_time - 24h, not shift rows."""
        intensity_df, weather_df, mix_df = curated_sample_data
        t_issue = pd.Timestamp("2026-03-30 00:00:00+00:00")
        t_expected_24h = t_issue - pd.Timedelta(hours=24)

        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[t_issue],
            horizons=[1],
            require_aux=True,
        )

        feat_val = features["actual_lag_24h"].iloc[0]
        int_ts = pd.to_datetime(intensity_df["period_start_utc"], utc=True)
        source_val = intensity_df.loc[int_ts == t_expected_24h, "actual"].iloc[0]

        assert feat_val == source_val

    def test_bank_holidays_presence(self) -> None:
        """UK bank holidays table must cover 2024-2027."""
        years = {d.year for d in UK_BANK_HOLIDAYS}
        assert years == {2024, 2025, 2026, 2027}
        assert date(2024, 12, 25) in UK_BANK_HOLIDAYS
        assert date(2025, 12, 25) in UK_BANK_HOLIDAYS
        assert date(2026, 12, 25) in UK_BANK_HOLIDAYS
        assert is_uk_bank_holiday(date(2026, 12, 25)) is True
        assert is_uk_bank_holiday(date(2026, 12, 24)) is False

    def test_missing_required_columns_raises(self) -> None:
        """Missing period_start_utc or actual raises ValueError."""
        valid_cols = ["period_start_utc", "actual"]
        df_no_utc = pd.DataFrame({"actual": [100]})
        assert_injection_changed(
            valid_cols,
            list(df_no_utc.columns),
            label="intensity missing period_start_utc",
        )
        with pytest.raises(ValueError, match="period_start_utc"):
            build_example_frame(df_no_utc)

        df_no_act = pd.DataFrame({"period_start_utc": [pd.Timestamp.now()]})
        assert_injection_changed(
            valid_cols,
            list(df_no_act.columns),
            label="intensity missing actual",
        )
        with pytest.raises(ValueError, match="actual"):
            build_example_frame(df_no_act)


class TestIssueAnchorEuropeLondon:
    """Validate 00:00 Europe/London issue anchor invariants."""

    def test_issue_instants_bst_and_gmt(self) -> None:
        """In BST the instant is 23:00Z of the previous UTC date; in GMT it is 00:00Z."""
        # BST day: 2025-06-01 (Europe/London is UTC+1)
        bst_issue = issue_time_for_date("2025-06-01")
        assert bst_issue == pd.Timestamp("2025-05-31 23:00:00+00:00")
        assert bst_issue.strftime("%Y-%m-%d %H:%M:%SZ") == "2025-05-31 23:00:00Z"

        # GMT day: 2025-01-01 (Europe/London is UTC+0)
        gmt_issue = issue_time_for_date("2025-01-01")
        assert gmt_issue == pd.Timestamp("2025-01-01 00:00:00+00:00")
        assert gmt_issue.strftime("%Y-%m-%d %H:%M:%SZ") == "2025-01-01 00:00:00Z"

    def test_consecutive_issue_instants_spacing(self) -> None:
        """Consecutive issue instants across the schedule are 23, 24, or 25 hours apart."""
        schedule = default_issue_schedule(date(2024, 11, 1), date(2026, 11, 1))
        diffs = [schedule[i + 1] - schedule[i] for i in range(len(schedule) - 1)]
        distinct_diff_hours = {int(d.total_seconds() / 3600) for d in diffs}
        assert distinct_diff_hours == {23, 24, 25}

    def test_default_frame_issue_times_anchored_to_london_midnight(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """The default frame issues must be anchored to 00:00 Europe/London for each local date."""
        intensity_df, weather_df, mix_df = curated_sample_data
        # Sample two dates: one GMT, one BST
        gmt_issue = issue_time_for_date("2025-01-15")
        bst_issue = issue_time_for_date("2025-07-15")
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[gmt_issue, bst_issue],
            horizons=[1],
            require_aux=True,
        )
        assert features["issue_time"].iloc[0] == pd.Timestamp("2025-01-15 00:00:00+00:00")
        assert features["issue_time"].iloc[1] == pd.Timestamp("2025-07-14 23:00:00+00:00")


class TestHorizonAbsoluteOffsetDST:
    """Validate that horizon is an absolute offset spanning 48 hours across DST days."""

    @pytest.mark.parametrize(
        ("dst_date", "expected_local_periods"),
        [
            ("2025-03-30", 46),  # Spring clocks go forward (2025)
            ("2025-10-26", 50),  # Autumn clocks go back (2025)
            ("2026-03-29", 46),  # Spring clocks go forward (2026)
            ("2026-10-25", 50),  # Autumn clocks go back (2026)
        ],
        ids=["2025-spring", "2025-autumn", "2026-spring", "2026-autumn"],
    )
    def test_horizon_absolute_offset_on_dst_transition_day(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        dst_date: str,
        expected_local_periods: int,
    ) -> None:
        """On DST days with 46 or 50 local half-hours, horizons are strictly 1..96 over 48 hours."""
        intensity_df, weather_df, mix_df = curated_sample_data
        t_issue = issue_time_for_date(dst_date)

        frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            issue_times=[t_issue],
            horizons=range(1, 97),
            require_aux=True,
        )

        # 1. Exactly 96 horizons, 1..96
        assert len(frame) == 96
        assert list(frame["horizon"]) == list(range(1, 97))

        # 2. 96 distinct target_time values spanning exactly 48 hours from issue_time
        target_times = frame["target_time"]
        assert target_times.nunique() == 96
        assert target_times.is_monotonic_increasing

        span_from_issue = target_times.max() - t_issue
        assert span_from_issue == pd.Timedelta(hours=48), (
            f"Expected span of 48 hours from issue_time {t_issue}, got {span_from_issue}"
        )

        # Target range: from t_issue + 30m to t_issue + 48h
        assert target_times.iloc[0] == t_issue + pd.Timedelta(minutes=30)
        assert target_times.iloc[-1] == t_issue + pd.Timedelta(hours=48)

        # 3. No gap and no duplicate in target sequence (every step is exactly 30 minutes)
        target_diffs = target_times.diff().iloc[1:]
        assert (target_diffs == pd.Timedelta(minutes=30)).all(), (
            f"Found gaps or irregular steps in target sequence: {target_diffs.unique()}"
        )

        # 4. Target time must equal issue_time + horizon * 30 minutes exactly for every row
        expected_targets = [t_issue + pd.Timedelta(minutes=30 * h) for h in range(1, 97)]
        assert list(target_times) == expected_targets


class TestSplitByIssueDate:
    """Validate split_by_issue_date with embargo gap preventing target leakage."""

    def test_split_by_issue_date_embargo_and_target_overlap(self) -> None:
        """Synthetic frame split must drop embargo_days, be disjoint, and avoid target overlap."""
        # Create a synthetic frame of 60 consecutive days from 2025-01-01 to 2025-03-01
        start_d = date(2025, 1, 1)
        end_d = date(2025, 3, 1)
        issue_times = default_issue_schedule(start_d, end_d)
        n_issues = len(issue_times)
        horizons = list(range(1, 97))
        n_h = len(horizons)

        issue_rep = np.repeat(issue_times, n_h)
        horizon_rep = np.tile(horizons, n_issues)
        target_rep = pd.DatetimeIndex(issue_rep) + pd.to_timedelta(horizon_rep * 30, unit="m")

        # Create target y, with a few null y rows scattered
        y_vals = np.random.RandomState(42).randn(len(issue_rep))
        y_vals[10] = np.nan
        y_vals[500] = np.nan
        # Also null at tail
        y_vals[-10:] = np.nan

        synthetic_frame = pd.DataFrame({
            "issue_time": issue_rep,
            "horizon": horizon_rep,
            "target_time": target_rep,
            "y": y_vals,
        })
        for col in FEATURE_COLUMNS:
            if col != "horizon":
                synthetic_frame[col] = 1.0

        train_end = "2025-01-20"
        val_end = "2025-02-10"
        embargo = 4

        train, val, test = split_by_issue_date(
            synthetic_frame,
            train_end=train_end,
            val_end=val_end,
            embargo_days=embargo,
        )

        # 1. Null y rows are excluded from every split
        assert train["y"].isna().sum() == 0
        assert val["y"].isna().sum() == 0
        assert test["y"].isna().sum() == 0

        # 2. Splits are disjoint by issue_time
        train_issues = set(train["issue_time"])
        val_issues = set(val["issue_time"])
        test_issues = set(test["issue_time"])
        assert train_issues.isdisjoint(val_issues)
        assert val_issues.isdisjoint(test_issues)
        assert train_issues.isdisjoint(test_issues)

        # 3. Check issue-date bounds and number of dropped issue dates at each boundary
        train_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in train_issues})
        val_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in val_issues})
        test_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in test_issues})

        assert train_dates[-1] == date(2025, 1, 20)
        assert val_dates[0] == date(2025, 1, 25)  # 21, 22, 23, 24 dropped = 4 days
        dropped_b1 = (val_dates[0] - train_dates[-1]).days - 1
        assert dropped_b1 >= embargo
        assert dropped_b1 == 4

        assert val_dates[-1] == date(2025, 2, 10)
        assert test_dates[0] == date(2025, 2, 15)  # 11, 12, 13, 14 dropped = 4 days
        dropped_b2 = (test_dates[0] - val_dates[-1]).days - 1
        assert dropped_b2 >= embargo
        assert dropped_b2 == 4

        # 4. Target overlap prevention:
        # max(target_time) of train < min(target_time) of validation - 48h
        # max(target_time) of validation < min(target_time) of test - 48h
        assert train["target_time"].max() < val["target_time"].min() - pd.Timedelta(hours=48)
        assert val["target_time"].max() < test["target_time"].min() - pd.Timedelta(hours=48)


class TestPositiveModelInputSelection:
    """Validate select_model_inputs and select_training_arrays as positive selections."""

    def test_select_model_inputs_exact_columns_and_order(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """select_model_inputs returns exactly FEATURE_COLUMNS in exact order."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issues = [pd.Timestamp("2025-01-01 00:00:00+00:00")]
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=[1, 48],
            require_aux=True,
        )

        X = select_model_inputs(frame)
        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == len(frame)
        assert "y" not in X.columns
        assert "issue_time" not in X.columns
        assert "target_time" not in X.columns

    def test_select_training_arrays_drops_null_y_and_asserts_non_null(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """select_training_arrays drops rows with null y and returns strictly non-null (X, y)."""
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issues = [pd.Timestamp("2025-01-01 00:00:00+00:00")]
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=sample_issues,
            horizons=[1, 2, 3, 4],
            require_aux=True,
        )
        # Deliberately inject a null y in one row
        frame_with_null = frame.copy()
        frame_with_null.loc[1, "y"] = np.nan

        assert_injection_changed(
            frame["y"],
            frame_with_null["y"],
            label="injected null y into training frame",
        )

        X, y = select_training_arrays(frame_with_null)
        assert len(X) == 3
        assert len(y) == 3
        assert not y.isna().any(), "Invariant violated: null label in training y"
        assert list(X.columns) == FEATURE_COLUMNS
        assert (X.index == y.index).all()

    def test_select_training_arrays_raises_on_missing_or_all_null_y(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """select_training_arrays raises ValueError when y is missing or entirely null."""
        intensity_df, weather_df, mix_df = curated_sample_data
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[pd.Timestamp("2025-01-01 00:00:00+00:00")],
            horizons=[1, 2],
            require_aux=True,
        )

        frame_no_y = frame.drop(columns=["y"])
        assert_injection_changed(
            list(frame.columns),
            list(frame_no_y.columns),
            label="training arrays missing y column",
        )
        with pytest.raises(ValueError, match="y"):
            select_training_arrays(frame_no_y)

        frame_all_null = frame.copy()
        frame_all_null["y"] = np.nan
        assert_injection_changed(
            frame["y"],
            frame_all_null["y"],
            label="training arrays all null y",
        )
        with pytest.raises(ValueError, match="(?i)(valid.*training.*rows|null)"):
            select_training_arrays(frame_all_null)

    def test_model_feature_columns_is_subset_of_feature_columns(self) -> None:
        """The model feature list must be a subset of FEATURE_COLUMNS."""
        from gridcast.features import MODEL_FEATURE_COLUMNS

        assert isinstance(MODEL_FEATURE_COLUMNS, list)
        assert set(MODEL_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS))

    def test_select_model_inputs_raises_on_missing_column(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """select_model_inputs raises ValueError when a declared feature column is missing."""
        intensity_df, weather_df, mix_df = curated_sample_data
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[pd.Timestamp("2025-01-01 00:00:00+00:00")],
            horizons=[1],
            require_aux=True,
        )
        bad_frame = frame.drop(columns=["temperature"])
        assert_injection_changed(
            list(frame.columns),
            list(bad_frame.columns),
            label="model inputs missing temperature column",
        )
        with pytest.raises(ValueError, match="(?i)(missing.*column|temperature)"):
            select_model_inputs(bad_frame)

    def test_select_model_inputs_raises_on_extra_column(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """select_model_inputs raises ValueError when the frame carries an undeclared column."""
        intensity_df, weather_df, mix_df = curated_sample_data
        frame = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[pd.Timestamp("2025-01-01 00:00:00+00:00")],
            horizons=[1],
            require_aux=True,
        )
        bad_frame = frame.copy()
        bad_frame["sneaky_extra_column"] = 42.0
        assert_injection_changed(
            list(frame.columns),
            list(bad_frame.columns),
            label="model inputs undeclared extra column",
        )
        with pytest.raises(ValueError, match="(?i)(undeclared.*column|sneaky_extra_column)"):
            select_model_inputs(bad_frame)


class TestLeadingDateTrimming:
    """Validate that incomplete leading issue dates from warmup are dropped."""

    def test_leading_issue_dates_dropped_and_first_date_has_zero_null_features(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """The default frame must drop warmup dates; its first date has zero feature nulls."""
        intensity_df, weather_df, mix_df = curated_sample_data
        frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            require_aux=True,
        )

        issue_dates = sorted({ts.tz_convert(LONDON_TZ).date() for ts in frame["issue_time"]})
        first_issue_date = issue_dates[0]

        # 1. Warmup of 7 days (168h lag) dropped: first date must be 2024-11-08
        assert first_issue_date == date(2024, 11, 8), (
            f"Expected first issue date 2024-11-08 after 7-day warmup drop, got {first_issue_date}"
        )

        # 2. Assert the first issue date in the frame has zero nulls across FEATURE_COLUMNS
        first_date_mask = frame["issue_time"].dt.tz_convert(LONDON_TZ).dt.date == first_issue_date
        first_date_frame = frame[first_date_mask]
        feature_nulls = first_date_frame[FEATURE_COLUMNS].isna().sum()
        assert (feature_nulls == 0).all(), (
            f"First issue date {first_issue_date} has nulls in feature columns:\n"
            f"{feature_nulls[feature_nulls > 0]}"
        )

        # 3. Target y may still have nulls at the tail
        tail_nulls = frame["y"].isna().sum()
        assert tail_nulls > 0, "Expected unsettled targets at the tail of the dataset to be null"


class TestPerHorizonWeatherLead:
    """Validate per-horizon weather lead: previous_day1 for h <= 48, previous_day2 for h > 48."""

    def test_weather_lead_selection_per_horizon(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-01-20 00:00:00+00:00")
        sample_horizons = [24, 72]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[sample_issue],
            horizons=sample_horizons,
            require_aux=True,
        )

        w_ts = pd.to_datetime(weather_df["period_start_utc"], utc=True)
        s_d1 = pd.Series(
            weather_df["temperature_2m_previous_day1"].to_numpy(dtype=float),
            index=w_ts,
        )
        s_d2 = pd.Series(
            weather_df["temperature_2m_previous_day2"].to_numpy(dtype=float),
            index=w_ts,
        )

        # At h = 24 (<= 48): equals previous_day1 and differs from previous_day2
        row_24 = features[features["horizon"] == 24].iloc[0]
        t_24 = sample_issue + pd.Timedelta(minutes=30 * 24)
        ch_24 = t_24.floor("h")
        assert s_d1.loc[ch_24] != s_d2.loc[ch_24], (
            "Precondition failed: day1 and day2 must differ at h=24"
        )
        assert row_24["temperature"] == pytest.approx(s_d1.loc[ch_24], rel=1e-5)
        assert row_24["temperature"] != pytest.approx(s_d2.loc[ch_24], rel=1e-5)

        # At h = 72 (> 48): equals previous_day2
        row_72 = features[features["horizon"] == 72].iloc[0]
        t_72 = sample_issue + pd.Timedelta(minutes=30 * 72)
        ch_72 = t_72.floor("h")
        assert row_72["temperature"] == pytest.approx(s_d2.loc[ch_72], rel=1e-5)

    def test_deliberate_build_with_day1_at_h72_raises_contract_violation(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        intensity_df, weather_df, mix_df = curated_sample_data
        sample_issue = pd.Timestamp("2025-01-20 00:00:00+00:00")
        override = {72: "temperature_2m_previous_day1"}
        assert_injection_changed(
            "temperature_2m_previous_day2",
            override[72],
            label="deliberate lead override day1 at h72",
        )
        with pytest.raises(ValueError, match="Horizon contract violation"):
            build_example_frame(
                intensity_df,
                weather_df,
                mix_df,
                issue_times=[sample_issue],
                horizons=[72],
                require_aux=True,
                weather_series_by_horizon=override,
            )


def naive_feature_oracle(
    intensity_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    mix_df: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
    feature_name: str,
) -> tuple[float, pd.Timestamp | str]:
    """Compute expected feature value and source timestamp via slow, obvious raw store lookups.

    Recomputes directly from raw store with slow, obvious pandas lookups
    (e.g., store.loc[store['period_start_utc'] == expected_ts, ...]) and
    never calls the feature builder.
    """
    target_time = issue_time + pd.Timedelta(minutes=30 * horizon)

    if feature_name == "horizon":
        return float(horizon), "N/A"

    if feature_name == "actual_recent_3h":
        expected_ts = issue_time - ACTUAL_AVAILABILITY_LAG
        rows = intensity_df.loc[intensity_df["period_start_utc"] == expected_ts, "actual"]
        val = float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else float("nan")
        return val, expected_ts

    if feature_name == "actual_lag_24h":
        expected_ts = issue_time - pd.Timedelta(hours=24)
        rows = intensity_df.loc[intensity_df["period_start_utc"] == expected_ts, "actual"]
        val = float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else float("nan")
        return val, expected_ts

    if feature_name == "actual_lag_48h":
        expected_ts = issue_time - pd.Timedelta(hours=48)
        rows = intensity_df.loc[intensity_df["period_start_utc"] == expected_ts, "actual"]
        val = float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else float("nan")
        return val, expected_ts

    if feature_name == "actual_lag_168h":
        expected_ts = issue_time - pd.Timedelta(hours=168)
        rows = intensity_df.loc[intensity_df["period_start_utc"] == expected_ts, "actual"]
        val = float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else float("nan")
        return val, expected_ts

    if feature_name.startswith("actual_rolling_"):
        t_end = issue_time - ACTUAL_AVAILABILITY_LAG
        if "3h" in feature_name:
            t_start = t_end - pd.Timedelta(hours=3)
        elif "24h" in feature_name:
            t_start = t_end - pd.Timedelta(hours=24)
        elif "168h" in feature_name:
            t_start = t_end - pd.Timedelta(hours=168)
        else:
            raise ValueError(f"Unknown rolling feature: {feature_name}")

        window_rows = intensity_df.loc[
            (intensity_df["period_start_utc"] > t_start)
            & (intensity_df["period_start_utc"] <= t_end),
            "actual",
        ]
        source_ts_str = f"{t_start} to {t_end}"
        if "mean" in feature_name:
            return float(window_rows.mean()), source_ts_str
        if "std" in feature_name:
            val = window_rows.std()
            return (float(val) if pd.notna(val) else float("nan")), source_ts_str

    target_local = target_time.tz_convert(LONDON_TZ)
    if feature_name == "hour_of_day":
        return float(target_local.hour), target_time
    if feature_name == "day_of_week":
        return float(target_local.dayofweek), target_time
    if feature_name == "month":
        return float(target_local.month), target_time
    if feature_name == "is_weekend":
        return (1.0 if target_local.dayofweek >= 5 else 0.0), target_time
    if feature_name == "is_uk_bank_holiday":
        return (1.0 if target_local.date() in UK_BANK_HOLIDAYS else 0.0), target_time
    if feature_name == "sin_hour":
        hour_cont = target_local.hour + target_local.minute / 60.0
        return float(np.sin(2.0 * np.pi * hour_cont / 24.0)), target_time
    if feature_name == "cos_hour":
        hour_cont = target_local.hour + target_local.minute / 60.0
        return float(np.cos(2.0 * np.pi * hour_cont / 24.0)), target_time
    if feature_name == "sin_day_of_year":
        doy_cont = target_local.dayofyear - 1.0
        return float(np.sin(2.0 * np.pi * doy_cont / 365.25)), target_time
    if feature_name == "cos_day_of_year":
        doy_cont = target_local.dayofyear - 1.0
        return float(np.cos(2.0 * np.pi * doy_cont / 365.25)), target_time

    if feature_name in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
        target_hour = target_time.floor("h")
        col = (
            "temperature_2m_previous_day1"
            if horizon <= 48
            else "temperature_2m_previous_day2"
        )
        target_rows = weather_df.loc[weather_df["period_start_utc"] == target_hour, col]
        # Boundary semantics: when the hour containing the target is absent from the
        # weather store or null, the weather feature and both rolling statistics are null.
        # A 24-hour mean computed over a shorter window is a different statistic and must
        # not be substituted.
        if target_rows.empty or pd.isna(target_rows.iloc[0]):
            return float("nan"), target_hour

        if feature_name == "temperature":
            return float(target_rows.iloc[0]), target_hour

        t_start = target_hour - pd.Timedelta(hours=24)
        window_rows = weather_df.loc[
            (weather_df["period_start_utc"] > t_start)
            & (weather_df["period_start_utc"] <= target_hour),
            col,
        ]
        source_ts_str = f"{t_start} to {target_hour}"
        if feature_name == "temperature_24h_mean":
            return float(window_rows.mean()), source_ts_str
        if feature_name == "temperature_24h_min":
            return float(window_rows.min()), source_ts_str

    if feature_name.startswith("mix_share_"):
        fuel = feature_name.removeprefix("mix_share_")
        expected_ts = issue_time - pd.Timedelta(hours=48)
        col = fuel if fuel in mix_df.columns else feature_name
        rows = mix_df.loc[mix_df["period_start_utc"] == expected_ts, col]
        val = float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else float("nan")
        return val, expected_ts

    raise ValueError(f"Unknown feature: {feature_name}")


@pytest.fixture(scope="module")
def dense_grid_frame(
    curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """Build dense example frame across all DST and gap-adjacent test dates."""
    intensity_df, weather_df, mix_df = curated_sample_data
    issue_times = [issue_time_for_date(d) for d in DENSE_GRID_TEST_DATES]
    return build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=issue_times,
        horizons=range(1, 97),
        require_aux=True,
    )


class TestDenseValueGridOracle:
    """Validate builder values against naive raw store lookups across dense grid."""

    def test_dense_grid_coverage_property(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """The grid's date set must be a superset of the computed coverage property.

        Rule: first seven + last seven issue dates in the frame, plus the four DST
        transition days, plus the gap-adjacent dates already covered. Deleting any
        date from the grid's date set must fail this assertion.
        """
        intensity_df, weather_df, mix_df = curated_sample_data
        canonical_frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            require_aux=True,
        )
        required_dates = derive_required_grid_dates(canonical_frame)
        grid_dates_set = set(DENSE_GRID_TEST_DATES)

        # 1. Assert the grid's date set is a superset of all required dates
        missing_dates = required_dates - grid_dates_set
        assert not missing_dates, (
            f"Dense grid fails coverage rule: missing required dates: {sorted(missing_dates)}"
        )
        assert grid_dates_set.issuperset(required_dates)

        # 2. Deleting any date from list fails suite (demonstrated with tail date)
        issue_dates = sorted(
            {ts.tz_convert(LONDON_TZ).date() for ts in canonical_frame["issue_time"]}
        )
        tail_date = issue_dates[-1].isoformat()
        mutated_grid_set = grid_dates_set - {tail_date}
        assert not mutated_grid_set.issuperset(required_dates), (
            f"Deleting tail date '{tail_date}' must fail the superset coverage check"
        )

    def test_weather_store_boundary_target_hour_absent_yields_null(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """For issue 2026-10-31 at h=96, target_time is 2026-11-02T00:00Z (past store end).

        Assert builder and oracle both yield null and agree for temperature and both
        rolling statistics (temperature_24h_mean, temperature_24h_min).
        """
        intensity_df, weather_df, mix_df = curated_sample_data
        issue_time = issue_time_for_date("2026-10-31")
        h = 96

        frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            issue_times=[issue_time],
            horizons=range(1, 97),
            require_aux=True,
        )
        row_96 = frame[frame["horizon"] == h].iloc[0]

        for col in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
            builder_val = row_96[col]
            oracle_val, source_ts = naive_feature_oracle(
                intensity_df,
                weather_df,
                mix_df,
                issue_time,
                h,
                col,
            )
            assert pd.isna(builder_val), (
                f"Builder must yield null for '{col}' at 2026-10-31 h=96, got {builder_val}"
            )
            assert pd.isna(oracle_val), (
                f"Oracle must yield null for '{col}' at 2026-10-31 h=96, got {oracle_val}"
            )
            assert math.isnan(float(builder_val)) and math.isnan(float(oracle_val))

    def test_tail_target_leak_reproduction_detected_by_dense_grid(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        dense_grid_frame: pd.DataFrame,
    ) -> None:
        """Reproduction of Defect 1: target leak confined to issue dates >= 2026-10-26.

        Asserts that injecting actual_recent_3h = y on tail dates is detected by the
        dense grid oracle check and raises an AssertionError with diagnostic naming
        date, horizon, column, expected and observed values.
        """
        intensity_df, weather_df, mix_df = curated_sample_data
        leak_date_str = "2026-10-26"
        leak_cutoff = issue_time_for_date(leak_date_str)

        leaked_frame = dense_grid_frame.copy()
        mask = leaked_frame["issue_time"] >= leak_cutoff
        assert mask.any(), "Precondition failed: tail leak mask must not be empty"

        orig_vals = leaked_frame.loc[mask, "actual_recent_3h"].copy()
        leaked_frame.loc[mask, "actual_recent_3h"] = leaked_frame.loc[mask, "y"]

        assert_injection_changed(
            orig_vals,
            leaked_frame.loc[mask, "actual_recent_3h"],
            label="tail target leak injection on issue dates >= 2026-10-26",
        )

        day_frame = leaked_frame[leaked_frame["issue_time"] == leak_cutoff]
        with pytest.raises(AssertionError) as exc_info:
            for h in range(1, 97):
                row = day_frame[day_frame["horizon"] == h].iloc[0]
                for feat in FEATURE_COLUMNS:
                    builder_val = row[feat]
                    oracle_val, source_ts = naive_feature_oracle(
                        intensity_df,
                        weather_df,
                        mix_df,
                        leak_cutoff,
                        h,
                        feat,
                    )
                    if pd.isna(builder_val) and pd.isna(oracle_val):
                        continue
                    if pd.isna(builder_val) or pd.isna(oracle_val) or not math.isclose(
                        float(builder_val),
                        float(oracle_val),
                        rel_tol=1e-5,
                        abs_tol=1e-7,
                    ):
                        raise AssertionError(
                            f"Dense grid mismatch for column '{feat}' at issue date "
                            f"{leak_date_str}, horizon {h}:\n"
                            f"  issue_date:       {leak_date_str}\n"
                            f"  horizon:          {h}\n"
                            f"  column:           {feat}\n"
                            f"  expected value:   {oracle_val}\n"
                            f"  observed value:   {builder_val}\n"
                            f"  source timestamp: {source_ts}"
                        )

        diag = str(exc_info.value)
        assert "column 'actual_recent_3h'" in diag
        assert f"issue date {leak_date_str}" in diag
        assert "horizon 1" in diag
        assert "expected value" in diag
        assert "observed value" in diag

    @pytest.mark.parametrize(
        "test_date",
        DENSE_GRID_TEST_DATES,
        ids=DENSE_GRID_TEST_DATES,
    )
    def test_dense_grid_matches_naive_store_oracle(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        dense_grid_frame: pd.DataFrame,
        test_date: str,
    ) -> None:
        """Assert every declared feature across all 96 horizons equals naive store oracle."""
        intensity_df, weather_df, mix_df = curated_sample_data
        issue_time = issue_time_for_date(test_date)

        day_frame = dense_grid_frame[dense_grid_frame["issue_time"] == issue_time]
        assert len(day_frame) == 96

        for h in range(1, 97):
            row = day_frame[day_frame["horizon"] == h].iloc[0]
            for feat in FEATURE_COLUMNS:
                builder_val = row[feat]
                oracle_val, source_ts = naive_feature_oracle(
                    intensity_df,
                    weather_df,
                    mix_df,
                    issue_time,
                    h,
                    feat,
                )
                if pd.isna(builder_val) and pd.isna(oracle_val):
                    continue
                if pd.isna(builder_val) or pd.isna(oracle_val) or not math.isclose(
                    float(builder_val),
                    float(oracle_val),
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                ):
                    raise AssertionError(
                        f"Dense grid mismatch for column '{feat}' at issue date {test_date}, "
                        f"horizon {h}:\n"
                        f"  issue_date:       {test_date}\n"
                        f"  horizon:          {h}\n"
                        f"  column:           {feat}\n"
                        f"  expected value:   {oracle_val}\n"
                        f"  observed value:   {builder_val}\n"
                        f"  source timestamp: {source_ts}"
                    )


class TestAssertInjectionChangedHelper:
    """Verify that assert_injection_changed enforces structural proof of mutation."""

    def test_assert_injection_changed_raises_on_equal_scalars(self) -> None:
        with pytest.raises(AssertionError, match="scalar equal"):
            assert_injection_changed(42, 42, label="scalar equal")
        with pytest.raises(AssertionError, match="both None"):
            assert_injection_changed(None, None, label="both None")
        with pytest.raises(AssertionError, match="both NaN"):
            assert_injection_changed(float("nan"), np.nan, label="both NaN")
        with pytest.raises(AssertionError, match="pd.NA and None"):
            assert_injection_changed(pd.NA, None, label="pd.NA and None")

    def test_assert_injection_changed_passes_on_different_scalars(self) -> None:
        assert_injection_changed(42, 43, label="different ints")
        assert_injection_changed(None, 42, label="None vs int")
        assert_injection_changed(np.nan, 1.0, label="nan vs float")
        assert_injection_changed(True, 1, label="bool vs int")

    def test_assert_injection_changed_raises_on_equal_series_and_frames(self) -> None:
        s1 = pd.Series([1.0, 2.0])
        s2 = pd.Series([1.0, 2.0])
        with pytest.raises(AssertionError, match="differing elements: 0"):
            assert_injection_changed(s1, s2, label="equal series")

        s_null1 = pd.Series([np.nan, np.nan])
        s_null2 = pd.Series([np.nan, np.nan])
        with pytest.raises(AssertionError, match="differing elements: 0"):
            assert_injection_changed(s_null1, s_null2, label="all-null series")

        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(AssertionError, match="differing elements: 0"):
            assert_injection_changed(df1, df2, label="equal df")

        df_null1 = pd.DataFrame({"a": [np.nan, np.nan]})
        df_null2 = pd.DataFrame({"a": [np.nan, np.nan]})
        with pytest.raises(AssertionError, match="differing elements: 0"):
            assert_injection_changed(df_null1, df_null2, label="all-null df")

    def test_assert_injection_changed_passes_on_mutated_series_and_frames(self) -> None:
        s1 = pd.Series([1.0, 2.0])
        s2 = pd.Series([1.0, 3.0])
        assert_injection_changed(s1, s2, label="mutated series")

        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"a": [1.0, 3.0]})
        assert_injection_changed(df1, df2, label="mutated df")
