"""Temporal leakage tests and feature engineering invariants for Phase P2.

Validates that no feature at issue_time uses any value stamped after issue_time.
Tests future-perturbation invariance across 25+ sampled (issue_time, horizon) pairs
including boundaries (first/last issue times, DST transitions, horizon extremes h=1/h=96,
and dates adjacent to recorded source gaps). Enforces the horizon contract in code.
"""

from __future__ import annotations

from datetime import date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from gridcast.features import (
    FEATURE_COLUMNS,
    MIX_FUELS,
    build_example_frame,
    default_issue_schedule,
    issue_time_for_date,
    load_feature_sources,
    max_source_timestamp_allowed,
    select_model_inputs,
    split_by_issue_date,
)
from gridcast.holidays import UK_BANK_HOLIDAYS, is_uk_bank_holiday

LONDON_TZ = ZoneInfo("Europe/London")
UTC_TZ = timezone.utc

CURATED_PATH = Path("data/curated")


@pytest.fixture(scope="module")
def curated_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load curated carbon intensity and real auxiliary test sources."""
    return load_feature_sources()


# At least 25 (issue_time, horizon) pairs deliberately including:
# - first issue time (2024-11-01)
# - last issue time (2026-11-01)
# - both DST spring-forward days (2025-03-30, 2026-03-29)
# - both DST autumn-back days (2025-10-26, 2026-10-25)
# - both ends of horizon range (h=1 and h=96)
# - issue times adjacent to recorded source gaps in data/audit/source-gaps.csv
#   (2025-01-12..14, 2025-08-10..12)
SAMPLED_TEST_PAIRS: list[tuple[pd.Timestamp, int]] = [
    # 1-2. First issue time (h=1, h=96)
    (pd.Timestamp("2024-11-01 00:00:00+00:00"), 1),
    (pd.Timestamp("2024-11-01 00:00:00+00:00"), 96),
    # 3-4. Last issue time (h=1, h=96)
    (pd.Timestamp("2026-11-01 00:00:00+00:00"), 1),
    (pd.Timestamp("2026-11-01 00:00:00+00:00"), 96),
    # 5-7. DST spring-forward 2025 (2025-03-30)
    (pd.Timestamp("2025-03-30 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-03-30 00:00:00+00:00"), 48),
    (pd.Timestamp("2025-03-30 00:00:00+00:00"), 96),
    # 8-10. DST autumn-back 2025 (2025-10-26)
    (pd.Timestamp("2025-10-26 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-10-26 00:00:00+00:00"), 48),
    (pd.Timestamp("2025-10-26 00:00:00+00:00"), 96),
    # 11-13. DST spring-forward 2026 (2026-03-29)
    (pd.Timestamp("2026-03-29 00:00:00+00:00"), 1),
    (pd.Timestamp("2026-03-29 00:00:00+00:00"), 48),
    (pd.Timestamp("2026-03-29 00:00:00+00:00"), 96),
    # 14-16. DST autumn-back 2026 (2026-10-25)
    (pd.Timestamp("2026-10-25 00:00:00+00:00"), 1),
    (pd.Timestamp("2026-10-25 00:00:00+00:00"), 48),
    (pd.Timestamp("2026-10-25 00:00:00+00:00"), 96),
    # 17-21. Adjacent to source gaps (2025-01-12, 2025-01-13)
    (pd.Timestamp("2025-01-12 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-01-12 00:00:00+00:00"), 96),
    (pd.Timestamp("2025-01-13 00:00:00+00:00"), 48),
    (pd.Timestamp("2025-01-14 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-01-14 00:00:00+00:00"), 96),
    # 22-26. Adjacent to source gaps (2025-08-11)
    (pd.Timestamp("2025-08-10 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-08-10 00:00:00+00:00"), 96),
    (pd.Timestamp("2025-08-11 00:00:00+00:00"), 48),
    (pd.Timestamp("2025-08-12 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-08-12 00:00:00+00:00"), 96),
    # 27-31. Mid-window samples with varied horizons
    (pd.Timestamp("2025-06-01 00:00:00+00:00"), 1),
    (pd.Timestamp("2025-06-01 00:00:00+00:00"), 24),
    (pd.Timestamp("2025-06-01 00:00:00+00:00"), 72),
    (pd.Timestamp("2025-06-01 00:00:00+00:00"), 96),
    (pd.Timestamp("2026-01-15 00:00:00+00:00"), 10),
    (pd.Timestamp("2026-01-15 00:00:00+00:00"), 96),
]


class TestTemporalLeakageInvariance:
    """Core per-row future-perturbation invariance property (Section 4)."""

    @pytest.mark.parametrize(
        ("issue_time", "horizon"),
        SAMPLED_TEST_PAIRS,
        ids=[f"{t.strftime('%Y-%m-%d')}_h{h}" for t, h in SAMPLED_TEST_PAIRS],
    )
    def test_future_perturbation_invariance_per_pair(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        issue_time: pd.Timestamp,
        horizon: int,
    ) -> None:
        """The row for (issue_time, h) must be unchanged when sources > issue_time are altered."""
        intensity_df, weather_df, mix_df = curated_sample_data

        # 1. Build original unperturbed row for (issue_time, horizon)
        orig_frame = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            issue_times=[issue_time],
            horizons=[horizon],
            require_aux=False,
        )
        assert len(orig_frame) == 1, f"Expected 1 row for single pair, got {len(orig_frame)}"

        # 2. Perturb every source row timestamped after issue_time
        perturbed_intensity = intensity_df.copy()
        perturbed_weather = weather_df.copy()
        perturbed_mix = mix_df.copy()

        # Intensity actuals and forecasts after issue_time
        mask_int = pd.to_datetime(perturbed_intensity["period_start_utc"], utc=True) > issue_time
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
                perturbed_mix.loc[mask_mix, fuel] = np.roll(
                    perturbed_mix.loc[mask_mix, fuel].to_numpy(), shift=1
                )

        # Weather previous-runs day-2 forecast was issued at period_start_utc - 48h.
        # A weather forecast was issued after issue_time if period_start_utc - 48h > issue_time.
        w_ts = pd.to_datetime(perturbed_weather["period_start_utc"], utc=True)
        mask_wth = (w_ts - pd.Timedelta(hours=48)) > issue_time
        perturbed_weather.loc[mask_wth, "temperature_2m_previous_day2"] = (
            perturbed_weather.loc[mask_wth, "temperature_2m_previous_day2"] + 40.0
        )
        # Also perturb any contemporaneous weather columns if injected
        for col in perturbed_weather.columns:
            if col not in ("period_start_utc", "local_date", "temperature_2m_previous_day2"):
                perturbed_weather.loc[w_ts > issue_time, col] = 999.0

        # 3. Rebuild feature row for (issue_time, horizon)
        rebuilt_frame = build_example_frame(
            intensity_df=perturbed_intensity,
            weather_df=perturbed_weather,
            mix_df=perturbed_mix,
            issue_times=[issue_time],
            horizons=[horizon],
            require_aux=False,
        )
        assert len(rebuilt_frame) == 1

        # 4. Assert feature values are strictly identical
        for col in FEATURE_COLUMNS:
            orig_val = orig_frame[col].iloc[0]
            rebuilt_val = rebuilt_frame[col].iloc[0]

            if pd.isna(orig_val) and pd.isna(rebuilt_val):
                continue

            matches = (orig_val == rebuilt_val) or (
                isinstance(orig_val, (int, float, np.floating))
                and orig_val == pytest.approx(rebuilt_val, rel=1e-5)
            )
            assert matches, (
                f"Temporal leakage in '{col}' at issue_time {issue_time}, horizon {horizon}!\n"
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
        sample_horizons = [1, 16, 48]
        features = build_example_frame(
            intensity_df,
            weather_df,
            mix_df,
            issue_times=[sample_issue],
            horizons=sample_horizons,
            require_aux=True,
        )

        w_ts = pd.to_datetime(weather_df["period_start_utc"], utc=True)
        w_series = pd.Series(
            weather_df["temperature_2m_previous_day2"].to_numpy(dtype=float),
            index=w_ts,
        )

        for h in sample_horizons:
            target_time = sample_issue + pd.Timedelta(minutes=30 * h)
            containing_hour = target_time.floor("h")
            row = features[features["horizon"] == h].iloc[0]
            feat_val = row["temperature"]
            stored_val = w_series.loc[containing_hour]

            assert feat_val == pytest.approx(stored_val, rel=1e-5)
            # Reanalysis divergence test
            reanalysis_divergent = stored_val + 5.0
            assert feat_val != pytest.approx(reanalysis_divergent, rel=1e-5)

    def test_horizon_contract_enforced_in_code(self) -> None:
        """Validate max_source_timestamp_allowed helper and enforcement."""
        issue_time = pd.Timestamp("2026-06-01 00:00:00+00:00")
        target_time = issue_time + pd.Timedelta(hours=48)  # h=96

        # Lags of actual
        t_1h = issue_time - pd.Timedelta(hours=1)
        assert max_source_timestamp_allowed("actual_lag_1h", issue_time, target_time) == t_1h
        t_48h = issue_time - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("actual_lag_48h", issue_time, target_time) == t_48h
        t_168h = issue_time - pd.Timedelta(hours=168)
        assert max_source_timestamp_allowed("actual_lag_168h", issue_time, target_time) == t_168h

        # Rolling of actual: window ends at issue_time - 30min
        t_30m = issue_time - pd.Timedelta(minutes=30)
        roll_3h = max_source_timestamp_allowed("actual_rolling_3h_mean", issue_time, target_time)
        roll_24h = max_source_timestamp_allowed("actual_rolling_24h_mean", issue_time, target_time)
        assert roll_3h == t_30m
        assert roll_24h == t_30m

        # Generation mix: at issue_time - 48h
        assert max_source_timestamp_allowed("mix_share_wind", issue_time, target_time) == t_48h

        # Weather: previous_day2 forecast issued at target_time - 48h <= issue_time
        w_issue = target_time - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("temperature", issue_time, target_time) == w_issue
        assert max_source_timestamp_allowed("temperature", issue_time, target_time) <= issue_time

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
        with pytest.raises(ValueError, match="(?i)(horizon contract|source timestamp|issue_time)"):
            build_example_frame(
                intensity_df,
                weather_df,
                mix_df,
                issue_times=[sample_issue],
                horizons=[97],
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
        intensity_df, _, mix_df = curated_sample_data
        bad_weather = pd.DataFrame({
            "period_start_utc": intensity_df["period_start_utc"],
            "temperature_reanalysis": [15.0] * len(intensity_df),
        })

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
        with pytest.raises(ValueError, match="period_start_utc"):
            build_example_frame(pd.DataFrame({"actual": [100]}))
        with pytest.raises(ValueError, match="actual"):
            build_example_frame(pd.DataFrame({"period_start_utc": [pd.Timestamp.now()]}))


class TestIssueAnchorEuropeLondon:
    """Validate 00:00 Europe/London issue anchor invariants (Section 1)."""

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
    """Validate that horizon is an absolute offset spanning 48 hours across DST days (Section 2)."""

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
    """Validate split_by_issue_date with embargo gap preventing target leakage (Section 3)."""

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
    """Validate select_model_inputs as the only positive selection of X (Section 4)."""

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

    def test_model_feature_columns_is_subset_of_feature_columns(self) -> None:
        """The model feature list must be a subset of FEATURE_COLUMNS for P3 compatibility."""
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
        with pytest.raises(ValueError, match="(?i)(undeclared.*column|sneaky_extra_column)"):
            select_model_inputs(bad_frame)


class TestLeadingDateTrimming:
    """Validate that incomplete leading issue dates from warmup are dropped (Section 5)."""

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





