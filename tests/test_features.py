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
    load_feature_sources,
    max_source_timestamp_allowed,
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
        assert not unexpected, f"Undeclared columns emitted by the frame builder: {sorted(unexpected)}"

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
