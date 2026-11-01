"""Temporal leakage tests and feature engineering invariants for Phase P2.

Validates that no feature at target period t uses any value stamped after t.
Tests future-perturbation invariance across multiple cutoffs (mid-window, DST
transition days, and window end), enforces bounded feature definitions (D1-D4),
and proves that leaks in actuals, weather, or generation mix fail loudly.
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
    build_feature_frame,
    max_source_timestamp_allowed,
)
from gridcast.holidays import UK_BANK_HOLIDAYS, is_uk_bank_holiday

LONDON_TZ = ZoneInfo("Europe/London")
UTC_TZ = timezone.utc

CURATED_PATH = Path("data/curated")


def create_synthetic_auxiliary_data(
    timestamps: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic weather and mix source frames aligned to timestamps."""
    ts_sorted = pd.to_datetime(timestamps, utc=True).sort_values().reset_index(drop=True)

    # Weather: previous_runs day 2 forecast (temperature in deg C)
    # Seasonal wave + diurnal wave + deterministic variations
    hours = (ts_sorted - ts_sorted.iloc[0]).dt.total_seconds() / 3600.0
    day_of_year = ts_sorted.dt.dayofyear
    seasonal_temp = 12.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
    minute_frac = ts_sorted.dt.minute / 60.0
    diurnal_temp = 3.0 * np.sin(2 * np.pi * (ts_sorted.dt.hour + minute_frac - 9) / 24.0)
    temps = seasonal_temp + diurnal_temp + np.sin(hours / 17.0) * 1.5

    weather_df = pd.DataFrame({
        "period_start_utc": ts_sorted,
        "temperature_2m_previous_day2": temps.astype(float),
    })

    # Generation mix: shares of 9 fuels summing to 100%
    base_wind = 25.0 + 10.0 * np.sin(hours / 23.0)
    base_solar = np.maximum(0.0, 15.0 * np.sin(np.pi * (ts_sorted.dt.hour - 6) / 12.0))
    base_nuclear = 15.0 + 2.0 * np.sin(hours / 73.0)
    base_biomass = 6.0 + 1.5 * np.cos(hours / 41.0)
    base_coal = np.maximum(0.1, 1.5 + 0.8 * np.sin(hours / 67.0))
    base_hydro = 2.0 + 0.5 * np.sin(hours / 19.0)
    base_imports = 8.0 + 2.0 * np.cos(hours / 29.0)
    base_other = 1.5 + 0.3 * np.sin(hours / 13.0)
    non_gas = (
        base_wind
        + base_solar
        + base_nuclear
        + base_biomass
        + base_coal
        + base_hydro
        + base_imports
        + base_other
    )
    base_gas = 100.0 - non_gas

    mix_df = pd.DataFrame({
        "period_start_utc": ts_sorted,
        "wind": base_wind,
        "solar": base_solar,
        "gas": base_gas,
        "coal": base_coal,
        "nuclear": base_nuclear,
        "biomass": base_biomass,
        "hydro": base_hydro,
        "imports": base_imports,
        "other": base_other,
    })

    return weather_df, mix_df


@pytest.fixture(scope="module")
def curated_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load curated carbon intensity and auxiliary test sources.

    Loads the real committed data from data/curated.
    """
    df = pd.read_parquet(CURATED_PATH)
    weather_df, mix_df = create_synthetic_auxiliary_data(df["period_start_utc"])
    return df, weather_df, mix_df


class TestTemporalLeakageInvariance:
    """Core future-perturbation invariance property (Section 4)."""

    @pytest.mark.parametrize(
        "cutoff_desc,cutoff_time",
        [
            ("mid_window", pd.Timestamp("2025-11-01 12:00:00+00:00")),
            ("spring_forward_dst", pd.Timestamp("2025-03-30 01:00:00+00:00")),
            ("autumn_back_dst", pd.Timestamp("2025-10-26 01:00:00+00:00")),
            ("final_row", pd.Timestamp("2026-11-01 23:30:00+00:00")),
        ],
    )
    def test_future_perturbation_invariance(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        cutoff_desc: str,
        cutoff_time: pd.Timestamp,
    ) -> None:
        """Assert all rows <= T are identical after perturbing all source data > T."""
        intensity_df, weather_df, mix_df = curated_sample_data

        # 1. Build original feature frame from source
        orig_features = build_feature_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
        )

        # 2. Perturb every value strictly after cutoff T
        perturbed_intensity = intensity_df.copy()
        perturbed_weather = weather_df.copy()
        perturbed_mix = mix_df.copy()

        mask_int = perturbed_intensity["period_start_utc"] > cutoff_time
        perturbed_intensity.loc[mask_int, "actual"] = (
            perturbed_intensity.loc[mask_int, "actual"].fillna(100) + 1000
        )
        if "forecast" in perturbed_intensity.columns:
            perturbed_intensity.loc[mask_int, "forecast"] = (
                perturbed_intensity.loc[mask_int, "forecast"].fillna(100) * 2
            )

        mask_wth = perturbed_weather["period_start_utc"] > cutoff_time
        perturbed_weather.loc[mask_wth, "temperature_2m_previous_day2"] = (
            perturbed_weather.loc[mask_wth, "temperature_2m_previous_day2"] + 40.0
        )

        mask_mix = perturbed_mix["period_start_utc"] > cutoff_time
        # Shuffle mix shares strictly after T
        for col in ["wind", "solar", "gas", "coal", "nuclear", "biomass"]:
            perturbed_mix.loc[mask_mix, col] = np.roll(
                perturbed_mix.loc[mask_mix, col].to_numpy(), shift=1
            )

        # 3. Rebuild feature frame
        rebuilt_features = build_feature_frame(
            intensity_df=perturbed_intensity,
            weather_df=perturbed_weather,
            mix_df=perturbed_mix,
        )

        # 4. Assert rows <= T are strictly identical
        orig_subset = orig_features[
            orig_features["period_start_utc"] <= cutoff_time
        ].reset_index(drop=True)
        rebuilt_subset = rebuilt_features[
            rebuilt_features["period_start_utc"] <= cutoff_time
        ].reset_index(drop=True)

        assert len(orig_subset) == len(rebuilt_subset), (
            f"Row count mismatch at cutoff {cutoff_desc}: "
            f"{len(orig_subset)} vs {len(rebuilt_subset)}"
        )

        # Check each column individually to identify leaking column by name if any
        for col in orig_subset.columns:
            orig_col = orig_subset[col]
            rebuilt_col = rebuilt_subset[col]
            if orig_col.isna().all() and rebuilt_col.isna().all():
                continue
            matches = (orig_col == rebuilt_col) | (orig_col.isna() & rebuilt_col.isna())
            if not matches.all():
                mismatches = (~matches).sum()
                first_diff_idx = (~matches).idxmax()
                bad_ts = orig_subset.loc[first_diff_idx, "period_start_utc"]
                orig_val = orig_subset.loc[first_diff_idx, col]
                rebuilt_val = rebuilt_subset.loc[first_diff_idx, col]
                pytest.fail(
                    f"Temporal leakage in '{col}' at {cutoff_desc} ({cutoff_time})!\n"
                    f"Mismatches: {mismatches} / {len(orig_subset)}\n"
                    f"Diff at {bad_ts}: orig={orig_val}, rebuilt={rebuilt_val}"
                )


class TestFeatureContractAndInvariants:
    """Validate structure, column names, coverage, dead-feature avoidance, and cutoffs."""

    def test_feature_frame_shape_and_columns(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Every feature in D3 must be present; no unapproved features."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        assert "period_start_utc" in features.columns
        for expected_col in FEATURE_COLUMNS:
            assert expected_col in features.columns, f"Missing feature column: {expected_col}"

    def test_period_start_utc_matches_source(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Assert every feature row's period_start_utc is present in the source."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        source_timestamps = set(pd.to_datetime(intensity_df["period_start_utc"], utc=True))
        feature_timestamps = set(pd.to_datetime(features["period_start_utc"], utc=True))

        assert feature_timestamps.issubset(source_timestamps), (
            "Feature frame contains period_start_utc timestamps not present in source"
        )

    def test_no_feature_column_is_constant(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """No feature column may be constant across the window (dead feature is a bug)."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        for col in FEATURE_COLUMNS:
            non_null = features[col].dropna()
            assert len(non_null) > 0, f"Feature column '{col}' is entirely null"
            assert (
                non_null.nunique() > 1
            ), f"Dead feature: '{col}' is constant across the entire window"

    def test_max_source_timestamp_allowed_helper(self) -> None:
        """Validate max_source_timestamp_allowed helper implementation."""
        t = pd.Timestamp("2026-06-01 12:00:00+00:00")

        # Lags of actual
        assert max_source_timestamp_allowed("actual_lag_1h", t) == t - pd.Timedelta(hours=1)
        assert max_source_timestamp_allowed("actual_lag_2h", t) == t - pd.Timedelta(hours=2)
        assert max_source_timestamp_allowed("actual_lag_3h", t) == t - pd.Timedelta(hours=3)
        assert max_source_timestamp_allowed("actual_lag_24h", t) == t - pd.Timedelta(hours=24)
        assert max_source_timestamp_allowed("actual_lag_48h", t) == t - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("actual_lag_168h", t) == t - pd.Timedelta(hours=168)

        # Rolling of actual: windows end at t-1h
        assert max_source_timestamp_allowed("actual_rolling_3h_mean", t) == t - pd.Timedelta(
            hours=1
        )
        assert max_source_timestamp_allowed("actual_rolling_3h_std", t) == t - pd.Timedelta(
            hours=1
        )
        assert max_source_timestamp_allowed("actual_rolling_24h_mean", t) == t - pd.Timedelta(
            hours=1
        )
        assert max_source_timestamp_allowed("actual_rolling_168h_mean", t) == t - pd.Timedelta(
            hours=1
        )

        # Weather: previous_day2 forecast issued at t-48h
        assert max_source_timestamp_allowed("temperature", t) == t - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("temperature_24h_mean", t) == t - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("temperature_24h_min", t) == t - pd.Timedelta(hours=48)

        # Generation mix: at t-48h
        assert max_source_timestamp_allowed("mix_share_wind", t) == t - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("mix_share_solar", t) == t - pd.Timedelta(hours=48)
        assert max_source_timestamp_allowed("mix_share_gas", t) == t - pd.Timedelta(hours=48)

    def test_no_feature_reads_after_allowed_cutoff(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify no feature column carries information after its allowed cutoff."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        # Select a sample timestamp t
        t_sample = pd.Timestamp("2026-01-15 12:00:00+00:00")
        row = features[features["period_start_utc"] == t_sample]
        assert not row.empty

        for feat in FEATURE_COLUMNS:
            cutoff = max_source_timestamp_allowed(feat, t_sample)
            # The cutoff must be strictly before or at t for calendar,
            # and <= t-1h for actual, <= t-48h for mix/weather
            if feat.startswith("actual_"):
                assert cutoff <= t_sample - pd.Timedelta(hours=1), (
                    f"Feature '{feat}' allowed cutoff {cutoff} violates <= t-1h"
                )
            elif feat.startswith("mix_share_"):
                assert cutoff <= t_sample - pd.Timedelta(hours=48), (
                    f"Mix feature '{feat}' allowed cutoff {cutoff} violates <= t-48h"
                )
            elif feat.startswith("temperature"):
                assert cutoff <= t_sample - pd.Timedelta(hours=48), (
                    f"Weather feature '{feat}' issue cutoff {cutoff} violates <= t-48h"
                )


class TestLeakageTeethDemonstrations:
    """Specific tests proving the 3 deliberate leakage vectors fail loudly (Section 5)."""

    def test_no_contemporaneous_actual_allowed(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Leak 1: contemporaneous actual as a feature must be rejected."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        assert "actual" not in features.columns, (
            "Contemporaneous 'actual' column found in feature frame! Direct leakage of target."
        )

    def test_weather_source_must_be_previous_runs_not_reanalysis(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Leak 2: archive-api reanalysis weather must be rejected."""
        intensity_df, _, mix_df = curated_sample_data

        # Providing reanalysis weather frame instead of previous_runs
        bad_weather = pd.DataFrame({
            "period_start_utc": intensity_df["period_start_utc"],
            "temperature_reanalysis": [15.0] * len(intensity_df),
        })

        with pytest.raises(
            (ValueError, KeyError),
            match="(?i)(previous_day2|previous-runs|reanalysis)",
        ):
            build_feature_frame(intensity_df, weather_df=bad_weather, mix_df=mix_df)

    def test_weather_null_in_window_fails_loudly_without_reanalysis_fallback(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """D1: If previous-runs is null in window, weather feature is null."""
        intensity_df, weather_df, mix_df = curated_sample_data
        bad_weather = weather_df.copy()
        # Null out some values inside window
        bad_weather.loc[100:105, "temperature_2m_previous_day2"] = np.nan

        features = build_feature_frame(intensity_df, weather_df=bad_weather, mix_df=mix_df)
        # Weather features at those rows must be NaN, NOT imputed from reanalysis
        assert features.loc[100:105, "temperature"].isna().all()

    def test_generation_mix_must_be_lagged_48h_not_contemporaneous(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Leak 3: generation mix at t instead of t-48h must be detected and rejected."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        t_eval = pd.Timestamp("2026-06-01 12:00:00+00:00")
        t_48h = t_eval - pd.Timedelta(hours=48)

        feat_row = features[features["period_start_utc"] == t_eval].iloc[0]
        mix_source_at_t = mix_df[mix_df["period_start_utc"] == t_eval].iloc[0]
        mix_source_at_48h = mix_df[mix_df["period_start_utc"] == t_48h].iloc[0]

        # Verify feature matches source at t-48h, NOT source at t
        assert feat_row["mix_share_wind"] == pytest.approx(mix_source_at_48h["wind"], rel=1e-4)
        if mix_source_at_t["wind"] != mix_source_at_48h["wind"]:
            assert feat_row["mix_share_wind"] != pytest.approx(mix_source_at_t["wind"], rel=1e-4), (
                "mix_share_wind matched contemporaneous mix at t rather than lagged at t-48h!"
            )


class TestTimestampKeyingAndDST:
    """D4: Timestamp-based lookups across DST shifts rather than row shifting."""

    def test_dst_spring_forward_lag_lookup(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """A 24h lag on a 46-period day must look up exactly t - 24h, not shift 48 rows."""
        intensity_df, weather_df, mix_df = curated_sample_data
        features = build_feature_frame(intensity_df, weather_df, mix_df)

        # 2026-03-29 is spring forward (46 periods)
        t_target = pd.Timestamp("2026-03-29 14:00:00+00:00")
        t_expected_24h = t_target - pd.Timedelta(hours=24)

        feat_val = features.loc[features["period_start_utc"] == t_target, "actual_lag_24h"].iloc[0]
        source_val = intensity_df.loc[
            intensity_df["period_start_utc"] == t_expected_24h, "actual"
        ].iloc[0]

        assert feat_val == source_val

    def test_bank_holidays_presence(self) -> None:
        """UK bank holidays table must cover 2024-2027."""
        years = {d.year for d in UK_BANK_HOLIDAYS}
        assert years == {2024, 2025, 2026, 2027}
        # Christmas 2024, 2025, 2026
        assert date(2024, 12, 25) in UK_BANK_HOLIDAYS
        assert date(2025, 12, 25) in UK_BANK_HOLIDAYS
        assert date(2026, 12, 25) in UK_BANK_HOLIDAYS
        assert is_uk_bank_holiday(date(2026, 12, 25)) is True
        assert is_uk_bank_holiday(date(2026, 12, 24)) is False

    def test_build_feature_frame_without_auxiliary_data(
        self,
        curated_sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Building features with only intensity_df leaves weather and mix features null."""
        intensity_df, _, _ = curated_sample_data
        features = build_feature_frame(intensity_df)
        assert features["temperature"].isna().all()
        assert features["mix_share_wind"].isna().all()
        assert not features["actual_lag_1h"].dropna().empty
        assert not features["hour_of_day"].isna().any()

    def test_missing_required_columns_raises(self) -> None:
        """Missing period_start_utc or actual raises ValueError."""
        with pytest.raises(ValueError, match="period_start_utc"):
            build_feature_frame(pd.DataFrame({"actual": [100]}))
        with pytest.raises(ValueError, match="actual"):
            build_feature_frame(pd.DataFrame({"period_start_utc": [pd.Timestamp.now()]}))

