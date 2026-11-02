"""Feature extraction and temporal leakage prevention for Phase P2.

Core Properties:
- The product forecasts 48 hours ahead from a single issue time.
- The example frame is keyed by (issue_time, horizon), where
  target_time = issue_time + horizon * 30min for horizon in 1..96 (DEFAULT_HORIZONS).
- Horizon contract (ADR-011): No feature at issue_time may read any source value
  stamped after issue_time.
- All lags and rolling statistics are keyed and resolved by timestamp lookup,
  never by row position, preserving temporal correctness across DST transitions.
- Weather is Open-Meteo previous-runs day-2 forecast at target_time (issued 48 hours
  prior at target_time - 48h <= issue_time), mapped via containing hour target_time.floor("h").
- Generation mix fuel shares are lagged to issue_time - 48h.
- Target column 'y' is the realized actual at target_time (nullable for unsettled targets).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_HORIZONS: Final[range] = range(1, 97)

LONDON_TZ: Final[ZoneInfo] = ZoneInfo("Europe/London")

CURATED_DIR_DEFAULT: Final[Path] = Path("data/curated")
WEATHER_DIR_DEFAULT: Final[Path] = Path("data/weather")
MIX_DIR_DEFAULT: Final[Path] = Path("data/mix")
AUDIT_DIR_DEFAULT: Final[Path] = Path("data/audit")

MIX_FUELS: Final[list[str]] = [
    "wind",
    "solar",
    "gas",
    "coal",
    "nuclear",
    "biomass",
]

FEATURE_COLUMNS: Final[list[str]] = [
    # Forecast horizon (step in half-hours ahead)
    "horizon",
    # Lags of actual relative to issue_time
    "actual_lag_1h",
    "actual_lag_2h",
    "actual_lag_3h",
    "actual_lag_24h",
    "actual_lag_48h",
    "actual_lag_168h",
    # Rolling stats of actual (windows ending strictly at issue_time - 30min)
    "actual_rolling_3h_mean",
    "actual_rolling_3h_std",
    "actual_rolling_24h_mean",
    "actual_rolling_24h_std",
    "actual_rolling_168h_mean",
    "actual_rolling_168h_std",
    # Calendar features at target_time
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_uk_bank_holiday",
    "sin_hour",
    "cos_hour",
    "sin_day_of_year",
    "cos_day_of_year",
    # Weather features: previous-runs day-2 forecast at target_time (issued <= issue_time)
    "temperature",
    "temperature_24h_mean",
    "temperature_24h_min",
    # Generation mix features (lagged to issue_time - 48h)
    "mix_share_wind",
    "mix_share_solar",
    "mix_share_gas",
    "mix_share_coal",
    "mix_share_nuclear",
    "mix_share_biomass",
]

__all__: list[str] = [
    "DEFAULT_HORIZONS",
    "FEATURE_COLUMNS",
    "MIX_FUELS",
    "build_example_frame",
    "load_feature_sources",
    "max_source_timestamp_allowed",
]


def load_feature_sources(
    curated_dir: Path | str = CURATED_DIR_DEFAULT,
    weather_dir: Path | str = WEATHER_DIR_DEFAULT,
    mix_dir: Path | str = MIX_DIR_DEFAULT,
    audit_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load curated carbon intensity, weather, and generation mix DataFrames.

    Args:
        curated_dir: Directory containing curated carbon intensity partitions.
        weather_dir: Directory containing weather partitions.
        mix_dir: Directory containing generation mix partitions.
        audit_dir: Directory containing audit files (such as source-gaps.csv).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (intensity_df, weather_df, mix_df)

    Raises:
        FileNotFoundError: If any directory does not exist or contains no parquet partitions.
        ValueError: If any dataset is empty or if previous-runs weather contains nulls
                    for periods at or after 2024-11-01.
    """
    c_path = Path(curated_dir)
    w_path = Path(weather_dir)
    m_path = Path(mix_dir)

    for p, name in [
        (c_path, "curated intensity"),
        (w_path, "weather"),
        (m_path, "generation mix"),
    ]:
        if not p.exists() or not any(p.glob("date=*.parquet")):
            raise FileNotFoundError(
                f"Source store '{name}' at '{p}' is missing or has no partition files."
            )

    intensity_df = pd.read_parquet(c_path)
    weather_df = pd.read_parquet(w_path)
    mix_df = pd.read_parquet(m_path)

    if intensity_df.empty:
        raise ValueError(f"Curated intensity dataset at '{c_path}' is empty.")
    if weather_df.empty:
        raise ValueError(f"Weather dataset at '{w_path}' is empty.")
    if mix_df.empty:
        raise ValueError(f"Generation mix dataset at '{m_path}' is empty.")

    w_ts = pd.to_datetime(weather_df["period_start_utc"], utc=True)
    post_cutoff = w_ts >= pd.Timestamp("2024-11-01 00:00:00+00:00")
    if "temperature_2m_previous_day2" in weather_df.columns:
        null_mask = post_cutoff & weather_df["temperature_2m_previous_day2"].isna()
        if null_mask.any():
            bad_date = weather_df.loc[null_mask, "period_start_utc"].iloc[0]
            raise ValueError(
                f"Previous-runs weather has null value at or after 2024-11-01 (date: {bad_date}). "
                "Silent reanalysis fallback is strictly prohibited."
            )

    # Report source gaps if recorded
    a_path = (
        Path(audit_dir)
        if audit_dir is not None
        else (
            c_path.parent / "audit"
            if (c_path.parent / "audit").exists()
            else AUDIT_DIR_DEFAULT
        )
    )
    sg_path = a_path / "source-gaps.csv"
    if sg_path.exists():
        try:
            sg_df = pd.read_csv(sg_path)
            if not sg_df.empty and "store" in sg_df.columns:
                mix_g = sg_df[sg_df["store"] == "mix"]
                weather_g = sg_df[sg_df["store"] == "weather"]
                if not mix_g.empty or not weather_g.empty:
                    print("\nFeature sources coverage audit:")
                    if not mix_g.empty:
                        reasons = ", ".join(mix_g["reason"].unique())
                        print(f"  mix source gaps:     {len(mix_g)} periods affected ({reasons})")
                    if not weather_g.empty:
                        reasons = ", ".join(weather_g["reason"].unique())
                        print(f"  weather source gaps: {len(weather_g)} hours affected ({reasons})")
        except Exception:
            pass

    return intensity_df, weather_df, mix_df


def max_source_timestamp_allowed(
    feature_name: str,
    issue_time: pd.Timestamp,
    target_time: pd.Timestamp,
) -> pd.Timestamp:
    """Return the latest source timestamp a feature is permitted to read.

    Enforces the horizon contract (ADR-011):
    - Lags of actual: issue_time - N hours
    - Rolling stats of actual: issue_time - 30min
      (all rolling windows strictly end at issue_time - 30min)
    - Generation mix: issue_time - 48h
      (published after period; observation lagged to issue time)
    - Weather: target_time - 48h
      (previous-runs day 2 forecast issued at target_time - 48h <= issue_time)
    - Calendar / Horizon: issue_time (deterministic coordinates of target period)
    """
    issue_ts = pd.Timestamp(issue_time)
    target_ts = pd.Timestamp(target_time)

    if feature_name == "actual_lag_1h":
        return issue_ts - pd.Timedelta(hours=1)
    if feature_name == "actual_lag_2h":
        return issue_ts - pd.Timedelta(hours=2)
    if feature_name == "actual_lag_3h":
        return issue_ts - pd.Timedelta(hours=3)
    if feature_name == "actual_lag_24h":
        return issue_ts - pd.Timedelta(hours=24)
    if feature_name == "actual_lag_48h":
        return issue_ts - pd.Timedelta(hours=48)
    if feature_name == "actual_lag_168h":
        return issue_ts - pd.Timedelta(hours=168)

    if feature_name.startswith("actual_rolling_"):
        return issue_ts - pd.Timedelta(minutes=30)

    if feature_name.startswith("mix_share_"):
        return issue_ts - pd.Timedelta(hours=48)

    if feature_name in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
        return target_ts - pd.Timedelta(hours=48)

    if feature_name == "horizon" or feature_name in (
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "is_uk_bank_holiday",
        "sin_hour",
        "cos_hour",
        "sin_day_of_year",
        "cos_day_of_year",
    ):
        return issue_ts

    raise ValueError(f"Unknown feature: {feature_name}")


def build_example_frame(
    intensity_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    mix_df: pd.DataFrame | None = None,
    issue_times: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
    horizons: Sequence[int] | range = DEFAULT_HORIZONS,
    require_aux: bool = True,
) -> pd.DataFrame:
    """Build the example frame keyed by (issue_time, horizon) without leakage.

    Args:
        intensity_df: Curated carbon intensity DataFrame containing 'period_start_utc'
                      and 'actual'.
        weather_df: Weather DataFrame containing 'period_start_utc' and
                    'temperature_2m_previous_day2'. Reanalysis weather is strictly rejected.
        mix_df: Generation mix DataFrame containing 'period_start_utc' and fuel shares
                ('wind', 'solar', 'gas', 'coal', 'nuclear', 'biomass').
        issue_times: Schedule of issue timestamps. Defaults to one per day at 00:00 UTC
                     across the intensity window.
        horizons: Sequence of horizon steps in half-hours ahead (default: 1..96).
        require_aux: If True, raises ValueError if any weather or mix column is entirely null
                     or if any previous-runs temperature is null at or after 2024-11-01.

    Returns:
        pd.DataFrame with keys ('issue_time', 'horizon', 'target_time'), target 'y',
        and FEATURE_COLUMNS.
    """
    raise NotImplementedError("build_example_frame contract defined; implementation pending.")
