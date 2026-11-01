"""Feature extraction and temporal leakage prevention for Phase P2.

Core Property:
No feature at target period t may use any value stamped after t.
All lags and rolling statistics are keyed and resolved by timestamp lookup,
never by row position, preserving temporal correctness across DST transitions.

Bounded Feature Set (ADR-006 / D3):
- Lags of actual: t-1h, t-2h, t-3h, t-24h, t-48h, t-168h
- Rolling of actual: mean and std over 3h, 24h, 168h (windows ending at t-1h)
- Calendar: hour of day, day of week, month, is-weekend, is-UK-bank-holiday,
  sin/cos encodings of hour and day-of-year
- Weather: temperature from previous-runs day-2 forecast at t, plus 24h mean and min ending at t
- Generation mix: shares of wind, solar, gas, coal, nuclear, biomass at t-48h
"""

from __future__ import annotations

from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gridcast.holidays import UK_BANK_HOLIDAYS

# Required to remain empty list for smoke test stub validation
__all__: list[str] = []

LONDON_TZ: Final[ZoneInfo] = ZoneInfo("Europe/London")

FEATURE_COLUMNS: Final[list[str]] = [
    # Lags of actual
    "actual_lag_1h",
    "actual_lag_2h",
    "actual_lag_3h",
    "actual_lag_24h",
    "actual_lag_48h",
    "actual_lag_168h",
    # Rolling stats of actual (ending at t-1h)
    "actual_rolling_3h_mean",
    "actual_rolling_3h_std",
    "actual_rolling_24h_mean",
    "actual_rolling_24h_std",
    "actual_rolling_168h_mean",
    "actual_rolling_168h_std",
    # Calendar features
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_uk_bank_holiday",
    "sin_hour",
    "cos_hour",
    "sin_day_of_year",
    "cos_day_of_year",
    # Weather features (D1 previous-runs day 2)
    "temperature",
    "temperature_24h_mean",
    "temperature_24h_min",
    # Generation mix features (lagged to t-48h)
    "mix_share_wind",
    "mix_share_solar",
    "mix_share_gas",
    "mix_share_coal",
    "mix_share_nuclear",
    "mix_share_biomass",
]

MIX_FUELS: Final[list[str]] = [
    "wind",
    "solar",
    "gas",
    "coal",
    "nuclear",
    "biomass",
]


def max_source_timestamp_allowed(feature_name: str, target_time: pd.Timestamp) -> pd.Timestamp:
    """Return the latest source timestamp a feature was permitted to read for target_time.

    Enforces the leakage barrier:
    - Actual lags: t - N hours
    - Actual rolling stats: t - 1 hour (all rolling windows strictly end at t - 1h)
    - Weather: t - 48 hours (previous-runs day 2 forecast issued at t-48h)
    - Generation mix: t - 48 hours (published after period; observation lagged to issue time)
    - Calendar: target_time (calendar position of the target period itself)
    """
    ts = pd.Timestamp(target_time)

    # Lags of actual
    if feature_name == "actual_lag_1h":
        return ts - pd.Timedelta(hours=1)
    if feature_name == "actual_lag_2h":
        return ts - pd.Timedelta(hours=2)
    if feature_name == "actual_lag_3h":
        return ts - pd.Timedelta(hours=3)
    if feature_name == "actual_lag_24h":
        return ts - pd.Timedelta(hours=24)
    if feature_name == "actual_lag_48h":
        return ts - pd.Timedelta(hours=48)
    if feature_name == "actual_lag_168h":
        return ts - pd.Timedelta(hours=168)

    # Rolling statistics of actual: each window ends at t - 1h
    if feature_name.startswith("actual_rolling_"):
        return ts - pd.Timedelta(hours=1)

    # Generation mix: lagged to t - 48h
    if feature_name.startswith("mix_share_"):
        return ts - pd.Timedelta(hours=48)

    # Weather: previous-runs day-2 forecast issued 48 hours earlier
    if feature_name in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
        return ts - pd.Timedelta(hours=48)

    # Calendar features
    return ts


def build_feature_frame(
    intensity_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    mix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the bounded feature frame from source DataFrames without temporal leakage.

    Args:
        intensity_df: Curated carbon intensity DataFrame containing 'period_start_utc'
                      and 'actual'.
        weather_df: Weather DataFrame containing 'period_start_utc' and
                    'temperature_2m_previous_day2'. Reanalysis weather is strictly rejected.
        mix_df: Generation mix DataFrame containing 'period_start_utc' and fuel shares
                ('wind', 'solar', 'gas', 'coal', 'nuclear', 'biomass').

    Returns:
        pd.DataFrame sorted by 'period_start_utc' with exactly the bounded feature columns.
    """
    if "period_start_utc" not in intensity_df.columns:
        raise ValueError("intensity_df must contain 'period_start_utc'")
    if "actual" not in intensity_df.columns:
        raise ValueError("intensity_df must contain 'actual'")

    # Sort source strictly by period_start_utc
    sorted_intensity = intensity_df.sort_values("period_start_utc").reset_index(drop=True)
    timestamps = pd.to_datetime(sorted_intensity["period_start_utc"], utc=True)

    # Keyed lookup mapping for actuals by timestamp (never row-shifted)
    actual_series = pd.Series(
        sorted_intensity["actual"].to_numpy(dtype=float),
        index=timestamps,
    )

    result = pd.DataFrame({"period_start_utc": timestamps})

    # 1. Lags of actual
    for lag_h in [1, 2, 3, 24, 48, 168]:
        lookup_ts = timestamps - pd.Timedelta(hours=lag_h)
        result[f"actual_lag_{lag_h}h"] = lookup_ts.map(actual_series).to_numpy(dtype=float)

    # 2. Rolling statistics of actual (windows ending strictly at t-1h)
    t_minus_1h = timestamps - pd.Timedelta(hours=1)

    # Precompute rolling series on actual_series
    roll_3h = actual_series.rolling("3h", min_periods=1)
    roll_3h_mean = roll_3h.mean()
    roll_3h_std = roll_3h.std()

    roll_24h = actual_series.rolling("24h", min_periods=1)
    roll_24h_mean = roll_24h.mean()
    roll_24h_std = roll_24h.std()

    roll_168h = actual_series.rolling("168h", min_periods=1)
    roll_168h_mean = roll_168h.mean()
    roll_168h_std = roll_168h.std()

    result["actual_rolling_3h_mean"] = t_minus_1h.map(roll_3h_mean).to_numpy(dtype=float)
    result["actual_rolling_3h_std"] = t_minus_1h.map(roll_3h_std).to_numpy(dtype=float)
    result["actual_rolling_24h_mean"] = t_minus_1h.map(roll_24h_mean).to_numpy(dtype=float)
    result["actual_rolling_24h_std"] = t_minus_1h.map(roll_24h_std).to_numpy(dtype=float)
    result["actual_rolling_168h_mean"] = t_minus_1h.map(roll_168h_mean).to_numpy(dtype=float)
    result["actual_rolling_168h_std"] = t_minus_1h.map(roll_168h_std).to_numpy(dtype=float)

    # 3. Calendar features (Europe/London diurnal clock and UK bank holidays)
    london_ts = timestamps.dt.tz_convert(LONDON_TZ)
    hour = london_ts.dt.hour
    minute = london_ts.dt.minute
    day_of_week = london_ts.dt.dayofweek
    month = london_ts.dt.month
    day_of_year = london_ts.dt.dayofyear
    cal_date = london_ts.dt.date

    result["hour_of_day"] = hour.to_numpy(dtype=int)
    result["day_of_week"] = day_of_week.to_numpy(dtype=int)
    result["month"] = month.to_numpy(dtype=int)
    result["is_weekend"] = (day_of_week >= 5).astype(int).to_numpy()
    result["is_uk_bank_holiday"] = cal_date.isin(UK_BANK_HOLIDAYS).astype(int).to_numpy()

    hour_continuous = hour.to_numpy(dtype=float) + minute.to_numpy(dtype=float) / 60.0
    result["sin_hour"] = np.sin(2.0 * np.pi * hour_continuous / 24.0)
    result["cos_hour"] = np.cos(2.0 * np.pi * hour_continuous / 24.0)

    doy_continuous = day_of_year.to_numpy(dtype=float) - 1.0
    result["sin_day_of_year"] = np.sin(2.0 * np.pi * doy_continuous / 365.25)
    result["cos_day_of_year"] = np.cos(2.0 * np.pi * doy_continuous / 365.25)

    # 4. Weather features (D1 — previous-runs forecast issued 48 hours earlier)
    # Check for invalid reanalysis sources
    w_source = weather_df if weather_df is not None else sorted_intensity

    # Reject archive-api reanalysis loudly
    reanalysis_cols = [
        c for c in w_source.columns if "reanalysis" in c.lower() or "archive" in c.lower()
    ]
    if reanalysis_cols:
        raise ValueError(
            f"Weather from archive-api reanalysis is strictly forbidden (found {reanalysis_cols}). "
            "Must use Open-Meteo previous-runs day 2 forecast (temperature_2m_previous_day2)."
        )

    temp_col = None
    if "temperature_2m_previous_day2" in w_source.columns:
        temp_col = "temperature_2m_previous_day2"
    elif "temperature" in w_source.columns and "temperature" in getattr(
        w_source, "attrs", {}
    ).get("source", ""):
        temp_col = "temperature"

    if temp_col is not None:
        w_ts = pd.to_datetime(w_source["period_start_utc"], utc=True)
        w_series = pd.Series(
            w_source[temp_col].to_numpy(dtype=float),
            index=w_ts,
        ).sort_index()

        # Temperature at t from previous-runs day 2
        result["temperature"] = timestamps.map(w_series).to_numpy(dtype=float)

        # 24h mean and 24h min of D1 temperatures ending at t
        w_roll_24h = w_series.rolling("24h", min_periods=1)
        w_24h_mean = w_roll_24h.mean()
        w_24h_min = w_roll_24h.min()

        result["temperature_24h_mean"] = timestamps.map(w_24h_mean).to_numpy(dtype=float)
        result["temperature_24h_min"] = timestamps.map(w_24h_min).to_numpy(dtype=float)
    else:
        # If no previous-runs weather provided, weather features remain null
        result["temperature"] = np.nan
        result["temperature_24h_mean"] = np.nan
        result["temperature_24h_min"] = np.nan

    # 5. Generation mix features (D2 — lagged to t-48h)
    m_source = mix_df if mix_df is not None else sorted_intensity
    t_minus_48h = timestamps - pd.Timedelta(hours=48)

    for fuel in MIX_FUELS:
        feature_name = f"mix_share_{fuel}"
        if fuel in m_source.columns:
            m_ts = pd.to_datetime(m_source["period_start_utc"], utc=True)
            fuel_series = pd.Series(
                m_source[fuel].to_numpy(dtype=float),
                index=m_ts,
            )
            result[feature_name] = t_minus_48h.map(fuel_series).to_numpy(dtype=float)
        elif feature_name in m_source.columns:
            m_ts = pd.to_datetime(m_source["period_start_utc"], utc=True)
            fuel_series = pd.Series(
                m_source[feature_name].to_numpy(dtype=float),
                index=m_ts,
            )
            result[feature_name] = t_minus_48h.map(fuel_series).to_numpy(dtype=float)
        else:
            result[feature_name] = np.nan

    # Enforce exact column order: period_start_utc + FEATURE_COLUMNS
    final_cols = ["period_start_utc"] + FEATURE_COLUMNS
    return result[final_cols].copy()
