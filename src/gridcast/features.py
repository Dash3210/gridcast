"""Feature extraction and temporal leakage prevention for Phase P2.

Core Properties:
- The product forecasts 48 hours ahead from a single issue time.
- The example frame is keyed by (issue_time, horizon), where
  target_time = issue_time + horizon * 30min for horizon in 1..96 (DEFAULT_HORIZONS).
- Horizon contract (ADR-011): No feature at issue_time may read any source value
  stamped after issue_time.
- All lags and rolling statistics are keyed and resolved by timestamp lookup,
  never by row position, preserving temporal correctness across DST transitions.
- Weather lead is per-horizon: short horizons (h <= 48) consume a 24-hour-old forecast
  (temperature_2m_previous_day1), long horizons (h > 48) consume a 48-hour-old forecast
  (temperature_2m_previous_day2). Both satisfy the horizon contract (issued at or before
  issue_time), and that is by design.
- Generation mix fuel shares are lagged to issue_time - 48h.
- Target column 'y' is the realized actual at target_time (nullable for unsettled targets).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gridcast.holidays import UK_BANK_HOLIDAYS

DEFAULT_HORIZONS: Final[range] = range(1, 97)

LONDON_TZ: Final[ZoneInfo] = ZoneInfo("Europe/London")

# Module-level constant: availability lag for actual carbon intensity observations
ACTUAL_AVAILABILITY_LAG: Final[pd.Timedelta] = pd.Timedelta(hours=3)

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
    # Most recent actual available at issue time: the actual for the half hour ending
    # immediately before issue_time - ACTUAL_AVAILABILITY_LAG
    "actual_recent_3h",
    # Historical lags of actual relative to issue_time
    "actual_lag_24h",
    "actual_lag_48h",
    "actual_lag_168h",
    # Rolling stats of actual (windows ending strictly at issue_time - ACTUAL_AVAILABILITY_LAG)
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

MODEL_FEATURE_COLUMNS: Final[list[str]] = list(FEATURE_COLUMNS)

ALLOWED_FRAME_COLUMNS: Final[set[str]] = {
    "issue_time",
    "horizon",
    "target_time",
    "y",
    *FEATURE_COLUMNS,
}

__all__: list[str] = [
    "ACTUAL_AVAILABILITY_LAG",
    "ALLOWED_FRAME_COLUMNS",
    "DEFAULT_HORIZONS",
    "FEATURE_COLUMNS",
    "MIX_FUELS",
    "MODEL_FEATURE_COLUMNS",
    "build_example_frame",
    "default_issue_schedule",
    "issue_time_for_date",
    "load_feature_sources",
    "max_source_timestamp_allowed",
    "select_model_inputs",
    "split_by_issue_date",
]


def issue_time_for_date(d: date | str) -> pd.Timestamp:
    """Return the issue instant (00:00 Europe/London) converted to UTC.

    - In BST the instant is 23:00Z of the previous UTC date.
    - In GMT it is 00:00Z.
    """
    if isinstance(d, str):
        target_d = date.fromisoformat(d)
    elif isinstance(d, (datetime, pd.Timestamp)):
        target_d = d.date()
    else:
        target_d = d
    local_midnight = datetime(target_d.year, target_d.month, target_d.day, 0, 0, tzinfo=LONDON_TZ)
    return pd.Timestamp(local_midnight).tz_convert("UTC")


def default_issue_schedule(start_date: date | str, end_date: date | str) -> pd.DatetimeIndex:
    """Generate daily issue instants at 00:00 Europe/London across date range in UTC."""
    s = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    e = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    return pd.date_range(start=s, end=e, freq="D", tz=LONDON_TZ).tz_convert("UTC")


def split_by_issue_date(
    frame: pd.DataFrame,
    train_end: date | str,
    val_end: date | str,
    embargo_days: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature frame into (train, val, test) by issue_date with embargo boundaries.

    - Splits are cut on issue_time (Europe/London local issue date), never on target_time.
    - At least embargo_days issue dates are dropped at the train/validation boundary
      and at the validation/test boundary (dropped, not assigned to either side).
    - Rows with null y are excluded from every split.
    - No issue date appears in more than one split.

    Args:
        frame: DataFrame containing 'issue_time', 'target_time', and 'y'.
        train_end: End date for the train split (inclusive).
        val_end: End date for the validation split (inclusive).
        embargo_days: Number of issue dates to drop between consecutive splits (default: 4).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if "issue_time" not in frame.columns:
        raise ValueError("frame must contain 'issue_time'")
    if "y" not in frame.columns:
        raise ValueError("frame must contain 'y'")

    t_end = date.fromisoformat(train_end) if isinstance(train_end, str) else train_end
    v_end = date.fromisoformat(val_end) if isinstance(val_end, str) else val_end

    if t_end >= v_end:
        raise ValueError(f"train_end ({t_end}) must be strictly before val_end ({v_end})")

    val_start = t_end + timedelta(days=embargo_days + 1)
    if val_start > v_end:
        raise ValueError(
            f"embargo_days ({embargo_days}) is too large: "
            f"validation start ({val_start}) > val_end ({v_end})"
        )
    test_start = v_end + timedelta(days=embargo_days + 1)

    issue_ts = pd.to_datetime(frame["issue_time"], utc=True)
    issue_dates = issue_ts.dt.tz_convert(LONDON_TZ).dt.date

    train_mask = issue_dates <= t_end
    val_mask = (issue_dates >= val_start) & (issue_dates <= v_end)
    test_mask = issue_dates >= test_start

    # Exclude rows where y is null
    valid_y = frame["y"].notna()

    train = frame[train_mask & valid_y].copy()
    val = frame[val_mask & valid_y].copy()
    test = frame[test_mask & valid_y].copy()

    return train, val, test


def select_model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract model feature matrix X as a positive selection of FEATURE_COLUMNS.

    This is the only supported way to build X. Negative selection (such as
    df.drop(columns=['y'])) is strictly forbidden.

    Args:
        frame: DataFrame produced by build_example_frame or split_by_issue_date.

    Returns:
        pd.DataFrame containing exactly FEATURE_COLUMNS in declared order.

    Raises:
        ValueError: If any declared feature column is missing or if frame carries
                    undeclared columns not in ALLOWED_FRAME_COLUMNS.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing declared feature columns: {missing}")

    extra = set(frame.columns) - ALLOWED_FRAME_COLUMNS
    if extra:
        raise ValueError(f"Frame carries undeclared columns: {sorted(extra)}")

    return frame[FEATURE_COLUMNS].copy()


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
    for col in ("temperature_2m_previous_day1", "temperature_2m_previous_day2"):
        if col in weather_df.columns:
            null_mask = post_cutoff & weather_df[col].isna()
            if null_mask.any():
                bad_date = weather_df.loc[null_mask, "period_start_utc"].iloc[0]
                raise ValueError(
                    f"Previous-runs weather has null {col} at or after 2024-11-01 "
                    f"(date: {bad_date}). Silent reanalysis fallback is strictly prohibited."
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

    if feature_name == "actual_recent_3h":
        return issue_ts - ACTUAL_AVAILABILITY_LAG
    if feature_name == "actual_lag_24h":
        return issue_ts - pd.Timedelta(hours=24)
    if feature_name == "actual_lag_48h":
        return issue_ts - pd.Timedelta(hours=48)
    if feature_name == "actual_lag_168h":
        return issue_ts - pd.Timedelta(hours=168)

    if feature_name.startswith("actual_rolling_"):
        return issue_ts - ACTUAL_AVAILABILITY_LAG

    if feature_name.startswith("mix_share_"):
        return issue_ts - pd.Timedelta(hours=48)

    if feature_name in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
        lead_h = 24 if (target_ts - issue_ts) <= pd.Timedelta(hours=24) else 48
        return target_ts - pd.Timedelta(hours=lead_h)

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
    trim_leading: bool | None = None,
    weather_series_by_horizon: dict[int, str] | None = None,
    actual_availability_lag: pd.Timedelta = ACTUAL_AVAILABILITY_LAG,
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
    if "period_start_utc" not in intensity_df.columns:
        raise ValueError("intensity_df must contain 'period_start_utc'")
    if "actual" not in intensity_df.columns:
        raise ValueError("intensity_df must contain 'actual'")
    if intensity_df.empty:
        raise ValueError("intensity_df is empty")

    sorted_intensity = intensity_df.sort_values("period_start_utc").reset_index(drop=True)
    intensity_ts = pd.to_datetime(sorted_intensity["period_start_utc"], utc=True)

    # Actuals indexed by timestamp for timestamp lookup
    actual_series = pd.Series(
        sorted_intensity["actual"].to_numpy(dtype=float),
        index=intensity_ts,
    ).sort_index()

    # 1. Resolve issue times: anchored to 00:00 Europe/London for each local date (ADR-012)
    if issue_times is None:
        min_date = intensity_ts.min().tz_convert(LONDON_TZ).date()
        max_date = intensity_ts.max().tz_convert(LONDON_TZ).date()
        issue_dt_index = default_issue_schedule(min_date, max_date)
    else:
        resolved_issues: list[pd.Timestamp] = []
        for it in issue_times:
            if isinstance(it, (str, date)) and not isinstance(it, (datetime, pd.Timestamp)):
                resolved_issues.append(issue_time_for_date(it))
            else:
                ts = pd.Timestamp(it)
                if ts.tzinfo is None:
                    resolved_issues.append(ts.tz_localize(LONDON_TZ).tz_convert("UTC"))
                else:
                    resolved_issues.append(ts.tz_convert("UTC"))
        issue_dt_index = pd.DatetimeIndex(resolved_issues)

    if len(issue_dt_index) == 0:
        raise ValueError("issue_times resolved to an empty schedule")

    # 2. Resolve horizons
    horizons_arr = np.asarray(list(horizons), dtype=int)
    if len(horizons_arr) == 0:
        raise ValueError("horizons sequence cannot be empty")
    if (horizons_arr < 1).any():
        raise ValueError("horizons must be positive integers (>= 1)")

    # 3. Form the grid of (issue_time, horizon)
    n_issues = len(issue_dt_index)
    n_horizons = len(horizons_arr)

    issue_times_rep = pd.DatetimeIndex(np.repeat(issue_dt_index, n_horizons))
    horizons_rep = np.tile(horizons_arr, n_issues)
    target_times_rep = issue_times_rep + pd.to_timedelta(horizons_rep * 30, unit="m")

    # 4. Enforce Horizon Contract in code (ADR-011)
    # The weather forecast series used depends on the horizon:
    # h <= 48: Open-Meteo previous-runs day-1 forecast (temperature_2m_previous_day1),
    #          issued at target_time - 24h.
    # h > 48:  Open-Meteo previous-runs day-2 forecast (temperature_2m_previous_day2),
    #          issued at target_time - 48h.
    # Short horizons consume a 24-hour-old forecast, long horizons a 48-hour-old one,
    # and that is by design.
    lead_hours_by_h: dict[int, int] = {}
    for h in np.unique(horizons_arr):
        s_name = (weather_series_by_horizon or {}).get(
            int(h),
            "temperature_2m_previous_day1" if h <= 48 else "temperature_2m_previous_day2",
        )
        if s_name == "temperature_2m_previous_day1":
            lead_hours_by_h[int(h)] = 24
        elif s_name == "temperature_2m_previous_day2":
            lead_hours_by_h[int(h)] = 48
        else:
            raise ValueError(f"Unknown weather series name: {s_name}")

    lead_hours_rep = pd.to_timedelta([lead_hours_by_h[int(h)] for h in horizons_rep], unit="h")
    w_forecast_issue_ts = target_times_rep - lead_hours_rep
    bad_mask = w_forecast_issue_ts > issue_times_rep
    if bad_mask.any():
        bad_idx = int(bad_mask.argmax())
        bad_issue = issue_times_rep[bad_idx]
        bad_target = target_times_rep[bad_idx]
        bad_h = horizons_rep[bad_idx]
        bad_issue_str = w_forecast_issue_ts[bad_idx]
        raise ValueError(
            f"Horizon contract violation: horizon {bad_h} requires weather forecast for "
            f"target {bad_target} issued at {bad_issue_str} > issue_time {bad_issue}."
        )

    result = pd.DataFrame({
        "issue_time": issue_times_rep,
        "horizon": horizons_rep,
        "target_time": target_times_rep,
    })

    # 5. Target variable y (actual at target_time; may be NaN for unsettled targets)
    result["y"] = pd.Series(target_times_rep).map(actual_series).to_numpy(dtype=float)

    # 6. Recent and historical lags of actual relative to issue_time
    # actual_recent_3h: the actual for the half hour ending immediately
    # before issue_time - ACTUAL_AVAILABILITY_LAG
    recent_ts = issue_dt_index - actual_availability_lag
    recent_vals = recent_ts.map(actual_series).to_numpy(dtype=float)
    result["actual_recent_3h"] = np.repeat(recent_vals, n_horizons)

    for lag_h in [24, 48, 168]:
        lag_ts = issue_dt_index - pd.Timedelta(hours=lag_h)
        lag_vals = lag_ts.map(actual_series).to_numpy(dtype=float)
        result[f"actual_lag_{lag_h}h"] = np.repeat(lag_vals, n_horizons)

    # 7. Rolling statistics of actual ending strictly at issue_time - actual_availability_lag
    t_end = issue_dt_index - actual_availability_lag
    roll_3h = actual_series.rolling("3h", min_periods=1)
    roll_24h = actual_series.rolling("24h", min_periods=1)
    roll_168h = actual_series.rolling("168h", min_periods=1)

    result["actual_rolling_3h_mean"] = np.repeat(
        t_end.map(roll_3h.mean()).to_numpy(dtype=float), n_horizons
    )
    result["actual_rolling_3h_std"] = np.repeat(
        t_end.map(roll_3h.std()).to_numpy(dtype=float), n_horizons
    )
    result["actual_rolling_24h_mean"] = np.repeat(
        t_end.map(roll_24h.mean()).to_numpy(dtype=float), n_horizons
    )
    result["actual_rolling_24h_std"] = np.repeat(
        t_end.map(roll_24h.std()).to_numpy(dtype=float), n_horizons
    )
    result["actual_rolling_168h_mean"] = np.repeat(
        t_end.map(roll_168h.mean()).to_numpy(dtype=float), n_horizons
    )
    result["actual_rolling_168h_std"] = np.repeat(
        t_end.map(roll_168h.std()).to_numpy(dtype=float), n_horizons
    )

    # 8. Calendar features of target_time (Europe/London clock and UK bank holidays)
    target_london = target_times_rep.tz_convert(LONDON_TZ)
    hour = target_london.hour
    minute = target_london.minute
    day_of_week = target_london.dayofweek
    month = target_london.month
    day_of_year = target_london.dayofyear
    cal_date = target_london.date

    result["hour_of_day"] = hour.to_numpy(dtype=int)
    result["day_of_week"] = day_of_week.to_numpy(dtype=int)
    result["month"] = month.to_numpy(dtype=int)
    result["is_weekend"] = (day_of_week.to_numpy() >= 5).astype(int)
    result["is_uk_bank_holiday"] = (
        pd.Series(cal_date).isin(UK_BANK_HOLIDAYS).to_numpy(dtype=int)
    )

    hour_continuous = hour.to_numpy(dtype=float) + minute.to_numpy(dtype=float) / 60.0
    result["sin_hour"] = np.sin(2.0 * np.pi * hour_continuous / 24.0)
    result["cos_hour"] = np.cos(2.0 * np.pi * hour_continuous / 24.0)

    doy_continuous = day_of_year.to_numpy(dtype=float) - 1.0
    result["sin_day_of_year"] = np.sin(2.0 * np.pi * doy_continuous / 365.25)
    result["cos_day_of_year"] = np.cos(2.0 * np.pi * doy_continuous / 365.25)

    # 9. Weather features (Open-Meteo previous-runs per-horizon forecast at target_time)
    # Short horizons consume a 24-hour-old forecast, long horizons a 48-hour-old one,
    # and that is by design.
    if weather_df is not None:
        reanalysis_cols = [
            c for c in weather_df.columns if "reanalysis" in c.lower() or "archive" in c.lower()
        ]
        if reanalysis_cols:
            raise ValueError(
                f"Weather from archive-api reanalysis is forbidden (found {reanalysis_cols}). "
                "Must use Open-Meteo previous-runs forecast "
                "(temperature_2m_previous_day1 / previous_day2)."
            )

        w_ts = pd.to_datetime(weather_df["period_start_utc"], utc=True)
        weather_stats: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
        for col_name in ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]:
            if col_name in weather_df.columns:
                s = pd.Series(
                    weather_df[col_name].to_numpy(dtype=float),
                    index=w_ts,
                ).sort_index()
                s_roll = s.rolling("24h", min_periods=1)
                weather_stats[col_name] = (s, s_roll.mean(), s_roll.min())

        containing_hours = target_times_rep.floor("h")
        temps = np.full(len(result), np.nan, dtype=float)
        means = np.full(len(result), np.nan, dtype=float)
        mins = np.full(len(result), np.nan, dtype=float)

        for h in np.unique(horizons_arr):
            h_mask = horizons_rep == h
            series_name = (weather_series_by_horizon or {}).get(
                int(h),
                "temperature_2m_previous_day1" if h <= 48 else "temperature_2m_previous_day2",
            )
            if series_name in weather_stats:
                s, s_mean, s_min = weather_stats[series_name]
                ch_for_h = containing_hours[h_mask]
                temps[h_mask] = ch_for_h.map(s).to_numpy(dtype=float)
                means[h_mask] = ch_for_h.map(s_mean).to_numpy(dtype=float)
                mins[h_mask] = ch_for_h.map(s_min).to_numpy(dtype=float)

        result["temperature"] = temps
        result["temperature_24h_mean"] = means
        result["temperature_24h_min"] = mins
    else:
        result["temperature"] = np.nan
        result["temperature_24h_mean"] = np.nan
        result["temperature_24h_min"] = np.nan

    # 10. Generation mix features (fuel shares lagged to issue_time - 48h)
    mix_lagged_ts = issue_dt_index - pd.Timedelta(hours=48)
    if mix_df is not None:
        m_ts = pd.to_datetime(mix_df["period_start_utc"], utc=True)
        for fuel in MIX_FUELS:
            col_name = f"mix_share_{fuel}"
            if fuel in mix_df.columns:
                fuel_series = pd.Series(
                    mix_df[fuel].to_numpy(dtype=float),
                    index=m_ts,
                ).sort_index()
                fuel_vals = mix_lagged_ts.map(fuel_series).to_numpy(dtype=float)
                result[col_name] = np.repeat(fuel_vals, n_horizons)
            elif col_name in mix_df.columns:
                fuel_series = pd.Series(
                    mix_df[col_name].to_numpy(dtype=float),
                    index=m_ts,
                ).sort_index()
                fuel_vals = mix_lagged_ts.map(fuel_series).to_numpy(dtype=float)
                result[col_name] = np.repeat(fuel_vals, n_horizons)
            else:
                result[col_name] = np.nan
    else:
        for fuel in MIX_FUELS:
            result[f"mix_share_{fuel}"] = np.nan

    # 11. Auxiliary requirements validation
    if require_aux:
        for col in ["temperature", "temperature_24h_mean", "temperature_24h_min"]:
            if result[col].isna().all():
                raise ValueError(
                    f"Weather feature column '{col}' is entirely null with require_aux=True."
                )
        for fuel in MIX_FUELS:
            col_name = f"mix_share_{fuel}"
            if result[col_name].isna().all():
                raise ValueError(
                    f"Mix feature column '{col_name}' is entirely null with require_aux=True."
                )

        if weather_df is not None:
            w_ts_valid = pd.to_datetime(weather_df["period_start_utc"], utc=True)
            max_available_weather = w_ts_valid.max()
            post_cutoff = (target_times_rep >= pd.Timestamp("2024-11-01 00:00:00+00:00")) & (
                target_times_rep <= max_available_weather
            )
            null_temps = post_cutoff & result["temperature"].isna()
            if null_temps.any():
                bad_ts = target_times_rep[null_temps][0]
                raise ValueError(
                    f"Previous-runs temperature null at/after 2024-11-01 for {bad_ts}. "
                    "Silent reanalysis fallback is strictly prohibited."
                )

    # 12. Make mix nulls explainable at point of use without imputation
    if "mix_share_wind" in result.columns:
        null_count = int(result["mix_share_wind"].isna().sum())
        if null_count > 0:
            sg_path = AUDIT_DIR_DEFAULT / "source-gaps.csv"
            mix_gaps_count = 0
            if sg_path.exists():
                try:
                    sg_df = pd.read_csv(sg_path)
                    if not sg_df.empty and "store" in sg_df.columns:
                        mix_gaps_count = len(sg_df[sg_df["store"] == "mix"])
                except Exception:
                    pass
            warmup_issues = int(result.loc[result["mix_share_wind"].isna(), "issue_time"].nunique())
            print(
                f"Generation mix features: {null_count} null rows across "
                f"{warmup_issues} issue times "
                f"({mix_gaps_count} source gaps: upstream_empty; unimputed)"
            )

    # 13. Drop incomplete leading issue dates from warmup (Section 5)
    should_trim = (issue_times is None) if trim_leading is None else trim_leading
    if should_trim and not result.empty:
        sg_path = AUDIT_DIR_DEFAULT / "source-gaps.csv"
        mix_gaps_ts: set[pd.Timestamp] = set()
        weather_gaps_ts: set[pd.Timestamp] = set()
        if sg_path.exists():
            try:
                sg_df = pd.read_csv(sg_path)
                if not sg_df.empty and "store" in sg_df.columns:
                    m_gaps = sg_df[sg_df["store"] == "mix"]
                    w_gaps = sg_df[sg_df["store"] == "weather"]
                    mix_gaps_ts = set(pd.to_datetime(m_gaps["period_start_utc"], utc=True))
                    weather_gaps_ts = set(pd.to_datetime(w_gaps["period_start_utc"], utc=True))
            except Exception:
                pass

        issue_dates = result["issue_time"].dt.tz_convert(LONDON_TZ).dt.date
        unique_dates = sorted(issue_dates.unique())

        first_complete_date = None
        for d in unique_dates:
            day_mask = issue_dates == d
            day_df = result[day_mask]

            is_complete = True
            for col in FEATURE_COLUMNS:
                null_mask = day_df[col].isna()
                if not null_mask.any():
                    continue

                if col.startswith("mix_share_"):
                    null_issue_times = day_df.loc[null_mask, "issue_time"]
                    lagged_ts = null_issue_times - pd.Timedelta(hours=48)
                    unexplained = [ts for ts in lagged_ts if ts not in mix_gaps_ts]
                    if unexplained:
                        is_complete = False
                        break
                elif col in ("temperature", "temperature_24h_mean", "temperature_24h_min"):
                    null_target_times = day_df.loc[null_mask, "target_time"]
                    containing_hours = null_target_times.dt.floor("h")
                    unexplained = [ts for ts in containing_hours if ts not in weather_gaps_ts]
                    if unexplained:
                        is_complete = False
                        break
                else:
                    is_complete = False
                    break

            if is_complete:
                first_complete_date = d
                break

        if first_complete_date is not None:
            dropped_dates = [d for d in unique_dates if d < first_complete_date]
            if dropped_dates:
                result = result[issue_dates >= first_complete_date].copy()

    # Enforce exact column ordering: keys + target y + FEATURE_COLUMNS
    feature_cols_rest = [c for c in FEATURE_COLUMNS if c != "horizon"]
    final_cols = ["issue_time", "horizon", "target_time", "y"] + feature_cols_rest
    return result[final_cols].copy()
