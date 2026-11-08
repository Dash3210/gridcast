"""FastAPI service for 48-hour carbon intensity forecasts.

Provides startup servability verification and feature resolution against declared
model booster artifacts without hand-written feature lists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from gridcast.features import (
    CURATED_DIR_DEFAULT,
    MIX_DIR_DEFAULT,
    build_example_frame,
    issue_time_for_date,
)
from gridcast.models import load_model_booster

__all__: list[str] = []


def _check_source_availability(
    declared_features: Sequence[str],
    intensity_df: pd.DataFrame | None,
    curated_dir: Path | str | None,
    weather_df: pd.DataFrame | None,
    weather_dir: Path | str | None,
    mix_df: pd.DataFrame | None,
    mix_dir: Path | str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Verify source stores and load DataFrames, failing if any feature's source is missing."""
    # 1. Intensity actuals
    if intensity_df is None:
        c_p = Path(curated_dir) if curated_dir is not None else None
        if c_p is None or not c_p.exists() or not any(c_p.glob("date=*.parquet")):
            for feat in declared_features:
                if feat.startswith("actual_"):
                    raise RuntimeError(
                        f"Startup check failed: model feature '{feat}' is unobtainable "
                        f"because curated intensity source '{c_p}' is missing or empty."
                    )
            raise RuntimeError(
                f"Startup check failed: curated intensity source '{c_p}' is missing or empty."
            )
        intensity_df = pd.read_parquet(c_p)
    if intensity_df.empty:
        for feat in declared_features:
            if feat.startswith("actual_"):
                raise RuntimeError(
                    f"Startup check failed: model feature '{feat}' is unobtainable "
                    f"because curated intensity source is empty."
                )

    # 2. Weather
    has_weather_features = any(f.startswith("temperature") for f in declared_features)
    if weather_df is None:
        w_p = Path(weather_dir) if weather_dir is not None else None
        if w_p is None or not w_p.exists() or not any(w_p.glob("date=*.parquet")):
            if has_weather_features:
                for feat in declared_features:
                    if feat.startswith("temperature"):
                        raise RuntimeError(
                            f"Startup check failed: model feature '{feat}' is unobtainable "
                            f"because weather source '{w_p}' is missing or empty."
                        )
            weather_df = None
        else:
            weather_df = pd.read_parquet(w_p)
    if weather_df is not None and weather_df.empty and has_weather_features:
        for feat in declared_features:
            if feat.startswith("temperature"):
                raise RuntimeError(
                    f"Startup check failed: model feature '{feat}' is unobtainable "
                    f"because weather source is empty."
                )

    # 3. Generation mix
    has_mix_features = any(f.startswith("mix_share_") for f in declared_features)
    if mix_df is None:
        m_p = Path(mix_dir) if mix_dir is not None else None
        if m_p is None or not m_p.exists() or not any(m_p.glob("date=*.parquet")):
            if has_mix_features:
                for feat in declared_features:
                    if feat.startswith("mix_share_"):
                        raise RuntimeError(
                            f"Startup check failed: model feature '{feat}' is unobtainable "
                            f"because generation mix source '{m_p}' is missing or empty."
                        )
            mix_df = None
        else:
            mix_df = pd.read_parquet(m_p)
    if mix_df is not None and mix_df.empty and has_mix_features:
        for feat in declared_features:
            if feat.startswith("mix_share_"):
                raise RuntimeError(
                    f"Startup check failed: model feature '{feat}' is unobtainable "
                    f"because generation mix source is empty."
                )

    return intensity_df, weather_df, mix_df


def resolve_features_for_issue(
    issue_time: pd.Timestamp | str | datetime,
    booster: lgb.Booster | None = None,
    model_path: Path | str = "data/models/lgbm_model.txt",
    curated_dir: Path | str | None = CURATED_DIR_DEFAULT,
    weather_dir: Path | str | None = None,
    mix_dir: Path | str | None = MIX_DIR_DEFAULT,
    intensity_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    mix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve every feature the model declares from configured sources for an issue instant.

    Args:
        issue_time: Issue instant at 00:00 Europe/London (as Timestamp, date string, or datetime).
        booster: Optional trained LightGBM Booster instance. Loaded from model_path if None.
        model_path: Path to model artifact text file.
        curated_dir: Path to directory of curated carbon intensity partitions.
        weather_dir: Path to directory of weather partitions.
        mix_dir: Path to directory of generation mix partitions.
        intensity_df: Optional preloaded curated intensity DataFrame.
        weather_df: Optional preloaded weather DataFrame.
        mix_df: Optional preloaded generation mix DataFrame.

    Returns:
        pd.DataFrame containing exactly the model's declared features across 96 horizons.

    Raises:
        RuntimeError: If any feature declared by the booster artifact is unobtainable from
                      the configured sources.
    """
    if booster is None:
        booster = load_model_booster(model_path)

    declared_features: list[str] = booster.feature_name()
    if not declared_features:
        raise RuntimeError("Model artifact declares no features.")

    # Normalise issue_time to UTC Timestamp
    if isinstance(issue_time, str):
        try:
            issue_ts = issue_time_for_date(issue_time)
        except Exception:
            issue_ts = pd.Timestamp(issue_time)
            if issue_ts.tzinfo is None:
                issue_ts = issue_ts.tz_localize("UTC")
            else:
                issue_ts = issue_ts.tz_convert("UTC")
    elif isinstance(issue_time, (date, datetime)):
        issue_ts = issue_time_for_date(issue_time)
    elif isinstance(issue_time, pd.Timestamp):
        issue_ts = (
            issue_time.tz_localize("UTC")
            if issue_time.tzinfo is None
            else issue_time.tz_convert("UTC")
        )
    else:
        raise TypeError(f"Unsupported issue_time type: {type(issue_time)}")

    # Verify sources and load if necessary
    int_df, w_df, m_df = _check_source_availability(
        declared_features=declared_features,
        intensity_df=intensity_df,
        curated_dir=curated_dir,
        weather_df=weather_df,
        weather_dir=weather_dir,
        mix_df=mix_df,
        mix_dir=mix_dir,
    )

    has_weather_features = any(f.startswith("temperature") for f in declared_features)
    try:
        frame = build_example_frame(
            intensity_df=int_df,
            weather_df=w_df,
            mix_df=m_df,
            issue_times=[issue_ts],
            include_weather=has_weather_features,
            require_aux=False,
        )
    except Exception as e:
        raise RuntimeError(f"Startup check failed: feature frame construction failed: {e}") from e

    # Check that every feature declared in booster is present and not entirely null
    missing_columns = [f for f in declared_features if f not in frame.columns]
    if missing_columns:
        raise RuntimeError(
            f"Startup check failed: model feature '{missing_columns[0]}' is unobtainable "
            f"(missing from generated frame columns)."
        )

    null_features = [f for f in declared_features if frame[f].isna().all()]
    if null_features:
        raise RuntimeError(
            f"Startup check failed: model feature '{null_features[0]}' is unobtainable "
            f"(all values null for issue instant {issue_ts})."
        )

    # Return exactly the declared feature columns in declared order
    return frame[declared_features].copy()


def check_startup_servability(
    issue_time: pd.Timestamp | str | datetime | None = None,
    booster: lgb.Booster | None = None,
    model_path: Path | str = "data/models/lgbm_model.txt",
    curated_dir: Path | str | None = CURATED_DIR_DEFAULT,
    weather_dir: Path | str | None = None,
    mix_dir: Path | str | None = MIX_DIR_DEFAULT,
    intensity_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    mix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Verify that all features declared by the model can be resolved at service startup.

    Refuses to start and raises RuntimeError naming any missing feature if unobtainable.
    """
    if issue_time is None:
        if (
            intensity_df is not None
            and not intensity_df.empty
            and "period_start_utc" in intensity_df.columns
        ):
            max_ts = pd.to_datetime(intensity_df["period_start_utc"], utc=True).max()
            issue_time = issue_time_for_date(max_ts.tz_convert("Europe/London").date())
        elif (
            curated_dir is not None
            and Path(curated_dir).exists()
            and any(Path(curated_dir).glob("date=*.parquet"))
        ):
            partitions = sorted(Path(curated_dir).glob("date=*.parquet"))
            latest_date_str = partitions[-1].name.split("=")[-1].replace(".parquet", "")
            issue_time = issue_time_for_date(latest_date_str)
        else:
            issue_time = issue_time_for_date(datetime.now(timezone.utc).date())

    return resolve_features_for_issue(
        issue_time=issue_time,
        booster=booster,
        model_path=model_path,
        curated_dir=curated_dir,
        weather_dir=weather_dir,
        mix_dir=mix_dir,
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
    )
