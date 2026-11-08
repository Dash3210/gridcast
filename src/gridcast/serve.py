"""FastAPI service for 48-hour carbon intensity forecasts.

Provides:
- GET /health: 200; model version, training date, data_freshness_at; non-200 when older than 48h
- GET /forecast?hours=48: exactly 96 points (2 per hour), timestamps strictly increasing at 30 min,
  lower <= value <= upper for all points
- GET /metrics: parses as Prometheus exposition; request count, latency histogram, last
  measured MAE vs operator and vs persistence from scores.csv; absent candidate exposed with label
- GET /model: matches MODEL_CARD: version, content sha256, training window, feature count,
  interval sha256
- POST /reload: 401 without/wrong token, 503 unconfigured, 200 with right token, 500 when
  artifact missing
- Feature resolution and servability verification against declared booster features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from gridcast.features import (
    CURATED_DIR_DEFAULT,
    LONDON_TZ,
    MIX_DIR_DEFAULT,
    build_example_frame,
    issue_time_for_date,
    select_model_inputs,
)
from gridcast.models import get_model_artifact_hash, load_model_booster, predict_lgbm

__all__: list[str] = []

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: Path = Path("data/models/lgbm_model.txt")
DEFAULT_INTERVAL_PATH: Path = Path("data/models/interval_v2.json")
DEFAULT_SCORES_PATH: Path = Path("data/scoreboard/scores.csv")
MODEL_VERSION: str = "lgbm-v2"
TRAINING_WINDOW: str = "2024-11-08 to 2026-11-07"
TRAINING_DATE: str = "2026-11-07"

# Single source of truth for the publish deadline: nightly ingest cron runs at 02:17 UTC
# with measured 6.5-7.3h platform ingest delay (landing at 08:50-09:36 UTC).
# Adding ~1h margin establishes the 11:00 UTC publish deadline.
PUBLISH_DEADLINE_UTC: str = "11:00 UTC"


def parse_publish_deadline(deadline_str: str = PUBLISH_DEADLINE_UTC) -> tuple[int, int]:
    """Parse hour and minute from the single publish deadline constant string."""
    time_part = deadline_str.split()[0]
    hour_str, minute_str = time_part.split(":")
    return int(hour_str), int(minute_str)


def get_curated_freshness(
    curated_dir: Path | str = CURATED_DIR_DEFAULT,
) -> tuple[datetime | None, float]:
    """Return curated freshness token: (newest_period_start_utc, latest_partition_mtime)."""
    c_p = Path(curated_dir)
    if not c_p.exists():
        return (None, 0.0)
    partitions = sorted(c_p.glob("date=*.parquet"))
    if not partitions:
        partitions = sorted(c_p.glob("*.parquet"))
    if not partitions:
        return (None, 0.0)
    latest_file = partitions[-1]
    try:
        mtime = latest_file.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        freshness_dt: datetime | None = get_data_freshness_at(c_p)
    except Exception:
        freshness_dt = None
    return (freshness_dt, mtime)

FEATURE_CACHE_TTL_SECONDS: float = 60.0
FEATURE_CACHE_MAX_ENTRIES: int = 64


class FeatureSourceGapError(RuntimeError):
    """Raised when a required model feature is unobtainable or null due to a source gap."""


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
                    raise FeatureSourceGapError(
                        f"Startup check failed: model feature '{feat}' is unobtainable "
                        f"because curated intensity source is missing or empty."
                    )
            raise FeatureSourceGapError(
                "Startup check failed: curated intensity source is missing or empty."
            )
        intensity_df = pd.read_parquet(c_p)
    if intensity_df.empty:
        for feat in declared_features:
            if feat.startswith("actual_"):
                raise FeatureSourceGapError(
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
                        raise FeatureSourceGapError(
                            f"Startup check failed: model feature '{feat}' is unobtainable "
                            f"because weather source is missing or empty."
                        )
            weather_df = None
        else:
            weather_df = pd.read_parquet(w_p)
    if weather_df is not None and weather_df.empty and has_weather_features:
        for feat in declared_features:
            if feat.startswith("temperature"):
                raise FeatureSourceGapError(
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
                        raise FeatureSourceGapError(
                            f"Startup check failed: model feature '{feat}' is unobtainable "
                            f"because generation mix source is missing or empty."
                        )
            mix_df = None
        else:
            mix_df = pd.read_parquet(m_p)
    if mix_df is not None and mix_df.empty and has_mix_features:
        for feat in declared_features:
            if feat.startswith("mix_share_"):
                raise FeatureSourceGapError(
                    f"Startup check failed: model feature '{feat}' is unobtainable "
                    f"because generation mix source is empty."
                )

    return intensity_df, weather_df, mix_df


def resolve_features_for_issue(
    issue_time: pd.Timestamp | str | datetime,
    booster: lgb.Booster | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
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
        FeatureSourceGapError: If any feature declared by the booster artifact is unobtainable
                              or contains null values from the configured sources.
    """
    if booster is None:
        booster = load_model_booster(model_path)

    declared_features: list[str] = booster.feature_name()
    if not declared_features:
        raise FeatureSourceGapError("Model artifact declares no features.")

    # Normalise issue_time to UTC Timestamp
    if isinstance(issue_time, str):
        try:
            issue_ts = issue_time_for_date(issue_time)
        except Exception:
            try:
                issue_ts = pd.Timestamp(issue_time)
                if issue_ts is pd.NaT:
                    raise ValueError("NaT timestamp")
            except Exception as e:
                raise ValueError("Invalid issue_time") from e
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
        raise FeatureSourceGapError(
            f"Startup check failed: feature frame construction failed: {e}"
        ) from e

    # Check that every feature declared in booster is present in generated frame
    missing_columns = [f for f in declared_features if f not in frame.columns]
    if missing_columns:
        raise FeatureSourceGapError(
            f"Startup check failed: model feature '{missing_columns[0]}' is unobtainable "
            f"(missing from generated frame columns)."
        )

    # Check for any nulls in required features: refuse with FeatureSourceGapError
    for feat in declared_features:
        if frame[feat].isna().any():
            null_count = int(frame[feat].isna().sum())
            raise FeatureSourceGapError(
                f"Startup or serving check failed: model feature '{feat}' contains "
                f"{null_count} null value(s) due to upstream source gap for {issue_ts}."
            )

    # Return exactly the declared feature columns in declared order
    return frame[declared_features].copy()


def check_startup_servability(
    issue_time: pd.Timestamp | str | datetime | None = None,
    booster: lgb.Booster | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    curated_dir: Path | str | None = CURATED_DIR_DEFAULT,
    weather_dir: Path | str | None = None,
    mix_dir: Path | str | None = MIX_DIR_DEFAULT,
    intensity_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    mix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Verify that all features declared by the model can be resolved at service startup.

    Refuses to start and raises FeatureSourceGapError naming any missing feature if unobtainable.
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
            and (
                any(Path(curated_dir).glob("date=*.parquet"))
                or any(Path(curated_dir).glob("*.parquet"))
            )
        ):
            partitions = sorted(Path(curated_dir).glob("date=*.parquet"))
            if not partitions:
                partitions = sorted(Path(curated_dir).glob("*.parquet"))
            latest_date_str = partitions[-1].name.split("=")[-1].replace(".parquet", "")
            try:
                issue_time = issue_time_for_date(latest_date_str)
            except Exception:
                df = pd.read_parquet(partitions[-1])
                max_ts = pd.to_datetime(df["period_start_utc"], utc=True).max()
                issue_time = issue_time_for_date(max_ts.tz_convert("Europe/London").date())
        else:
            issue_time = issue_time_for_date(datetime.now(LONDON_TZ).date())

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


def get_data_freshness_at(curated_dir: Path | str = CURATED_DIR_DEFAULT) -> datetime:
    """Return newest period_start_utc across curated partitions."""
    c_p = Path(curated_dir)
    if not c_p.exists():
        raise FileNotFoundError(f"Curated archive directory '{c_p}' does not exist.")

    partitions = sorted(c_p.glob("date=*.parquet"))
    if not partitions:
        partitions = sorted(c_p.glob("*.parquet"))
    if not partitions:
        raise ValueError(f"No parquet partitions found in curated store '{c_p}'.")

    # Inspect the latest partition to obtain max period_start_utc
    latest_file = partitions[-1]
    df = pd.read_parquet(latest_file)
    if "period_start_utc" not in df.columns or df.empty:
        raise ValueError(
            f"Curated partition '{latest_file.name}' is empty or missing 'period_start_utc'."
        )
    max_ts = pd.to_datetime(df["period_start_utc"], utc=True).max()
    return max_ts.to_pydatetime()


def _file_sha256(path: Path | str) -> str:
    """Compute LF-normalised sha256 of text file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    text = p.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable snapshot of model booster, interval quantiles, and telemetry state."""

    booster: lgb.Booster
    interval_data: dict[str, Any]
    content_sha256: str
    interval_sha256: str
    version: str = MODEL_VERSION
    training_window: str = TRAINING_WINDOW
    training_date: str = TRAINING_DATE
    generation: int = 1
    state_version: int = 1

    @property
    def feature_count(self) -> int:
        return len(self.booster.feature_name())


class ServiceState:
    """Holds active service configuration and thread-safe reference to ModelSnapshot."""

    def __init__(
        self,
        snapshot: ModelSnapshot,
        model_path: Path,
        interval_path: Path,
        curated_dir: Path,
        scores_path: Path,
    ) -> None:
        self._snapshot = snapshot
        self.model_path = model_path
        self.interval_path = interval_path
        self.curated_dir = curated_dir
        self.scores_path = scores_path
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._cached_status: (
            tuple[
                str,
                date,
                datetime | None,
                pd.Timestamp,
                bool,
                str | None,
                pd.DataFrame | None,
            ]
            | None
        ) = None
        self._feature_cache: dict[
            tuple[pd.Timestamp, str, Any], tuple[float, pd.DataFrame]
        ] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @property
    def snapshot(self) -> ModelSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def booster(self) -> lgb.Booster:
        return self.snapshot.booster

    @property
    def interval_data(self) -> dict[str, Any]:
        return self.snapshot.interval_data

    @property
    def content_sha256(self) -> str:
        return self.snapshot.content_sha256

    @property
    def interval_sha256(self) -> str:
        return self.snapshot.interval_sha256

    @property
    def version(self) -> str:
        return self.snapshot.version

    @property
    def training_window(self) -> str:
        return self.snapshot.training_window

    @property
    def training_date(self) -> str:
        return self.snapshot.training_date

    @property
    def feature_count(self) -> int:
        return self.snapshot.feature_count

    def get_cached_features(
        self,
        issue_time: pd.Timestamp,
        model_sha256: str,
        freshness: Any = None,
    ) -> pd.DataFrame | None:
        """Lookup cached resolved feature frame for issue instant if not expired."""
        now = time.monotonic()
        key = (issue_time, model_sha256, freshness)
        with self._lock:
            entry = self._feature_cache.get(key)
            if entry is not None:
                cached_time, cached_df = entry
                if now - cached_time < FEATURE_CACHE_TTL_SECONDS:
                    self.cache_hits += 1
                    return cached_df.copy()
                self._feature_cache.pop(key, None)
            self.cache_misses += 1
            return None

    def set_cached_features(
        self,
        issue_time: pd.Timestamp,
        model_sha256: str,
        freshness: Any,
        df: pd.DataFrame,
    ) -> None:
        """Store resolved feature frame in bounded cache with current monotonic timestamp."""
        now = time.monotonic()
        key = (issue_time, model_sha256, freshness)
        with self._lock:
            if len(self._feature_cache) >= FEATURE_CACHE_MAX_ENTRIES:
                expired = [
                    k
                    for k, (t, _) in self._feature_cache.items()
                    if now - t >= FEATURE_CACHE_TTL_SECONDS
                ]
                for k in expired:
                    self._feature_cache.pop(k, None)
                if len(self._feature_cache) >= FEATURE_CACHE_MAX_ENTRIES:
                    oldest_key = next(iter(self._feature_cache))
                    self._feature_cache.pop(oldest_key, None)
            self._feature_cache[key] = (now, df.copy())

    def clear_cache(self) -> None:
        """Clear both issue status cache and explicit-date feature cache."""
        with self._lock:
            self._feature_cache.clear()
            self._cached_status = None

    def swap_snapshot(self, new_snapshot: ModelSnapshot) -> None:
        """Publish a new immutable snapshot with a single assignment under lock and clear caches."""
        with self._lock:
            self._snapshot = new_snapshot
            self._cached_status = None
            self._feature_cache.clear()


def get_served_issue_status(
    st: ServiceState,
    snap: ModelSnapshot | None = None,
) -> tuple[pd.Timestamp, bool, str | None, pd.DataFrame | None]:
    """Determine the served issue timestamp, whether it is the current issue, and fallback info.

    Returns:
        tuple of:
        - served_issue_time (Timestamp in UTC)
        - is_current_issue (bool, True only when served issue is today's issue anchor)
        - fallback_reason (str | None, reason naming null feature and date when
          is_current_issue is False)
        - feature_df (DataFrame | None, pre-resolved features for the served issue if available)
    """
    if snap is None:
        snap = st.snapshot

    today_date = datetime.now(LONDON_TZ).date()
    today_anchor_ts = issue_time_for_date(today_date)
    today_str = today_date.isoformat()

    try:
        freshness_dt: datetime | None = get_data_freshness_at(st.curated_dir)
    except Exception:
        freshness_dt = None

    with st._lock:
        if st._cached_status is not None:
            c_sha, c_date, c_freshness, c_ts, c_is_curr, c_reason, c_df = st._cached_status
            if (
                c_sha == snap.content_sha256
                and c_date == today_date
                and c_freshness == freshness_dt
            ):
                return c_ts, c_is_curr, c_reason, c_df

    try:
        df = resolve_features_for_issue(
            issue_time=today_anchor_ts,
            booster=snap.booster,
            curated_dir=st.curated_dir,
            mix_dir=MIX_DIR_DEFAULT,
        )
        with st._lock:
            st._cached_status = (
                snap.content_sha256,
                today_date,
                freshness_dt,
                today_anchor_ts,
                True,
                None,
                df,
            )
        return today_anchor_ts, True, None, df
    except FeatureSourceGapError as exc:
        fallback_reason = f"Today's issue ({today_str}) is not buildable: {exc}"
    except Exception as exc:
        fallback_reason = f"Today's issue ({today_str}) is not buildable: {exc}"

    partitions: list[Path] = []
    if st.curated_dir.exists():
        partitions = sorted(st.curated_dir.glob("date=*.parquet"))
        if not partitions:
            partitions = sorted(st.curated_dir.glob("*.parquet"))

    newest_ts: pd.Timestamp | None = None
    newest_df: pd.DataFrame | None = None

    for p in reversed(partitions):
        cand_str = p.name.split("=")[-1].replace(".parquet", "")
        try:
            cand_ts = issue_time_for_date(cand_str)
        except Exception:
            continue
        if cand_ts >= today_anchor_ts:
            continue
        try:
            cand_df = resolve_features_for_issue(
                issue_time=cand_ts,
                booster=snap.booster,
                curated_dir=st.curated_dir,
                mix_dir=MIX_DIR_DEFAULT,
            )
            newest_ts = cand_ts
            newest_df = cand_df
            break
        except Exception:
            continue

    if newest_ts is None:
        if partitions:
            cand_str = partitions[-1].name.split("=")[-1].replace(".parquet", "")
            try:
                newest_ts = issue_time_for_date(cand_str)
            except Exception:
                newest_ts = today_anchor_ts
        else:
            newest_ts = today_anchor_ts

    with st._lock:
        st._cached_status = (
            snap.content_sha256,
            today_date,
            freshness_dt,
            newest_ts,
            False,
            fallback_reason,
            newest_df,
        )
    return newest_ts, False, fallback_reason, newest_df


def _parse_and_validate_anchor(
    date_str: str | None,
    issue_time_str: str | None,
) -> pd.Timestamp | None:
    """Parse and validate date and issue_time parameters at the serving boundary.

    Enforces that the resolved anchor instant is strictly 00:00 Europe/London.
    Invalid formats, path traversals, or impossible dates raise 422 with a structured error
    that does not echo the input verbatim or any path.
    """
    if date_str is None and issue_time_str is None:
        return None

    # Guard against null bytes and control characters
    if date_str is not None and ("\x00" in date_str or "%00" in date_str):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ValidationError",
                "detail": "Invalid date parameter.",
                "message": "Invalid date parameter.",
            },
        )
    if issue_time_str is not None and ("\x00" in issue_time_str or "%00" in issue_time_str):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ValidationError",
                "detail": "Invalid issue_time parameter.",
                "message": "Invalid issue_time parameter.",
            },
        )

    # Length guard to prevent parser DoS or memory issues
    if date_str is not None and len(date_str) > 64:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ValidationError",
                "detail": "Invalid date parameter.",
                "message": "Invalid date parameter.",
            },
        )
    if issue_time_str is not None and len(issue_time_str) > 64:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ValidationError",
                "detail": "Invalid issue_time parameter.",
                "message": "Invalid issue_time parameter.",
            },
        )

    parsed_date_val: date | None = None
    if date_str is not None:
        try:
            parsed_date_val = date.fromisoformat(date_str.strip())
        except Exception:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "detail": "Invalid date parameter.",
                    "message": "Invalid date parameter.",
                },
            ) from None

    parsed_ts: pd.Timestamp | None = None
    if issue_time_str is not None:
        clean_issue_str = issue_time_str.strip()
        # Handle + in query strings unquoted to space before timezone offset (e.g. " 01:00")
        clean_issue_str = re.sub(
            r"(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s+([0-9]{2}:?[0-9]{2})$",
            r"\1+\2",
            clean_issue_str,
        )
        try:
            parsed_ts = pd.Timestamp(clean_issue_str)
            if parsed_ts is pd.NaT:
                raise ValueError("NaT timestamp")
        except Exception:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "detail": "Invalid issue_time parameter.",
                    "message": "Invalid issue_time parameter.",
                },
            ) from None

        if parsed_ts.tzinfo is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "detail": (
                        "issue_time requires an explicit UTC offset (e.g. 'Z' or '+00:00'). "
                        "date=YYYY-MM-DD is the supported form for date-only queries."
                    ),
                    "message": (
                        "issue_time requires an explicit UTC offset (e.g. 'Z' or '+00:00'). "
                        "date=YYYY-MM-DD is the supported form for date-only queries."
                    ),
                },
            )
        parsed_ts = parsed_ts.tz_convert("UTC")

        # Verify that the instant is exactly 00:00 Europe/London
        ts_london = parsed_ts.tz_convert(LONDON_TZ)
        if (
            ts_london.hour != 0
            or ts_london.minute != 0
            or ts_london.second != 0
            or ts_london.microsecond != 0
            or getattr(ts_london, "nanosecond", 0) != 0
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "detail": "Anchors must be 00:00 Europe/London.",
                    "message": "Anchors must be 00:00 Europe/London.",
                },
            )

    if parsed_date_val is not None:
        date_anchor = issue_time_for_date(parsed_date_val)
        if parsed_ts is not None and date_anchor != parsed_ts:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "detail": "date and issue_time specify conflicting anchors.",
                    "message": "date and issue_time specify conflicting anchors.",
                },
            )
        return date_anchor

    return parsed_ts


def create_app(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    interval_path: Path | str = DEFAULT_INTERVAL_PATH,
    curated_dir: Path | str = CURATED_DIR_DEFAULT,
    scores_path: Path | str = DEFAULT_SCORES_PATH,
) -> FastAPI:
    """Factory creating and configuring the gridcast FastAPI forecasting application."""
    m_p = Path(model_path)
    i_p = Path(interval_path)
    c_p = Path(curated_dir)
    s_p = Path(scores_path)

    # Initialise booster and intervals
    booster = load_model_booster(m_p)
    interval_data: dict[str, Any] = {}
    if i_p.exists():
        interval_data = json.loads(i_p.read_text(encoding="utf-8"))
    content_sha256 = get_model_artifact_hash(m_p)
    interval_sha256 = _file_sha256(i_p) if i_p.exists() else ""

    init_snapshot = ModelSnapshot(
        booster=booster,
        interval_data=interval_data,
        content_sha256=content_sha256,
        interval_sha256=interval_sha256,
        state_version=1,
    )
    service_state = ServiceState(
        snapshot=init_snapshot,
        model_path=m_p,
        interval_path=i_p,
        curated_dir=c_p,
        scores_path=s_p,
    )

    registry = CollectorRegistry()
    requests_total = Counter(
        "gridcast_requests_total",
        "Total count of HTTP requests",
        ["endpoint", "status"],
        registry=registry,
    )
    request_duration = Histogram(
        "gridcast_request_duration_seconds",
        "HTTP request latency in seconds",
        ["endpoint"],
        registry=registry,
    )
    last_measured_mae = Gauge(
        "gridcast_last_measured_mae",
        "Last measured MAE from scoreboard scores.csv",
        ["series", "status", "plane", "horizons"],
        registry=registry,
    )
    last_measured_mae_samples = Gauge(
        "gridcast_last_measured_mae_samples",
        "Sample count (row count) for last measured MAE metric",
        ["series", "plane"],
        registry=registry,
    )
    served_issue_age_seconds = Gauge(
        "gridcast_served_issue_age_seconds",
        "Age of served issue in seconds relative to current time",
        registry=registry,
    )
    publish_lateness_seconds = Gauge(
        "gridcast_publish_lateness_seconds",
        "Publish lateness in seconds relative to publish deadline (negative before deadline)",
        registry=registry,
    )

    application = FastAPI(
        title="gridcast",
        description="Forecasting service for Great Britain's half-hourly grid carbon intensity.",
        version="0.1.0",
    )
    application.state.service = service_state
    application.state.registry = registry
    application.state.requests_total = requests_total
    application.state.request_duration = request_duration
    application.state.last_measured_mae = last_measured_mae
    application.state.last_measured_mae_samples = last_measured_mae_samples
    application.state.served_issue_age_seconds = served_issue_age_seconds
    application.state.publish_lateness_seconds = publish_lateness_seconds

    @application.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Response:
        start_time = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or "unmatched"
            requests_total.labels(endpoint=endpoint, status=str(status_code)).inc()
            request_duration.labels(endpoint=endpoint).observe(elapsed)

    @application.exception_handler(FeatureSourceGapError)
    async def feature_gap_handler(
        _request: Request, exc: FeatureSourceGapError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "FeatureSourceGapError",
                "detail": str(exc),
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled server exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "Internal Server Error: An unexpected error occurred.",
            },
        )

    def _sync_served_issue_age() -> float:
        st = application.state.service
        snap = st.snapshot
        served_ts, _, _, _ = get_served_issue_status(st, snap)
        now_utc = datetime.now(timezone.utc)
        served_utc = served_ts.tz_convert(timezone.utc)
        age_seconds = max(0.0, (now_utc - served_utc).total_seconds())
        served_issue_age_seconds.set(age_seconds)
        return age_seconds

    def _compute_publish_lateness(
        now_utc: datetime,
        is_current: bool,
    ) -> tuple[float, str]:
        """Compute publish lateness in seconds and status ('on_time' | 'late')."""
        deadline_h, deadline_m = parse_publish_deadline(PUBLISH_DEADLINE_UTC)
        today_deadline = datetime(
            now_utc.year,
            now_utc.month,
            now_utc.day,
            deadline_h,
            deadline_m,
            tzinfo=timezone.utc,
        )
        if is_current:
            next_deadline = today_deadline + timedelta(days=1)
            lateness = (now_utc - next_deadline).total_seconds()
            status = "on_time"
        else:
            lateness = (now_utc - today_deadline).total_seconds()
            status = "late" if lateness > 0 else "on_time"
        return lateness, status

    def _sync_publish_lateness() -> tuple[float, str]:
        st = application.state.service
        snap = st.snapshot
        _, is_current, _, _ = get_served_issue_status(st, snap)
        now_utc = datetime.now(timezone.utc)
        lateness_sec, pub_status = _compute_publish_lateness(now_utc, is_current)
        publish_lateness_seconds.set(lateness_sec)
        return lateness_sec, pub_status

    def _sync_mae_metrics() -> None:
        """Sync last measured MAE metrics and sample counts from scores.csv."""
        sp = application.state.service.scores_path
        if not sp.exists():
            return
        try:
            scores_df = pd.read_csv(sp)
            if scores_df.empty:
                return

            def _get_horizons_label(sub_df: pd.DataFrame, default: str = "settled") -> str:
                if "horizon" in sub_df.columns and not sub_df.empty:
                    h_vals = pd.to_numeric(sub_df["horizon"], errors="coerce").dropna()
                    if not h_vals.empty:
                        min_h = int(h_vals.min())
                        max_h = int(h_vals.max())
                        if min_h == 1 and max_h >= 49:
                            return "settled"
                        return f"{min_h}-{max_h}"
                return default

            # Persistence series
            if "model_version" in scores_df.columns and "abs_error" in scores_df.columns:
                pers_mask = scores_df["model_version"].str.startswith("persistence")
                pers_sub = scores_df.loc[pers_mask].dropna(subset=["abs_error"]).copy()
                pers_sub = pers_sub[pers_sub["abs_error"] != ""]
                pers_h_label = _get_horizons_label(pers_sub, default="settled")
                if not pers_sub.empty:
                    pers_sub["abs_error"] = pers_sub["abs_error"].astype(float)
                    if "horizon" in pers_sub.columns:
                        p_val = float(pers_sub.groupby("horizon")["abs_error"].mean().mean())
                    else:
                        p_val = float(pers_sub["abs_error"].mean())
                    last_measured_mae.labels(
                        series="persistence",
                        status="settled",
                        plane="live_scoreboard",
                        horizons=pers_h_label,
                    ).set(p_val)
                    last_measured_mae_samples.labels(
                        series="persistence",
                        plane="live_scoreboard",
                    ).set(float(len(pers_sub)))
                else:
                    last_measured_mae.labels(
                        series="persistence",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="settled",
                    ).set(float("nan"))
                    last_measured_mae_samples.labels(
                        series="persistence",
                        plane="live_scoreboard",
                    ).set(0.0)

            # Operator series
            if "neso_abs_error" in scores_df.columns:
                oper_mask = (
                    scores_df["neso_abs_error"].notna()
                    & (scores_df["neso_abs_error"] != "")
                )
                oper_sub = scores_df.loc[oper_mask].copy()
                oper_h_label = _get_horizons_label(oper_sub, default="1-33")
                if not oper_sub.empty:
                    oper_sub["neso_abs_error"] = oper_sub["neso_abs_error"].astype(float)
                    if "horizon" in oper_sub.columns:
                        o_val = float(oper_sub.groupby("horizon")["neso_abs_error"].mean().mean())
                    else:
                        o_val = float(oper_sub["neso_abs_error"].mean())
                    # Do not call operator settled; label partial with its horizon range
                    last_measured_mae.labels(
                        series="operator",
                        status="partial",
                        plane="live_scoreboard",
                        horizons=oper_h_label,
                    ).set(o_val)
                    last_measured_mae_samples.labels(
                        series="operator",
                        plane="live_scoreboard",
                    ).set(float(len(oper_sub)))
                else:
                    last_measured_mae.labels(
                        series="operator",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="partial",
                    ).set(float("nan"))
                    last_measured_mae_samples.labels(
                        series="operator",
                        plane="live_scoreboard",
                    ).set(0.0)

            # Candidate series (lgbm-v2)
            if "model_version" in scores_df.columns and "abs_error" in scores_df.columns:
                cand_mask = scores_df["model_version"] == "lgbm-v2"
                cand_sub = scores_df.loc[cand_mask].dropna(subset=["abs_error"]).copy()
                cand_sub = cand_sub[cand_sub["abs_error"] != ""]
                cand_h_label = _get_horizons_label(cand_sub, default="settled")
                if not cand_sub.empty:
                    cand_sub["abs_error"] = cand_sub["abs_error"].astype(float)
                    if "horizon" in cand_sub.columns:
                        c_val = float(cand_sub.groupby("horizon")["abs_error"].mean().mean())
                    else:
                        c_val = float(cand_sub["abs_error"].mean())
                    last_measured_mae.labels(
                        series="candidate",
                        status="settled",
                        plane="live_scoreboard",
                        horizons=cand_h_label,
                    ).set(c_val)
                    last_measured_mae_samples.labels(
                        series="candidate",
                        plane="live_scoreboard",
                    ).set(float(len(cand_sub)))
                else:
                    # Expose absent series with label and NaN, no plausible constant
                    last_measured_mae.labels(
                        series="candidate",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="settled",
                    ).set(float("nan"))
                    last_measured_mae_samples.labels(
                        series="candidate",
                        plane="live_scoreboard",
                    ).set(0.0)
        except Exception:
            pass

    @application.get("/health")
    def get_health() -> JSONResponse:
        st = application.state.service
        snap = st.snapshot
        try:
            freshness_dt = get_data_freshness_at(st.curated_dir)
            freshness_iso = freshness_dt.isoformat()
        except Exception as e:
            logger.warning("Failed to inspect data freshness: %s", e)
            freshness_iso = "unavailable"

        served_ts, is_current, _, _ = get_served_issue_status(st, snap)

        payload: dict[str, Any] = {
            "status": "healthy",
            "model_version": snap.version,
            "version": snap.version,
            "training_date": snap.training_date,
            "training_window": snap.training_window,
            "data_freshness_at": freshness_iso,
            "served_issue_time": served_ts.isoformat(),
            "is_current_issue": is_current,
            "content_sha256": snap.content_sha256,
            "sha256": snap.content_sha256,
            "artifact_sha256": snap.content_sha256,
            "feature_count": snap.feature_count,
            "interval_sha256": snap.interval_sha256,
            "generation": snap.generation,
        }
        return JSONResponse(status_code=200, content=payload)

    @application.get("/ready")
    def get_ready() -> JSONResponse:
        st = application.state.service
        snap = st.snapshot
        served_ts, is_current, fallback_reason, feature_df = get_served_issue_status(st, snap)
        age_seconds = _sync_served_issue_age()
        lateness_seconds, publish_status = _sync_publish_lateness()

        is_servable = feature_df is not None

        payload: dict[str, Any] = {
            "status": "ready" if is_servable else "not_ready",
            "publish_status": publish_status,
            "publish_lateness_seconds": lateness_seconds,
            "served_issue_time": served_ts.isoformat(),
            "served_issue_age_seconds": age_seconds,
            "is_current_issue": is_current,
            "publish_deadline_utc": PUBLISH_DEADLINE_UTC,
            "deadline": PUBLISH_DEADLINE_UTC,
            "generation": snap.generation,
        }
        if not is_servable:
            payload["detail"] = (
                "No servable forecast issue is available."
                if not fallback_reason
                else f"No servable forecast issue: {fallback_reason}"
            )
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(status_code=200, content=payload)

    @application.get("/forecast")
    def get_forecast(
        hours: int = Query(
            default=48, description="Forecast horizon in hours counted from anchor (1..48)"
        ),
        date: str | None = Query(default=None, description="Optional Europe/London issue date"),
        issue_time: str | None = Query(default=None, description="Optional ISO issue timestamp"),
    ) -> dict[str, Any]:
        # Parameter validation: 1..48 integer
        if hours < 1 or hours > 48:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "message": "hours must be an integer between 1 and 48",
                    "accepted_range": [1, 48],
                },
            )

        anchor_ts = _parse_and_validate_anchor(date, issue_time)

        st = application.state.service
        snap = st.snapshot
        today_date = datetime.now(LONDON_TZ).date()
        today_anchor = issue_time_for_date(today_date)
        generated_at = datetime.now(timezone.utc).isoformat()

        fallback_reason: str | None = None

        if anchor_ts is None:
            # Default request: serve newest buildable issue
            served_ts, is_current, fallback_reason, feature_df = get_served_issue_status(st, snap)
            if feature_df is None:
                feature_df = resolve_features_for_issue(
                    issue_time=served_ts,
                    booster=snap.booster,
                    curated_dir=st.curated_dir,
                    mix_dir=MIX_DIR_DEFAULT,
                )
        else:
            # Explicit date or issue_time anchor
            freshness = get_curated_freshness(st.curated_dir)
            cached_df = st.get_cached_features(anchor_ts, snap.content_sha256, freshness)
            if cached_df is not None:
                feature_df = cached_df
            else:
                feature_df = resolve_features_for_issue(
                    issue_time=anchor_ts,
                    booster=snap.booster,
                    curated_dir=st.curated_dir,
                    mix_dir=MIX_DIR_DEFAULT,
                )
                st.set_cached_features(anchor_ts, snap.content_sha256, freshness, feature_df)
            served_ts = anchor_ts
            is_current = (served_ts == today_anchor)
            if not is_current:
                _, _, today_reason, _ = get_served_issue_status(st, snap)
                if today_reason:
                    fallback_reason = today_reason
                else:
                    fallback_reason = (
                        f"Explicit issue {served_ts.date().isoformat()} requested "
                        f"instead of current issue ({today_date.isoformat()})."
                    )

        # Select declared inputs and predict
        X_mat = select_model_inputs(feature_df, feature_columns=snap.booster.feature_name())
        predictions = predict_lgbm(snap.booster, X_mat)

        num_points = hours * 2
        quantiles = snap.interval_data.get("quantiles", {})

        points: list[dict[str, Any]] = []
        for h in range(1, num_points + 1):
            pred_val = float(predictions[h - 1])
            q_info = quantiles.get(str(h), {"q10": 0.0, "q90": 0.0})
            q10 = float(q_info.get("q10", 0.0))
            q90 = float(q_info.get("q90", 0.0))

            # Bounds bracketing by construction
            lower_bound = pred_val + q10
            upper_bound = pred_val + q90
            target_ts = served_ts + timedelta(minutes=30 * h)

            points.append({
                "horizon": h,
                "timestamp": target_ts.isoformat(),
                "target_time": target_ts.isoformat(),
                "value": pred_val,
                "lower": lower_bound,
                "upper": upper_bound,
            })

        gen_ts = pd.Timestamp(generated_at)
        retrospective_horizons = sum(
            1 for pt in points if pd.Timestamp(pt["target_time"]) < gen_ts
        )
        contains_future_points = any(
            pd.Timestamp(pt["target_time"]) >= gen_ts for pt in points
        )

        response_dict: dict[str, Any] = {
            "issue_time": served_ts.isoformat(),
            "generated_at": generated_at,
            "is_current_issue": is_current,
            "retrospective_horizons": retrospective_horizons,
            "contains_future_points": contains_future_points,
            "hours": hours,
            "model_version": snap.version,
            "content_sha256": snap.content_sha256,
            "interval_sha256": snap.interval_sha256,
            "model_hash": snap.content_sha256,
            "interval_hash": snap.interval_sha256,
            "state_version": snap.state_version,
            "generation": snap.generation,
            "points": points,
        }
        if not is_current and fallback_reason is not None:
            response_dict["fallback_reason"] = fallback_reason

        return response_dict

    @application.get("/metrics")
    def get_metrics() -> Response:
        _sync_mae_metrics()
        _sync_served_issue_age()
        _sync_publish_lateness()
        output = generate_latest(application.state.registry)
        return Response(content=output, media_type=CONTENT_TYPE_LATEST)

    @application.get("/model")
    def get_model() -> dict[str, Any]:
        snap = application.state.service.snapshot
        return {
            "version": snap.version,
            "model_version": snap.version,
            "content_sha256": snap.content_sha256,
            "sha256": snap.content_sha256,
            "artifact_sha256": snap.content_sha256,
            "training_window": snap.training_window,
            "training_date": snap.training_date,
            "feature_count": snap.feature_count,
            "interval_sha256": snap.interval_sha256,
            "generation": snap.generation,
        }

    @application.post("/reload")
    def post_reload(
        request: Request,
    ) -> dict[str, Any]:
        expected_token = os.environ.get("GRIDCAST_RELOAD_TOKEN")
        if not expected_token:
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable: Reload token not configured in environment.",
            )

        # Extract token ONLY from Authorization: Bearer <token> header or X-Reload-Token header
        provided_token = None
        auth_hdr = request.headers.get("Authorization")
        if auth_hdr:
            parts = auth_hdr.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                provided_token = parts[1].strip()
        elif request.headers.get("X-Reload-Token"):
            provided_token = request.headers.get("X-Reload-Token", "").strip()

        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid reload token.",
            )

        st = application.state.service
        # Hold reload writer lock across load, generation increment and swap
        with st._reload_lock:
            try:
                new_booster = load_model_booster(st.model_path)
                if not new_booster.feature_name():
                    raise ValueError("Reloaded booster declares no features.")

                new_interval_data: dict[str, Any] = {}
                if st.interval_path.exists():
                    new_interval_data = json.loads(st.interval_path.read_text(encoding="utf-8"))
                if not new_interval_data or "quantiles" not in new_interval_data:
                    raise ValueError("Reloaded interval artifact is missing or invalid.")

                new_content_sha256 = get_model_artifact_hash(st.model_path)
                new_interval_sha256 = _file_sha256(st.interval_path)
            except Exception as e:
                # On missing or corrupt artifact, return 500 without leaking host paths
                logger.error("Failed to reload model artifact: %s", e, exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Internal Server Error: Failed to reload model artifact.",
                ) from e

            # Atomic state swap under lock with strictly increasing generation
            curr_snap = st.snapshot
            new_snap = ModelSnapshot(
                booster=new_booster,
                interval_data=new_interval_data,
                content_sha256=new_content_sha256,
                interval_sha256=new_interval_sha256,
                version=curr_snap.version,
                training_window=curr_snap.training_window,
                training_date=curr_snap.training_date,
                generation=curr_snap.generation + 1,
                state_version=curr_snap.state_version + 1,
            )
            st.swap_snapshot(new_snap)

            return {
                "status": "reloaded",
                "model_version": new_snap.version,
                "generation": new_snap.generation,
            }

    return application


app: FastAPI = create_app()


def run_serve_cli(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    interval_path: Path | str = DEFAULT_INTERVAL_PATH,
    curated_dir: Path | str = CURATED_DIR_DEFAULT,
    scores_path: Path | str = DEFAULT_SCORES_PATH,
) -> int:
    """Run forecast API service with uvicorn."""
    import uvicorn

    server_app = create_app(
        model_path=model_path,
        interval_path=interval_path,
        curated_dir=curated_dir,
        scores_path=scores_path,
    )
    uvicorn.run(server_app, host=host, port=port)
    return 0
