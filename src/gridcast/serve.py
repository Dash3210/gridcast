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
import os
import secrets
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

DEFAULT_MODEL_PATH: Path = Path("data/models/lgbm_model.txt")
DEFAULT_INTERVAL_PATH: Path = Path("data/models/interval_v2.json")
DEFAULT_SCORES_PATH: Path = Path("data/scoreboard/scores.csv")
MODEL_VERSION: str = "lgbm-v2"
TRAINING_WINDOW: str = "2024-11-08 to 2026-11-07"
TRAINING_DATE: str = "2026-11-07"


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
                        f"because curated intensity source '{c_p}' is missing or empty."
                    )
            raise FeatureSourceGapError(
                f"Startup check failed: curated intensity source '{c_p}' is missing or empty."
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
                            f"because weather source '{w_p}' is missing or empty."
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
                            f"because generation mix source '{m_p}' is missing or empty."
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


@dataclass
class ServiceState:
    """Holds active in-memory model booster, interval quantiles, and telemetry state."""

    booster: lgb.Booster
    interval_data: dict[str, Any]
    model_path: Path
    interval_path: Path
    curated_dir: Path
    scores_path: Path
    content_sha256: str
    interval_sha256: str
    version: str = MODEL_VERSION
    training_window: str = TRAINING_WINDOW
    training_date: str = TRAINING_DATE
    _cached_status: (
        tuple[date, datetime | None, pd.Timestamp, bool, str | None, pd.DataFrame | None] | None
    ) = None

    @property
    def feature_count(self) -> int:
        return len(self.booster.feature_name())


def get_served_issue_status(
    st: ServiceState,
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
    today_date = datetime.now(LONDON_TZ).date()
    today_anchor_ts = issue_time_for_date(today_date)
    today_str = today_date.isoformat()

    try:
        freshness_dt: datetime | None = get_data_freshness_at(st.curated_dir)
    except Exception:
        freshness_dt = None

    if st._cached_status is not None:
        c_date, c_freshness, c_ts, c_is_curr, c_reason, c_df = st._cached_status
        if c_date == today_date and c_freshness == freshness_dt:
            return c_ts, c_is_curr, c_reason, c_df

    try:
        df = resolve_features_for_issue(
            issue_time=today_anchor_ts,
            booster=st.booster,
            curated_dir=st.curated_dir,
            mix_dir=MIX_DIR_DEFAULT,
        )
        st._cached_status = (today_date, freshness_dt, today_anchor_ts, True, None, df)
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
                booster=st.booster,
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

    st._cached_status = (
        today_date,
        freshness_dt,
        newest_ts,
        False,
        fallback_reason,
        newest_df,
    )
    return newest_ts, False, fallback_reason, newest_df


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

    service_state = ServiceState(
        booster=booster,
        interval_data=interval_data,
        model_path=m_p,
        interval_path=i_p,
        curated_dir=c_p,
        scores_path=s_p,
        content_sha256=content_sha256,
        interval_sha256=interval_sha256,
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

    @application.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Response:
        start_time = time.perf_counter()
        endpoint = request.url.path
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

    def _sync_mae_metrics() -> None:
        """Sync last measured MAE metrics from scores.csv."""
        sp = application.state.service.scores_path
        if not sp.exists():
            return
        try:
            scores_df = pd.read_csv(sp)
            if scores_df.empty:
                return

            def _get_horizons_label(sub_df: pd.DataFrame, default: str = "settled") -> str:
                if "horizon" in sub_df.columns and not sub_df.empty:
                    h_vals = pd.to_numeric(sub_df["horizon"], errors="coerce")
                    if (h_vals >= 49).any():
                        return "settled"
                    return "partial"
                return default

            # Persistence series
            if "model_version" in scores_df.columns and "abs_error" in scores_df.columns:
                pers_mask = scores_df["model_version"].str.startswith("persistence")
                pers_sub = scores_df.loc[pers_mask, "abs_error"].dropna()
                pers_sub = pers_sub[pers_sub != ""]
                pers_h_label = _get_horizons_label(scores_df.loc[pers_mask], default="settled")
                if not pers_sub.empty:
                    p_val = float(pers_sub.astype(float).mean())
                    last_measured_mae.labels(
                        series="persistence",
                        status="settled",
                        plane="live_scoreboard",
                        horizons=pers_h_label,
                    ).set(p_val)
                else:
                    last_measured_mae.labels(
                        series="persistence",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="settled",
                    ).set(float("nan"))

            # Operator series
            if "neso_abs_error" in scores_df.columns:
                oper_mask = (
                    scores_df["neso_abs_error"].notna()
                    & (scores_df["neso_abs_error"] != "")
                )
                oper_sub = scores_df.loc[oper_mask, "neso_abs_error"]
                oper_h_label = _get_horizons_label(scores_df.loc[oper_mask], default="partial")
                if not oper_sub.empty:
                    o_val = float(oper_sub.astype(float).mean())
                    last_measured_mae.labels(
                        series="operator",
                        status="settled",
                        plane="live_scoreboard",
                        horizons=oper_h_label,
                    ).set(o_val)
                else:
                    last_measured_mae.labels(
                        series="operator",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="partial",
                    ).set(float("nan"))

            # Candidate series (lgbm-v2)
            if "model_version" in scores_df.columns and "abs_error" in scores_df.columns:
                cand_mask = scores_df["model_version"] == "lgbm-v2"
                cand_sub = scores_df.loc[cand_mask, "abs_error"].dropna()
                cand_sub = cand_sub[cand_sub != ""]
                cand_h_label = _get_horizons_label(scores_df.loc[cand_mask], default="settled")
                if not cand_sub.empty:
                    c_val = float(cand_sub.astype(float).mean())
                    last_measured_mae.labels(
                        series="candidate",
                        status="settled",
                        plane="live_scoreboard",
                        horizons=cand_h_label,
                    ).set(c_val)
                else:
                    # Expose absent series with label and NaN, no plausible constant
                    last_measured_mae.labels(
                        series="candidate",
                        status="absent",
                        plane="live_scoreboard",
                        horizons="settled",
                    ).set(float("nan"))
        except Exception:
            pass

    @application.get("/health")
    def get_health() -> JSONResponse:
        st = application.state.service
        try:
            freshness_dt = get_data_freshness_at(st.curated_dir)
            now_utc = datetime.now(timezone.utc)
            is_stale = (now_utc - freshness_dt) > timedelta(hours=48)
            freshness_iso = freshness_dt.isoformat()
        except Exception as e:
            freshness_dt = datetime.min.replace(tzinfo=timezone.utc)
            is_stale = True
            freshness_iso = str(e)

        served_ts, is_current, _, _ = get_served_issue_status(st)

        payload: dict[str, Any] = {
            "status": "unhealthy" if is_stale else "healthy",
            "model_version": st.version,
            "version": st.version,
            "training_date": st.training_date,
            "training_window": st.training_window,
            "data_freshness_at": freshness_iso,
            "served_issue_time": served_ts.isoformat(),
            "is_current_issue": is_current,
            "content_sha256": st.content_sha256,
            "sha256": st.content_sha256,
            "artifact_sha256": st.content_sha256,
            "feature_count": st.feature_count,
            "interval_sha256": st.interval_sha256,
        }
        if is_stale:
            payload["detail"] = (
                f"Curated data freshness ({freshness_iso}) is older than 48 hours."
            )
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(status_code=200, content=payload)

    @application.get("/forecast")
    def get_forecast(
        hours: int = Query(default=48, description="Forecast horizon in hours (1..48)"),
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

        st = application.state.service
        today_date = datetime.now(LONDON_TZ).date()
        today_anchor = issue_time_for_date(today_date)
        generated_at = datetime.now(timezone.utc).isoformat()

        target_param = issue_time or date
        fallback_reason: str | None = None

        if target_param is None:
            # Default request: serve newest buildable issue
            served_ts, is_current, fallback_reason, feature_df = get_served_issue_status(st)
            if feature_df is None:
                feature_df = resolve_features_for_issue(
                    issue_time=served_ts,
                    booster=st.booster,
                    curated_dir=st.curated_dir,
                    mix_dir=MIX_DIR_DEFAULT,
                )
        else:
            # Explicit date or timestamp: refuse to build if source gap exists
            if isinstance(target_param, str):
                try:
                    req_ts = issue_time_for_date(target_param)
                except Exception:
                    req_ts = pd.Timestamp(target_param)
                    if req_ts.tzinfo is None:
                        req_ts = req_ts.tz_localize("UTC")
                    else:
                        req_ts = req_ts.tz_convert("UTC")
            else:
                req_ts = issue_time_for_date(target_param)

            feature_df = resolve_features_for_issue(
                issue_time=req_ts,
                booster=st.booster,
                curated_dir=st.curated_dir,
                mix_dir=MIX_DIR_DEFAULT,
            )
            served_ts = req_ts
            is_current = (served_ts == today_anchor)
            if not is_current:
                _, _, today_reason, _ = get_served_issue_status(st)
                if today_reason:
                    fallback_reason = today_reason
                else:
                    fallback_reason = (
                        f"Explicit issue {served_ts.date().isoformat()} requested "
                        f"instead of current issue ({today_date.isoformat()})."
                    )

        # Select declared inputs and predict
        X_mat = select_model_inputs(feature_df, feature_columns=st.booster.feature_name())
        predictions = predict_lgbm(st.booster, X_mat)

        num_points = hours * 2
        quantiles = st.interval_data.get("quantiles", {})

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

        response_dict: dict[str, Any] = {
            "issue_time": served_ts.isoformat(),
            "generated_at": generated_at,
            "is_current_issue": is_current,
            "retrospective_horizons": retrospective_horizons,
            "hours": hours,
            "points": points,
        }
        if not is_current and fallback_reason is not None:
            response_dict = {
                "issue_time": served_ts.isoformat(),
                "generated_at": generated_at,
                "is_current_issue": is_current,
                "retrospective_horizons": retrospective_horizons,
                "fallback_reason": fallback_reason,
                "hours": hours,
                "points": points,
            }

        return response_dict

    @application.get("/metrics")
    def get_metrics() -> Response:
        _sync_mae_metrics()
        output = generate_latest(application.state.registry)
        return Response(content=output, media_type=CONTENT_TYPE_LATEST)

    @application.get("/model")
    def get_model() -> dict[str, Any]:
        st = application.state.service
        return {
            "version": st.version,
            "model_version": st.version,
            "content_sha256": st.content_sha256,
            "sha256": st.content_sha256,
            "artifact_sha256": st.content_sha256,
            "training_window": st.training_window,
            "training_date": st.training_date,
            "feature_count": st.feature_count,
            "interval_sha256": st.interval_sha256,
        }

    @application.post("/reload")
    async def post_reload(
        request: Request,
        token: str | None = Query(default=None),
    ) -> dict[str, str]:
        expected_token = os.environ.get("GRIDCAST_RELOAD_TOKEN")
        if not expected_token:
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable: Reload token not configured in environment.",
            )

        # Extract token from Authorization header, X-Reload-Token header, query, or body
        provided_token = None
        auth_hdr = request.headers.get("Authorization")
        if auth_hdr:
            provided_token = auth_hdr[7:] if auth_hdr.startswith("Bearer ") else auth_hdr
        elif request.headers.get("X-Reload-Token"):
            provided_token = request.headers.get("X-Reload-Token")
        elif token:
            provided_token = token
        else:
            try:
                body = await request.json()
                if isinstance(body, dict) and "token" in body:
                    provided_token = str(body["token"])
            except Exception:
                pass

        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid reload token.",
            )

        st = application.state.service
        # Attempt to load new artifacts safely without mutating current state
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
            # On missing or corrupt artifact, return 500 while preserving in-memory state
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: Failed to reload model artifact: {e}",
            ) from e

        # Atomic state update
        st.booster = new_booster
        st.interval_data = new_interval_data
        st.content_sha256 = new_content_sha256
        st.interval_sha256 = new_interval_sha256
        st._cached_status = None

        return {"status": "reloaded", "model_version": st.version}

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
