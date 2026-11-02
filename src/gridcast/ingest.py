"""Half-hourly ingest into append-only Parquet partitions.

Implements Phase P1 of gridcast:
- Fetch GB carbon intensity data from official NESO API addressed by Europe/London date.
- Normalise raw payloads into strict curated schema with UTC-aware timestamps.
- Idempotent upsert into daily Parquet partitions at `data/curated/date=<YYYY-MM-DD>.parquet`.
- Distinguish between pending actuals and historical gaps.
- Two-tier storage with raw gzipped HTTP cache and 90-day retention pruning.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LONDON_TZ = ZoneInfo("Europe/London")
UTC_TZ = timezone.utc

API_BASE_URL = "https://api.carbonintensity.org.uk"
DEFAULT_USER_AGENT = "gridcast/0.1.0 (+https://github.com/Dash3210/gridcast)"

CURATED_DIR_DEFAULT = Path("data/curated")
WEATHER_DIR_DEFAULT = Path("data/weather")
MIX_DIR_DEFAULT = Path("data/mix")
CACHE_DIR_DEFAULT = Path("data/cache/http")
WEATHER_CACHE_DIR_DEFAULT = Path("data/cache/weather")
MIX_CACHE_DIR_DEFAULT = Path("data/cache/mix")
AUDIT_DIR_DEFAULT = Path("data/audit")
SOURCE_GAPS_PATH_DEFAULT = AUDIT_DIR_DEFAULT / "source-gaps.csv"

OPEN_METEO_BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ARCHIVE_FLOOR = date(2017, 9, 12)

__all__: list[str] = [
    "ARCHIVE_FLOOR",
    "AUDIT_DIR_DEFAULT",
    "CACHE_DIR_DEFAULT",
    "CURATED_DIR_DEFAULT",
    "Client",
    "DayIngestResult",
    "Gap",
    "IngestSummary",
    "MIX_CACHE_DIR_DEFAULT",
    "MIX_DIR_DEFAULT",
    "OPEN_METEO_BASE_URL",
    "SOURCE_GAPS_PATH_DEFAULT",
    "SourceGap",
    "UpsertResult",
    "WEATHER_CACHE_DIR_DEFAULT",
    "WEATHER_DIR_DEFAULT",
    "check_mix_coverage_and_retry",
    "check_weather_coverage_and_retry",
    "dst_row_count",
    "expected_row_count",
    "expected_weather_row_count",
    "fetch_day",
    "fetch_mix_chunk",
    "fetch_range_for_day",
    "fetch_weather_chunk",
    "generate_monthly_chunks",
    "ingest_day",
    "ingest_mix_day",
    "ingest_mix_range",
    "ingest_range",
    "ingest_weather_day",
    "ingest_weather_range",
    "is_cache_usable",
    "is_day_complete",
    "is_mix_day_complete",
    "is_weather_day_complete",
    "mix_payload_to_frame",
    "partition_path",
    "payload_to_frame",
    "prune_cache",
    "read_partition",
    "reconcile_recent",
    "record_conflicts",
    "record_gaps",
    "record_source_gaps",
    "upsert_day",
    "upsert_mix_day",
    "upsert_weather_day",
    "weather_payload_to_frame",
    "write_partition_atomic",
]


@dataclass(frozen=True)
class UpsertResult:
    """Outcome of partition upsert."""

    rows_before: int
    rows_after: int
    duplicates_dropped: int
    rows_refreshed: int = 0


@dataclass(frozen=True)
class Gap:
    """An audited data gap (missing actual intensity or omitted period withheld by upstream)."""

    local_date: str
    period_start_utc: str
    reason: str


@dataclass(frozen=True)
class SourceGap:
    """An audited source gap (missing data in upstream source store like weather or mix)."""

    store: str
    local_date: str
    period_start_utc: str
    reason: str


@dataclass(frozen=True)
class DayIngestResult:
    """Result of ingesting a single local date."""

    local_date: date
    status: str  # "written", "skipped", "no_data", "failed"
    rows_written: int
    duplicates_dropped: int
    gaps_recorded: int
    rows_refreshed: int = 0
    nulls_remaining: int = 0
    error: str | None = None
    used_fallback: bool = False
    nulls_forecast: int = 0
    missing_periods: int = 0


@dataclass(frozen=True)
class IngestSummary:
    """Summary of backfill or multi-date ingest execution."""

    dates_requested: int
    dates_written: int
    dates_skipped: int
    rows_total: int
    duplicates_dropped: int
    rows_refreshed: int = 0
    gaps_recorded: int = 0
    nulls_remaining: int = 0
    failures: int = 0
    used_fallback: int = 0
    nulls_forecast: int = 0
    missing_periods: int = 0
    periods_recovered: int = 0


class Client:
    """HTTP client for the GB Carbon Intensity API with retry, backoff and rate limiting."""

    def __init__(
        self,
        session: requests.Session | None = None,
        base_url: str = API_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        min_request_interval: float = 0.4,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.time,
        max_attempts: int = 5,
        base_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        backoff_cap: float = 30.0,
        jitter_fn: Callable[[float], float] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.min_request_interval = min_request_interval
        self.sleep_fn = sleep_fn
        self.time_fn = time_fn
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.backoff_factor = backoff_factor
        self.backoff_cap = backoff_cap
        self.jitter_fn = jitter_fn
        self._last_request_time: float = 0.0
        self.request_count: int = 0

    def _rate_limit(self) -> None:
        elapsed = self.time_fn() - self._last_request_time
        if elapsed < self.min_request_interval:
            self.sleep_fn(self.min_request_interval - elapsed)
        self._last_request_time = self.time_fn()

    def _fetch_url(self, url: str) -> dict:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        for attempt in range(1, self.max_attempts + 1):
            self._rate_limit()
            self.request_count += 1
            try:
                response = self.session.get(url, headers=headers, timeout=30)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == self.max_attempts:
                    raise
                delay = min(
                    self.base_backoff * (self.backoff_factor ** (attempt - 1)),
                    self.backoff_cap,
                )
                jitter = self.jitter_fn(delay) if self.jitter_fn else random.uniform(0, 0.5)
                self.sleep_fn(delay + jitter)
                continue

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
                    if attempt == self.max_attempts:
                        raise
                    delay = min(
                        self.base_backoff * (self.backoff_factor ** (attempt - 1)),
                        self.backoff_cap,
                    )
                    jitter = self.jitter_fn(delay) if self.jitter_fn else random.uniform(0, 0.5)
                    self.sleep_fn(delay + jitter)
                    continue

            # 429 or 5xx: retry
            if status == 429 or 500 <= status <= 599:
                if attempt == self.max_attempts:
                    response.raise_for_status()
                delay = min(
                    self.base_backoff * (self.backoff_factor ** (attempt - 1)),
                    self.backoff_cap,
                )
                jitter = self.jitter_fn(delay) if self.jitter_fn else random.uniform(0, 0.5)
                self.sleep_fn(delay + jitter)
                continue

            # Other 4xx errors: raise immediately without retrying
            response.raise_for_status()

        raise RuntimeError(f"Failed to fetch {url} after {self.max_attempts} attempts.")

    def get_day(self, local_date: date | str) -> dict:
        """Fetch raw payload for a local Europe/London day.

        Retries on 429/5xx and JSONDecodeError.
        """
        d_str = local_date.isoformat() if isinstance(local_date, date) else str(local_date)
        url = f"{self.base_url}/intensity/date/{d_str}"
        return self._fetch_url(url)

    def get_range(
        self,
        start_utc: datetime | str,
        end_utc: datetime | str,
    ) -> dict:
        """Fetch raw payload for a UTC range.

        Retries on 429/5xx and JSONDecodeError.
        """
        if isinstance(start_utc, datetime):
            s_dt = (
                start_utc.astimezone(UTC_TZ)
                if start_utc.tzinfo is not None
                else start_utc.replace(tzinfo=UTC_TZ)
            )
            s_str = s_dt.strftime("%Y-%m-%dT%H:%MZ")
        else:
            s_str = str(start_utc)

        if isinstance(end_utc, datetime):
            e_dt = (
                end_utc.astimezone(UTC_TZ)
                if end_utc.tzinfo is not None
                else end_utc.replace(tzinfo=UTC_TZ)
            )
            e_str = e_dt.strftime("%Y-%m-%dT%H:%MZ")
        else:
            e_str = str(end_utc)

        url = f"{self.base_url}/intensity/{s_str}/{e_str}"
        return self._fetch_url(url)

    def get_generation_range(
        self,
        start_utc: datetime | str,
        end_utc: datetime | str,
    ) -> dict:
        """Fetch generation mix for a UTC range."""
        if isinstance(start_utc, datetime):
            s_dt = (
                start_utc.astimezone(UTC_TZ)
                if start_utc.tzinfo is not None
                else start_utc.replace(tzinfo=UTC_TZ)
            )
            s_str = s_dt.strftime("%Y-%m-%dT%H:%MZ")
        else:
            s_str = str(start_utc)

        if isinstance(end_utc, datetime):
            e_dt = (
                end_utc.astimezone(UTC_TZ)
                if end_utc.tzinfo is not None
                else end_utc.replace(tzinfo=UTC_TZ)
            )
            e_str = e_dt.strftime("%Y-%m-%dT%H:%MZ")
        else:
            e_str = str(end_utc)

        url = f"{self.base_url}/generation/{s_str}/{e_str}"
        return self._fetch_url(url)

    def get_weather_previous_runs(
        self,
        start_date: date | str,
        end_date: date | str,
        latitude: float = 51.5074,
        longitude: float = -0.1278,
    ) -> dict:
        """Fetch hourly previous_day1 and previous_day2 from Open-Meteo previous-runs API."""
        s_str = start_date.isoformat() if isinstance(start_date, date) else str(start_date)
        e_str = end_date.isoformat() if isinstance(end_date, date) else str(end_date)
        url = (
            f"{OPEN_METEO_BASE_URL}?"
            f"latitude={latitude}&longitude={longitude}&"
            f"hourly=temperature_2m_previous_day1,temperature_2m_previous_day2&"
            f"start_date={s_str}&end_date={e_str}&timezone=UTC"
        )
        return self._fetch_url(url)


def partition_path(local_date: date | str, base_dir: Path | None = None) -> Path:
    """Return path to the curated Parquet partition file for a given local date."""
    d_str = local_date.isoformat() if isinstance(local_date, date) else str(local_date)
    target_dir = base_dir if base_dir is not None else CURATED_DIR_DEFAULT
    return target_dir / f"date={d_str}.parquet"


def cache_path(local_date: date | str, cache_dir: Path | None = None) -> Path:
    """Return path to the raw HTTP cache file for a given local date."""
    d_str = local_date.isoformat() if isinstance(local_date, date) else str(local_date)
    target_dir = cache_dir if cache_dir is not None else CACHE_DIR_DEFAULT
    return target_dir / f"{d_str}.json.gz"


def fetch_range_for_day(client: Client, local_date: date | str) -> dict:
    """Fetch range endpoint for a local Europe/London day and filter to that day."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=LONDON_TZ)
    next_d = d + timedelta(days=1)
    end_local = datetime(next_d.year, next_d.month, next_d.day, 0, 0, tzinfo=LONDON_TZ)

    start_utc = start_local.astimezone(UTC_TZ)
    end_utc = end_local.astimezone(UTC_TZ)

    start_str = start_utc.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_utc.strftime("%Y-%m-%dT%H:%MZ")

    raw_range = client.get_range(start_str, end_str)
    raw_data = raw_range.get("data", [])
    if not raw_data:
        return {"data": []}

    kept = []
    for item in raw_data:
        start_ts = pd.to_datetime(item["from"], utc=True)
        item_local_date = start_ts.tz_convert(LONDON_TZ).date()
        if item_local_date == d:
            kept.append(item)

    return {"data": kept}


def is_cache_usable(
    payload: dict,
    local_date: date | str,
    today: date | None = None,
    now: datetime | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    gaps_path: Path | None = None,
) -> bool:
    """Predicate determining if cached payload is settled and safe to use.

    Settled condition:
    - Date is strictly before today in Europe/London; and
    - Cached payload records + recorded missing periods >= dst_row_count(d); and
    - Every null actual for any period ending >= 2 hours ago is already in gaps.csv.
    """
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    eval_today = today if today is not None else datetime.now(LONDON_TZ).date()
    if d >= eval_today:
        return False

    if not isinstance(payload, dict):
        return False

    records = payload.get("data")
    if not isinstance(records, list):
        return False

    if gaps_path is not None:
        target_gaps = gaps_path
    elif audit_dir is not None:
        target_gaps = audit_dir / "gaps.csv"
    elif curated_dir is not None and (curated_dir / "gaps.csv").exists():
        target_gaps = curated_dir / "gaps.csv"
    elif curated_dir is not None and curated_dir != CURATED_DIR_DEFAULT:
        target_gaps = curated_dir.parent / "audit" / "gaps.csv"
    else:
        target_gaps = AUDIT_DIR_DEFAULT / "gaps.csv"

    recorded_missing_periods = 0
    recorded_gaps: set[tuple[str, str]] = set()
    if target_gaps.exists():
        with open(target_gaps, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 2 and row[0] == str(d):
                        try:
                            ts_norm = pd.Timestamp(row[1]).isoformat()
                        except Exception:
                            ts_norm = row[1]
                        recorded_gaps.add((row[0], ts_norm))
                        if len(row) >= 3 and row[2] == "missing_period":
                            recorded_missing_periods += 1

    if len(records) + recorded_missing_periods < dst_row_count(d):
        return False

    eval_now = now if now is not None else datetime.now(UTC_TZ)

    for item in records:
        if not isinstance(item, dict):
            return False
        intensity = item.get("intensity")
        if not isinstance(intensity, dict):
            return False
        actual = intensity.get("actual")
        if actual is None or pd.isna(actual):
            to_val = item.get("to")
            from_val = item.get("from")
            if not to_val or not from_val:
                return False
            period_end = pd.to_datetime(to_val, utc=True).to_pydatetime()
            if (eval_now - period_end) >= timedelta(hours=2):
                try:
                    from_ts = pd.Timestamp(from_val).isoformat()
                except Exception:
                    from_ts = from_val
                if (str(d), from_ts) not in recorded_gaps:
                    return False

    return True


def _fetch_day_with_fallback(
    client: Client,
    local_date: date | str,
    cache_dir: Path | None = None,
    force: bool = False,
    today: date | None = None,
    now: datetime | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
) -> tuple[dict, bool]:
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    c_path = cache_path(d, cache_dir)
    if not force and c_path.exists():
        try:
            with gzip.open(c_path, "rt", encoding="utf-8") as f:
                cached_payload = json.load(f)
            if is_cache_usable(
                cached_payload,
                d,
                today=today,
                now=now,
                curated_dir=curated_dir,
                audit_dir=audit_dir,
            ):
                return cached_payload, False
            logger.info("Cache for %s is unsettled or incomplete; refetching live", d)
        except Exception as exc:
            logger.warning("Failed to read cache for %s (%s); refetching", d, exc)

    used_fallback = False
    try:
        payload = client.get_day(d)
    except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
        logger.warning("Date endpoint malformed for %s; falling back to range", d)
        payload = fetch_range_for_day(client, d)
        used_fallback = True

    if "data" not in payload:
        raise ValueError(f"Payload for {d} missing required 'data' key")

    if not payload["data"] and d >= ARCHIVE_FLOOR and not used_fallback:
        logger.info("Empty data for %s confirmed via range endpoint", d)
        range_payload = fetch_range_for_day(client, d)
        if range_payload.get("data"):
            payload = range_payload
            used_fallback = True
        else:
            payload = {"data": []}

    # Write to raw cache buffer (overwrite so next run benefits)
    c_path.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = c_path.with_name(f".{c_path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(temp_cache, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(temp_cache, c_path)
    finally:
        if temp_cache.exists():
            temp_cache.unlink(missing_ok=True)

    return payload, used_fallback


def fetch_day(
    client: Client,
    local_date: date | str,
    cache_dir: Path | None = None,
    force: bool = False,
    today: date | None = None,
    now: datetime | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
) -> dict:
    """Fetch raw JSON payload for a local date, utilizing the gzipped HTTP cache buffer."""
    payload, _ = _fetch_day_with_fallback(
        client=client,
        local_date=local_date,
        cache_dir=cache_dir,
        force=force,
        today=today,
        now=now,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
    )
    return payload


def payload_to_frame(
    payload: dict,
    local_date: date | str,
    ingested_at: datetime | None = None,
) -> pd.DataFrame:
    """Normalise raw API payload into a DataFrame enforcing the curated schema contract."""
    requested_date = (
        date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    )
    ingested_ts = (
        pd.to_datetime(ingested_at, utc=True)
        if ingested_at is not None
        else pd.Timestamp.now(tz=UTC_TZ)
    )

    if "data" not in payload:
        raise ValueError(f"Payload for {requested_date} missing required 'data' key")

    data_items = payload["data"]
    if not data_items:
        # Empty schema-compliant DataFrame
        empty_df = pd.DataFrame(
            columns=[
                "period_start_utc",
                "period_end_utc",
                "utc_date",
                "local_date",
                "forecast",
                "actual",
                "index",
                "ingested_at",
                "first_seen_at",
            ]
        )
        empty_df["period_start_utc"] = pd.to_datetime([], utc=True)
        empty_df["period_end_utc"] = pd.to_datetime([], utc=True)
        empty_df["utc_date"] = pd.Series([], dtype="object")
        empty_df["local_date"] = pd.Series([], dtype="object")
        empty_df["forecast"] = pd.Series([], dtype="Int32")
        empty_df["actual"] = pd.Series([], dtype="Int32")
        empty_df["index"] = pd.Series([], dtype="string")
        empty_df["ingested_at"] = pd.to_datetime([], utc=True)
        empty_df["first_seen_at"] = pd.to_datetime([], utc=True)
        return empty_df

    rows = []
    for item in data_items:
        if "intensity" not in item or item["intensity"] is None:
            raise ValueError(
                f"Record for period {item.get('from')} missing required 'intensity' object"
            )

        start_utc = pd.to_datetime(item["from"], utc=True)
        end_utc = pd.to_datetime(item["to"], utc=True)

        if end_utc <= start_utc:
            raise ValueError(
                f"Invalid period: period_end_utc ({end_utc}) <= period_start_utc ({start_utc})"
            )
        if (end_utc - start_utc) != timedelta(minutes=30):
            raise ValueError(
                f"Invalid period duration: {end_utc - start_utc} is not exactly 30 minutes"
            )

        intensity = item["intensity"]

        # local_date: Europe/London date of period_start_utc
        row_local_date = start_utc.tz_convert(LONDON_TZ).date()
        if row_local_date != requested_date:
            raise ValueError(
                f"Record for period {item.get('from')} has local_date {row_local_date}, "
                f"expected requested date {requested_date}"
            )

        act_val = intensity.get("actual")
        first_seen_val = (
            ingested_ts if act_val is not None and not pd.isna(act_val) else pd.NaT
        )
        rows.append(
            {
                "period_start_utc": start_utc,
                "period_end_utc": end_utc,
                "utc_date": start_utc.date(),
                "local_date": row_local_date,
                "forecast": intensity.get("forecast"),
                "actual": act_val,
                "index": intensity.get("index"),
                "ingested_at": ingested_ts,
                "first_seen_at": first_seen_val,
            }
        )

    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"], utc=True)
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")

    # Sort strictly by period_start_utc
    df = df.sort_values(by="period_start_utc").reset_index(drop=True)

    if not df["period_start_utc"].is_unique:
        raise ValueError(f"Duplicate period_start_utc encountered in payload for {requested_date}")

    return df


def read_partition(local_date: date | str, base_dir: Path | None = None) -> pd.DataFrame | None:
    """Read an existing Parquet partition if it exists, preserving dtypes."""
    p_path = partition_path(local_date, base_dir)
    if not p_path.exists():
        return None
    df = pd.read_parquet(p_path)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    if "first_seen_at" in df.columns:
        df["first_seen_at"] = pd.to_datetime(df["first_seen_at"], utc=True)
    else:
        df["first_seen_at"] = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns, UTC]")
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")
    return df


def write_partition_atomic(
    frame: pd.DataFrame,
    local_date: date | str,
    base_dir: Path | None = None,
) -> Path:
    """Write DataFrame to Parquet atomically using a temporary file in the same directory."""
    target_path = partition_path(local_date, base_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path_str = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return target_path


def record_conflicts(
    conflicts: list[tuple[str, str, str, str]],
    base_dir: Path | None = None,
    audit_dir: Path | None = None,
    conflicts_path: Path | None = None,
) -> None:
    """Record upstream conflicts/retractions in conflicts.csv without duplication."""
    if not conflicts:
        return
    if conflicts_path is not None:
        target_path = conflicts_path
    elif audit_dir is not None:
        target_path = audit_dir / "conflicts.csv"
    elif base_dir is not None:
        if base_dir.name == "curated":
            target_path = base_dir.parent / "audit" / "conflicts.csv"
        elif (base_dir / "conflicts.csv").parent.exists() or base_dir.name == "audit":
            target_path = base_dir / "conflicts.csv"
        else:
            target_path = base_dir / "conflicts.csv"
    else:
        target_path = AUDIT_DIR_DEFAULT / "conflicts.csv"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_conflicts: set[tuple[str, str, str, str]] = set()
    if target_path.exists():
        with open(target_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 4:
                        try:
                            ts_norm = pd.Timestamp(row[1]).isoformat()
                        except Exception:
                            ts_norm = row[1]
                        existing_conflicts.add((row[0], ts_norm, row[2], str(row[3])))

    new_conflicts = []
    for c in conflicts:
        try:
            ts_norm = pd.Timestamp(c[1]).isoformat()
        except Exception:
            ts_norm = c[1]
        key = (c[0], ts_norm, c[2], str(c[3]))
        if key not in existing_conflicts:
            new_conflicts.append((c[0], ts_norm, c[2], str(c[3])))
            existing_conflicts.add(key)

    if new_conflicts:
        file_exists = target_path.exists()
        with open(target_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["local_date", "period_start_utc", "field", "stored_value"])
            for c in new_conflicts:
                writer.writerow([c[0], c[1], c[2], c[3]])


def upsert_day(
    frame: pd.DataFrame,
    local_date: date | str,
    base_dir: Path | None = None,
    audit_dir: Path | None = None,
) -> UpsertResult:
    """Idempotently upsert frame into partition, coalescing non-null values."""
    existing = read_partition(local_date, base_dir)

    incoming_unique = frame["period_start_utc"].nunique()
    incoming_duplicates = len(frame) - incoming_unique

    if existing is None or existing.empty:
        rows_before = 0
        rows_refreshed = 0
        duplicates_dropped = incoming_duplicates
        deduped = frame.drop_duplicates(subset=["period_start_utc"], keep="last").copy()
        if "first_seen_at" not in deduped.columns:
            deduped["first_seen_at"] = pd.Series(
                [pd.NaT] * len(deduped), dtype="datetime64[ns, UTC]"
            )
            deduped.loc[deduped["actual"].notna(), "first_seen_at"] = deduped.loc[
                deduped["actual"].notna(), "ingested_at"
            ]
        else:
            deduped["first_seen_at"] = pd.to_datetime(deduped["first_seen_at"], utc=True)
            deduped.loc[deduped["actual"].isna(), "first_seen_at"] = pd.NaT
            need_ts = deduped["actual"].notna() & deduped["first_seen_at"].isna()
            deduped.loc[need_ts, "first_seen_at"] = deduped.loc[need_ts, "ingested_at"]
    else:
        rows_before = len(existing)
        existing_unique = existing["period_start_utc"].nunique()
        existing_duplicates = len(existing) - existing_unique
        duplicates_dropped = existing_duplicates + incoming_duplicates
        rows_refreshed = 0

        records: dict[pd.Timestamp, dict] = {}

        # 1. Populate from existing rows sorted by ingested_at ascending
        existing_sorted = existing.sort_values(by=["period_start_utc", "ingested_at"])
        for _, row in existing_sorted.iterrows():
            key = row["period_start_utc"]
            row_dict = row.to_dict()
            if key not in records:
                records[key] = row_dict
            else:
                prior = records[key]
                if pd.notna(row_dict["forecast"]) and row_dict["forecast"] is not None:
                    prior["forecast"] = row_dict["forecast"]
                if pd.notna(row_dict["actual"]) and row_dict["actual"] is not None:
                    prior_actual_was_null = pd.isna(prior.get("actual"))
                    prior["actual"] = row_dict["actual"]
                    prior["index"] = row_dict["index"]
                    if prior_actual_was_null and pd.isna(prior.get("first_seen_at")):
                        prior["first_seen_at"] = (
                            row_dict.get("first_seen_at") or row_dict["ingested_at"]
                        )
                prior["ingested_at"] = max(prior["ingested_at"], row_dict["ingested_at"])

        # 2. Coalesce incoming rows: atomic (actual, index), independent forecast, conflicts
        conflicts: list[tuple[str, str, str, str]] = []
        frame_sorted = frame.sort_values(by=["period_start_utc", "ingested_at"])
        for _, row in frame_sorted.iterrows():
            key = row["period_start_utc"]
            row_dict = row.to_dict()
            if key not in records:
                if pd.notna(row_dict.get("actual")) and row_dict["actual"] is not None:
                    if "first_seen_at" not in row_dict or pd.isna(row_dict["first_seen_at"]):
                        row_dict["first_seen_at"] = row_dict["ingested_at"]
                else:
                    row_dict["first_seen_at"] = pd.NaT
                records[key] = row_dict
            else:
                prior = records[key]
                row_changed = False

                # Forecast handling
                in_fc = row_dict["forecast"]
                prior_fc = prior["forecast"]
                if pd.notna(in_fc) and in_fc is not None:
                    if pd.isna(prior_fc) or prior_fc != in_fc:
                        prior["forecast"] = in_fc
                        row_changed = True
                else:
                    if pd.notna(prior_fc) and prior_fc is not None:
                        conflicts.append((
                            str(local_date),
                            pd.Timestamp(key).isoformat(),
                            "forecast",
                            str(prior_fc),
                        ))

                # Actual and index handling (atomic observation)
                in_act = row_dict["actual"]
                prior_act = prior["actual"]
                if pd.notna(in_act) and in_act is not None:
                    in_idx = row_dict["index"]
                    prior_idx = prior["index"]
                    act_changed = pd.isna(prior_act) or prior_act != in_act
                    idx_changed = pd.isna(prior_idx) or prior_idx != in_idx
                    if act_changed or idx_changed:
                        prior["actual"] = in_act
                        prior["index"] = in_idx
                        row_changed = True

                    if pd.isna(prior_act):
                        # Transition null -> non-null: set first_seen_at to ingest instant
                        prior["first_seen_at"] = row_dict["ingested_at"]
                    else:
                        # Already non-null: keep stored first_seen_at. Never overwrite it.
                        pass
                else:
                    if pd.notna(prior_act) and prior_act is not None:
                        conflicts.append((
                            str(local_date),
                            pd.Timestamp(key).isoformat(),
                            "actual",
                            str(prior_act),
                        ))

                if row_changed:
                    rows_refreshed += 1

                prior["ingested_at"] = max(prior["ingested_at"], row_dict["ingested_at"])
                prior["period_end_utc"] = row_dict["period_end_utc"]
                prior["utc_date"] = row_dict["utc_date"]
                prior["local_date"] = row_dict["local_date"]

        if conflicts:
            target_audit = audit_dir
            if target_audit is None and base_dir is not None:
                if base_dir.name == "curated":
                    target_audit = base_dir.parent / "audit"
                elif base_dir.name == "audit":
                    target_audit = base_dir
            record_conflicts(conflicts, audit_dir=target_audit)

        deduped = pd.DataFrame(list(records.values()))

    columns = [
        "period_start_utc",
        "period_end_utc",
        "utc_date",
        "local_date",
        "forecast",
        "actual",
        "index",
        "ingested_at",
        "first_seen_at",
    ]
    deduped = deduped[columns]

    deduped["period_start_utc"] = pd.to_datetime(deduped["period_start_utc"], utc=True)
    deduped["period_end_utc"] = pd.to_datetime(deduped["period_end_utc"], utc=True)
    deduped["ingested_at"] = pd.to_datetime(deduped["ingested_at"], utc=True)
    deduped["first_seen_at"] = pd.to_datetime(deduped["first_seen_at"], utc=True)
    deduped["forecast"] = deduped["forecast"].astype("Int32")
    deduped["actual"] = deduped["actual"].astype("Int32")
    deduped["index"] = deduped["index"].astype("string")

    # Invariant: null actual never carries a first_seen_at
    deduped.loc[deduped["actual"].isna(), "first_seen_at"] = pd.NaT

    # Sort strictly by period_start_utc
    deduped = deduped.sort_values(by="period_start_utc").reset_index(drop=True)

    write_partition_atomic(deduped, local_date, base_dir)
    return UpsertResult(
        rows_before=rows_before,
        rows_after=len(deduped),
        duplicates_dropped=duplicates_dropped,
        rows_refreshed=rows_refreshed,
    )


def dst_row_count(local_date: date | str) -> int:
    """Compute full-day half-hourly row count (46, 48, or 50) for a local date in Europe/London."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=LONDON_TZ)
    next_d = d + timedelta(days=1)
    end_local = datetime(next_d.year, next_d.month, next_d.day, 0, 0, tzinfo=LONDON_TZ)

    start_utc = start_local.astimezone(UTC_TZ)
    end_utc = end_local.astimezone(UTC_TZ)

    duration_seconds = (end_utc - start_utc).total_seconds()
    return int(duration_seconds // 1800)


def expected_row_count(local_date: date | str, today: date | None = None) -> int | None:
    """Compute expected half-hourly row count for a local date using Europe/London convention."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    if today is not None and d >= today:
        return None
    return dst_row_count(d)


def record_gaps(
    frame: pd.DataFrame,
    today: datetime | None = None,
    gaps_path: Path | None = None,
    local_date: date | None = None,
    eval_instant: datetime | None = None,
    eval_today: date | None = None,
    audit_dir: Path | None = None,
) -> list[Gap]:
    """Identify missing actuals older than 2h and missing periods, recording without duplication."""
    instant = eval_instant if eval_instant is not None else today
    if instant is None:
        eval_time = datetime.now(UTC_TZ)
    elif isinstance(instant, date) and not hasattr(instant, "hour"):
        raise TypeError(f"record_gaps requires a datetime instant, not a date: {instant!r}")
    elif hasattr(instant, "hour"):
        eval_time = (
            instant
            if getattr(instant, "tzinfo", None) is not None
            else instant.replace(tzinfo=UTC_TZ)
        )
    else:
        raise TypeError(f"record_gaps requires a datetime instant, got: {type(instant).__name__}")

    if eval_today is not None:
        eval_today_date = eval_today
    else:
        eval_today_date = eval_time.astimezone(LONDON_TZ).date()

    if local_date is not None and local_date > eval_today_date:
        return []

    if gaps_path is not None:
        target_gaps = gaps_path
    elif audit_dir is not None:
        target_gaps = audit_dir / "gaps.csv"
    else:
        target_gaps = AUDIT_DIR_DEFAULT / "gaps.csv"

    # Read existing gap records
    existing_rows: list[dict[str, str]] = []
    if target_gaps.exists():
        with open(target_gaps, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 3:
                        try:
                            ts_norm = pd.Timestamp(row[1]).isoformat()
                        except Exception:
                            ts_norm = row[1]
                        existing_rows.append({
                            "local_date": row[0],
                            "period_start_utc": ts_norm,
                            "reason": row[2],
                        })

    # Reconcile gaps for the evaluated date
    eval_date_str = (
        str(local_date)
        if local_date is not None
        else (
            str(frame["local_date"].iloc[0])
            if not frame.empty and "local_date" in frame.columns
            else None
        )
    )

    all_present_starts: set[str] = set()
    actual_present_starts: set[str] = set()
    if not frame.empty and "period_start_utc" in frame.columns:
        for _, row in frame.iterrows():
            try:
                ts_str = pd.Timestamp(row["period_start_utc"]).isoformat()
            except Exception:
                ts_str = str(row["period_start_utc"])
            all_present_starts.add(ts_str)
            if "actual" in frame.columns and pd.notna(row["actual"]):
                actual_present_starts.add(ts_str)

    kept_rows: list[dict[str, str]] = []
    reconciled_any = False
    for r in existing_rows:
        if eval_date_str is not None and r["local_date"] == eval_date_str:
            if r["reason"] == "missing_actual" and r["period_start_utc"] in actual_present_starts:
                reconciled_any = True
                continue
            if r["reason"] == "missing_period" and r["period_start_utc"] in all_present_starts:
                reconciled_any = True
                continue
        kept_rows.append(r)

    kept_keys: set[tuple[str, str]] = {
        (r["local_date"], r["period_start_utc"]) for r in kept_rows
    }
    new_gaps: list[Gap] = []

    # 1. Missing periods for settled dates
    if local_date is not None and local_date < eval_today_date:
        exp_count = dst_row_count(local_date)
        start_local = datetime(
            local_date.year, local_date.month, local_date.day, 0, 0, tzinfo=LONDON_TZ
        )
        start_utc = start_local.astimezone(UTC_TZ)
        expected_starts = [start_utc + timedelta(minutes=30 * i) for i in range(exp_count)]
        present_starts = set(pd.to_datetime(frame["period_start_utc"], utc=True))

        for p_start in expected_starts:
            ts_p_start = pd.Timestamp(p_start)
            if ts_p_start not in present_starts:
                l_date_str = str(local_date)
                start_str = ts_p_start.isoformat()
                if (l_date_str, start_str) not in kept_keys:
                    gap = Gap(
                        local_date=l_date_str,
                        period_start_utc=start_str,
                        reason="missing_period",
                    )
                    new_gaps.append(gap)
                    kept_keys.add((l_date_str, start_str))

    # 2. Missing actuals (null actuals ended >= 2 hours ago)
    null_actuals = frame[frame["actual"].isna()]
    for _, row in null_actuals.iterrows():
        period_end = row["period_end_utc"]
        if isinstance(period_end, pd.Timestamp):
            period_end_dt = period_end.to_pydatetime()
        else:
            period_end_dt = period_end

        # A null actual is pending if ended < 2 hours ago; a gap otherwise
        if (eval_time - period_end_dt) >= timedelta(hours=2):
            l_date_str = str(row["local_date"])
            start_str = pd.Timestamp(row["period_start_utc"]).isoformat()
            if (l_date_str, start_str) not in kept_keys:
                gap = Gap(
                    local_date=l_date_str,
                    period_start_utc=start_str,
                    reason="missing_actual",
                )
                new_gaps.append(gap)
                kept_keys.add((l_date_str, start_str))

    final_rows = kept_rows + [
        {"local_date": g.local_date, "period_start_utc": g.period_start_utc, "reason": g.reason}
        for g in new_gaps
    ]

    if not final_rows:
        if target_gaps.exists():
            target_gaps.unlink(missing_ok=True)
    elif new_gaps or reconciled_any or not target_gaps.exists():
        target_gaps.parent.mkdir(parents=True, exist_ok=True)
        with open(target_gaps, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["local_date", "period_start_utc", "reason"])
            for r in final_rows:
                writer.writerow([r["local_date"], r["period_start_utc"], r["reason"]])

    return new_gaps


def is_day_complete(
    local_date: date | str,
    frame: pd.DataFrame | None,
    gaps_path: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    today: date | None = None,
) -> bool:
    """Predicate determining if a past date is analysed to completion.

    A past date is complete when:
    - Date is strictly before today in Europe/London;
    - Partition exists and period_start_utc is unique;
    - rows + recorded_gaps == expected_rows for its DST type; and
    - Every null actual in the partition is already present in gaps.csv.
    """
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    eval_today = today if today is not None else datetime.now(LONDON_TZ).date()
    if d >= eval_today:
        return False

    if frame is None or frame.empty:
        return False

    if not frame["period_start_utc"].is_unique:
        return False

    if gaps_path is not None:
        target_gaps = gaps_path
    elif audit_dir is not None:
        target_gaps = audit_dir / "gaps.csv"
    elif curated_dir is not None and (curated_dir / "gaps.csv").exists():
        target_gaps = curated_dir / "gaps.csv"
    elif curated_dir is not None and curated_dir != CURATED_DIR_DEFAULT:
        target_gaps = curated_dir.parent / "audit" / "gaps.csv"
    else:
        target_gaps = AUDIT_DIR_DEFAULT / "gaps.csv"

    recorded_missing_periods = 0
    missing_actual_gaps: set[tuple[str, str]] = set()
    if target_gaps.exists():
        with open(target_gaps, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 2 and row[0] == str(d):
                        try:
                            ts_norm = pd.Timestamp(row[1]).isoformat()
                        except Exception:
                            ts_norm = row[1]
                        reason = row[2] if len(row) >= 3 else "missing_actual"
                        if reason == "missing_period":
                            recorded_missing_periods += 1
                        else:
                            missing_actual_gaps.add((row[0], ts_norm))

    expected = dst_row_count(d)
    if len(frame) + recorded_missing_periods != expected:
        return False

    null_actuals = frame[frame["actual"].isna()]
    for _, row in null_actuals.iterrows():
        try:
            ts_norm = pd.Timestamp(row["period_start_utc"]).isoformat()
        except Exception:
            ts_norm = str(row["period_start_utc"])
        if (str(d), ts_norm) not in missing_actual_gaps:
            return False

    return True


def ingest_day(
    local_date: date | str,
    client: Client | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    today: date | None = None,
    ingested_at: datetime | None = None,
) -> DayIngestResult:
    """Fetch, normalise, record gaps, and upsert a single local day."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date

    p_path = partition_path(d, curated_dir)

    # Resumable backfill: skip only if partition is analysed to completion:
    # - date is strictly before today in Europe/London
    # - partition exists and period_start_utc is unique
    # - rows + recorded_gaps == expected_rows for its DST type
    # - every null actual in the partition is already present in gaps.csv
    eval_today = today if today is not None else datetime.now(LONDON_TZ).date()

    if not force and d < eval_today and p_path.exists():
        existing = read_partition(d, curated_dir)
        if is_day_complete(
            d,
            existing,
            curated_dir=curated_dir,
            audit_dir=audit_dir,
            today=eval_today,
        ):
            assert existing is not None
            nulls_act = int(existing["actual"].isna().sum())
            return DayIngestResult(
                local_date=d,
                status="skipped",
                rows_written=len(existing),
                duplicates_dropped=0,
                gaps_recorded=0,
                rows_refreshed=0,
                nulls_remaining=nulls_act,
                used_fallback=False,
                nulls_forecast=int(existing["forecast"].isna().sum()),
                missing_periods=max(0, dst_row_count(d) - len(existing)),
            )

    client_inst = client or Client()
    now_ingested = ingested_at or datetime.now(UTC_TZ)
    payload, used_fallback = _fetch_day_with_fallback(
        client=client_inst,
        local_date=d,
        cache_dir=cache_dir,
        force=force,
        today=eval_today,
        now=now_ingested,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
    )

    if "data" not in payload:
        raise ValueError(f"Payload for {d} missing required 'data' key")

    if not payload["data"]:
        return DayIngestResult(
            local_date=d,
            status="no_data",
            rows_written=0,
            duplicates_dropped=0,
            gaps_recorded=0,
            rows_refreshed=0,
            nulls_remaining=0,
            used_fallback=used_fallback,
            nulls_forecast=0,
            missing_periods=0,
        )

    df = payload_to_frame(payload, d, ingested_at=now_ingested)

    upsert_res = upsert_day(df, d, base_dir=curated_dir, audit_dir=audit_dir)

    # Gaps must be computed on the merged partition
    merged = read_partition(d, curated_dir)
    if merged is None:
        merged = df

    if audit_dir is not None:
        gaps_path = audit_dir / "gaps.csv"
    elif curated_dir is not None and (curated_dir / "gaps.csv").exists():
        gaps_path = curated_dir / "gaps.csv"
    elif curated_dir is not None and curated_dir != CURATED_DIR_DEFAULT:
        gaps_path = curated_dir.parent / "audit" / "gaps.csv"
    else:
        gaps_path = AUDIT_DIR_DEFAULT / "gaps.csv"

    gaps = record_gaps(
        merged,
        today=now_ingested,
        gaps_path=gaps_path,
        local_date=d,
        eval_today=eval_today,
        audit_dir=audit_dir,
    )

    nulls_remaining = int(merged["actual"].isna().sum())
    nulls_forecast = int(merged["forecast"].isna().sum())
    missing_periods = max(0, dst_row_count(d) - len(merged)) if d < eval_today else 0

    return DayIngestResult(
        local_date=d,
        status="written",
        rows_written=upsert_res.rows_after,
        duplicates_dropped=upsert_res.duplicates_dropped,
        gaps_recorded=len(gaps),
        rows_refreshed=upsert_res.rows_refreshed,
        nulls_remaining=nulls_remaining,
        used_fallback=used_fallback,
        nulls_forecast=nulls_forecast,
        missing_periods=missing_periods,
    )


def ingest_range(
    start: date,
    end: date,
    force: bool = False,
    client: Client | None = None,
    cache_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    today: date | None = None,
) -> IngestSummary:
    """Iterate local dates from start to end, running resilient backfill with progress reporting."""
    if start > end:
        raise ValueError(f"Start date {start} must be on or before end date {end}")

    current_today = today or datetime.now(LONDON_TZ).date()
    client_inst = client or Client()

    dates_requested = 0
    dates_written = 0
    dates_skipped = 0
    rows_total = 0
    duplicates_dropped = 0
    rows_refreshed = 0
    gaps_recorded = 0
    nulls_remaining = 0
    failures = 0
    used_fallback = 0
    nulls_forecast = 0
    missing_periods = 0

    curr = start
    while curr <= end:
        dates_requested += 1
        try:
            res = ingest_day(
                local_date=curr,
                client=client_inst,
                force=force,
                cache_dir=cache_dir,
                curated_dir=curated_dir,
                audit_dir=audit_dir,
                today=current_today,
            )
            if res.status == "skipped":
                dates_skipped += 1
                rows_total += res.rows_written
                print(f"{curr}: skipped (complete, {res.rows_written} rows)")
            elif res.status == "no_data":
                print(f"{curr}: no data available (0 rows)")
            elif res.status == "written":
                dates_written += 1
                rows_total += res.rows_written
                duplicates_dropped += res.duplicates_dropped
                rows_refreshed += res.rows_refreshed
                gaps_recorded += res.gaps_recorded
                nulls_remaining += res.nulls_remaining
                missing_periods += res.missing_periods
                if res.used_fallback:
                    used_fallback += 1
                    nulls_forecast += res.nulls_forecast
                    if res.nulls_forecast > 0:
                        s_sfx = "s" if res.nulls_forecast != 1 else ""
                        v_verb = "lacks" if res.nulls_forecast == 1 else "lack"
                        lack_str = f" ({res.nulls_forecast} record{s_sfx} {v_verb} a forecast)"
                    else:
                        lack_str = ""
                    print(f"{curr}: {res.rows_written} rows written via range fallback{lack_str}")
                else:
                    s_miss = "s" if res.missing_periods != 1 else ""
                    miss_str = (
                        f", {res.missing_periods} missing period{s_miss}"
                        if res.missing_periods > 0
                        else ""
                    )
                    print(
                        f"{curr}: {res.rows_written} rows written "
                        f"({res.duplicates_dropped} duplicates dropped, "
                        f"{res.rows_refreshed} rows refreshed, "
                        f"{res.gaps_recorded} gaps recorded"
                        f"{miss_str}, "
                        f"{res.nulls_remaining} actuals pending)"
                    )
        except Exception as exc:
            failures += 1
            print(f"{curr}: FAILED ({exc})")
            logger.error("Failed to ingest date %s: %s", curr, exc, exc_info=True)

        curr += timedelta(days=1)

    print("\nIngest summary:")
    print(f"  dates requested:    {dates_requested}")
    print(f"  dates written:      {dates_written}")
    print(f"  dates skipped:      {dates_skipped}")
    print(f"  rows total:         {rows_total}")
    print(f"  duplicates dropped: {duplicates_dropped}")
    print(f"  rows refreshed:     {rows_refreshed}")
    print(f"  gaps recorded:      {gaps_recorded}")
    print(f"  nulls remaining:    {nulls_remaining}")
    print(f"  nulls forecast:     {nulls_forecast}")
    print(f"  used fallback:      {used_fallback}")
    if failures > 0:
        print(f"  failures:           {failures}")

    if failures == 0:
        pruned = prune_cache(keep_days=90, cache_dir=cache_dir)
        if pruned:
            print(f"Pruned {len(pruned)} cache file(s) older than 90 days.")

    return IngestSummary(
        dates_requested=dates_requested,
        dates_written=dates_written,
        dates_skipped=dates_skipped,
        rows_total=rows_total,
        duplicates_dropped=duplicates_dropped,
        rows_refreshed=rows_refreshed,
        gaps_recorded=gaps_recorded,
        nulls_remaining=nulls_remaining,
        failures=failures,
        used_fallback=used_fallback,
        nulls_forecast=nulls_forecast,
        missing_periods=missing_periods,
    )


def prune_cache(
    keep_days: int = 90,
    cache_dir: Path | None = None,
    now: float | None = None,
) -> list[Path]:
    """Prune raw HTTP cache files older than keep_days; never touches curated table."""
    target_cache = cache_dir if cache_dir is not None else CACHE_DIR_DEFAULT
    current_time = now if now is not None else time.time()
    cutoff_seconds = keep_days * 86400

    pruned: list[Path] = []
    if not target_cache.exists():
        return pruned

    for entry in target_cache.glob("*.json.gz"):
        try:
            stat = entry.stat()
            if (current_time - stat.st_mtime) > cutoff_seconds:
                entry.unlink(missing_ok=True)
                pruned.append(entry)
        except OSError as exc:
            logger.warning("Error pruning cache file %s: %s", entry, exc)

    return pruned


def reconcile_recent(
    window_days: int = 3,
    client: Client | None = None,
    cache_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    today: date | None = None,
    now: datetime | None = None,
) -> int:
    """Reconcile recent partitions that contain pending null actuals or gaps.

    Selects dates in [today - (window_days - 1), today] whose partition contains
    a null actual, OR which appear in data/audit/gaps.csv.
    For each qualifying date, bypasses the cache and fetches live (at most one request
    per date per run), coalescing observations into the curated partition.

    Returns the count of dates reconciled.
    """
    if window_days < 1:
        raise ValueError(f"window_days must be at least 1, got {window_days}")

    target_curated = curated_dir if curated_dir is not None else CURATED_DIR_DEFAULT
    if audit_dir is not None:
        target_audit = audit_dir
    elif target_curated.name == "curated":
        target_audit = target_curated.parent / "audit"
    else:
        target_audit = target_curated
    gaps_file = target_audit / "gaps.csv"

    eval_today = today if today is not None else datetime.now(LONDON_TZ).date()
    start_date = eval_today - timedelta(days=window_days - 1)

    # 1. Read dates recorded in gaps.csv
    gap_dates: set[str] = set()
    if gaps_file.exists():
        try:
            with open(gaps_file, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if row:
                            gap_dates.add(row[0])
        except Exception as exc:
            logger.warning("Error reading gaps file %s: %s", gaps_file, exc)

    # 2. Identify qualifying dates in [start_date, eval_today]
    qualifying_dates: list[date] = []
    curr = start_date
    while curr <= eval_today:
        d_str = str(curr)
        qualifies = False

        if d_str in gap_dates:
            qualifies = True
        else:
            p_path = partition_path(curr, target_curated)
            if p_path.exists():
                part = read_partition(curr, target_curated)
                if part is not None and "actual" in part.columns and part["actual"].isna().any():
                    qualifies = True

        if qualifies:
            qualifying_dates.append(curr)

        curr += timedelta(days=1)

    # 3. If nothing qualifies, make no requests at all
    if not qualifying_dates:
        print("reconciled: 0 date(s)")
        return 0

    client_inst = client or Client()
    now_ts = now or datetime.now(UTC_TZ)

    reconciled_count = 0
    for d in qualifying_dates:
        try:
            res = ingest_day(
                local_date=d,
                client=client_inst,
                force=True,  # Bypass cache and fetch live
                cache_dir=cache_dir,
                curated_dir=target_curated,
                audit_dir=target_audit,
                today=eval_today,
                ingested_at=now_ts,
            )
            reconciled_count += 1
            print(
                f"{d}: reconciled ({res.rows_refreshed} rows refreshed, "
                f"{res.nulls_remaining} actuals pending)"
            )
        except Exception as exc:
            print(f"{d}: reconcile FAILED ({exc})")
            logger.error("Failed to reconcile date %s: %s", d, exc, exc_info=True)

    print(f"reconciled: {reconciled_count} date(s)")
    return reconciled_count


def generate_monthly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into monthly intervals."""
    if start > end:
        raise ValueError(f"Start date {start} must be on or before end date {end}")
    chunks: list[tuple[date, date]] = []
    curr = start
    while curr <= end:
        if curr.month == 12:
            next_month = date(curr.year + 1, 1, 1)
        else:
            next_month = date(curr.year, curr.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((curr, chunk_end))
        curr = chunk_end + timedelta(days=1)
    return chunks


def expected_weather_row_count(local_date: date | str) -> int:
    """Return expected hourly row count (24, 23 on spring DST, 25 on autumn DST)."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=LONDON_TZ)
    next_d = d + timedelta(days=1)
    end_local = datetime(next_d.year, next_d.month, next_d.day, 0, 0, tzinfo=LONDON_TZ)
    diff_s = (end_local.astimezone(UTC_TZ) - start_local.astimezone(UTC_TZ)).total_seconds()
    return int(diff_s / 3600.0)


def is_weather_day_complete(
    local_date: date | str,
    weather_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> bool:
    """Assert whether a local date partition for weather exists and is complete."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    target_dir = weather_dir if weather_dir is not None else WEATHER_DIR_DEFAULT
    p = partition_path(d, base_dir=target_dir)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p)
        rows = len(df)
        gaps_count = 0
        target_gaps = (
            source_gaps_path
            if source_gaps_path is not None
            else (
                audit_dir / "source-gaps.csv"
                if audit_dir is not None
                else (
                    target_dir.parent / "audit" / "source-gaps.csv"
                    if target_dir != WEATHER_DIR_DEFAULT
                    else AUDIT_DIR_DEFAULT / "source-gaps.csv"
                )
            )
        )
        if target_gaps.exists():
            g_df = pd.read_csv(target_gaps)
            if not g_df.empty and "store" in g_df.columns:
                gaps_count = len(
                    g_df[
                        (g_df["store"] == "weather")
                        & (g_df["local_date"].astype(str) == str(d))
                    ]
                )

        if rows + gaps_count != expected_weather_row_count(d):
            return False
        if "temperature_2m_previous_day1" not in df.columns:
            return False
        if "temperature_2m_previous_day2" not in df.columns:
            return False
        if df["temperature_2m_previous_day1"].isna().any():
            return False
        return not df["temperature_2m_previous_day2"].isna().any()
    except Exception:
        return False


def is_mix_day_complete(
    local_date: date | str,
    mix_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> bool:
    """Assert whether a local date partition for generation mix exists and is complete."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    target_dir = mix_dir if mix_dir is not None else MIX_DIR_DEFAULT
    p = partition_path(d, base_dir=target_dir)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p)
        rows = len(df)
        gaps_count = 0
        target_gaps = (
            source_gaps_path
            if source_gaps_path is not None
            else (
                audit_dir / "source-gaps.csv"
                if audit_dir is not None
                else (
                    target_dir.parent / "audit" / "source-gaps.csv"
                    if target_dir != MIX_DIR_DEFAULT
                    else AUDIT_DIR_DEFAULT / "source-gaps.csv"
                )
            )
        )
        if target_gaps.exists():
            g_df = pd.read_csv(target_gaps)
            if not g_df.empty and "store" in g_df.columns:
                gaps_count = len(
                    g_df[
                        (g_df["store"] == "mix")
                        & (g_df["local_date"].astype(str) == str(d))
                    ]
                )

        if rows + gaps_count != dst_row_count(d):
            return False
        for fuel in ["wind", "solar", "gas", "coal", "nuclear", "biomass"]:
            if fuel not in df.columns or df[fuel].isna().all():
                return False
        return True
    except Exception:
        return False


def weather_payload_to_frame(payload: dict) -> pd.DataFrame:
    """Normalise Open-Meteo previous-runs weather payload to DataFrame."""
    h = payload.get("hourly", {})
    times = h.get("time", [])
    temps_d1 = h.get("temperature_2m_previous_day1", [])
    temps_d2 = h.get("temperature_2m_previous_day2", [])
    if not times:
        empty_df = pd.DataFrame(
            columns=[
                "period_start_utc",
                "local_date",
                "temperature_2m_previous_day1",
                "temperature_2m_previous_day2",
            ]
        )
        empty_df["period_start_utc"] = pd.to_datetime([], utc=True)
        empty_df["local_date"] = pd.Series([], dtype="object")
        empty_df["temperature_2m_previous_day1"] = pd.Series([], dtype="float64")
        empty_df["temperature_2m_previous_day2"] = pd.Series([], dtype="float64")
        return empty_df

    data_dict: dict[str, object] = {
        "period_start_utc": pd.to_datetime(times, utc=True),
    }
    if temps_d1:
        data_dict["temperature_2m_previous_day1"] = pd.Series(temps_d1, dtype="float64")
    if temps_d2:
        data_dict["temperature_2m_previous_day2"] = pd.Series(temps_d2, dtype="float64")

    df = pd.DataFrame(data_dict)
    df["local_date"] = df["period_start_utc"].dt.tz_convert(LONDON_TZ).dt.date
    df = (
        df.sort_values("period_start_utc")
        .drop_duplicates(subset=["period_start_utc"])
        .reset_index(drop=True)
    )

    # Check for null previous-runs temperature at or after 2024-11-01
    post_cutoff = df["period_start_utc"] >= pd.Timestamp("2024-11-01 00:00:00+00:00")
    for col in ("temperature_2m_previous_day1", "temperature_2m_previous_day2"):
        if col in df.columns:
            null_temps = post_cutoff & df[col].isna()
            if null_temps.any():
                bad_date = df.loc[null_temps, "period_start_utc"].iloc[0]
                raise ValueError(
                    f"Open-Meteo previous-runs returned null {col} for period {bad_date}. "
                    "Reanalysis fallback is strictly prohibited."
                )

    out_cols = [
        c
        for c in [
            "period_start_utc",
            "local_date",
            "temperature_2m_previous_day1",
            "temperature_2m_previous_day2",
        ]
        if c in df.columns
    ]
    return df[out_cols]


def mix_payload_to_frame(payload: dict) -> pd.DataFrame:
    """Normalise NESO generation mix payload to DataFrame."""
    data_items = payload.get("data", [])
    if not data_items:
        empty_df = pd.DataFrame(columns=["period_start_utc", "local_date"])
        empty_df["period_start_utc"] = pd.to_datetime([], utc=True)
        empty_df["local_date"] = pd.Series([], dtype="object")
        return empty_df

    rows = []
    for item in data_items:
        start_utc = pd.to_datetime(item["from"], utc=True)
        row_local_date = start_utc.tz_convert(LONDON_TZ).date()
        row_dict: dict[str, object] = {
            "period_start_utc": start_utc,
            "local_date": row_local_date,
        }
        for f in item.get("generationmix", []):
            fuel_key = str(f["fuel"]).lower().replace(" ", "_")
            perc_val = f.get("perc")
            row_dict[fuel_key] = float(perc_val) if perc_val is not None else float("nan")
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df = (
        df.sort_values("period_start_utc")
        .drop_duplicates(subset=["period_start_utc"])
        .reset_index(drop=True)
    )
    return df


def fetch_weather_chunk(
    client: Client,
    start_date: date,
    end_date: date,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Fetch raw Open-Meteo previous-runs JSON for a date range with raw cache."""
    c_dir = cache_dir if cache_dir is not None else WEATHER_CACHE_DIR_DEFAULT
    c_path = c_dir / f"{start_date.isoformat()}_{end_date.isoformat()}.json.gz"

    if c_path.exists() and not force:
        try:
            with gzip.open(c_path, "rt", encoding="utf-8") as f:
                cached_data = json.load(f)
                h_data = cached_data.get("hourly", {})
                if (
                    "temperature_2m_previous_day1" in h_data
                    and "temperature_2m_previous_day2" in h_data
                ):
                    return cached_data
        except Exception:
            logger.warning("Corrupted weather cache %s; refetching", c_path)

    s_fetch = start_date - timedelta(days=1)
    e_fetch = end_date + timedelta(days=1)
    payload = client.get_weather_previous_runs(start_date=s_fetch, end_date=e_fetch)

    c_dir.mkdir(parents=True, exist_ok=True)
    temp_cache = c_dir / f"{c_path.name}.tmp.{os.getpid()}"
    try:
        with gzip.open(temp_cache, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(temp_cache, c_path)
    finally:
        if temp_cache.exists():
            temp_cache.unlink(missing_ok=True)

    return payload


def fetch_mix_chunk(
    client: Client,
    start_date: date,
    end_date: date,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Fetch raw NESO generation mix JSON for a date range with raw cache."""
    c_dir = cache_dir if cache_dir is not None else MIX_CACHE_DIR_DEFAULT
    c_path = c_dir / f"{start_date.isoformat()}_{end_date.isoformat()}.json.gz"

    if c_path.exists() and not force:
        try:
            with gzip.open(c_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Corrupted mix cache %s; refetching", c_path)

    start_local = datetime(
        start_date.year, start_date.month, start_date.day, 0, 0, tzinfo=LONDON_TZ
    )
    end_local = datetime(
        end_date.year, end_date.month, end_date.day, 0, 0, tzinfo=LONDON_TZ
    ) + timedelta(days=1)
    start_utc = start_local.astimezone(UTC_TZ)
    end_utc = end_local.astimezone(UTC_TZ)

    payload = client.get_generation_range(start_utc=start_utc, end_utc=end_utc)

    c_dir.mkdir(parents=True, exist_ok=True)
    temp_cache = c_dir / f"{c_path.name}.tmp.{os.getpid()}"
    try:
        with gzip.open(temp_cache, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(temp_cache, c_path)
    finally:
        if temp_cache.exists():
            temp_cache.unlink(missing_ok=True)

    return payload


def upsert_weather_day(
    day_df: pd.DataFrame,
    local_date: date | str,
    base_dir: Path | None = None,
) -> UpsertResult:
    """Upsert hourly weather DataFrame into daily partition atomically."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    target_dir = base_dir if base_dir is not None else WEATHER_DIR_DEFAULT
    p_path = partition_path(d, base_dir=target_dir)

    df_to_write = day_df[day_df["local_date"] == d].copy()
    df_to_write = df_to_write.sort_values("period_start_utc").reset_index(drop=True)

    incoming_unique = df_to_write["period_start_utc"].nunique()
    incoming_duplicates = len(df_to_write) - incoming_unique

    if not p_path.exists():
        write_partition_atomic(df_to_write, d, base_dir=target_dir)
        return UpsertResult(
            rows_before=0,
            rows_after=len(df_to_write),
            duplicates_dropped=incoming_duplicates,
        )

    existing = pd.read_parquet(p_path)
    rows_before = len(existing)
    existing_duplicates = len(existing) - existing["period_start_utc"].nunique()
    duplicates_dropped = existing_duplicates + incoming_duplicates

    combined = pd.concat([existing, df_to_write], ignore_index=True)
    combined = combined.drop_duplicates(subset=["period_start_utc"], keep="last")
    target_cols = [
        "period_start_utc",
        "local_date",
        "temperature_2m_previous_day1",
        "temperature_2m_previous_day2",
    ]
    present_cols = [c for c in target_cols if c in combined.columns]
    combined = combined[present_cols]
    combined = combined.sort_values("period_start_utc").reset_index(drop=True)

    if len(existing) == len(combined) and existing.equals(combined):
        return UpsertResult(
            rows_before=rows_before, rows_after=len(existing), duplicates_dropped=duplicates_dropped
        )

    write_partition_atomic(combined, d, base_dir=target_dir)
    return UpsertResult(
        rows_before=rows_before, rows_after=len(combined), duplicates_dropped=duplicates_dropped
    )


def upsert_mix_day(
    day_df: pd.DataFrame,
    local_date: date | str,
    base_dir: Path | None = None,
) -> UpsertResult:
    """Upsert generation mix DataFrame into daily partition atomically."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    target_dir = base_dir if base_dir is not None else MIX_DIR_DEFAULT
    p_path = partition_path(d, base_dir=target_dir)

    df_to_write = day_df[day_df["local_date"] == d].copy()
    df_to_write = df_to_write.sort_values("period_start_utc").reset_index(drop=True)

    incoming_unique = df_to_write["period_start_utc"].nunique()
    incoming_duplicates = len(df_to_write) - incoming_unique

    if not p_path.exists():
        write_partition_atomic(df_to_write, d, base_dir=target_dir)
        return UpsertResult(
            rows_before=0,
            rows_after=len(df_to_write),
            duplicates_dropped=incoming_duplicates,
        )

    existing = pd.read_parquet(p_path)
    rows_before = len(existing)
    existing_duplicates = len(existing) - existing["period_start_utc"].nunique()
    duplicates_dropped = existing_duplicates + incoming_duplicates

    combined = pd.concat([existing, df_to_write], ignore_index=True)
    combined = combined.drop_duplicates(subset=["period_start_utc"], keep="last")
    combined = combined.sort_values("period_start_utc").reset_index(drop=True)

    if len(existing) == len(combined) and existing.equals(combined):
        return UpsertResult(
            rows_before=rows_before, rows_after=len(existing), duplicates_dropped=duplicates_dropped
        )

    write_partition_atomic(combined, d, base_dir=target_dir)
    return UpsertResult(
        rows_before=rows_before, rows_after=len(combined), duplicates_dropped=duplicates_dropped
    )


def record_source_gaps(
    store: str,
    missing_periods: list[tuple[date, pd.Timestamp]] | list[tuple[str, str]],
    present_periods: set[pd.Timestamp] | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
    reason: str = "upstream_empty",
) -> list[SourceGap]:
    """Record unrecoverable source gaps in data/audit/source-gaps.csv with reconciliation.

    - Write one row per period that remains missing after the retry, with store in
      weather|mix and reason describing the cause (upstream_empty).
    - Created lazily and deleted when empty, exactly like gaps.csv.
    - Reconciliation applies: if a later fetch fills a recorded gap, drop the row.
    - Preserves unique (store, period_start_utc).
    """
    if source_gaps_path is not None:
        target_path = source_gaps_path
    elif audit_dir is not None:
        target_path = audit_dir / "source-gaps.csv"
    else:
        target_path = AUDIT_DIR_DEFAULT / "source-gaps.csv"

    existing_rows: list[dict[str, str]] = []
    if target_path.exists():
        with open(target_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 4:
                        try:
                            ts_norm = pd.Timestamp(row[2]).isoformat()
                        except Exception:
                            ts_norm = row[2]
                        existing_rows.append({
                            "store": row[0],
                            "local_date": row[1],
                            "period_start_utc": ts_norm,
                            "reason": row[3],
                        })

    # Reconcile existing rows for this store against present_periods
    kept_rows: list[dict[str, str]] = []
    reconciled_any = False
    present_ts_set: set[str] = set()
    if present_periods:
        for p in present_periods:
            try:
                present_ts_set.add(pd.Timestamp(p).isoformat())
            except Exception:
                present_ts_set.add(str(p))

    for r in existing_rows:
        if r["store"] == store and r["period_start_utc"] in present_ts_set:
            reconciled_any = True
            continue
        kept_rows.append(r)

    kept_keys = {(r["store"], r["period_start_utc"]) for r in kept_rows}
    new_gaps: list[SourceGap] = []

    for item in missing_periods:
        l_date, p_start = item
        l_date_str = str(l_date)
        try:
            start_str = pd.Timestamp(p_start).isoformat()
        except Exception:
            start_str = str(p_start)

        if (store, start_str) not in kept_keys:
            gap = SourceGap(
                store=store,
                local_date=l_date_str,
                period_start_utc=start_str,
                reason=reason,
            )
            new_gaps.append(gap)
            kept_keys.add((store, start_str))

    final_rows = kept_rows + [
        {
            "store": g.store,
            "local_date": g.local_date,
            "period_start_utc": g.period_start_utc,
            "reason": g.reason,
        }
        for g in new_gaps
    ]

    # Sort final rows for deterministic output: store, local_date, period_start_utc
    final_rows.sort(key=lambda r: (r["store"], r["local_date"], r["period_start_utc"]))

    if not final_rows:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
    elif new_gaps or reconciled_any or not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.parent / f"{target_path.name}.tmp.{os.getpid()}"
        try:
            with open(temp_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["store", "local_date", "period_start_utc", "reason"])
                for r in final_rows:
                    writer.writerow(
                        [r["store"], r["local_date"], r["period_start_utc"], r["reason"]]
                    )
            os.replace(temp_file, target_path)
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    return new_gaps


def check_mix_coverage_and_retry(
    start_date: date,
    end_date: date,
    client: Client | None = None,
    mix_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Post-fetch coverage check with targeted retry for generation mix.

    - Compares fetched period starts against curated grid for those dates (exact half hour).
    - For each missing period, re-requests window beginning at period_start_utc + 30min
      (which includes the period as a straddling period before 'from').
    - Dedupes and keeps whatever lands.
    - Records unrecoverable gaps into source-gaps.csv.

    Returns (periods_recovered, gaps_recorded).
    """
    client_inst = client or Client()
    m_dir = mix_dir if mix_dir is not None else MIX_DIR_DEFAULT
    c_dir = curated_dir if curated_dir is not None else (
        m_dir.parent / "curated" if m_dir != MIX_DIR_DEFAULT else CURATED_DIR_DEFAULT
    )
    a_dir = audit_dir if audit_dir is not None else (
        m_dir.parent / "audit" if m_dir != MIX_DIR_DEFAULT else AUDIT_DIR_DEFAULT
    )

    all_curated_periods: list[tuple[date, pd.Timestamp]] = []
    all_present_starts: set[pd.Timestamp] = set()

    curr = start_date
    while curr <= end_date:
        c_p = partition_path(curr, base_dir=c_dir)
        if c_p.exists():
            c_df = pd.read_parquet(c_p)
            for ts in pd.to_datetime(c_df["period_start_utc"], utc=True):
                all_curated_periods.append((curr, ts))
        else:
            count = dst_row_count(curr)
            start_local = datetime(curr.year, curr.month, curr.day, 0, 0, tzinfo=LONDON_TZ)
            start_utc = start_local.astimezone(UTC_TZ)
            for i in range(count):
                period_ts = pd.Timestamp(start_utc + timedelta(minutes=30 * i))
                all_curated_periods.append((curr, period_ts))

        m_p = partition_path(curr, base_dir=m_dir)
        if m_p.exists():
            m_df = pd.read_parquet(m_p)
            all_present_starts.update(pd.to_datetime(m_df["period_start_utc"], utc=True))

        curr += timedelta(days=1)

    missing = [(d, ts) for d, ts in all_curated_periods if ts not in all_present_starts]

    if not missing:
        record_source_gaps(
            store="mix",
            missing_periods=[],
            present_periods=all_present_starts,
            audit_dir=a_dir,
            source_gaps_path=source_gaps_path,
        )
        return 0, 0

    periods_recovered = 0
    for _d, p_start in missing:
        s_dt = p_start.astimezone(UTC_TZ) + timedelta(minutes=30)
        e_dt = s_dt + timedelta(minutes=30)
        retry_start = s_dt.strftime("%Y-%m-%dT%H:%MZ")
        retry_end = e_dt.strftime("%Y-%m-%dT%H:%MZ")
        try:
            payload = client_inst.get_generation_range(start_utc=retry_start, end_utc=retry_end)
            frame = mix_payload_to_frame(payload)
            if not frame.empty:
                for ret_date, grp in frame.groupby("local_date"):
                    upsert_mix_day(grp, ret_date, base_dir=m_dir)
        except Exception as exc:
            logger.warning("Targeted retry for %s failed: %s", p_start, exc)

    recovered_present: set[pd.Timestamp] = set()
    curr = start_date
    while curr <= end_date:
        m_p = partition_path(curr, base_dir=m_dir)
        if m_p.exists():
            m_df = pd.read_parquet(m_p)
            recovered_present.update(pd.to_datetime(m_df["period_start_utc"], utc=True))
        curr += timedelta(days=1)

    still_missing: list[tuple[date, pd.Timestamp]] = []
    for d, ts in missing:
        if ts in recovered_present:
            periods_recovered += 1
        else:
            still_missing.append((d, ts))

    record_source_gaps(
        store="mix",
        missing_periods=still_missing,
        present_periods=recovered_present,
        audit_dir=a_dir,
        source_gaps_path=source_gaps_path,
        reason="upstream_empty",
    )

    return periods_recovered, len(still_missing)


def check_weather_coverage_and_retry(
    start_date: date,
    end_date: date,
    client: Client | None = None,
    weather_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Post-fetch coverage check with targeted retry for weather.

    - Compares fetched period starts against curated grid for those dates (containing hour).
    - For each missing hour, re-requests Open-Meteo previous-runs for that date.
    - Dedupes and keeps whatever lands.
    - Records unrecoverable gaps into source-gaps.csv.

    Returns (hours_recovered, gaps_recorded).
    """
    client_inst = client or Client()
    w_dir = weather_dir if weather_dir is not None else WEATHER_DIR_DEFAULT
    c_dir = curated_dir if curated_dir is not None else (
        w_dir.parent / "curated" if w_dir != WEATHER_DIR_DEFAULT else CURATED_DIR_DEFAULT
    )
    a_dir = audit_dir if audit_dir is not None else (
        w_dir.parent / "audit" if w_dir != WEATHER_DIR_DEFAULT else AUDIT_DIR_DEFAULT
    )

    all_curated_hours: list[tuple[date, pd.Timestamp]] = []
    all_present_starts: set[pd.Timestamp] = set()

    curr = start_date
    while curr <= end_date:
        c_p = partition_path(curr, base_dir=c_dir)
        if c_p.exists():
            c_df = pd.read_parquet(c_p)
            p_starts = pd.to_datetime(c_df["period_start_utc"], utc=True)
            unique_hours = sorted({ts.floor("h") for ts in p_starts})
            for h in unique_hours:
                all_curated_hours.append((curr, h))
        else:
            count = expected_weather_row_count(curr)
            start_local = datetime(curr.year, curr.month, curr.day, 0, 0, tzinfo=LONDON_TZ)
            start_utc = start_local.astimezone(UTC_TZ)
            for i in range(count):
                hour_ts = pd.Timestamp(start_utc + timedelta(hours=i))
                all_curated_hours.append((curr, hour_ts))

        w_p = partition_path(curr, base_dir=w_dir)
        if w_p.exists():
            w_df = pd.read_parquet(w_p)
            all_present_starts.update(pd.to_datetime(w_df["period_start_utc"], utc=True))

        curr += timedelta(days=1)

    missing = [(d, h) for d, h in all_curated_hours if h not in all_present_starts]

    if not missing:
        record_source_gaps(
            store="weather",
            missing_periods=[],
            present_periods=all_present_starts,
            audit_dir=a_dir,
            source_gaps_path=source_gaps_path,
        )
        return 0, 0

    hours_recovered = 0
    missing_dates = sorted({d for d, _ in missing})
    for d in missing_dates:
        try:
            payload = client_inst.get_weather_previous_runs(start_date=d, end_date=d)
            frame = weather_payload_to_frame(payload)
            day_data = frame[frame["local_date"] == d]
            if not day_data.empty:
                upsert_weather_day(day_data, d, base_dir=w_dir)
        except Exception as exc:
            logger.warning("Targeted weather retry for %s failed: %s", d, exc)

    recovered_present: set[pd.Timestamp] = set()
    curr = start_date
    while curr <= end_date:
        w_p = partition_path(curr, base_dir=w_dir)
        if w_p.exists():
            w_df = pd.read_parquet(w_p)
            recovered_present.update(pd.to_datetime(w_df["period_start_utc"], utc=True))
        curr += timedelta(days=1)

    still_missing: list[tuple[date, pd.Timestamp]] = []
    for d, h in missing:
        if h in recovered_present:
            hours_recovered += 1
        else:
            still_missing.append((d, h))

    record_source_gaps(
        store="weather",
        missing_periods=still_missing,
        present_periods=recovered_present,
        audit_dir=a_dir,
        source_gaps_path=source_gaps_path,
        reason="upstream_empty",
    )

    return hours_recovered, len(still_missing)


def ingest_weather_day(
    local_date: date | str,
    client: Client | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
    weather_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> DayIngestResult:
    """Ingest a single local day of weather data."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    w_dir = weather_dir if weather_dir is not None else WEATHER_DIR_DEFAULT
    if not force and is_weather_day_complete(
        d, weather_dir=w_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
    ):
        existing_p = partition_path(d, base_dir=w_dir)
        rows = len(pd.read_parquet(existing_p))
        return DayIngestResult(
            local_date=d,
            status="skipped",
            rows_written=rows,
            duplicates_dropped=0,
            gaps_recorded=0,
        )

    client_inst = client or Client()
    payload = fetch_weather_chunk(
        client=client_inst,
        start_date=d,
        end_date=d,
        cache_dir=cache_dir,
        force=force,
    )
    frame = weather_payload_to_frame(payload)
    day_frame = frame[frame["local_date"] == d]
    if day_frame.empty:
        check_weather_coverage_and_retry(
            start_date=d,
            end_date=d,
            client=client_inst,
            weather_dir=w_dir,
            curated_dir=curated_dir,
            audit_dir=audit_dir,
            source_gaps_path=source_gaps_path,
            force=force,
        )
        return DayIngestResult(
            local_date=d,
            status="no_data",
            rows_written=0,
            duplicates_dropped=0,
            gaps_recorded=0,
        )

    res = upsert_weather_day(day_frame, d, base_dir=w_dir)
    recovered, gaps_recorded = check_weather_coverage_and_retry(
        start_date=d,
        end_date=d,
        client=client_inst,
        weather_dir=w_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        source_gaps_path=source_gaps_path,
        force=force,
    )
    return DayIngestResult(
        local_date=d,
        status="written",
        rows_written=res.rows_after + recovered,
        duplicates_dropped=res.duplicates_dropped,
        gaps_recorded=gaps_recorded,
    )


def ingest_weather_range(
    start: date,
    end: date,
    force: bool = False,
    client: Client | None = None,
    cache_dir: Path | None = None,
    weather_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> IngestSummary:
    """Ingest weather forecast data for a range of dates using monthly chunks."""
    if start > end:
        raise ValueError(f"Start date {start} must be on or before end date {end}")

    client_inst = client or Client()
    chunks = generate_monthly_chunks(start, end)
    w_dir = weather_dir if weather_dir is not None else WEATHER_DIR_DEFAULT
    c_dir = cache_dir if cache_dir is not None else WEATHER_CACHE_DIR_DEFAULT

    dates_requested = 0
    dates_written = 0
    dates_skipped = 0
    rows_total = 0
    duplicates_dropped = 0
    failures = 0
    initial_request_count = getattr(client_inst, "request_count", 0)

    for c_start, c_end in chunks:
        chunk_dates = [c_start + timedelta(days=i) for i in range((c_end - c_start).days + 1)]
        need_fetch = force or any(
            not is_weather_day_complete(
                d, weather_dir=w_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
            )
            for d in chunk_dates
        )

        chunk_df: pd.DataFrame | None = None
        if need_fetch:
            try:
                payload = fetch_weather_chunk(
                    client=client_inst,
                    start_date=c_start,
                    end_date=c_end,
                    cache_dir=c_dir,
                    force=force,
                )
                chunk_df = weather_payload_to_frame(payload)
            except Exception as exc:
                failures += len(chunk_dates)
                print(f"Weather chunk {c_start} to {c_end}: FAILED ({exc})")
                logger.error(
                    "Failed weather chunk %s to %s: %s", c_start, c_end, exc, exc_info=True
                )
                continue

        for d in chunk_dates:
            dates_requested += 1
            if not force and is_weather_day_complete(
                d, weather_dir=w_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
            ):
                dates_skipped += 1
                existing_p = partition_path(d, base_dir=w_dir)
                existing_rows = len(pd.read_parquet(existing_p))
                rows_total += existing_rows
                print(f"{d}: skipped (complete, {existing_rows} rows)")
                continue

            if chunk_df is None:
                continue

            day_data = chunk_df[chunk_df["local_date"] == d]
            if day_data.empty:
                print(f"{d}: no weather data available (0 rows)")
                continue

            res = upsert_weather_day(day_data, d, base_dir=w_dir)
            dates_written += 1
            rows_total += res.rows_after
            duplicates_dropped += res.duplicates_dropped
            print(
                f"{d}: {res.rows_after} weather rows written "
                f"({res.duplicates_dropped} duplicates dropped)"
            )

    recovered, source_gaps_count = check_weather_coverage_and_retry(
        start_date=start,
        end_date=end,
        client=client_inst,
        weather_dir=w_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        source_gaps_path=source_gaps_path,
        force=force,
    )

    requests_made = getattr(client_inst, "request_count", 0) - initial_request_count

    print("\nWeather ingest summary:")
    print(f"  dates requested:    {dates_requested}")
    print(f"  dates written:      {dates_written}")
    print(f"  dates skipped:      {dates_skipped}")
    print(f"  rows total:         {rows_total + recovered}")
    print(f"  duplicates dropped: {duplicates_dropped}")
    print(f"  hours recovered:    {recovered}")
    print(f"  source gaps:        {source_gaps_count}")
    print(f"  requests made:      {requests_made}")
    if failures > 0:
        print(f"  failures:           {failures}")

    return IngestSummary(
        dates_requested=dates_requested,
        dates_written=dates_written,
        dates_skipped=dates_skipped,
        rows_total=rows_total + recovered,
        duplicates_dropped=duplicates_dropped,
        failures=failures,
        periods_recovered=recovered,
        gaps_recorded=source_gaps_count,
    )


def ingest_mix_day(
    local_date: date | str,
    client: Client | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
    mix_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> DayIngestResult:
    """Ingest a single local day of generation mix data."""
    d = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    m_dir = mix_dir if mix_dir is not None else MIX_DIR_DEFAULT
    if not force and is_mix_day_complete(
        d, mix_dir=m_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
    ):
        existing_p = partition_path(d, base_dir=m_dir)
        rows = len(pd.read_parquet(existing_p))
        return DayIngestResult(
            local_date=d,
            status="skipped",
            rows_written=rows,
            duplicates_dropped=0,
            gaps_recorded=0,
        )

    client_inst = client or Client()
    payload = fetch_mix_chunk(
        client=client_inst,
        start_date=d,
        end_date=d,
        cache_dir=cache_dir,
        force=force,
    )
    frame = mix_payload_to_frame(payload)
    day_frame = frame[frame["local_date"] == d]
    if day_frame.empty:
        check_mix_coverage_and_retry(
            start_date=d,
            end_date=d,
            client=client_inst,
            mix_dir=m_dir,
            curated_dir=curated_dir,
            audit_dir=audit_dir,
            source_gaps_path=source_gaps_path,
            force=force,
        )
        return DayIngestResult(
            local_date=d,
            status="no_data",
            rows_written=0,
            duplicates_dropped=0,
            gaps_recorded=0,
        )

    res = upsert_mix_day(day_frame, d, base_dir=m_dir)
    recovered, gaps_recorded = check_mix_coverage_and_retry(
        start_date=d,
        end_date=d,
        client=client_inst,
        mix_dir=m_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        source_gaps_path=source_gaps_path,
        force=force,
    )
    return DayIngestResult(
        local_date=d,
        status="written",
        rows_written=res.rows_after + recovered,
        duplicates_dropped=res.duplicates_dropped,
        gaps_recorded=gaps_recorded,
    )


def ingest_mix_range(
    start: date,
    end: date,
    force: bool = False,
    client: Client | None = None,
    cache_dir: Path | None = None,
    mix_dir: Path | None = None,
    curated_dir: Path | None = None,
    audit_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> IngestSummary:
    """Ingest generation mix data for a range of dates using monthly chunks."""
    if start > end:
        raise ValueError(f"Start date {start} must be on or before end date {end}")

    client_inst = client or Client()
    chunks = generate_monthly_chunks(start, end)
    m_dir = mix_dir if mix_dir is not None else MIX_DIR_DEFAULT
    c_dir = cache_dir if cache_dir is not None else MIX_CACHE_DIR_DEFAULT

    dates_requested = 0
    dates_written = 0
    dates_skipped = 0
    rows_total = 0
    duplicates_dropped = 0
    failures = 0
    initial_request_count = getattr(client_inst, "request_count", 0)

    for c_start, c_end in chunks:
        chunk_dates = [c_start + timedelta(days=i) for i in range((c_end - c_start).days + 1)]
        need_fetch = force or any(
            not is_mix_day_complete(
                d, mix_dir=m_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
            )
            for d in chunk_dates
        )

        chunk_df: pd.DataFrame | None = None
        if need_fetch:
            try:
                payload = fetch_mix_chunk(
                    client=client_inst,
                    start_date=c_start,
                    end_date=c_end,
                    cache_dir=c_dir,
                    force=force,
                )
                chunk_df = mix_payload_to_frame(payload)
            except Exception as exc:
                failures += len(chunk_dates)
                print(f"Generation mix chunk {c_start} to {c_end}: FAILED ({exc})")
                logger.error(
                    "Failed mix chunk %s to %s: %s", c_start, c_end, exc, exc_info=True
                )
                continue

        for d in chunk_dates:
            dates_requested += 1
            if not force and is_mix_day_complete(
                d, mix_dir=m_dir, audit_dir=audit_dir, source_gaps_path=source_gaps_path
            ):
                dates_skipped += 1
                existing_p = partition_path(d, base_dir=m_dir)
                existing_rows = len(pd.read_parquet(existing_p))
                rows_total += existing_rows
                print(f"{d}: skipped (complete, {existing_rows} rows)")
                continue

            if chunk_df is None:
                continue

            day_data = chunk_df[chunk_df["local_date"] == d]
            if day_data.empty:
                print(f"{d}: no mix data available (0 rows)")
                continue

            res = upsert_mix_day(day_data, d, base_dir=m_dir)
            dates_written += 1
            rows_total += res.rows_after
            duplicates_dropped += res.duplicates_dropped
            print(
                f"{d}: {res.rows_after} mix rows written "
                f"({res.duplicates_dropped} duplicates dropped)"
            )

    recovered, source_gaps_count = check_mix_coverage_and_retry(
        start_date=start,
        end_date=end,
        client=client_inst,
        mix_dir=m_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        source_gaps_path=source_gaps_path,
        force=force,
    )

    requests_made = getattr(client_inst, "request_count", 0) - initial_request_count

    print("\nGeneration mix ingest summary:")
    print(f"  dates requested:    {dates_requested}")
    print(f"  dates written:      {dates_written}")
    print(f"  dates skipped:      {dates_skipped}")
    print(f"  rows total:         {rows_total + recovered}")
    print(f"  duplicates dropped: {duplicates_dropped}")
    print(f"  periods recovered:  {recovered}")
    print(f"  source gaps:        {source_gaps_count}")
    print(f"  requests made:      {requests_made}")
    if failures > 0:
        print(f"  failures:           {failures}")

    return IngestSummary(
        dates_requested=dates_requested,
        dates_written=dates_written,
        dates_skipped=dates_skipped,
        rows_total=rows_total + recovered,
        duplicates_dropped=duplicates_dropped,
        failures=failures,
        periods_recovered=recovered,
        gaps_recorded=source_gaps_count,
    )



