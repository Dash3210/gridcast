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
CACHE_DIR_DEFAULT = Path("data/cache/http")
AUDIT_DIR_DEFAULT = Path("data/audit")

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
    "UpsertResult",
    "dst_row_count",
    "expected_row_count",
    "fetch_day",
    "fetch_range_for_day",
    "ingest_day",
    "ingest_range",
    "is_cache_usable",
    "is_day_complete",
    "partition_path",
    "payload_to_frame",
    "prune_cache",
    "read_partition",
    "record_conflicts",
    "record_gaps",
    "upsert_day",
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

        rows.append(
            {
                "period_start_utc": start_utc,
                "period_end_utc": end_utc,
                "utc_date": start_utc.date(),
                "local_date": row_local_date,
                "forecast": intensity.get("forecast"),
                "actual": intensity.get("actual"),
                "index": intensity.get("index"),
                "ingested_at": ingested_ts,
            }
        )

    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
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
                    prior["actual"] = row_dict["actual"]
                    prior["index"] = row_dict["index"]
                prior["ingested_at"] = max(prior["ingested_at"], row_dict["ingested_at"])

        # 2. Coalesce incoming rows: atomic (actual, index), independent forecast, conflicts
        conflicts: list[tuple[str, str, str, str]] = []
        frame_sorted = frame.sort_values(by=["period_start_utc", "ingested_at"])
        for _, row in frame_sorted.iterrows():
            key = row["period_start_utc"]
            row_dict = row.to_dict()
            if key not in records:
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
    ]
    deduped = deduped[columns]

    deduped["period_start_utc"] = pd.to_datetime(deduped["period_start_utc"], utc=True)
    deduped["period_end_utc"] = pd.to_datetime(deduped["period_end_utc"], utc=True)
    deduped["ingested_at"] = pd.to_datetime(deduped["ingested_at"], utc=True)
    deduped["forecast"] = deduped["forecast"].astype("Int32")
    deduped["actual"] = deduped["actual"].astype("Int32")
    deduped["index"] = deduped["index"].astype("string")

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
