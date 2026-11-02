"""Tests for half-hourly ingest pipeline, schema adherence, and storage invariants."""

import json
import os
import socket
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import requests

from gridcast.ingest import (
    Client,
    dst_row_count,
    expected_row_count,
    expected_weather_row_count,
    fetch_day,
    fetch_mix_chunk,
    fetch_weather_chunk,
    generate_monthly_chunks,
    ingest_day,
    ingest_mix_day,
    ingest_mix_range,
    ingest_range,
    ingest_weather_day,
    ingest_weather_range,
    is_mix_day_complete,
    is_weather_day_complete,
    mix_payload_to_frame,
    partition_path,
    payload_to_frame,
    prune_cache,
    read_partition,
    reconcile_recent,
    record_gaps,
    upsert_day,
    upsert_mix_day,
    upsert_weather_day,
    weather_payload_to_frame,
    write_partition_atomic,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def normal_payload() -> dict:
    with open(FIXTURES_DIR / "day_normal.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def spring_payload() -> dict:
    with open(FIXTURES_DIR / "day_spring_forward.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def autumn_payload() -> dict:
    with open(FIXTURES_DIR / "day_autumn_back.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def empty_payload() -> dict:
    with open(FIXTURES_DIR / "day_empty.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def partial_payload() -> dict:
    with open(FIXTURES_DIR / "day_partial_today.json", encoding="utf-8") as f:
        return json.load(f)


def test_schema_assertion(normal_payload: dict) -> None:
    """1. Schema assertion: exact curated columns, correct dtypes, strictly increasing."""
    ingested_at = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = payload_to_frame(normal_payload, date(2026, 10, 15), ingested_at)

    expected_columns = [
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
    assert list(df.columns) == expected_columns

    # Dtype contract
    assert str(df["period_start_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(df["period_end_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(df["ingested_at"].dtype) == "datetime64[ns, UTC]"
    assert str(df["first_seen_at"].dtype) == "datetime64[ns, UTC]"
    assert str(df["forecast"].dtype) == "Int32"
    assert str(df["actual"].dtype) == "Int32"
    assert str(df["index"].dtype) in ("string", "string[python]")

    # Check date column values are datetime.date
    assert all(isinstance(v, date) for v in df["utc_date"])
    assert all(isinstance(v, date) for v in df["local_date"])

    # Uniqueness and strictly increasing
    assert df["period_start_utc"].is_unique
    assert df["period_start_utc"].is_monotonic_increasing
    assert len(df) == 48


def test_dst_correctness(autumn_payload: dict, spring_payload: dict) -> None:
    """2. DST correctness: 50 periods and 46 periods, local_date matching requested date."""
    ingested_at = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Autumn back (50 periods)
    req_autumn = date(2026, 10, 25)
    df_autumn = payload_to_frame(autumn_payload, req_autumn, ingested_at)
    assert len(df_autumn) == 50
    assert df_autumn["period_start_utc"].iloc[0] == pd.Timestamp("2026-10-24T23:00:00Z")
    assert all(d == req_autumn for d in df_autumn["local_date"])

    # Spring forward (46 periods)
    req_spring = date(2026, 3, 29)
    df_spring = payload_to_frame(spring_payload, req_spring, ingested_at)
    assert len(df_spring) == 46
    assert df_spring["period_start_utc"].iloc[0] == pd.Timestamp("2026-03-29T00:00:00Z")
    assert all(d == req_spring for d in df_spring["local_date"])


def test_idempotency(normal_payload: dict, tmp_path: Path) -> None:
    """3. Idempotency: upserting same day twice yields identical row count and parquet bytes."""
    ingested_at = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    target_date = date(2026, 10, 15)
    df = payload_to_frame(normal_payload, target_date, ingested_at)

    res1 = upsert_day(df, target_date, base_dir=tmp_path)
    p_path = partition_path(target_date, base_dir=tmp_path)
    assert p_path.exists()
    bytes1 = p_path.read_bytes()

    res2 = upsert_day(df, target_date, base_dir=tmp_path)
    bytes2 = p_path.read_bytes()

    assert res1.rows_after == len(df)
    assert res2.rows_after == len(df)
    assert res2.duplicates_dropped == 0
    assert res2.rows_refreshed == 0
    assert bytes1 == bytes2


def test_pending_fills_in(normal_payload: dict, tmp_path: Path) -> None:
    """4. Pending fills in: second upsert with known actuals replaces nulls without duplicates."""
    target_date = date(2026, 10, 15)
    t1 = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 15, 23, 59, 0, tzinfo=timezone.utc)

    # Make a copy of normal_payload where the last 10 periods have actual = None
    partial_data = json.loads(json.dumps(normal_payload))
    for item in partial_data["data"][-10:]:
        item["intensity"]["actual"] = None

    df_initial = payload_to_frame(partial_data, target_date, t1)
    assert df_initial["actual"].isna().sum() == 10

    res1 = upsert_day(df_initial, target_date, base_dir=tmp_path)
    assert res1.rows_after == 48

    # Later frame has all actuals populated and later ingested_at
    df_later = payload_to_frame(normal_payload, target_date, t2)
    assert not df_later["actual"].isna().any()

    res2 = upsert_day(df_later, target_date, base_dir=tmp_path)
    assert res2.rows_after == 48
    assert res2.duplicates_dropped == 0
    assert res2.rows_refreshed == 10

    df_read = read_partition(target_date, base_dir=tmp_path)
    assert df_read is not None
    assert len(df_read) == 48
    assert not df_read["actual"].isna().any()


def test_first_seen_at_lifecycle(normal_payload: dict, tmp_path: Path) -> None:
    """Assert first_seen_at transitions on arrival and stays unchanged on re-ingest."""
    target_date = date(2026, 10, 15)
    t1 = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 15, 13, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)

    # 1. First payload where one actual is null
    p1 = json.loads(json.dumps(normal_payload))
    p1["data"][5]["intensity"]["actual"] = None
    p1["data"][5]["intensity"]["index"] = None

    df1 = payload_to_frame(p1, target_date, t1)
    assert pd.isna(df1.loc[5, "actual"])
    assert pd.isna(df1.loc[5, "first_seen_at"])
    # Assert a null actual never carries a first_seen_at
    assert df1.loc[df1["actual"].isna(), "first_seen_at"].isna().all()

    upsert_day(df1, target_date, base_dir=tmp_path)
    read1 = read_partition(target_date, base_dir=tmp_path)
    assert read1 is not None
    assert pd.isna(read1.loc[5, "actual"])
    assert pd.isna(read1.loc[5, "first_seen_at"])
    assert read1.loc[read1["actual"].isna(), "first_seen_at"].isna().all()
    assert read1.loc[0, "first_seen_at"] == pd.Timestamp(t1)

    # 2. Later payload where it is known
    p2 = json.loads(json.dumps(normal_payload))
    known_val = normal_payload["data"][5]["intensity"]["actual"]
    assert known_val is not None
    df2 = payload_to_frame(p2, target_date, t2)
    assert df2.loc[5, "actual"] == known_val

    upsert_day(df2, target_date, base_dir=tmp_path)
    read2 = read_partition(target_date, base_dir=tmp_path)
    assert read2 is not None
    # Assert first_seen_at is set on the transition
    assert read2.loc[5, "actual"] == known_val
    assert read2.loc[5, "first_seen_at"] == pd.Timestamp(t2)
    assert read2.loc[0, "first_seen_at"] == pd.Timestamp(t1)

    # 3. Third identical ingest
    df3 = payload_to_frame(p2, target_date, t3)
    upsert_day(df3, target_date, base_dir=tmp_path)
    read3 = read_partition(target_date, base_dir=tmp_path)
    assert read3 is not None
    # Assert unchanged by a third identical ingest
    assert read3.loc[5, "first_seen_at"] == pd.Timestamp(t2)
    assert read3.loc[0, "first_seen_at"] == pd.Timestamp(t1)
    assert read3.loc[read3["actual"].isna(), "first_seen_at"].isna().all()


def test_pending_vs_gap(tmp_path: Path) -> None:
    """5. Pending vs gap: pending is not logged, gap is logged once; no interpolation."""
    eval_time = datetime(2026, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
    gaps_file = tmp_path / "gaps.csv"

    # Row 1: ended 10 mins ago -> Pending (< 2 hours)
    end_recent = eval_time - timedelta(minutes=10)
    start_recent = end_recent - timedelta(minutes=30)

    # Row 2: ended 24 hours ago -> Gap (>= 2 hours)
    end_old = eval_time - timedelta(days=1)
    start_old = end_old - timedelta(minutes=30)

    df = pd.DataFrame(
        [
            {
                "period_start_utc": pd.Timestamp(start_recent),
                "period_end_utc": pd.Timestamp(end_recent),
                "utc_date": start_recent.date(),
                "local_date": start_recent.date(),
                "forecast": 50,
                "actual": None,
                "index": "low",
                "ingested_at": eval_time,
            },
            {
                "period_start_utc": pd.Timestamp(start_old),
                "period_end_utc": pd.Timestamp(end_old),
                "utc_date": start_old.date(),
                "local_date": start_old.date(),
                "forecast": 60,
                "actual": None,
                "index": "moderate",
                "ingested_at": eval_time,
            },
        ]
    )
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")

    # First call records only the old row
    gaps = record_gaps(df, today=eval_time, gaps_path=gaps_file)
    assert len(gaps) == 1
    assert gaps[0].period_start_utc == pd.Timestamp(start_old).isoformat()

    # Second call appends no duplicates
    gaps_again = record_gaps(df, today=eval_time, gaps_path=gaps_file)
    assert len(gaps_again) == 0

    lines = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 gap record
    assert lines[0] == "local_date,period_start_utc,reason"
    assert pd.Timestamp(start_old).isoformat() in lines[1]


def test_retry_behaviour() -> None:
    """6. Retry: 503 retries and succeeds on 3rd attempt; 400 raises immediately; 429 retries."""
    sleep_calls: list[float] = []

    def mock_sleep(s: float) -> None:
        sleep_calls.append(s)

    mock_session = MagicMock()
    client = Client(
        session=mock_session,
        sleep_fn=mock_sleep,
        time_fn=lambda: 1000.0,
        min_request_interval=0.0,
        jitter_fn=lambda _: 0.0,
        base_backoff=1.0,
        backoff_factor=2.0,
    )

    # Scenario A: 503, 503, 200
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"data": []}

    mock_session.get.side_effect = [resp_503, resp_503, resp_200]
    data = client.get_day(date(2026, 10, 15))
    assert data == {"data": []}
    assert mock_session.get.call_count == 3
    assert len(sleep_calls) == 2
    assert sleep_calls == [1.0, 2.0]

    # Scenario B: 400 Client Error -> does NOT retry
    mock_session.reset_mock()
    mock_session.get.side_effect = None
    sleep_calls.clear()

    resp_400 = MagicMock()
    resp_400.status_code = 400
    resp_400.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
    mock_session.get.side_effect = [resp_400]

    with pytest.raises(requests.HTTPError) as exc_info:
        client.get_day(date(2026, 10, 15))
    assert "400" in str(exc_info.value)
    assert mock_session.get.call_count == 1
    assert len(sleep_calls) == 0

    # Scenario C: 429 Rate Limit -> retried
    mock_session.reset_mock()
    mock_session.get.side_effect = None
    sleep_calls.clear()

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    mock_session.get.side_effect = [resp_429, resp_200]

    data_429 = client.get_day(date(2026, 10, 15))
    assert data_429 == {"data": []}
    assert mock_session.get.call_count == 2
    assert len(sleep_calls) == 1


def test_prune_cache(tmp_path: Path) -> None:
    """7. Prune: raw cache older than 90 days pruned; newer cache and curated files kept."""
    cache_http = tmp_path / "data" / "cache" / "http"
    curated = tmp_path / "data" / "curated"
    cache_http.mkdir(parents=True, exist_ok=True)
    curated.mkdir(parents=True, exist_ok=True)

    old_cache = cache_http / "2026-01-01.json.gz"
    old_cache.write_bytes(b"old_data")
    new_cache = cache_http / "2026-10-30.json.gz"
    new_cache.write_bytes(b"new_data")

    old_curated = curated / "date=2026-01-01.parquet"
    old_curated.write_bytes(b"curated_data")

    # Set mtime
    now = time.time()
    os.utime(old_cache, (now - 100 * 86400, now - 100 * 86400))
    os.utime(new_cache, (now - 10 * 86400, now - 10 * 86400))
    os.utime(old_curated, (now - 100 * 86400, now - 100 * 86400))

    pruned = prune_cache(keep_days=90, cache_dir=cache_http)
    assert old_cache in pruned
    assert not old_cache.exists()
    assert new_cache.exists()
    assert old_curated.exists()


def test_no_network_in_tests() -> None:
    """8. No network in tests: attempting a socket connection raises RuntimeError."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError, match="Network access is strictly forbidden"):
        s.connect(("8.8.8.8", 53))


def test_expected_row_count() -> None:
    """Test expected row count computed from Europe/London convention."""
    today = date(2026, 11, 1)

    # Settled past days
    assert expected_row_count(date(2026, 3, 29), today=today) == 46
    assert expected_row_count(date(2026, 10, 25), today=today) == 50
    assert expected_row_count(date(2026, 10, 15), today=today) == 48

    # Today and future days
    assert expected_row_count(date(2026, 11, 1), today=today) is None
    assert expected_row_count(date(2026, 11, 5), today=today) is None


def test_empty_payload_handling(empty_payload: dict) -> None:
    """Empty payload returns empty frame with correct schema."""
    df = payload_to_frame(
        empty_payload, date(2017, 9, 11), datetime(2026, 11, 1, tzinfo=timezone.utc)
    )
    assert len(df) == 0
    assert list(df.columns) == [
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


def test_client_rate_limiting() -> None:
    """Client enforces min_request_interval between calls."""
    sleep_calls: list[float] = []
    current_time = [100.0]

    def mock_time() -> float:
        return current_time[0]

    def mock_sleep(s: float) -> None:
        sleep_calls.append(s)
        current_time[0] += s

    mock_session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": []}
    mock_session.get.return_value = resp

    client = Client(
        session=mock_session,
        min_request_interval=0.4,
        sleep_fn=mock_sleep,
        time_fn=mock_time,
    )

    client.get_day(date(2026, 10, 15))
    assert len(sleep_calls) == 0

    # Immediate second call without elapsed time
    client.get_day(date(2026, 10, 16))
    assert len(sleep_calls) == 1
    assert pytest.approx(sleep_calls[0], 0.01) == 0.4


def test_client_network_exceptions_and_exhaustion() -> None:
    """Client retries ConnectionError and Timeout; raises when attempts exhausted."""
    sleeps: list[float] = []
    mock_session = MagicMock()
    client = Client(
        session=mock_session,
        sleep_fn=lambda s: sleeps.append(s),
        time_fn=lambda: 1000.0,
        min_request_interval=0.0,
        max_attempts=3,
        jitter_fn=lambda _: 0.0,
    )

    # 1. ConnectionError retried then succeeds
    resp_200 = MagicMock(status_code=200)
    resp_200.json.return_value = {"data": []}
    mock_session.get.side_effect = [
        requests.exceptions.ConnectionError("network down"),
        resp_200,
    ]
    res = client.get_day(date(2026, 10, 15))
    assert res == {"data": []}
    assert mock_session.get.call_count == 2
    assert sleeps == [1.0]

    # 2. Timeout retried up to max_attempts then raises
    mock_session.reset_mock()
    mock_session.get.side_effect = requests.exceptions.Timeout("timed out")
    sleeps.clear()
    with pytest.raises(requests.exceptions.Timeout):
        client.get_day(date(2026, 10, 15))
    assert mock_session.get.call_count == 3
    assert len(sleeps) == 2

    # 3. HTTP 500 retried up to max_attempts then raises HTTPError
    mock_session.reset_mock()
    mock_session.get.side_effect = None
    sleeps.clear()
    resp_500 = MagicMock(status_code=500)
    resp_500.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")
    mock_session.get.return_value = resp_500

    with pytest.raises(requests.HTTPError):
        client.get_day(date(2026, 10, 15))
    assert mock_session.get.call_count == 3


def test_fetch_day_cache_flow(normal_payload: dict, tmp_path: Path) -> None:
    """fetch_day uses cache buffer, supports force, and recovers from corrupt cache."""
    cache_dir = tmp_path / "cache" / "http"
    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload
    d = date(2026, 10, 15)

    # 1. Initial fetch: no cache, calls client, writes cache
    data1 = fetch_day(mock_client, d, cache_dir=cache_dir)
    assert data1 == normal_payload
    assert mock_client.get_day.call_count == 1
    c_file = cache_dir / "2026-10-15.json.gz"
    assert c_file.exists()

    # 2. Second fetch: hits cache, client not called
    mock_client.reset_mock()
    data2 = fetch_day(mock_client, d, cache_dir=cache_dir)
    assert data2 == normal_payload
    assert mock_client.get_day.call_count == 0

    # 3. Force fetch: bypasses cache, client called, cache updated
    mock_client.reset_mock()
    data3 = fetch_day(mock_client, d, cache_dir=cache_dir, force=True)
    assert data3 == normal_payload
    assert mock_client.get_day.call_count == 1

    # 4. Corrupt cache: refetches
    c_file.write_bytes(b"invalid gzip content")
    mock_client.reset_mock()
    data4 = fetch_day(mock_client, d, cache_dir=cache_dir)
    assert data4 == normal_payload
    assert mock_client.get_day.call_count == 1


def test_payload_to_frame_duplicate_error() -> None:
    """payload_to_frame raises ValueError if duplicate from timestamps exist."""
    bad_payload = {
        "data": [
            {
                "from": "2026-10-15T00:00Z",
                "to": "2026-10-15T00:30Z",
                "intensity": {"forecast": 50, "actual": 50, "index": "low"},
            },
            {
                "from": "2026-10-15T00:00Z",
                "to": "2026-10-15T00:30Z",
                "intensity": {"forecast": 60, "actual": 60, "index": "low"},
            },
        ]
    }
    with pytest.raises(ValueError, match="Duplicate period_start_utc"):
        payload_to_frame(bad_payload, date(2026, 10, 15))


def test_ingest_day_lifecycle(normal_payload: dict, empty_payload: dict, tmp_path: Path) -> None:
    """ingest_day written, skipped when complete, force re-ingest, and no_data handling."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)

    # 1. First ingest -> written
    res1 = ingest_day(
        d,
        client=mock_client,
        cache_dir=cache,
        curated_dir=curated,
        today=today,
    )
    assert res1.status == "written"
    assert res1.rows_written == 48

    # 2. Second ingest without force -> skipped
    mock_client.reset_mock()
    res2 = ingest_day(
        d,
        client=mock_client,
        cache_dir=cache,
        curated_dir=curated,
        today=today,
    )
    assert res2.status == "skipped"
    assert res2.rows_written == 48
    assert mock_client.get_day.call_count == 0

    # 3. Third ingest with force=True -> written
    res3 = ingest_day(
        d,
        client=mock_client,
        force=True,
        cache_dir=cache,
        curated_dir=curated,
        today=today,
    )
    assert res3.status == "written"
    assert res3.rows_written == 48

    # 4. Ingest empty day -> no_data
    mock_client.reset_mock()
    mock_client.get_day.return_value = empty_payload
    d_empty = date(2017, 9, 11)
    res_empty = ingest_day(
        d_empty,
        client=mock_client,
        cache_dir=cache,
        curated_dir=curated,
        today=today,
    )
    assert res_empty.status == "no_data"
    assert res_empty.rows_written == 0


def test_ingest_range(normal_payload: dict, tmp_path: Path) -> None:
    """ingest_range iterates dates, validates start<=end, and continues past errors."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    mock_client = MagicMock()
    day_1 = date(2026, 10, 14)
    day_2 = date(2026, 10, 15)
    day_3 = date(2026, 10, 16)

    # Date order validation
    with pytest.raises(ValueError, match="Start date.*must be on or before end date"):
        ingest_range(start=day_2, end=day_1)

    # Mock client behavior: day_1 succeeds, day_2 raises error, day_3 succeeds
    def mock_get(d: date | str) -> dict:
        curr_d = date.fromisoformat(str(d))
        if str(curr_d) == "2026-10-15":
            raise RuntimeError("API timeout")
        day_delta = curr_d - date(2026, 10, 15)
        p = json.loads(json.dumps(normal_payload))
        for item in p["data"]:
            t_from = pd.to_datetime(item["from"], utc=True) + day_delta
            t_to = pd.to_datetime(item["to"], utc=True) + day_delta
            item["from"] = t_from.strftime("%Y-%m-%dT%H:%MZ")
            item["to"] = t_to.strftime("%Y-%m-%dT%H:%MZ")
        return p

    mock_client.get_day.side_effect = mock_get

    summary = ingest_range(
        start=day_1,
        end=day_3,
        client=mock_client,
        cache_dir=cache,
        curated_dir=curated,
        today=date(2026, 11, 1),
    )
    assert summary.dates_requested == 3
    assert summary.dates_written == 2
    assert summary.failures == 1
    assert summary.rows_total == 96
    assert summary.nulls_remaining == 0


def test_ingest_range_reporting_format(
    normal_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ingest_range formats progress line with actuals pending and summary with nulls remaining."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    today = date(2026, 10, 15)

    payload = json.loads(json.dumps(normal_payload))
    for item in payload["data"][-12:]:
        item["intensity"]["actual"] = None

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    summary = ingest_range(
        start=today,
        end=today,
        client=mock_client,
        cache_dir=cache,
        curated_dir=curated,
        today=today,
    )

    captured = capsys.readouterr().out
    expected_line = (
        f"48 rows written (0 duplicates dropped, 0 rows refreshed, "
        f"{summary.gaps_recorded} gaps recorded, 12 actuals pending)"
    )
    assert expected_line in captured
    assert "  nulls remaining:    12" in captured
    assert summary.nulls_remaining == 12


def test_cli_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI ingest options: --prune-cache, date validation, and routing."""
    from gridcast import cli

    # 1. Prune cache option exits 0
    monkeypatch.setattr(
        "gridcast.cli.prune_cache",
        lambda keep_days: [Path("test.json.gz")],
    )
    assert cli.main(["ingest", "--prune-cache"]) == 0

    # 2. Invalid date range (start > end) exits 2
    assert cli.main(["ingest", "--start", "2026-10-25", "--end", "2026-10-20"]) == 2

    # 3. Only --start provided without --end exits 2
    assert cli.main(["ingest", "--start", "2026-10-20"]) == 2

    # 4. Invalid date format exits 2
    assert cli.main(["ingest", "--start", "not-a-date", "--end", "2026-10-25"]) == 2

    # 5. Invalid days (< 1) exits 2
    assert cli.main(["ingest", "--days", "0"]) == 2

    # 6. Mock successful ingest_range
    mock_summary = MagicMock(failures=0)
    monkeypatch.setattr("gridcast.cli.ingest_range", lambda **kw: mock_summary)

    assert cli.main(["ingest", "--days", "3"]) == 0
    assert cli.main(["ingest", "--recent", "3"]) == 0
    assert cli.main(["ingest", "--start", "2026-10-20", "--end", "2026-10-22"]) == 0
    assert cli.main(["ingest"]) == 0  # default nightly path

    # 7. Ingest range with failure exits 1
    mock_fail_summary = MagicMock(failures=1)
    monkeypatch.setattr("gridcast.cli.ingest_range", lambda **kw: mock_fail_summary)
    assert cli.main(["ingest"]) == 1


def test_prune_cache_nonexistent(tmp_path: Path) -> None:
    """prune_cache on nonexistent directory returns empty list."""
    nonexistent = tmp_path / "does_not_exist"
    assert prune_cache(cache_dir=nonexistent) == []


def test_stale_nulls_are_refetched(normal_payload: dict, tmp_path: Path) -> None:
    """1. Past date partition with complete rows and a null actual is refetched and filled."""
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"

    # Create an existing partition with 48 rows, 1 of which has null actual
    df = payload_to_frame(normal_payload, d, datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    assert len(df) == 48
    df.loc[0, "actual"] = None
    assert df["actual"].isna().sum() == 1
    write_partition_atomic(df, d, base_dir=curated)

    # Client returns full payload where that actual is now present
    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=today,
    )

    # Must NOT be skipped
    assert res.status == "written"
    assert mock_client.get_day.call_count == 1
    assert res.nulls_remaining == 0

    # Partition must be filled (0 nulls remaining)
    updated = read_partition(d, curated)
    assert updated is not None
    assert len(updated) == 48
    assert updated["actual"].isna().sum() == 0


def test_today_is_never_skipped(normal_payload: dict, tmp_path: Path) -> None:
    """2. A partition for today with complete row count must fetch, not skip."""
    today = date(2026, 10, 15)
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"

    # Create an existing partition for today with 48 rows and no nulls
    df = payload_to_frame(normal_payload, today, datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc))
    write_partition_atomic(df, today, base_dir=curated)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        today,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=today,
    )

    assert res.status == "written"
    assert mock_client.get_day.call_count == 1


def test_complete_past_day_is_skipped(normal_payload: dict, tmp_path: Path) -> None:
    """3. A past date with complete rows and no nulls must skip."""
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"

    df = payload_to_frame(normal_payload, d, datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    write_partition_atomic(df, d, base_dir=curated)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=today,
    )

    assert res.status == "skipped"
    assert res.rows_written == 48
    assert res.nulls_remaining == 0
    assert mock_client.get_day.call_count == 0


def test_long_standing_null_is_recorded_as_gap(normal_payload: dict, tmp_path: Path) -> None:
    """4. Null actual whose period ended days ago appears in gaps.csv once across two runs."""
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    curated = tmp_path / "curated"
    audit = tmp_path / "audit"
    cache = tmp_path / "cache"
    gaps_file = audit / "gaps.csv"

    # Payload with 1 null actual
    payload = json.loads(json.dumps(normal_payload))
    payload["data"][0]["intensity"]["actual"] = None

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    # Run 1
    res1 = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        audit_dir=audit,
        cache_dir=cache,
        today=today,
    )
    assert res1.status == "written"
    assert res1.gaps_recorded == 1
    assert res1.nulls_remaining == 1

    lines_run1 = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_run1) == 2  # header + 1 gap

    # Run 2: ingest again (still returning null)
    res2 = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        audit_dir=audit,
        cache_dir=cache,
        today=today,
    )
    assert res2.status == "skipped"  # analysed to completion, skipped
    assert res2.gaps_recorded == 0  # not recorded again as gap (deduped)
    assert res2.nulls_remaining == 1

    lines_run2 = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_run2) == 2  # exactly once across two runs


def test_pending_null_is_not_a_gap(tmp_path: Path) -> None:
    """5. A null with a period that ended 10 minutes ago is recorded nowhere in gaps.csv."""
    eval_time = datetime(2026, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
    today = date(2026, 11, 1)
    curated = tmp_path / "curated"
    audit = tmp_path / "audit"
    cache = tmp_path / "cache"
    gaps_file = audit / "gaps.csv"

    end_recent = eval_time - timedelta(minutes=10)
    start_recent = end_recent - timedelta(minutes=30)

    payload = {
        "data": [
            {
                "from": start_recent.isoformat(),
                "to": end_recent.isoformat(),
                "intensity": {"forecast": 50, "actual": None, "index": "low"},
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    res = ingest_day(
        today,
        client=mock_client,
        curated_dir=curated,
        audit_dir=audit,
        cache_dir=cache,
        today=today,
        ingested_at=eval_time,
    )

    assert res.status == "written"
    assert res.gaps_recorded == 0
    assert res.nulls_remaining == 1
    if gaps_file.exists():
        assert len(gaps_file.read_text(encoding="utf-8").strip().splitlines()) <= 1


def test_malformed_json_is_retried() -> None:
    """1. Client retries malformed JSON with backoff and succeeds on 3rd attempt."""
    sleep_calls: list[float] = []
    mock_session = MagicMock()
    client = Client(
        session=mock_session,
        sleep_fn=lambda s: sleep_calls.append(s),
        time_fn=lambda: 1000.0,
        min_request_interval=0.0,
        max_attempts=3,
        base_backoff=1.0,
        backoff_factor=2.0,
        jitter_fn=lambda _: 0.0,
    )

    resp_bad1 = MagicMock(status_code=200)
    resp_bad1.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

    resp_bad2 = MagicMock(status_code=200)
    resp_bad2.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

    resp_good = MagicMock(status_code=200)
    resp_good.json.return_value = {"data": []}

    mock_session.get.side_effect = [resp_bad1, resp_bad2, resp_good]

    data = client.get_day(date(2025, 1, 12))
    assert data == {"data": []}
    assert mock_session.get.call_count == 3
    assert sleep_calls == [1.0, 2.0]


def test_malformed_json_falls_back_to_range(tmp_path: Path) -> None:
    """2. Date malformed, range valid: ingest falls back, used_fallback is True."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2025, 1, 12)

    base_start = datetime(2025, 1, 12, 0, 0, tzinfo=timezone.utc)
    range_data = []
    for i in range(48):
        p_start = base_start + timedelta(minutes=30 * i)
        p_end = p_start + timedelta(minutes=30)
        range_data.append(
            {
                "from": p_start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": p_end.strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": 100 + i, "actual": 90 + i, "index": "low"},
            }
        )

    mock_client = MagicMock()
    mock_client.get_day.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_client.get_range.return_value = {"data": range_data}

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )

    assert res.status == "written"
    assert res.used_fallback is True
    assert res.rows_written == 48

    p_df = read_partition(d, curated)
    assert p_df is not None
    assert len(p_df) == 48


def test_fallback_filters_to_local_day(tmp_path: Path) -> None:
    """3. Fallback range spanning two local dates on BST filters strictly to requested local day."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2025, 8, 11)  # BST date: UTC is 2025-08-10 23:00 to 2025-08-11 23:00

    range_data = []
    # 1 extra period on 2025-08-10 (22:30Z = 23:30 BST on 2025-08-10)
    range_data.append(
        {
            "from": "2025-08-10T22:30Z",
            "to": "2025-08-10T23:00Z",
            "intensity": {"forecast": 100, "actual": 95, "index": "low"},
        }
    )
    # 48 periods on 2025-08-11 (from 2025-08-10T23:00Z to 2025-08-11T23:00Z)
    base_start = datetime(2025, 8, 10, 23, 0, tzinfo=timezone.utc)
    for i in range(48):
        p_start = base_start + timedelta(minutes=30 * i)
        p_end = p_start + timedelta(minutes=30)
        range_data.append(
            {
                "from": p_start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": p_end.strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": 100 + i, "actual": 100 + i, "index": "low"},
            }
        )

    mock_client = MagicMock()
    mock_client.get_day.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_client.get_range.return_value = {"data": range_data}

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )

    assert res.status == "written"
    assert res.used_fallback is True
    assert res.rows_written == 48

    p_df = read_partition(d, curated)
    assert p_df is not None
    assert len(p_df) == 48
    assert all(ld == d for ld in p_df["local_date"])
    assert p_df["period_start_utc"].iloc[0] == pd.Timestamp("2025-08-10T23:00:00Z")


def test_fallback_preserves_null_forecast(tmp_path: Path) -> None:
    """4. Range fallback preserves null forecasts without imputing and counts nulls_forecast."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2025, 1, 12)

    range_data = []
    base_start = datetime(2025, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(48):
        p_start = base_start + timedelta(minutes=30 * i)
        p_end = p_start + timedelta(minutes=30)
        fc = None if i in (10, 20) else (150 + i)
        range_data.append(
            {
                "from": p_start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": p_end.strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": fc, "actual": 140 + i, "index": "moderate"},
            }
        )

    mock_client = MagicMock()
    mock_client.get_day.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_client.get_range.return_value = {"data": range_data}

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )

    assert res.status == "written"
    assert res.used_fallback is True
    assert res.nulls_forecast == 2

    p_df = read_partition(d, curated)
    assert p_df is not None
    assert len(p_df) == 48
    assert p_df["forecast"].isna().sum() == 2
    row_null_fc = p_df[p_df["forecast"].isna()].iloc[0]
    assert pd.isna(row_null_fc["forecast"])
    assert not pd.isna(row_null_fc["actual"])


def test_both_endpoints_broken_raises(tmp_path: Path) -> None:
    """5. If both endpoints fail, ingest_day raises and ingest_range reports failure."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2025, 1, 12)

    mock_client = MagicMock()
    mock_client.get_day.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_client.get_range.side_effect = requests.HTTPError("500 Server Error")

    with pytest.raises(requests.HTTPError):
        ingest_day(
            d,
            client=mock_client,
            curated_dir=curated,
            cache_dir=cache,
            today=date(2026, 11, 1),
        )

    summary = ingest_range(
        start=d,
        end=d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )
    assert summary.failures == 1
    assert summary.dates_written == 0
    assert not partition_path(d, curated).exists()


def test_empty_data_confirmed_once(tmp_path: Path) -> None:
    """6. Empty data on/after 2017-09-12 triggers 1 range call; pre-2017-09-12 triggers 0 calls."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"

    mock_client = MagicMock()
    mock_client.get_day.return_value = {"data": []}
    mock_client.get_range.return_value = {"data": []}

    d_post = date(2020, 1, 1)
    res_post = ingest_day(
        d_post,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )
    assert res_post.status == "no_data"
    assert mock_client.get_range.call_count == 1

    mock_client.reset_mock()
    d_pre = date(2017, 9, 5)
    res_pre = ingest_day(
        d_pre,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )
    assert res_pre.status == "no_data"
    assert mock_client.get_range.call_count == 0


def test_duplicate_reporting_separates_refresh(normal_payload: dict, tmp_path: Path) -> None:
    """7. Forcing refresh reports duplicates_dropped == 0 and rows_refreshed == 48."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2026, 10, 15)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res1 = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )
    assert res1.status == "written"
    assert res1.duplicates_dropped == 0
    assert res1.rows_refreshed == 0
    assert res1.rows_written == 48

    res2 = ingest_day(
        d,
        client=mock_client,
        force=True,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )
    assert res2.status == "written"
    assert res2.duplicates_dropped == 0
    assert res2.rows_refreshed == 0
    assert res2.rows_written == 48


def test_client_get_range() -> None:
    """Client.get_range formats datetime and string arguments and calls API."""
    mock_session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"test": 1}]}
    mock_session.get.return_value = resp

    client = Client(session=mock_session, min_request_interval=0.0)

    # 1. datetime with tzinfo
    t_start = datetime(2025, 1, 12, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2025, 1, 13, 0, 0, tzinfo=timezone.utc)
    res1 = client.get_range(t_start, t_end)
    assert res1 == {"data": [{"test": 1}]}
    assert "2025-01-12T00:00Z/2025-01-13T00:00Z" in mock_session.get.call_args[0][0]

    # 2. naive datetime and string arguments
    client.get_range(datetime(2025, 1, 12, 0, 0), "2025-01-13T00:00Z")
    assert "2025-01-12T00:00Z/2025-01-13T00:00Z" in mock_session.get.call_args[0][0]


def test_ingest_range_fallback_reporting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ingest_range reports range fallback in progress lines and summary."""
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"
    d = date(2025, 1, 12)

    base_start = datetime(2025, 1, 12, 0, 0, tzinfo=timezone.utc)
    range_data = []
    for i in range(48):
        p_start = base_start + timedelta(minutes=30 * i)
        p_end = p_start + timedelta(minutes=30)
        fc = None if i in (10, 20) else (150 + i)
        range_data.append(
            {
                "from": p_start.strftime("%Y-%m-%dT%H:%MZ"),
                "to": p_end.strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": fc, "actual": 140 + i, "index": "moderate"},
            }
        )

    mock_client = MagicMock()
    mock_client.get_day.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_client.get_range.return_value = {"data": range_data}

    summary = ingest_range(
        start=d,
        end=d,
        client=mock_client,
        curated_dir=curated,
        cache_dir=cache,
        today=date(2026, 11, 1),
    )

    captured = capsys.readouterr().out
    assert "48 rows written via range fallback (2 records lack a forecast)" in captured
    assert "  used fallback:      1" in captured
    assert "  nulls forecast:     2" in captured
    assert summary.used_fallback == 1
    assert summary.nulls_forecast == 2


def test_stale_cache_bypassed_for_unsettled_actuals(
    normal_payload: dict, tmp_path: Path
) -> None:
    """1. Stale cache with unsettled null actual is bypassed and client is called."""
    import gzip

    cache_dir = tmp_path / "cache" / "http"
    curated_dir = tmp_path / "curated"
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)

    # Cached payload has 48 records, but last 10 have actual=None (unsettled, ended > 2h ago)
    stale_payload = json.loads(json.dumps(normal_payload))
    for item in stale_payload["data"][-10:]:
        item["intensity"]["actual"] = None

    c_file = cache_dir / f"{d.isoformat()}.json.gz"
    with gzip.open(c_file, "wt", encoding="utf-8") as f:
        json.dump(stale_payload, f)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        d,
        client=mock_client,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        today=today,
    )

    assert mock_client.get_day.call_count == 1
    assert res.status == "written"
    assert res.nulls_remaining == 0

    # Cache file was overwritten with settled payload
    with gzip.open(c_file, "rt", encoding="utf-8") as f:
        updated_cache = json.load(f)
    actuals = [x["intensity"]["actual"] for x in updated_cache["data"]]
    assert None not in actuals


def test_cache_used_for_settled_historical_day(
    normal_payload: dict, tmp_path: Path
) -> None:
    """2. Settled cached historical day is served from cache with zero client calls."""
    import gzip

    cache_dir = tmp_path / "cache" / "http"
    curated_dir = tmp_path / "curated"
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)

    c_file = cache_dir / f"{d.isoformat()}.json.gz"
    with gzip.open(c_file, "wt", encoding="utf-8") as f:
        json.dump(normal_payload, f)

    mock_client = MagicMock()

    res = ingest_day(
        d,
        client=mock_client,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        today=today,
    )

    assert mock_client.get_day.call_count == 0
    assert res.status == "written"
    assert res.rows_written == 48


def test_today_always_fetched_live(normal_payload: dict, tmp_path: Path) -> None:
    """3. Today must always be fetched live even if cache exists."""
    import gzip

    cache_dir = tmp_path / "cache" / "http"
    curated_dir = tmp_path / "curated"
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date(2026, 10, 15)

    c_file = cache_dir / f"{today.isoformat()}.json.gz"
    with gzip.open(c_file, "wt", encoding="utf-8") as f:
        json.dump(normal_payload, f)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        today,
        client=mock_client,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        today=today,
    )

    assert mock_client.get_day.call_count == 1
    assert res.status == "written"


def test_overlong_partition_is_repaired(normal_payload: dict, tmp_path: Path) -> None:
    """4. Overlong partition (49 rows with a duplicate) is repaired to 48 unique rows."""
    curated_dir = tmp_path / "curated"
    cache_dir = tmp_path / "cache"
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)

    # Create 48-row partition
    df = payload_to_frame(normal_payload, d, datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    # Inject a duplicate row
    dup_row = df.iloc[[0]].copy()
    overlong_df = pd.concat([df, dup_row], ignore_index=True)
    assert len(overlong_df) == 49
    write_partition_atomic(overlong_df, d, base_dir=curated_dir)

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated_dir,
        cache_dir=cache_dir,
        today=today,
        force=False,
    )

    assert res.status == "written"  # Must NOT be skipped
    assert res.rows_written == 48
    assert res.duplicates_dropped == 1
    repaired = read_partition(d, curated_dir)
    assert repaired is not None
    assert len(repaired) == 48
    assert repaired["period_start_utc"].is_unique


def test_missing_period_recorded_as_gap(normal_payload: dict, tmp_path: Path) -> None:
    """5. Dropped period on settled date records missing_period gap and reports count."""
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    gaps_file = audit_dir / "gaps.csv"

    # Remove period at index 10
    payload = json.loads(json.dumps(normal_payload))
    removed_item = payload["data"].pop(10)
    assert len(payload["data"]) == 47

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    res = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=today,
    )

    assert res.status == "written"
    assert res.rows_written == 47
    assert res.missing_periods == 1
    assert res.gaps_recorded >= 1

    # Check parquet has 47 rows (no synthesised row)
    partition_df = read_partition(d, curated_dir)
    assert partition_df is not None
    assert len(partition_df) == 47

    # Check gaps.csv contains the missing period with reason missing_period
    assert gaps_file.exists()
    lines = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert any(
        f"{d.isoformat()},{pd.Timestamp(removed_item['from']).isoformat()},missing_period" in line
        for line in lines
    )


def test_inverted_period_rejected() -> None:
    """6a. Inverted period (end <= start) raises ValueError."""
    bad_payload = {
        "data": [
            {
                "from": "2026-10-15T23:45Z",
                "to": "2026-10-15T23:15Z",
                "intensity": {"forecast": 50, "actual": 50, "index": "low"},
            }
        ]
    }
    with pytest.raises(ValueError, match="period_end_utc <= period_start_utc|Invalid period"):
        payload_to_frame(bad_payload, date(2026, 10, 15))


def test_wrong_duration_rejected() -> None:
    """6b. Non-30-minute duration raises ValueError."""
    bad_payload = {
        "data": [
            {
                "from": "2026-10-15T00:00Z",
                "to": "2026-10-15T00:45Z",
                "intensity": {"forecast": 50, "actual": 50, "index": "low"},
            }
        ]
    }
    with pytest.raises(ValueError, match="30 minutes"):
        payload_to_frame(bad_payload, date(2026, 10, 15))


def test_null_never_overwrites_observation(normal_payload: dict, tmp_path: Path) -> None:
    """7. Incoming null actual never overwrites existing real observation (coalescing)."""
    curated_dir = tmp_path / "curated"
    d = date(2026, 10, 15)
    t1 = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)

    # Initial frame has actual=64 for period 0
    df1 = payload_to_frame(normal_payload, d, t1)
    df1.loc[0, "actual"] = 64
    upsert_day(df1, d, base_dir=curated_dir)

    # Later frame has actual=None for period 0, with later ingested_at
    payload2 = json.loads(json.dumps(normal_payload))
    payload2["data"][0]["intensity"]["actual"] = None
    df2 = payload_to_frame(payload2, d, t2)
    assert pd.isna(df2.loc[0, "actual"])

    upsert_day(df2, d, base_dir=curated_dir)

    updated = read_partition(d, curated_dir)
    assert updated is not None
    assert updated.loc[0, "actual"] == 64
    assert updated.loc[0, "ingested_at"] == pd.Timestamp(t2)


def test_payload_without_data_key_raises() -> None:
    """8a. Payload without 'data' key raises ValueError."""
    bad_payload = {"error": "Internal server error"}
    with pytest.raises(ValueError, match="data"):
        payload_to_frame(bad_payload, date(2026, 10, 15))


def test_record_without_intensity_raises() -> None:
    """8b. Record without 'intensity' object raises ValueError."""
    bad_payload = {
        "data": [
            {
                "from": "2026-10-15T00:00Z",
                "to": "2026-10-15T00:30Z",
            }
        ]
    }
    with pytest.raises(ValueError, match="intensity"):
        payload_to_frame(bad_payload, date(2026, 10, 15))


def test_prune_runs_after_ingest(normal_payload: dict, tmp_path: Path) -> None:
    """9. Stale cache file (>90 days old) is removed at the end of a successful ingest run."""
    cache_dir = tmp_path / "cache" / "http"
    curated_dir = tmp_path / "curated"
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)

    old_cache = cache_dir / "2026-01-01.json.gz"
    old_cache.write_bytes(b"old")
    now_ts = time.time()
    os.utime(old_cache, (now_ts - 100 * 86400, now_ts - 100 * 86400))

    mock_client = MagicMock()
    mock_client.get_day.return_value = normal_payload

    summary = ingest_range(
        start=d,
        end=d,
        client=mock_client,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        today=today,
    )

    assert summary.failures == 0
    assert not old_cache.exists()


def test_permanently_missing_actual_stops_refetching(normal_payload: dict, tmp_path: Path) -> None:
    """Past date with permanently missing actual makes exactly one client call across 3 runs."""
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    curated = tmp_path / "curated"
    cache = tmp_path / "cache"

    payload = json.loads(json.dumps(normal_payload))
    payload["data"][0]["intensity"]["actual"] = None

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    # Run 1: live fetch -> written, 1 gap recorded, 1 client call
    res1 = ingest_day(d, client=mock_client, curated_dir=curated, cache_dir=cache, today=today)
    assert res1.status == "written"
    assert res1.gaps_recorded == 1
    assert mock_client.get_day.call_count == 1

    # Run 2: date is analysed to completion -> skipped, 0 gaps recorded, 0 additional client calls
    res2 = ingest_day(d, client=mock_client, curated_dir=curated, cache_dir=cache, today=today)
    assert res2.status == "skipped"
    assert res2.gaps_recorded == 0
    assert mock_client.get_day.call_count == 1

    # Run 3: still skipped, 0 gaps recorded, exactly 1 client call total across all 3 runs
    res3 = ingest_day(d, client=mock_client, curated_dir=curated, cache_dir=cache, today=today)
    assert res3.status == "skipped"
    assert res3.gaps_recorded == 0
    assert mock_client.get_day.call_count == 1


def test_index_not_orphaned_from_actual(normal_payload: dict, tmp_path: Path) -> None:
    """actual and index are atomic; index stays 'high' when incoming actual is null."""
    curated_dir = tmp_path / "curated"
    d = date(2026, 10, 15)
    t1 = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)

    # Ingest 1: {forecast: 240, actual: 260, index: "high"}
    payload1 = json.loads(json.dumps(normal_payload))
    payload1["data"][0]["intensity"]["forecast"] = 240
    payload1["data"][0]["intensity"]["actual"] = 260
    payload1["data"][0]["intensity"]["index"] = "high"
    df1 = payload_to_frame(payload1, d, t1)
    upsert_day(df1, d, base_dir=curated_dir)

    # Ingest 2: {forecast: 40, actual: null, index: "very low"}
    payload2 = json.loads(json.dumps(normal_payload))
    payload2["data"][0]["intensity"]["forecast"] = 40
    payload2["data"][0]["intensity"]["actual"] = None
    payload2["data"][0]["intensity"]["index"] = "very low"
    df2 = payload_to_frame(payload2, d, t2)
    upsert_day(df2, d, base_dir=curated_dir)

    updated = read_partition(d, curated_dir)
    assert updated is not None
    row0 = updated.iloc[0]
    # Forecast is independent and updates to 40
    assert row0["forecast"] == 40
    # actual and index are atomic: actual stays 260, index stays "high" (not "very low")
    assert row0["actual"] == 260
    assert row0["index"] == "high"


def test_retraction_recorded_as_conflict(normal_payload: dict, tmp_path: Path) -> None:
    """A null arriving over a stored value writes one conflicts.csv row and keeps the value."""
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    d = date(2026, 10, 15)
    conflicts_file = audit_dir / "conflicts.csv"

    # Ingest 1: actual=260
    payload1 = json.loads(json.dumps(normal_payload))
    payload1["data"][0]["intensity"]["actual"] = 260
    payload1["data"][0]["intensity"]["index"] = "high"
    df1 = payload_to_frame(payload1, d)
    upsert_day(df1, d, base_dir=curated_dir, audit_dir=audit_dir)

    assert not conflicts_file.exists()

    # Ingest 2: upstream retracts actual to null
    payload2 = json.loads(json.dumps(normal_payload))
    payload2["data"][0]["intensity"]["actual"] = None
    payload2["data"][0]["intensity"]["index"] = "very low"
    df2 = payload_to_frame(payload2, d)
    upsert_day(df2, d, base_dir=curated_dir, audit_dir=audit_dir)

    # Stored observation is kept
    stored = read_partition(d, curated_dir)
    assert stored is not None
    assert stored.iloc[0]["actual"] == 260

    # Conflict is recorded in conflicts.csv
    assert conflicts_file.exists()
    lines = conflicts_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 record
    assert lines[0] == "local_date,period_start_utc,field,stored_value"
    p0_start = pd.Timestamp(normal_payload["data"][0]["from"]).isoformat()
    assert f"{d.isoformat()},{p0_start},actual,260" in lines[1]

    # Re-ingesting again does not duplicate the conflict record
    upsert_day(df2, d, base_dir=curated_dir, audit_dir=audit_dir)
    lines_after = conflicts_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == 2


def test_unchanged_reingest_reports_zero_refreshed(normal_payload: dict, tmp_path: Path) -> None:
    """An unchanged re-ingest must report rows_refreshed=0."""
    curated_dir = tmp_path / "curated"
    d = date(2026, 10, 15)

    df1 = payload_to_frame(normal_payload, d)
    res1 = upsert_day(df1, d, base_dir=curated_dir)
    assert res1.rows_after == 48

    df2 = payload_to_frame(normal_payload, d)
    res2 = upsert_day(df2, d, base_dir=curated_dir)
    assert res2.rows_refreshed == 0


def test_payload_for_wrong_day_rejected(normal_payload: dict) -> None:
    """payload_to_frame must reject any record whose local_date is not the requested date."""
    with pytest.raises(ValueError, match="local_date|requested"):
        payload_to_frame(normal_payload, date(2026, 10, 10))


def test_future_day_ingest_records_no_gaps(normal_payload: dict, tmp_path: Path) -> None:
    """Ingesting tomorrow's partial payload writes nulls and records zero gaps."""
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"
    today = date(2026, 10, 15)
    tomorrow = date(2026, 10, 16)

    payload = json.loads(json.dumps(normal_payload))
    for item in payload["data"]:
        ts_from = pd.to_datetime(item["from"], utc=True)
        ts_new = ts_from + timedelta(days=1)
        ts_end = pd.to_datetime(item["to"], utc=True) + timedelta(days=1)
        item["from"] = ts_new.strftime("%Y-%m-%dT%H:%MZ")
        item["to"] = ts_end.strftime("%Y-%m-%dT%H:%MZ")
        item["intensity"]["actual"] = None
        item["intensity"]["index"] = None

    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    res = ingest_day(
        tomorrow,
        client=mock_client,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=today,
    )
    assert res.status == "written"
    assert res.gaps_recorded == 0
    assert res.nulls_remaining == 48

    gaps_file = audit_dir / "gaps.csv"
    if gaps_file.exists():
        assert len(gaps_file.read_text(encoding="utf-8").strip().splitlines()) <= 1


def _build_today_payload_with_nulls_after_1942() -> dict:
    cutoff = datetime(2026, 11, 1, 19, 42, 0, tzinfo=timezone.utc)
    items = []
    for i in range(48):
        from_dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=30 * i)
        to_dt = from_dt + timedelta(minutes=30)
        from_str = from_dt.strftime("%Y-%m-%dT%H:%MZ")
        to_str = to_dt.strftime("%Y-%m-%dT%H:%MZ")
        if to_dt < cutoff:
            actual = 180
            index = "moderate"
        else:
            actual = None
            index = None
        items.append({
            "from": from_str,
            "to": to_str,
            "intensity": {
                "forecast": 175,
                "actual": actual,
                "index": index,
            },
        })
    return {"data": items}


def test_pending_periods_are_not_gaps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """1. Ingest today's payload at 20:11Z: nulls ending >= 19:42Z are pending, not gaps."""
    d = date(2026, 11, 1)
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    payload = _build_today_payload_with_nulls_after_1942()
    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    fake_now = datetime(2026, 11, 1, 20, 11, 0, tzinfo=timezone.utc)

    class MockDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return fake_now.astimezone(tz)
            return fake_now

    import gridcast.ingest
    monkeypatch.setattr(gridcast.ingest, "datetime", MockDT)

    res = ingest_day(
        local_date=d,
        client=mock_client,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=d,
    )

    assert res.status == "written"
    assert res.gaps_recorded == 0
    gaps_file = audit_dir / "gaps.csv"
    assert not gaps_file.exists()


def test_gap_only_after_two_hours(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """2. The same payload at 22:00Z records exactly one gap, for the 19:30 period only."""
    d = date(2026, 11, 1)
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    payload = _build_today_payload_with_nulls_after_1942()
    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    fake_now = datetime(2026, 11, 1, 22, 0, 0, tzinfo=timezone.utc)

    class MockDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return fake_now.astimezone(tz)
            return fake_now

    import gridcast.ingest
    monkeypatch.setattr(gridcast.ingest, "datetime", MockDT)

    res = ingest_day(
        local_date=d,
        client=mock_client,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=d,
    )

    assert res.status == "written"
    assert res.gaps_recorded == 1
    gaps_file = audit_dir / "gaps.csv"
    assert gaps_file.exists()
    lines = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 gap
    assert "2026-11-01T19:30:00+00:00" in lines[1] or "2026-11-01T19:30:00Z" in lines[1]
    assert "missing_actual" in lines[1]


def test_date_argument_cannot_shift_the_clock(tmp_path: Path) -> None:
    """3. Date argument cannot shift the clock: record_gaps rejects calendar date with TypeError."""
    d = date(2026, 11, 1)
    payload = _build_today_payload_with_nulls_after_1942()
    df = payload_to_frame(payload, d)
    with pytest.raises(TypeError, match="datetime"):
        record_gaps(df, today=d, gaps_path=tmp_path / "gaps.csv")  # type: ignore[arg-type]


def test_ingest_range_does_not_mark_today_pending_as_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """4. ingest_range over today with clock set to 20:11Z records zero gaps."""
    d = date(2026, 11, 1)
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    payload = _build_today_payload_with_nulls_after_1942()
    mock_client = MagicMock()
    mock_client.get_day.return_value = payload

    fake_now = datetime(2026, 11, 1, 20, 11, 0, tzinfo=timezone.utc)

    class MockDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return fake_now.astimezone(tz)
            return fake_now

    import gridcast.ingest
    monkeypatch.setattr(gridcast.ingest, "datetime", MockDT)

    summary = ingest_range(
        start=d,
        end=d,
        client=mock_client,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        today=d,
    )

    assert summary.failures == 0
    assert summary.dates_written == 1
    assert summary.gaps_recorded == 0
    assert summary.nulls_remaining > 0
    gaps_file = audit_dir / "gaps.csv"
    assert not gaps_file.exists()


def test_gap_reconciled_when_actual_arrives(normal_payload: dict, tmp_path: Path) -> None:
    """Gap reconciliation: drop missing_actual when actual arrives on force re-ingest."""
    d = date(2026, 10, 15)
    today = date(2026, 11, 1)
    curated_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"
    gaps_file = audit_dir / "gaps.csv"

    # Ingest 1: null actual older than 2 hours records a gap in gaps.csv
    payload1 = json.loads(json.dumps(normal_payload))
    payload1["data"][0]["intensity"]["actual"] = None
    mock_client = MagicMock()
    mock_client.get_day.return_value = payload1

    res1 = ingest_day(
        d,
        client=mock_client,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=today,
    )
    assert res1.status == "written"
    assert res1.gaps_recorded == 1
    assert gaps_file.exists()
    lines1 = gaps_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines1) == 2  # header + 1 gap row

    # Ingest 2: actual is finally published upstream (e.g. 260)
    # Re-ingest with force=True
    payload2 = json.loads(json.dumps(normal_payload))
    payload2["data"][0]["intensity"]["actual"] = 260
    mock_client.get_day.return_value = payload2

    res2 = ingest_day(
        d,
        client=mock_client,
        force=True,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        cache_dir=cache_dir,
        today=today,
    )
    assert res2.status == "written"
    # Parquet partition updated with non-null actual
    partition_df = read_partition(d, curated_dir)
    assert partition_df is not None
    assert partition_df.iloc[0]["actual"] == 260

    # gaps.csv reconciled: missing_actual dropped; file absent when empty
    assert not gaps_file.exists()


def _make_payload_for_date(d: date, actual: int | None = 150) -> dict:
    count = dst_row_count(d)
    start_local = datetime(d.year, d.month, d.day, 0, 0, tzinfo=ZoneInfo("Europe/London"))
    start_utc = start_local.astimezone(timezone.utc)
    data = []
    for i in range(count):
        p_from = start_utc + timedelta(minutes=30 * i)
        p_to = p_from + timedelta(minutes=30)
        data.append({
            "from": p_from.strftime("%Y-%m-%dT%H:%MZ"),
            "to": p_to.strftime("%Y-%m-%dT%H:%MZ"),
            "intensity": {
                "forecast": 140,
                "actual": actual,
                "index": "moderate",
            },
        })
    return {"data": data}


class TestReconcileRecent:
    """Test suite for gridcast ingest --reconcile-recent functionality."""

    def test_reconcile_recent_qualifying_and_non_qualifying(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify:
        - recent date with null is refetched
        - settled recent date with no nulls and no gap record is not fetched
        - date outside the window is never touched
        - call count is exactly one per qualifying date.
        """
        curated_dir = tmp_path / "curated"
        audit_dir = tmp_path / "audit"
        cache_dir = tmp_path / "cache"

        curated_dir.mkdir(parents=True)
        audit_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)

        today = date(2026, 11, 1)
        # Window of 3 days: [2026-10-30, 2026-10-31, 2026-11-01]
        # Outside window: 2026-10-29

        d_outside = date(2026, 10, 29)
        d_recent_null = date(2026, 10, 30)
        d_recent_settled = date(2026, 10, 31)
        d_today_gaps = date(2026, 11, 1)

        # 1. Create partition outside window with a null (should NOT be touched)
        p_outside = _make_payload_for_date(d_outside, actual=None)
        frame_outside = payload_to_frame(p_outside, d_outside)
        write_partition_atomic(frame_outside, d_outside, curated_dir)

        # 2. Create partition inside window with a null (qualifies and MUST be refetched)
        p_null = _make_payload_for_date(d_recent_null, actual=None)
        frame_null = payload_to_frame(p_null, d_recent_null)
        write_partition_atomic(frame_null, d_recent_null, curated_dir)

        # 3. Partition inside window with NO nulls / gaps (settled, MUST NOT be refetched)
        p_settled = _make_payload_for_date(d_recent_settled, actual=150)
        frame_settled = payload_to_frame(p_settled, d_recent_settled)
        write_partition_atomic(frame_settled, d_recent_settled, curated_dir)

        # 4. Create today partition with a gap record in gaps.csv (qualifies and MUST be refetched)
        p_today = _make_payload_for_date(d_today_gaps, actual=150)
        frame_today = payload_to_frame(p_today, d_today_gaps)
        write_partition_atomic(frame_today, d_today_gaps, curated_dir)
        gaps_file = audit_dir / "gaps.csv"
        gaps_file.write_text(
            f"local_date,period_start_utc,reason\n{d_today_gaps},2026-11-01T00:00:00Z,missing_actual\n",
            encoding="utf-8",
        )

        mock_client = MagicMock()
        mock_client.get_day.side_effect = lambda d: _make_payload_for_date(d, actual=160)

        # Run reconcile_recent with window_days=3
        reconciled_count = reconcile_recent(
            window_days=3,
            client=mock_client,
            cache_dir=cache_dir,
            curated_dir=curated_dir,
            audit_dir=audit_dir,
            today=today,
        )

        # Exactly 2 dates qualified: d_recent_null and d_today_gaps
        assert reconciled_count == 2

        # The call count is exactly one per qualifying date
        assert mock_client.get_day.call_count == 2
        called_dates = [call.args[0] for call in mock_client.get_day.call_args_list]
        assert called_dates == [d_recent_null, d_today_gaps]

        # Outside date was never touched
        assert d_outside not in called_dates

        # Settled date with no nulls was never touched
        assert d_recent_settled not in called_dates

    def test_reconcile_recent_zero_qualifying_makes_no_requests(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When no date qualifies, make no requests at all and report reconciled: 0."""
        curated_dir = tmp_path / "curated"
        audit_dir = tmp_path / "audit"
        curated_dir.mkdir(parents=True)
        audit_dir.mkdir(parents=True)

        today = date(2026, 11, 1)
        # Create complete partitions with no nulls
        for offset in range(3):
            d = today - timedelta(days=offset)
            frame = payload_to_frame(_make_payload_for_date(d, actual=150), d)
            write_partition_atomic(frame, d, curated_dir)

        mock_client = MagicMock()

        count = reconcile_recent(
            window_days=3,
            client=mock_client,
            curated_dir=curated_dir,
            audit_dir=audit_dir,
            today=today,
        )

        assert count == 0
        assert mock_client.get_day.call_count == 0
        captured = capsys.readouterr()
        assert "reconciled: 0 date(s)" in captured.out

    def test_cli_reconcile_recent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test CLI routing for gridcast ingest --reconcile-recent [--window-days 3]."""
        from gridcast import cli

        called_kwargs: dict[str, object] = {}

        def fake_reconcile(window_days: int = 3, **kwargs: object) -> int:
            called_kwargs["window_days"] = window_days
            called_kwargs.update(kwargs)
            return 1

        monkeypatch.setattr("gridcast.cli.reconcile_recent", fake_reconcile)

        code = cli.main(["ingest", "--reconcile-recent", "--window-days", "4"])
        assert code == 0
        assert called_kwargs.get("window_days") == 4


def test_weather_ingest_is_idempotent(tmp_path: Path) -> None:
    """Weather ingest run twice produces identical row count, no duplicates, identical parquet."""
    target_date = date(2025, 5, 1)
    # 24 hourly rows
    hours = pd.date_range("2025-05-01 00:00:00", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({
        "period_start_utc": hours,
        "local_date": target_date,
        "temperature_2m_previous_day2": [12.5 + i * 0.1 for i in range(24)],
    })

    res1 = upsert_weather_day(df, target_date, base_dir=tmp_path)
    p_path = tmp_path / f"date={target_date}.parquet"
    assert p_path.exists()
    bytes1 = p_path.read_bytes()

    res2 = upsert_weather_day(df, target_date, base_dir=tmp_path)
    bytes2 = p_path.read_bytes()

    assert res1.rows_after == 24
    assert res2.rows_after == 24
    assert res2.duplicates_dropped == 0
    assert bytes1 == bytes2


def test_mix_ingest_is_idempotent(tmp_path: Path) -> None:
    """Generation mix ingest run twice produces identical rows, no duplicates, identical parquet."""
    target_date = date(2025, 5, 1)
    # 48 half-hourly rows
    half_hours = pd.date_range("2025-05-01 00:00:00", periods=48, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "period_start_utc": half_hours,
        "local_date": target_date,
        "wind": [30.0] * 48,
        "solar": [10.0] * 48,
        "gas": [25.0] * 48,
        "coal": [1.0] * 48,
        "nuclear": [15.0] * 48,
        "biomass": [5.0] * 48,
        "hydro": [2.0] * 48,
        "imports": [10.0] * 48,
        "other": [2.0] * 48,
    })

    res1 = upsert_mix_day(df, target_date, base_dir=tmp_path)
    p_path = tmp_path / f"date={target_date}.parquet"
    assert p_path.exists()
    bytes1 = p_path.read_bytes()

    res2 = upsert_mix_day(df, target_date, base_dir=tmp_path)
    bytes2 = p_path.read_bytes()

    assert res1.rows_after == 48
    assert res2.rows_after == 48
    assert res2.duplicates_dropped == 0
    assert bytes1 == bytes2


def test_generate_monthly_chunks() -> None:
    """Test splitting dates into monthly chunks."""
    # Single month
    chunks = generate_monthly_chunks(date(2025, 3, 10), date(2025, 3, 25))
    assert chunks == [(date(2025, 3, 10), date(2025, 3, 25))]

    # Multi-month across years
    chunks2 = generate_monthly_chunks(date(2024, 11, 15), date(2025, 1, 10))
    assert chunks2 == [
        (date(2024, 11, 15), date(2024, 11, 30)),
        (date(2024, 12, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 10)),
    ]

    with pytest.raises(ValueError, match="must be on or before"):
        generate_monthly_chunks(date(2025, 1, 2), date(2025, 1, 1))


def test_expected_weather_row_count() -> None:
    """Verify expected hourly row counts across normal and DST transition days."""
    assert expected_weather_row_count(date(2026, 6, 1)) == 24
    assert expected_weather_row_count(date(2026, 3, 29)) == 23  # Spring forward
    assert expected_weather_row_count(date(2026, 10, 25)) == 25  # Autumn back


def test_is_weather_day_complete(tmp_path: Path) -> None:
    """Test is_weather_day_complete validation."""
    target_date = date(2026, 6, 1)
    # 1. Missing file
    assert is_weather_day_complete(target_date, weather_dir=tmp_path) is False

    # 2. Incomplete rows
    hours_20 = pd.date_range("2026-06-01 00:00:00", periods=20, freq="h", tz="UTC")
    df_incomplete = pd.DataFrame({
        "period_start_utc": hours_20,
        "local_date": target_date,
        "temperature_2m_previous_day2": [15.0] * 20,
    })
    upsert_weather_day(df_incomplete, target_date, base_dir=tmp_path)
    assert is_weather_day_complete(target_date, weather_dir=tmp_path) is False

    # 3. Complete rows but with NaN
    hours_24 = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h", tz="UTC")
    temps_nan = [15.0] * 23 + [np.nan]
    df_nan = pd.DataFrame({
        "period_start_utc": hours_24,
        "local_date": target_date,
        "temperature_2m_previous_day1": [16.0] * 24,
        "temperature_2m_previous_day2": temps_nan,
    })
    upsert_weather_day(df_nan, target_date, base_dir=tmp_path)
    assert is_weather_day_complete(target_date, weather_dir=tmp_path) is False

    # 4. Complete non-null
    df_valid = pd.DataFrame({
        "period_start_utc": hours_24,
        "local_date": target_date,
        "temperature_2m_previous_day1": [16.0] * 24,
        "temperature_2m_previous_day2": [15.0] * 24,
    })
    upsert_weather_day(df_valid, target_date, base_dir=tmp_path)
    assert is_weather_day_complete(target_date, weather_dir=tmp_path) is True


def test_is_mix_day_complete(tmp_path: Path) -> None:
    """Test is_mix_day_complete validation."""
    target_date = date(2026, 6, 1)
    assert is_mix_day_complete(target_date, mix_dir=tmp_path) is False

    half_hours = pd.date_range("2026-06-01 00:00:00", periods=48, freq="30min", tz="UTC")
    df_missing_fuel = pd.DataFrame({
        "period_start_utc": half_hours,
        "local_date": target_date,
        "wind": [30.0] * 48,
    })
    upsert_mix_day(df_missing_fuel, target_date, base_dir=tmp_path)
    assert is_mix_day_complete(target_date, mix_dir=tmp_path) is False

    df_full = pd.DataFrame({
        "period_start_utc": half_hours,
        "local_date": target_date,
        "wind": [30.0] * 48,
        "solar": [10.0] * 48,
        "gas": [25.0] * 48,
        "coal": [1.0] * 48,
        "nuclear": [15.0] * 48,
        "biomass": [5.0] * 48,
    })
    upsert_mix_day(df_full, target_date, base_dir=tmp_path)
    assert is_mix_day_complete(target_date, mix_dir=tmp_path) is True


def test_weather_payload_to_frame() -> None:
    """Test weather_payload_to_frame conversion and validation."""
    empty = weather_payload_to_frame({})
    assert empty.empty
    assert list(empty.columns) == [
        "period_start_utc",
        "local_date",
        "temperature_2m_previous_day1",
        "temperature_2m_previous_day2",
    ]

    valid_payload = {
        "hourly": {
            "time": ["2025-05-01T00:00", "2025-05-01T01:00"],
            "temperature_2m_previous_day1": [13.0, 13.5],
            "temperature_2m_previous_day2": [12.0, 12.5],
        }
    }
    df = weather_payload_to_frame(valid_payload)
    assert len(df) == 2
    assert df["temperature_2m_previous_day1"].iloc[0] == 13.0
    assert df["temperature_2m_previous_day2"].iloc[0] == 12.0

    bad_payload = {
        "hourly": {
            "time": ["2025-05-01T00:00"],
            "temperature_2m_previous_day1": [None],
            "temperature_2m_previous_day2": [12.0],
        }
    }
    with pytest.raises(ValueError, match="null temperature_2m_previous_day1"):
        weather_payload_to_frame(bad_payload)


def test_mix_payload_to_frame() -> None:
    """Test mix_payload_to_frame conversion."""
    empty = mix_payload_to_frame({})
    assert empty.empty
    assert "period_start_utc" in empty.columns

    valid_payload = {
        "data": [
            {
                "from": "2025-05-01T00:00Z",
                "to": "2025-05-01T00:30Z",
                "generationmix": [
                    {"fuel": "Wind", "perc": 35.5},
                    {"fuel": "Solar", "perc": 0.0},
                ],
            }
        ]
    }
    df = mix_payload_to_frame(valid_payload)
    assert len(df) == 1
    assert df["wind"].iloc[0] == 35.5
    assert df["solar"].iloc[0] == 0.0


def test_fetch_weather_and_mix_chunk_cache(tmp_path: Path) -> None:
    """Test caching behavior for weather and generation mix chunks."""
    cache_weather = tmp_path / "cache_w"
    cache_mix = tmp_path / "cache_m"

    mock_client = MagicMock()
    mock_client.get_weather_previous_runs.return_value = {
        "hourly": {
            "time": ["2025-05-01T00:00"],
            "temperature_2m_previous_day1": [16.0],
            "temperature_2m_previous_day2": [15.0],
        }
    }
    mock_client.get_generation_range.return_value = {
        "data": [
            {
                "from": "2025-05-01T00:00Z",
                "to": "2025-05-01T00:30Z",
                "generationmix": [{"fuel": "wind", "perc": 40}],
            }
        ]
    }

    # 1. First fetch writes cache
    d1 = date(2025, 5, 1)
    d2 = date(2025, 5, 2)
    p_w1 = fetch_weather_chunk(mock_client, d1, d2, cache_dir=cache_weather)
    assert mock_client.get_weather_previous_runs.call_count == 1
    assert (cache_weather / f"{d1}_{d2}.json.gz").exists()

    # 2. Second fetch reads cache without calling client
    p_w2 = fetch_weather_chunk(mock_client, d1, d2, cache_dir=cache_weather)
    assert mock_client.get_weather_previous_runs.call_count == 1
    assert p_w1 == p_w2

    # Repeat for mix
    p_m1 = fetch_mix_chunk(mock_client, d1, d2, cache_dir=cache_mix)
    assert mock_client.get_generation_range.call_count == 1
    p_m2 = fetch_mix_chunk(mock_client, d1, d2, cache_dir=cache_mix)
    assert mock_client.get_generation_range.call_count == 1
    assert p_m1 == p_m2


def test_ingest_weather_and_mix_day(tmp_path: Path) -> None:
    """Test ingest_weather_day and ingest_mix_day behavior."""
    w_dir = tmp_path / "weather"
    m_dir = tmp_path / "mix"
    target_date = date(2025, 1, 15)

    mock_client = MagicMock()
    # 24 hours of weather
    hours = [f"2025-01-15T{h:02d}:00" for h in range(24)]
    mock_client.get_weather_previous_runs.return_value = {
        "hourly": {
            "time": hours,
            "temperature_2m_previous_day1": [15.0] * 24,
            "temperature_2m_previous_day2": [14.0] * 24,
        }
    }
    # 48 periods of mix
    half_hours = [f"2025-01-15T{h:02d}:{m:02d}Z" for h in range(24) for m in (0, 30)]
    mock_client.get_generation_range.return_value = {
        "data": [
            {
                "from": ts,
                "to": ts,
                "generationmix": [
                    {"fuel": "wind", "perc": 30},
                    {"fuel": "solar", "perc": 10},
                    {"fuel": "gas", "perc": 25},
                    {"fuel": "coal", "perc": 1},
                    {"fuel": "nuclear", "perc": 15},
                    {"fuel": "biomass", "perc": 5},
                ],
            }
            for ts in half_hours
        ]
    }

    res_w = ingest_weather_day(target_date, client=mock_client, weather_dir=w_dir)
    assert res_w.status == "written"
    assert res_w.rows_written == 24

    # Run again: should skip
    res_w_skip = ingest_weather_day(target_date, client=mock_client, weather_dir=w_dir)
    assert res_w_skip.status == "skipped"

    res_m = ingest_mix_day(target_date, client=mock_client, mix_dir=m_dir)
    assert res_m.status == "written"
    assert res_m.rows_written == 48

    res_m_skip = ingest_mix_day(target_date, client=mock_client, mix_dir=m_dir)
    assert res_m_skip.status == "skipped"


def test_ingest_weather_and_mix_range_validation_and_failures(tmp_path: Path) -> None:
    """Test range validation and error handling in weather/mix ingest ranges."""
    with pytest.raises(ValueError, match="must be on or before"):
        ingest_weather_range(date(2025, 5, 2), date(2025, 5, 1))

    with pytest.raises(ValueError, match="must be on or before"):
        ingest_mix_range(date(2025, 5, 2), date(2025, 5, 1))

    # Test error handling when client raises
    failing_client = MagicMock()
    failing_client.get_weather_previous_runs.side_effect = RuntimeError("API down")
    failing_client.get_generation_range.side_effect = RuntimeError("API down")

    summary_w = ingest_weather_range(
        date(2025, 5, 1),
        date(2025, 5, 2),
        client=failing_client,
        weather_dir=tmp_path / "w",
        cache_dir=tmp_path / "cw",
        curated_dir=tmp_path / "c",
        audit_dir=tmp_path / "a",
    )
    assert summary_w.failures == 2

    summary_m = ingest_mix_range(
        date(2025, 5, 1),
        date(2025, 5, 2),
        client=failing_client,
        mix_dir=tmp_path / "m",
        cache_dir=tmp_path / "cm",
        curated_dir=tmp_path / "c",
        audit_dir=tmp_path / "a",
    )
    assert summary_m.failures == 2


def test_client_weather_and_mix_methods() -> None:
    """Test Client helper methods invoke correct URLs."""
    session_mock = MagicMock()
    resp_mock = MagicMock()
    resp_mock.status_code = 200
    resp_mock.json.return_value = {"ok": True}
    session_mock.get.return_value = resp_mock

    client = Client(session=session_mock, min_request_interval=0.0)

    client.get_generation_range("2025-05-01T00:00Z", "2025-05-02T00:00Z")
    assert "/generation/2025-05-01T00:00Z/2025-05-02T00:00Z" in session_mock.get.call_args[0][0]

    client.get_weather_previous_runs(date(2025, 5, 1), date(2025, 5, 2))
    called_url = session_mock.get.call_args[0][0]
    assert "previous-runs-api.open-meteo.com" in called_url
    assert "temperature_2m_previous_day1" in called_url
    assert "temperature_2m_previous_day2" in called_url
    assert "start_date=2025-05-01" in called_url


def test_cli_weather_and_mix_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI routing for gridcast ingest --weather and --mix."""
    from gridcast import cli

    weather_called = False
    mix_called = False

    def fake_weather_range(*args: object, **kwargs: object) -> object:
        nonlocal weather_called
        weather_called = True
        return MagicMock(failures=0)

    def fake_mix_range(*args: object, **kwargs: object) -> object:
        nonlocal mix_called
        mix_called = True
        return MagicMock(failures=0)

    monkeypatch.setattr("gridcast.cli.ingest_weather_range", fake_weather_range)
    monkeypatch.setattr("gridcast.cli.ingest_mix_range", fake_mix_range)

    code_w = cli.main(["ingest", "--weather", "--days", "2"])
    assert code_w == 0
    assert weather_called is True

    code_m = cli.main(["ingest", "--mix", "--days", "2"])
    assert code_m == 0
    assert mix_called is True


def test_mix_coverage_check_recovers_boundary_period(tmp_path: Path) -> None:
    """A stubbed endpoint drops straddling period; retry must recover it."""
    from gridcast.ingest import (
        Client,
        ingest_mix_range,
        partition_path,
        upsert_day,
    )

    curated_dir = tmp_path / "curated"
    mix_dir = tmp_path / "mix"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    d = date(2024, 12, 31)
    # Populate curated partition for 2024-12-31 with all 48 periods
    start_utc = datetime(2024, 12, 31, 0, 0, tzinfo=timezone.utc)
    periods = [start_utc + timedelta(minutes=30 * i) for i in range(48)]
    c_df = pd.DataFrame([
        {
            "period_start_utc": p,
            "period_end_utc": p + timedelta(minutes=30),
            "utc_date": p.date(),
            "local_date": d,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_utc,
        }
        for p in periods
    ])
    c_df["period_start_utc"] = pd.to_datetime(c_df["period_start_utc"], utc=True)
    c_df["period_end_utc"] = pd.to_datetime(c_df["period_end_utc"], utc=True)
    c_df["ingested_at"] = pd.to_datetime(c_df["ingested_at"], utc=True)
    c_df["forecast"] = c_df["forecast"].astype("Int32")
    c_df["actual"] = c_df["actual"].astype("Int32")
    c_df["index"] = c_df["index"].astype("string")
    upsert_day(c_df, d, base_dir=curated_dir)

    # Stub client: initial chunk drops 2024-12-31T23:30Z (returns 47 periods).
    # Retry window beginning at following midnight (2025-01-01T00:00Z) returns 2024-12-31T23:30Z!
    class StubClient(Client):
        def __init__(self) -> None:
            super().__init__()
            self.retry_called = False

        def get_generation_range(self, start_utc: datetime | str, end_utc: datetime | str) -> dict:
            s_str = str(start_utc)
            if "2025-01-01T00:00" in s_str:
                self.retry_called = True
                # Retry returns the straddling period
                return {
                    "data": [
                        {
                            "from": "2024-12-31T23:30Z",
                            "to": "2025-01-01T00:00Z",
                            "generationmix": [{"fuel": "wind", "perc": 40.0}],
                        }
                    ]
                }
            # Initial chunk returns 47 periods (drops the last period 23:30Z)
            items = []
            for p in periods[:-1]:
                p_str = p.strftime("%Y-%m-%dT%H:%MZ")
                p_end_str = (p + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
                items.append({
                    "from": p_str,
                    "to": p_end_str,
                    "generationmix": [{"fuel": "wind", "perc": 30.0}],
                })
            return {"data": items}

    client = StubClient()
    summary = ingest_mix_range(
        start=d,
        end=d,
        client=client,
        mix_dir=mix_dir,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
    )

    assert client.retry_called is True
    assert summary.periods_recovered == 1
    assert summary.gaps_recorded == 0

    p_mix = partition_path(d, base_dir=mix_dir)
    assert p_mix.exists()
    mix_df = pd.read_parquet(p_mix)
    assert len(mix_df) == 48
    assert pd.Timestamp("2024-12-31 23:30:00+00:00") in set(
        pd.to_datetime(mix_df["period_start_utc"], utc=True)
    )
    assert not (audit_dir / "source-gaps.csv").exists()


def test_unrecoverable_mix_period_recorded_as_source_gap(tmp_path: Path) -> None:
    """An empty upstream response yields exactly one source gap row across two runs."""
    from gridcast.ingest import (
        Client,
        ingest_mix_range,
        upsert_day,
    )

    curated_dir = tmp_path / "curated"
    mix_dir = tmp_path / "mix"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    d = date(2025, 1, 13)
    start_utc = datetime(2025, 1, 13, 0, 0, tzinfo=timezone.utc)
    periods = [start_utc + timedelta(minutes=30 * i) for i in range(48)]
    c_df = pd.DataFrame([
        {
            "period_start_utc": p,
            "period_end_utc": p + timedelta(minutes=30),
            "utc_date": p.date(),
            "local_date": d,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_utc,
        }
        for p in periods
    ])
    c_df["period_start_utc"] = pd.to_datetime(c_df["period_start_utc"], utc=True)
    c_df["period_end_utc"] = pd.to_datetime(c_df["period_end_utc"], utc=True)
    c_df["ingested_at"] = pd.to_datetime(c_df["ingested_at"], utc=True)
    c_df["forecast"] = c_df["forecast"].astype("Int32")
    c_df["actual"] = c_df["actual"].astype("Int32")
    c_df["index"] = c_df["index"].astype("string")
    upsert_day(c_df, d, base_dir=curated_dir)

    # Client returns 47 periods initially and empty on retry
    class EmptyRetryClient(Client):
        def get_generation_range(self, start_utc: datetime | str, end_utc: datetime | str) -> dict:
            # Retry for missing period 0 starts at 00:30Z -> return empty
            if "00:30" in str(start_utc):
                return {"data": []}
            # Initial chunk returns periods 1 through 47 (drops period 0: 00:00Z)
            items = []
            for p in periods[1:]:
                p_str = p.strftime("%Y-%m-%dT%H:%MZ")
                p_end_str = (p + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
                items.append({
                    "from": p_str,
                    "to": p_end_str,
                    "generationmix": [{"fuel": "wind", "perc": 30.0}],
                })
            return {"data": items}

    client = EmptyRetryClient()
    # First run
    s1 = ingest_mix_range(
        start=d,
        end=d,
        client=client,
        mix_dir=mix_dir,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
    )
    assert s1.gaps_recorded == 1
    assert s1.periods_recovered == 0

    gaps_file = audit_dir / "source-gaps.csv"
    assert gaps_file.exists()
    gaps_df1 = pd.read_csv(gaps_file)
    assert len(gaps_df1) == 1
    assert gaps_df1.iloc[0]["store"] == "mix"
    assert gaps_df1.iloc[0]["reason"] == "upstream_empty"

    # Second run: should remain exactly 1 row, no duplicates
    s2 = ingest_mix_range(
        start=d,
        end=d,
        client=client,
        mix_dir=mix_dir,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        force=True,
    )
    assert s2.dates_requested == 1
    assert gaps_file.exists()
    gaps_df2 = pd.read_csv(gaps_file)
    assert len(gaps_df2) == 1


def test_source_gap_reconciled_when_data_appears(tmp_path: Path) -> None:
    """A later successful fetch removes the recorded gap."""
    from gridcast.ingest import (
        Client,
        ingest_mix_range,
        upsert_day,
    )

    curated_dir = tmp_path / "curated"
    mix_dir = tmp_path / "mix"
    audit_dir = tmp_path / "audit"
    cache_dir = tmp_path / "cache"

    d = date(2025, 1, 13)
    start_utc = datetime(2025, 1, 13, 0, 0, tzinfo=timezone.utc)
    periods = [start_utc + timedelta(minutes=30 * i) for i in range(48)]
    c_df = pd.DataFrame([
        {
            "period_start_utc": p,
            "period_end_utc": p + timedelta(minutes=30),
            "utc_date": p.date(),
            "local_date": d,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_utc,
        }
        for p in periods
    ])
    c_df["period_start_utc"] = pd.to_datetime(c_df["period_start_utc"], utc=True)
    c_df["period_end_utc"] = pd.to_datetime(c_df["period_end_utc"], utc=True)
    c_df["ingested_at"] = pd.to_datetime(c_df["ingested_at"], utc=True)
    c_df["forecast"] = c_df["forecast"].astype("Int32")
    c_df["actual"] = c_df["actual"].astype("Int32")
    c_df["index"] = c_df["index"].astype("string")
    upsert_day(c_df, d, base_dir=curated_dir)

    # Pre-record a gap in source-gaps.csv for period 0
    audit_dir.mkdir(parents=True, exist_ok=True)
    gaps_file = audit_dir / "source-gaps.csv"
    p0_str = pd.Timestamp(periods[0]).isoformat()
    gaps_file.write_text(
        f"store,local_date,period_start_utc,reason\nmix,{d.isoformat()},{p0_str},upstream_empty\n",
        encoding="utf-8",
    )

    # Now a client returns all 48 periods
    class FullClient(Client):
        def get_generation_range(self, start_utc: datetime | str, end_utc: datetime | str) -> dict:
            items = []
            for p in periods:
                p_str = p.strftime("%Y-%m-%dT%H:%MZ")
                p_end_str = (p + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
                items.append({
                    "from": p_str,
                    "to": p_end_str,
                    "generationmix": [{"fuel": "wind", "perc": 30.0}],
                })
            return {"data": items}

    client = FullClient()
    ingest_mix_range(
        start=d,
        end=d,
        client=client,
        mix_dir=mix_dir,
        cache_dir=cache_dir,
        curated_dir=curated_dir,
        audit_dir=audit_dir,
        force=True,
    )

    # File should be deleted when empty
    assert not gaps_file.exists()


def test_weather_coverage_is_complete() -> None:
    """Zero missing hours across the window, no gaps file for weather."""
    curated = pd.read_parquet("data/curated")
    weather = pd.read_parquet("data/weather")

    c_starts = pd.to_datetime(curated["period_start_utc"], utc=True)
    w_starts = set(pd.to_datetime(weather["period_start_utc"], utc=True))

    c_hours = set(c_starts.dt.floor("h"))
    missing_hours = c_hours - w_starts
    assert len(missing_hours) == 0, f"Found missing weather hours: {missing_hours}"

    source_gaps = Path("data/audit/source-gaps.csv")
    if source_gaps.exists():
        gaps_df = pd.read_csv(source_gaps)
        weather_gaps = gaps_df[gaps_df["store"] == "weather"]
        assert len(weather_gaps) == 0, f"Found recorded weather gaps: {weather_gaps}"










