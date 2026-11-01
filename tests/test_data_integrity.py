"""Integrity tests for the committed curated dataset.

Fast, read-only, offline. These run against `data/curated/` in the working directory, so a fresh
clone can prove its own dataset is intact. A short day is allowed only when
`data/audit/gaps.csv` accounts for every missing period: an unexplained hole is a failure,
an explained one is an audited fact.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from gridcast.ingest import dst_row_count, partition_path

WINDOW_START = date(2024, 11, 1)
LONDON_TZ = ZoneInfo("Europe/London")

EXPECTED_COLUMNS = [
    "period_start_utc",
    "period_end_utc",
    "utc_date",
    "local_date",
    "forecast",
    "actual",
    "index",
    "ingested_at",
]


def _today() -> date:
    return datetime.now(LONDON_TZ).date()


def _gaps_by_date(curated_dir: Path, audit_dir: Path | None = None) -> dict[str, int]:
    """Count recorded gaps per local date. Absent file means no gaps, not a failure."""
    target_audit = (
        audit_dir
        if audit_dir is not None
        else (curated_dir.parent / "audit" if curated_dir.name == "curated" else curated_dir)
    )
    gaps_path = target_audit / "gaps.csv"
    if not gaps_path.exists():
        return {}
    frame = pd.read_csv(gaps_path)
    if frame.empty or "local_date" not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame["local_date"].astype(str).value_counts().items()}


def _assert_schema(frame: pd.DataFrame, day: date) -> None:
    assert list(frame.columns) == EXPECTED_COLUMNS, (
        f"Partition {day} columns {list(frame.columns)} do not match contract {EXPECTED_COLUMNS}"
    )
    assert str(frame["period_start_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["period_end_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["ingested_at"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["forecast"].dtype) == "Int32"
    assert str(frame["actual"].dtype) == "Int32"
    assert str(frame["index"].dtype) in ("string", "string[python]")
    assert frame["period_start_utc"].is_unique, f"Partition {day} has duplicate period_start_utc"
    assert frame["local_date"].nunique() == 1, f"Partition {day} spans more than one local date"
    assert str(frame["local_date"].iloc[0]) == str(day), (
        f"Partition {day} contains local_date {frame['local_date'].iloc[0]}"
    )
    assert (frame["utc_date"] == frame["period_start_utc"].dt.date).all(), (
        f"Partition {day} has utc_date values that do not match period_start_utc.date()"
    )
    durations = (frame["period_end_utc"] - frame["period_start_utc"]).unique()
    assert list(durations) == [pd.Timedelta(minutes=30)], (
        f"Partition {day} has periods that are not exactly 30 minutes: {durations}"
    )


def _expected_grid_for_day(day: date) -> set[pd.Timestamp]:
    count = dst_row_count(day)
    start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=LONDON_TZ)
    start_utc = start_local.astimezone(timezone.utc)
    return {pd.Timestamp(start_utc + timedelta(minutes=30 * i)) for i in range(count)}


def verify_curated_dataset_integrity(
    curated_dir: Path,
    today: date | None = None,
    window_start: date | None = None,
    audit_dir: Path | None = None,
    mix_dir: Path | None = None,
    weather_dir: Path | None = None,
    source_gaps_path: Path | None = None,
) -> None:
    assert curated_dir.exists(), "data/curated directory must exist"

    eval_today = today or _today()
    eval_window_start = window_start or WINDOW_START

    target_audit = (
        audit_dir
        if audit_dir is not None
        else (
            curated_dir.parent / "audit"
            if curated_dir.name == "curated"
            else curated_dir
        )
    )
    target_mix = (
        mix_dir
        if mix_dir is not None
        else (
            curated_dir.parent / "mix"
            if (curated_dir.parent / "mix").exists()
            else Path("data/mix")
        )
    )
    target_weather = (
        weather_dir
        if weather_dir is not None
        else (
            curated_dir.parent / "weather"
            if (curated_dir.parent / "weather").exists()
            else Path("data/weather")
        )
    )
    target_source_gaps = (
        source_gaps_path
        if source_gaps_path is not None
        else target_audit / "source-gaps.csv"
    )

    # 1. Any partition whose date falls outside [window_start, eval_today] is a failure
    partition_files = list(curated_dir.glob("date=*.parquet"))
    for p_file in partition_files:
        stem = p_file.stem
        date_str = stem.split("=", 1)[1] if "=" in stem else stem
        p_date = date.fromisoformat(date_str)
        assert eval_window_start <= p_date <= eval_today, (
            f"Partition date {p_date} falls outside [{eval_window_start}, {eval_today}]: "
            f"{p_file.name}"
        )

    # Read gaps.csv if present
    if audit_dir is not None:
        gaps_path = audit_dir / "gaps.csv"
    elif (curated_dir.parent / "audit" / "gaps.csv").exists():
        gaps_path = curated_dir.parent / "audit" / "gaps.csv"
    elif (curated_dir / "gaps.csv").exists():
        gaps_path = curated_dir / "gaps.csv"
    else:
        gaps_path = (
            curated_dir.parent / "audit" / "gaps.csv"
            if curated_dir.name == "curated"
            else curated_dir / "gaps.csv"
        )

    gaps_df = pd.DataFrame(columns=["local_date", "period_start_utc", "reason"])
    if gaps_path.exists():
        raw_gaps = pd.read_csv(gaps_path)
        if not raw_gaps.empty and "local_date" in raw_gaps.columns:
            gaps_df = raw_gaps
            assert not gaps_df.duplicated(subset=["local_date", "period_start_utc"]).any(), (
                "gaps.csv contains duplicated (local_date, period_start_utc) rows"
            )
            norm_periods = pd.to_datetime(gaps_df["period_start_utc"], utc=True)
            assert norm_periods.is_unique, (
                "gaps.csv contains multiple rows sharing the same period"
            )

    # Read source-gaps.csv if present
    source_gaps_df = pd.DataFrame(columns=["store", "local_date", "period_start_utc", "reason"])
    if target_source_gaps.exists():
        raw_sg = pd.read_csv(target_source_gaps)
        if not raw_sg.empty and "store" in raw_sg.columns:
            source_gaps_df = raw_sg
            # 3. source-gaps.csv, when present, has no duplicate (store, period_start_utc) pairs
            assert not source_gaps_df.duplicated(subset=["store", "period_start_utc"]).any(), (
                "source-gaps.csv contains duplicated (store, period_start_utc) rows"
            )

    starts: list[pd.Series] = []

    day = eval_window_start
    while day < eval_today:
        path = partition_path(day, base_dir=curated_dir)
        assert path.exists(), f"Missing partition for in-window date {day}: {path}"

        frame = pd.read_parquet(path)
        _assert_schema(frame, day)

        # Set-based completeness check
        partition_starts = set(pd.to_datetime(frame["period_start_utc"], utc=True))
        partition_hours = {p.floor("h") for p in partition_starts}
        day_gaps = gaps_df[gaps_df["local_date"].astype(str) == str(day)]
        if "reason" in day_gaps.columns:
            period_gaps = day_gaps[day_gaps["reason"] != "missing_actual"]
        else:
            period_gaps = day_gaps

        gap_starts = (
            set(pd.to_datetime(period_gaps["period_start_utc"], utc=True))
            if not period_gaps.empty
            else set()
        )

        # A gap timestamp must never coincide with a row present in the partition
        overlap = partition_starts.intersection(gap_starts)
        assert not overlap, (
            f"Partition {day} has gap timestamps coinciding with rows in parquet: {overlap}"
        )

        # Union must match expected half-hourly grid exactly
        expected_grid = _expected_grid_for_day(day)
        union_starts = partition_starts.union(gap_starts)
        assert union_starts == expected_grid, (
            f"Partition {day} timestamps + recorded gaps do not equal expected grid exactly. "
            f"Missing: {expected_grid - union_starts}, Extra: {union_starts - expected_grid}"
        )

        # Mix store coverage check
        if target_mix.exists():
            m_path = partition_path(day, base_dir=target_mix)
            if m_path.exists():
                m_frame = pd.read_parquet(m_path)
                m_starts = set(pd.to_datetime(m_frame["period_start_utc"], utc=True))
            else:
                m_starts = set()

            day_mix_gaps = source_gaps_df[
                (source_gaps_df["store"] == "mix")
                & (source_gaps_df["local_date"].astype(str) == str(day))
            ]
            mix_gap_starts = (
                set(pd.to_datetime(day_mix_gaps["period_start_utc"], utc=True))
                if not day_mix_gaps.empty
                else set()
            )

            # No recorded gap coincides with a row that exists in the store
            overlap_mix = m_starts.intersection(mix_gap_starts)
            assert not overlap_mix, (
                f"Partition {day} has recorded mix gaps coinciding with rows "
                f"present in mix store: {overlap_mix}"
            )

            # Every curated period has a mix row, or a source-gaps.csv row for that
            # period with store=mix
            mix_union = m_starts.union(mix_gap_starts)
            missing_from_mix = partition_starts - mix_union
            assert not missing_from_mix, (
                f"Curated partition {day} has mix periods missing from both "
                f"store and source-gaps.csv: {missing_from_mix}"
            )

        # Weather store coverage check
        if target_weather.exists():
            w_path = partition_path(day, base_dir=target_weather)
            if w_path.exists():
                w_frame = pd.read_parquet(w_path)
                w_starts = set(pd.to_datetime(w_frame["period_start_utc"], utc=True))
            else:
                w_starts = set()

            day_weather_gaps = source_gaps_df[
                (source_gaps_df["store"] == "weather")
                & (source_gaps_df["local_date"].astype(str) == str(day))
            ]
            weather_gap_starts = (
                set(pd.to_datetime(day_weather_gaps["period_start_utc"], utc=True))
                if not day_weather_gaps.empty
                else set()
            )

            # No recorded gap coincides with a row that exists in the store
            overlap_weather = w_starts.intersection(weather_gap_starts)
            assert not overlap_weather, (
                f"Partition {day} has recorded weather gaps coinciding with rows "
                f"present in weather store: {overlap_weather}"
            )

            # Every curated hour has a weather row, or a recorded weather gap
            weather_union = w_starts.union(weather_gap_starts)
            missing_from_weather = partition_hours - weather_union
            assert not missing_from_weather, (
                f"Curated partition {day} has weather hours missing from both "
                f"store and source-gaps.csv: {missing_from_weather}"
            )

        starts.append(frame["period_start_utc"])
        day += timedelta(days=1)

    # Today is in flight: validate it only if the nightly run has written it, and include it
    # in the global uniqueness and monotonicity checks
    today_path = partition_path(eval_today, base_dir=curated_dir)
    if today_path.exists():
        today_frame = pd.read_parquet(today_path)
        _assert_schema(today_frame, eval_today)
        starts.append(today_frame["period_start_utc"])

    if starts:
        combined = pd.concat(starts, ignore_index=True)
        assert combined.is_unique, "period_start_utc is not globally unique across partitions"
        assert combined.is_monotonic_increasing, (
            "period_start_utc is not monotonic increasing across partitions"
        )


def test_committed_dataset_integrity() -> None:
    verify_curated_dataset_integrity(Path("data/curated"))


def test_data_check_rejects_gap_overlapping_parquet(tmp_path: Path) -> None:
    """A gap timestamp must never coincide with a row present in the partition."""
    d = date(2026, 10, 20)
    today = date(2026, 10, 21)

    # Build 48 periods, then drop period 5 to have a 47-row partition
    start_utc = datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc)
    periods = [start_utc + timedelta(minutes=30 * i) for i in range(48)]
    rows = [
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
        for i, p in enumerate(periods)
        if i != 5  # drop period 5
    ]
    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")
    p_path = partition_path(d, base_dir=tmp_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p_path, index=False)

    # In gaps.csv, add row for period 0 (which exists in the parquet)
    gaps_file = tmp_path / "gaps.csv"
    p0_str = pd.Timestamp(periods[0]).isoformat()
    gaps_file.write_text(
        f"local_date,period_start_utc,reason\n{d.isoformat()},{p0_str},missing_period\n",
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(AssertionError):
        verify_curated_dataset_integrity(tmp_path, window_start=d, today=today)


def test_data_check_rejects_out_of_window_partition(tmp_path: Path) -> None:
    """Any partition whose date falls outside [2024-11-01, today] is a failure."""
    d = date(2026, 11, 5)  # in the future
    today = date(2026, 11, 1)

    start_utc = datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "period_start_utc": start_utc,
            "period_end_utc": start_utc + timedelta(minutes=30),
            "utc_date": start_utc.date(),
            "local_date": d,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_utc,
        }
    ]
    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")
    p_path = partition_path(d, base_dir=tmp_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p_path, index=False)

    import pytest
    with pytest.raises(AssertionError):
        verify_curated_dataset_integrity(tmp_path, window_start=today, today=today)


def test_data_check_includes_today_in_uniqueness(tmp_path: Path) -> None:
    """Include today's partition in global uniqueness and monotonicity checks."""
    d_yesterday = date(2026, 10, 31)
    d_today = date(2026, 11, 1)

    # Yesterday partition has 48 rows
    start_yesterday = datetime(2026, 10, 31, 0, 0, tzinfo=timezone.utc)
    periods_yesterday = [start_yesterday + timedelta(minutes=30 * i) for i in range(48)]
    rows_yesterday = [
        {
            "period_start_utc": p,
            "period_end_utc": p + timedelta(minutes=30),
            "utc_date": p.date(),
            "local_date": d_yesterday,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_yesterday,
        }
        for p in periods_yesterday
    ]
    df_y = pd.DataFrame(rows_yesterday)
    df_y["period_start_utc"] = pd.to_datetime(df_y["period_start_utc"], utc=True)
    df_y["period_end_utc"] = pd.to_datetime(df_y["period_end_utc"], utc=True)
    df_y["ingested_at"] = pd.to_datetime(df_y["ingested_at"], utc=True)
    df_y["forecast"] = df_y["forecast"].astype("Int32")
    df_y["actual"] = df_y["actual"].astype("Int32")
    df_y["index"] = df_y["index"].astype("string")
    p_y = partition_path(d_yesterday, base_dir=tmp_path)
    p_y.parent.mkdir(parents=True, exist_ok=True)
    df_y.to_parquet(p_y, index=False)

    # Today partition has a duplicate of yesterday's last period
    last_period = periods_yesterday[-1]
    rows_today = [
        {
            "period_start_utc": last_period,  # DUPLICATE spanning yesterday and today
            "period_end_utc": last_period + timedelta(minutes=30),
            "utc_date": last_period.date(),
            "local_date": d_today,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": start_yesterday,
        }
    ]
    df_t = pd.DataFrame(rows_today)
    df_t["period_start_utc"] = pd.to_datetime(df_t["period_start_utc"], utc=True)
    df_t["period_end_utc"] = pd.to_datetime(df_t["period_end_utc"], utc=True)
    df_t["ingested_at"] = pd.to_datetime(df_t["ingested_at"], utc=True)
    df_t["forecast"] = df_t["forecast"].astype("Int32")
    df_t["actual"] = df_t["actual"].astype("Int32")
    df_t["index"] = df_t["index"].astype("string")
    p_t = partition_path(d_today, base_dir=tmp_path)
    df_t.to_parquet(p_t, index=False)

    import pytest
    with pytest.raises(AssertionError):
        verify_curated_dataset_integrity(tmp_path, window_start=d_yesterday, today=d_today)


def test_data_check_rejects_duplicate_gaps(tmp_path: Path) -> None:
    """data-check rejects gaps.csv containing duplicate rows for the same period."""
    d = date(2026, 10, 20)
    today = date(2026, 10, 21)

    periods = sorted(_expected_grid_for_day(d))
    rows = [
        {
            "period_start_utc": p,
            "period_end_utc": p + timedelta(minutes=30),
            "utc_date": p.date(),
            "local_date": d,
            "forecast": 100,
            "actual": 100,
            "index": "low",
            "ingested_at": periods[0],
        }
        for i, p in enumerate(periods)
        if i != 5  # drop period 5
    ]
    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    df["forecast"] = df["forecast"].astype("Int32")
    df["actual"] = df["actual"].astype("Int32")
    df["index"] = df["index"].astype("string")
    p_path = partition_path(d, base_dir=tmp_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p_path, index=False)

    # In gaps.csv, add duplicate rows for missing period 5
    gaps_file = tmp_path / "gaps.csv"
    p5_str = pd.Timestamp(periods[5]).isoformat()
    gaps_file.write_text(
        f"local_date,period_start_utc,reason\n"
        f"{d.isoformat()},{p5_str},missing_period\n"
        f"{d.isoformat()},{p5_str},missing_period\n",
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(AssertionError, match="duplicate|unique"):
        verify_curated_dataset_integrity(tmp_path, window_start=d, today=today)





def test_recorded_gaps_are_justified() -> None:
    """Every gap row must fall on a half-hour boundary inside its own local date."""
    audit_dir = Path("data/audit")
    gaps_path = audit_dir / "gaps.csv"
    if not gaps_path.exists():
        return

    frame = pd.read_csv(gaps_path)
    assert not frame.duplicated(subset=["local_date", "period_start_utc"]).any(), (
        "gaps.csv contains duplicated (local_date, period_start_utc) rows"
    )
    assert pd.to_datetime(frame["period_start_utc"], utc=True).is_unique, (
        "gaps.csv contains multiple rows sharing the same period"
    )
    for _, row in frame.iterrows():
        start = pd.Timestamp(row["period_start_utc"])
        assert start.minute in (0, 30) and start.second == 0, f"Gap not on a half hour: {start}"
        aware = start if start.tzinfo else start.tz_localize("UTC")
        local = aware.tz_convert(LONDON_TZ)
        assert str(local.date()) == str(row["local_date"]), (
            f"Gap {start} is filed under {row['local_date']} but falls on {local.date()}"
        )


def test_data_check_fails_on_unrecorded_mix_hole(tmp_path: Path) -> None:
    """A curated period with neither a mix row nor a gap record makes data-check fail."""
    import pytest

    d = date(2026, 1, 20)
    today = date(2026, 1, 21)

    curated_dir = tmp_path / "curated"
    mix_dir = tmp_path / "mix"
    weather_dir = tmp_path / "weather"
    audit_dir = tmp_path / "audit"

    # Curated: all expected periods
    periods = sorted(_expected_grid_for_day(d))
    start_utc = periods[0]
    c_rows = [
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
    ]
    df_c = pd.DataFrame(c_rows)
    df_c["period_start_utc"] = pd.to_datetime(df_c["period_start_utc"], utc=True)
    df_c["period_end_utc"] = pd.to_datetime(df_c["period_end_utc"], utc=True)
    df_c["ingested_at"] = pd.to_datetime(df_c["ingested_at"], utc=True)
    df_c["forecast"] = df_c["forecast"].astype("Int32")
    df_c["actual"] = df_c["actual"].astype("Int32")
    df_c["index"] = df_c["index"].astype("string")
    c_p = partition_path(d, base_dir=curated_dir)
    c_p.parent.mkdir(parents=True, exist_ok=True)
    df_c.to_parquet(c_p, index=False)

    # Weather: all hours present
    hours = sorted({p.floor("h") for p in periods})
    w_rows = [
        {
            "period_start_utc": h,
            "local_date": d,
            "temperature_2m_previous_day2": 12.0,
        }
        for h in hours
    ]
    df_w = pd.DataFrame(w_rows)
    df_w["period_start_utc"] = pd.to_datetime(df_w["period_start_utc"], utc=True)
    w_p = partition_path(d, base_dir=weather_dir)
    w_p.parent.mkdir(parents=True, exist_ok=True)
    df_w.to_parquet(w_p, index=False)

    # Mix: drop period 5, and no source-gaps.csv row
    m_rows = [
        {
            "period_start_utc": p,
            "local_date": d,
            "wind": 30.0,
            "solar": 10.0,
            "gas": 25.0,
            "coal": 1.0,
            "nuclear": 15.0,
            "biomass": 5.0,
        }
        for i, p in enumerate(periods)
        if i != 5
    ]
    df_m = pd.DataFrame(m_rows)
    df_m["period_start_utc"] = pd.to_datetime(df_m["period_start_utc"], utc=True)
    m_p = partition_path(d, base_dir=mix_dir)
    m_p.parent.mkdir(parents=True, exist_ok=True)
    df_m.to_parquet(m_p, index=False)

    with pytest.raises(AssertionError, match="mix"):
        verify_curated_dataset_integrity(
            curated_dir=curated_dir,
            today=today,
            window_start=d,
            audit_dir=audit_dir,
            mix_dir=mix_dir,
            weather_dir=weather_dir,
        )


def test_data_check_fails_on_gap_overlapping_store_row(tmp_path: Path) -> None:
    """A gap recorded for a period that exists in the store fails."""
    import pytest

    d = date(2026, 1, 20)
    today = date(2026, 1, 21)

    curated_dir = tmp_path / "curated"
    mix_dir = tmp_path / "mix"
    weather_dir = tmp_path / "weather"
    audit_dir = tmp_path / "audit"

    periods = sorted(_expected_grid_for_day(d))
    start_utc = periods[0]
    c_rows = [
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
    ]
    df_c = pd.DataFrame(c_rows)
    df_c["period_start_utc"] = pd.to_datetime(df_c["period_start_utc"], utc=True)
    df_c["period_end_utc"] = pd.to_datetime(df_c["period_end_utc"], utc=True)
    df_c["ingested_at"] = pd.to_datetime(df_c["ingested_at"], utc=True)
    df_c["forecast"] = df_c["forecast"].astype("Int32")
    df_c["actual"] = df_c["actual"].astype("Int32")
    df_c["index"] = df_c["index"].astype("string")
    c_p = partition_path(d, base_dir=curated_dir)
    c_p.parent.mkdir(parents=True, exist_ok=True)
    df_c.to_parquet(c_p, index=False)

    hours = sorted({p.floor("h") for p in periods})
    df_w = pd.DataFrame({
        "period_start_utc": pd.to_datetime(hours, utc=True),
        "local_date": d,
        "temperature_2m_previous_day2": 12.0,
    })
    w_p = partition_path(d, base_dir=weather_dir)
    w_p.parent.mkdir(parents=True, exist_ok=True)
    df_w.to_parquet(w_p, index=False)

    # Mix: all 48 rows exist
    df_m = pd.DataFrame({
        "period_start_utc": pd.to_datetime(periods, utc=True),
        "local_date": d,
        "wind": 30.0,
        "solar": 10.0,
        "gas": 25.0,
        "coal": 1.0,
        "nuclear": 15.0,
        "biomass": 5.0,
    })
    m_p = partition_path(d, base_dir=mix_dir)
    m_p.parent.mkdir(parents=True, exist_ok=True)
    df_m.to_parquet(m_p, index=False)

    # But source-gaps.csv records a gap for period 0 (which exists in mix store)
    audit_dir.mkdir(parents=True, exist_ok=True)
    source_gaps = audit_dir / "source-gaps.csv"
    p0_str = pd.Timestamp(periods[0]).isoformat()
    source_gaps.write_text(
        f"store,local_date,period_start_utc,reason\nmix,{d.isoformat()},{p0_str},upstream_empty\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="coincid|overlap"):
        verify_curated_dataset_integrity(
            curated_dir=curated_dir,
            today=today,
            window_start=d,
            audit_dir=audit_dir,
            mix_dir=mix_dir,
            weather_dir=weather_dir,
        )


def test_recorded_source_gaps_are_justified() -> None:
    """Every source-gaps.csv row must have valid store, date, timestamp and reason."""
    audit_dir = Path("data/audit")
    source_gaps = audit_dir / "source-gaps.csv"
    if not source_gaps.exists():
        return

    frame = pd.read_csv(source_gaps)
    assert not frame.duplicated(subset=["store", "period_start_utc"]).any(), (
        "source-gaps.csv contains duplicated (store, period_start_utc) rows"
    )
    for _, row in frame.iterrows():
        assert row["store"] in ("mix", "weather")
        start = pd.Timestamp(row["period_start_utc"])
        if row["store"] == "mix":
            assert start.minute in (0, 30) and start.second == 0, (
                f"Mix gap not on half hour: {start}"
            )
        elif row["store"] == "weather":
            assert start.minute == 0 and start.second == 0, (
                f"Weather gap not on hour: {start}"
            )
        aware = start if start.tzinfo else start.tz_localize("UTC")
        local = aware.tz_convert(LONDON_TZ)
        assert str(local.date()) == str(row["local_date"]), (
            f"Gap {start} filed under {row['local_date']} but falls on {local.date()}"
        )

