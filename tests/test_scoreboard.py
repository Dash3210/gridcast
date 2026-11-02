"""Tests for persistence baseline forecast, scoreboard scoring, and docs generation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from gridcast import cli
from gridcast.scoreboard import (
    MODEL_VERSION,
    SCORE_COLUMNS,
    generate_scoreboard_markdown,
    issue_forecast,
    score_forecasts,
)


def _make_curated_partition(
    target_date: date,
    actual_by_hour: dict[int, int | None] | None = None,
    forecast_by_hour: dict[int, int | None] | None = None,
    default_actual: int | None = 150,
    default_forecast: int | None = 160,
) -> pd.DataFrame:
    """Create a minimal curated day partition with 48 half-hourly periods."""
    start_utc = pd.Timestamp(f"{target_date} 00:00:00", tz="UTC")
    periods = [start_utc + pd.Timedelta(minutes=30 * i) for i in range(48)]

    actuals: list[object] = []
    forecasts: list[object] = []
    for p in periods:
        h = p.hour
        act = (
            actual_by_hour[h]
            if (actual_by_hour and h in actual_by_hour)
            else default_actual
        )
        fct = (
            forecast_by_hour[h]
            if (forecast_by_hour and h in forecast_by_hour)
            else default_forecast
        )
        actuals.append(act if act is not None else pd.NA)
        forecasts.append(fct if fct is not None else pd.NA)

    return pd.DataFrame(
        {
            "period_start_utc": periods,
            "period_end_utc": [p + pd.Timedelta(minutes=30) for p in periods],
            "utc_date": [p.date() for p in periods],
            "local_date": [target_date] * 48,
            "forecast": pd.Series(forecasts, dtype="Int32"),
            "actual": pd.Series(actuals, dtype="Int32"),
            "index": ["moderate"] * 48,
            "ingested_at": [start_utc] * 48,
            "first_seen_at": [start_utc] * 48,
        }
    )


def test_forecast_issues_96_points(tmp_path: Path) -> None:
    """One issue yields 96 rows, absolute half-hour offsets, spanning 48 hours."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=175)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "scoreboard" / "forecasts.parquet"
    issue_time = "2026-11-02 00:00:00+00:00"

    df, new_count = issue_forecast(
        issue_time=issue_time,
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    assert len(df) == 96
    assert new_count == 96
    assert (df["model_version"] == "persistence-v0").all()
    assert list(df["horizon"]) == list(range(1, 97))

    expected_issue = pd.Timestamp(issue_time)
    assert (df["issue_time"] == expected_issue).all()
    assert (df["value"] == 175.0).all()

    expected_targets = [
        expected_issue + pd.Timedelta(minutes=30 * h) for h in range(1, 97)
    ]
    assert list(df["target_time"]) == expected_targets
    assert df["target_time"].iloc[0] == expected_issue + pd.Timedelta(minutes=30)
    assert df["target_time"].iloc[-1] == expected_issue + pd.Timedelta(hours=48)

    assert forecasts_path.exists()
    loaded = pd.read_parquet(forecasts_path)
    assert len(loaded) == 96
    assert (loaded["value"] == 175.0).all()


def test_forecast_uses_availability_allowance(tmp_path: Path) -> None:
    """The value equals stored actual at issue_time - 3h, and raises if missing."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()

    # issue_time: 2026-11-02 00:00:00+00:00; issue_time - 3h is 2026-11-01 21:00:00+00:00
    p_prev = _make_curated_partition(
        date(2026, 11, 1),
        actual_by_hour={21: 182, 22: 190, 20: 140},
        default_actual=100,
    )
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    df, _ = issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )
    assert (df["value"] == 182.0).all()

    # Null actual at 21:00 must raise rather than invent a number
    p_null = _make_curated_partition(
        date(2026, 11, 1),
        actual_by_hour={21: None},
        default_actual=100,
    )
    p_null.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    with pytest.raises(ValueError, match="is null"):
        issue_forecast(
            issue_time="2026-11-02 00:00:00+00:00",
            curated_dir=curated_dir,
            output_path=tmp_path / "f2.parquet",
        )

    # Missing partition / row must raise rather than invent a number
    empty_dir = tmp_path / "empty_curated"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="not found"):
        issue_forecast(
            issue_time="2026-11-02 00:00:00+00:00",
            curated_dir=empty_dir,
            output_path=tmp_path / "f3.parquet",
        )


def test_score_skips_unsettled_targets(tmp_path: Path) -> None:
    """A forecast whose target actual is null produces no score row."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # Target partition 2026-11-02 has null actuals
    p_today = _make_curated_partition(date(2026, 11, 2), default_actual=None)
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    summary = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )

    assert summary.points_scored == 0
    assert summary.total_points == 0
    assert scores_path.exists()
    scores_df = pd.read_csv(scores_path)
    assert len(scores_df) == 0
    assert list(scores_df.columns) == SCORE_COLUMNS


def test_score_is_idempotent(tmp_path: Path) -> None:
    """Scoring twice leaves one row per (issue_time, horizon, model_version)."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # Target partition 2026-11-02 has settled actuals for first 48 horizons
    p_today = _make_curated_partition(
        date(2026, 11, 2),
        default_actual=165,
        default_forecast=170,
    )
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    summary1 = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )
    assert summary1.points_scored == 47
    assert summary1.total_points == 47

    df1 = pd.read_csv(scores_path)
    assert len(df1) == 47

    # Second run must not duplicate any scored rows
    summary2 = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )
    assert summary2.points_scored == 0
    assert summary2.total_points == 47

    df2 = pd.read_csv(scores_path)
    assert len(df2) == 47
    keys = list(zip(df2["issue_time"], df2["horizon"], df2["model_version"], strict=True))
    assert len(keys) == len(set(keys))


def test_score_records_null_operator_forecast_as_null(tmp_path: Path) -> None:
    """Operator forecast is left null where null upstream and never imputed."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # 2026-11-02 has settled actuals, but hour 1 has null operator forecast
    p_today = _make_curated_partition(
        date(2026, 11, 2),
        actual_by_hour={1: 155},
        forecast_by_hour={1: None},
        default_actual=150,
        default_forecast=160,
    )
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )

    df = pd.read_csv(scores_path)
    # Horizon 2 corresponds to 01:00:00
    row_h2 = df[df["horizon"] == 2]
    assert len(row_h2) == 1
    assert pd.isna(row_h2["neso_forecast"].iloc[0])
    assert pd.isna(row_h2["neso_abs_error"].iloc[0])
    assert row_h2["actual"].iloc[0] == 155.0
    assert row_h2["abs_error"].iloc[0] == abs(160.0 - 155.0)


def test_scoreboard_markdown_regenerates(tmp_path: Path) -> None:
    """Running twice produces the same file for same scores, reflecting distinct dates present."""
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    # Distinct dates: 2 consecutive issue dates
    rows = [
        {
            "issue_time": "2026-11-02 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-02 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 155.0,
            "neso_forecast": 160.0,
            "abs_error": 5.0,
            "neso_abs_error": 5.0,
        },
        {
            "issue_time": "2026-11-03 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-03 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 140.0,
            "neso_forecast": 145.0,
            "abs_error": 10.0,
            "neso_abs_error": 5.0,
        },
    ]
    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(scores_path, index=False)

    content1 = generate_scoreboard_markdown(scores_df, output_path=md_path)
    content2 = generate_scoreboard_markdown(scores_df, output_path=md_path)

    assert content1 == content2
    assert "scoreboard started 2026-11-02; consecutive days recorded: 2" in content1
    assert "0\u201312h" in content1
    assert "12\u201324h" in content1
    assert "24\u201336h" in content1
    assert "36\u201348h" in content1
    assert "Scored points:** 2" in content1

    # Empty scores dataframe check
    empty_md_path = tmp_path / "empty_scoreboard.md"
    empty_content = generate_scoreboard_markdown(
        pd.DataFrame(columns=SCORE_COLUMNS),
        output_path=empty_md_path,
        start_date=date(2026, 11, 2),
    )
    assert "scoreboard started 2026-11-02; consecutive days recorded: 0" in empty_content
    assert "Scored points:** 0" in empty_content
    assert "Date range covered:** None" in empty_content


def test_cli_forecast_and_score(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI subcommands execute and print expected summaries."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    code_fc = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(curated_dir),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_fc == 0
    cap_fc = capsys.readouterr()
    assert "Issued 96 forecast points" in cap_fc.out
    assert "persistence-v0" in cap_fc.out

    # Re-issue (idempotent)
    code_fc2 = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(curated_dir),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_fc2 == 0
    cap_fc2 = capsys.readouterr()
    assert "New points written: 0" in cap_fc2.out

    code_sc = cli.main(
        [
            "score",
            "--curated-dir",
            str(curated_dir),
            "--forecasts-path",
            str(forecasts_path),
            "--scores-path",
            str(scores_path),
            "--scoreboard-md-path",
            str(md_path),
        ]
    )
    assert code_sc == 0
    cap_sc = capsys.readouterr()
    assert "points scored: 0" in cap_sc.out

    # Error handling in CLI
    code_err = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(tmp_path / "nonexistent"),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_err == 1
