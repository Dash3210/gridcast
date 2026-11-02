"""Persistence baseline forecast and scoreboard scoring for gridcast."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from gridcast.features import ACTUAL_AVAILABILITY_LAG, LONDON_TZ, issue_time_for_date

MODEL_VERSION: Final[str] = "persistence-v0"

DEFAULT_FORECASTS_PATH: Final[Path] = Path("data/scoreboard/forecasts.parquet")
DEFAULT_SCORES_PATH: Final[Path] = Path("data/scoreboard/scores.csv")
DEFAULT_SCOREBOARD_MD_PATH: Final[Path] = Path("docs/scoreboard.md")

FORECAST_COLUMNS: Final[list[str]] = [
    "issue_time",
    "horizon",
    "target_time",
    "model_version",
    "value",
    "issued_at",
]

SCORE_COLUMNS: Final[list[str]] = [
    "issue_time",
    "horizon",
    "target_time",
    "model_version",
    "forecast",
    "actual",
    "neso_forecast",
    "abs_error",
    "neso_abs_error",
]

HORIZON_BUCKETS: Final[list[tuple[str, int, int]]] = [
    ("0\u201312h", 1, 24),
    ("12\u201324h", 25, 48),
    ("24\u201336h", 49, 72),
    ("36\u201348h", 73, 96),
]


@dataclass(frozen=True)
class ScoreSummary:
    """Summary of scoring execution."""

    points_scored: int
    with_operator_forecast: int
    model_mae: float | None
    operator_mae: float | None
    total_points: int


def get_actual_at_timestamp(curated_dir: Path, target_ts: pd.Timestamp) -> float:
    """Lookup the settled actual carbon intensity at target_ts in curated archive."""
    if not curated_dir.exists():
        raise ValueError(f"Curated archive directory does not exist: {curated_dir}")

    ts_utc = pd.to_datetime(target_ts, utc=True)
    local_d = ts_utc.tz_convert(LONDON_TZ).date()
    target_partition = curated_dir / f"date={local_d}.parquet"

    if target_partition.exists():
        df = pd.read_parquet(target_partition)
        if "period_start_utc" in df.columns and "actual" in df.columns:
            matches = df[pd.to_datetime(df["period_start_utc"], utc=True) == ts_utc]
            if not matches.empty:
                val = matches["actual"].iloc[0]
                if pd.notna(val):
                    return float(val)
                msg = (
                    f"Actual observation at availability cutoff {ts_utc} "
                    f"is null in {target_partition.name}."
                )
                raise ValueError(msg)

    # Fallback to search all parquets in curated_dir (for custom fixtures or single files)
    for p in sorted(curated_dir.glob("*.parquet")):
        if p == target_partition:
            continue
        df = pd.read_parquet(p)
        if "period_start_utc" in df.columns and "actual" in df.columns:
            matches = df[pd.to_datetime(df["period_start_utc"], utc=True) == ts_utc]
            if not matches.empty:
                val = matches["actual"].iloc[0]
                if pd.notna(val):
                    return float(val)
                msg = (
                    f"Actual observation at availability cutoff {ts_utc} "
                    f"is null in {p.name}."
                )
                raise ValueError(msg)

    raise ValueError(
        f"Actual observation at availability cutoff {ts_utc} was not found in curated archive."
    )


def issue_forecast(
    issue_time: pd.Timestamp | datetime | str | None = None,
    date_val: date | str | None = None,
    curated_dir: Path = Path("data/curated"),
    output_path: Path = DEFAULT_FORECASTS_PATH,
    model_version: str = MODEL_VERSION,
) -> tuple[pd.DataFrame, int]:
    """Issue a 96-point half-hourly persistence baseline forecast for the next 48 hours."""
    if issue_time is not None:
        if isinstance(issue_time, (str, datetime, pd.Timestamp)):
            ts = pd.Timestamp(issue_time)
        else:
            raise ValueError(f"Unsupported issue_time type: {type(issue_time)}")
        if ts.tzinfo is None:
            resolved_issue = ts.tz_localize(LONDON_TZ).tz_convert("UTC")
        else:
            resolved_issue = ts.tz_convert("UTC")
    elif date_val is not None:
        resolved_issue = issue_time_for_date(date_val)
    else:
        today_local = datetime.now(LONDON_TZ).date()
        resolved_issue = issue_time_for_date(today_local)

    cutoff_ts = resolved_issue - ACTUAL_AVAILABILITY_LAG
    actual_val = get_actual_at_timestamp(curated_dir, cutoff_ts)

    issued_at_now = pd.Timestamp.now(timezone.utc)
    horizons = list(range(1, 97))
    target_times = [resolved_issue + pd.Timedelta(minutes=30 * h) for h in horizons]

    forecast_df = pd.DataFrame(
        {
            "issue_time": [resolved_issue] * 96,
            "horizon": horizons,
            "target_time": target_times,
            "model_version": [model_version] * 96,
            "value": [actual_val] * 96,
            "issued_at": [issued_at_now] * 96,
        }
    )

    forecast_df["issue_time"] = pd.to_datetime(forecast_df["issue_time"], utc=True)
    forecast_df["horizon"] = forecast_df["horizon"].astype("int64")
    forecast_df["target_time"] = pd.to_datetime(forecast_df["target_time"], utc=True)
    forecast_df["model_version"] = forecast_df["model_version"].astype("string")
    forecast_df["value"] = forecast_df["value"].astype("float64")
    forecast_df["issued_at"] = pd.to_datetime(forecast_df["issued_at"], utc=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_points_count = 96
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_keys = set(
            zip(
                pd.to_datetime(existing_df["issue_time"], utc=True),
                existing_df["horizon"].astype(int),
                existing_df["model_version"].astype(str),
                strict=False,
            )
        )
        mask = [
            (
                pd.to_datetime(row["issue_time"], utc=True),
                int(row["horizon"]),
                str(row["model_version"]),
            )
            not in existing_keys
            for _, row in forecast_df.iterrows()
        ]
        to_append = forecast_df[mask]
        new_points_count = len(to_append)
        if new_points_count > 0:
            combined = pd.concat([existing_df, to_append], ignore_index=True)
            combined.to_parquet(output_path, index=False)
    else:
        forecast_df.to_parquet(output_path, index=False)

    return forecast_df, new_points_count


def generate_scoreboard_markdown(
    scores_df: pd.DataFrame,
    output_path: Path = DEFAULT_SCOREBOARD_MD_PATH,
    start_date: date | str | None = None,
) -> str:
    """Generate docs/scoreboard.md from scores DataFrame."""
    if not scores_df.empty and "issue_time" in scores_df.columns:
        first_date = (
            pd.to_datetime(scores_df["issue_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date.min()
        )
    elif start_date is not None:
        first_date = (
            start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date))
        )
    else:
        first_date = datetime.now(LONDON_TZ).date()

    if scores_df.empty:
        consecutive_days = 0
        date_range_str = "None"
        total_points = 0
        model_mae_str = "N/A"
        operator_mae_str = "N/A"
    else:
        issue_dates = sorted(
            set(
                pd.to_datetime(scores_df["issue_time"], utc=True)
                .dt.tz_convert(LONDON_TZ)
                .dt.date
            )
        )
        consecutive_days = 1
        for i in range(1, len(issue_dates)):
            if issue_dates[i] == issue_dates[i - 1] + timedelta(days=1):
                consecutive_days += 1
            elif issue_dates[i] == issue_dates[i - 1]:
                continue
            else:
                break

        target_dates = (
            pd.to_datetime(scores_df["target_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        )
        min_td = target_dates.min()
        max_td = target_dates.max()
        date_range_str = f"{min_td} to {max_td}"

        total_points = len(scores_df)
        model_mae_str = f"{scores_df['abs_error'].astype(float).mean():.2f}"
        neso_non_null = scores_df[
            scores_df["neso_abs_error"].notna() & (scores_df["neso_abs_error"] != "")
        ]
        if not neso_non_null.empty:
            operator_mae_str = f"{neso_non_null['neso_abs_error'].astype(float).mean():.2f}"
        else:
            operator_mae_str = "N/A"

    bucket_rows: list[str] = []
    for label, h_min, h_max in HORIZON_BUCKETS:
        if not scores_df.empty:
            b_df = scores_df[
                (scores_df["horizon"].astype(int) >= h_min)
                & (scores_df["horizon"].astype(int) <= h_max)
            ]
            b_count = len(b_df)
            if b_count > 0:
                b_model_mae = f"{b_df['abs_error'].astype(float).mean():.2f}"
                b_neso = b_df[b_df["neso_abs_error"].notna() & (b_df["neso_abs_error"] != "")]
                b_oper_mae = (
                    f"{b_neso['neso_abs_error'].astype(float).mean():.2f}"
                    if not b_neso.empty
                    else "N/A"
                )
            else:
                b_model_mae = "N/A"
                b_oper_mae = "N/A"
        else:
            b_count = 0
            b_model_mae = "N/A"
            b_oper_mae = "N/A"
        bucket_rows.append(f"| {label} | {b_count} | {b_model_mae} | {b_oper_mae} |")

    bucket_table = "\n".join(bucket_rows)

    content = f"""# Scoreboard

scoreboard started {first_date}; consecutive days recorded: {consecutive_days}

## Overview

- **Date range covered:** {date_range_str}
- **Scored points:** {total_points}
- **Model MAE:** {model_mae_str}
- **Operator MAE:** {operator_mae_str}

## Horizon Performance

| Horizon Bucket | Scored Points | Model MAE | Operator MAE |
|---|---|---|---|
{bucket_table}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


def score_forecasts(
    curated_dir: Path = Path("data/curated"),
    forecasts_path: Path = DEFAULT_FORECASTS_PATH,
    scores_path: Path = DEFAULT_SCORES_PATH,
    scoreboard_md_path: Path = DEFAULT_SCOREBOARD_MD_PATH,
) -> ScoreSummary:
    """Find issued forecasts whose target_time now has a settled actual, score them,

    and update data/scoreboard/scores.csv and docs/scoreboard.md.
    """
    if scores_path.exists() and scores_path.stat().st_size > 0:
        existing_scores = pd.read_csv(scores_path)
        existing_keys = set(
            zip(
                pd.to_datetime(existing_scores["issue_time"], utc=True),
                existing_scores["horizon"].astype(int),
                existing_scores["model_version"].astype(str),
                strict=False,
            )
        )
    else:
        existing_scores = pd.DataFrame(columns=SCORE_COLUMNS)
        existing_keys = set()

    start_date: date | None = None
    if forecasts_path.exists():
        forecasts_df = pd.read_parquet(forecasts_path)
        if not forecasts_df.empty:
            start_date = (
                pd.to_datetime(forecasts_df["issue_time"], utc=True)
                .dt.tz_convert(LONDON_TZ)
                .dt.date.min()
            )
    else:
        forecasts_df = pd.DataFrame(columns=FORECAST_COLUMNS)

    if forecasts_df.empty:
        if not scores_path.exists() or scores_path.stat().st_size == 0:
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            existing_scores.to_csv(scores_path, index=False)
        generate_scoreboard_markdown(existing_scores, scoreboard_md_path, start_date=start_date)
        return ScoreSummary(0, 0, None, None, len(existing_scores))

    curated_cache: dict[date, pd.DataFrame] = {}

    def get_target_observation(target_ts: pd.Timestamp) -> tuple[float | None, float | None]:
        local_d = target_ts.tz_convert(LONDON_TZ).date()
        if local_d not in curated_cache:
            part_path = curated_dir / f"date={local_d}.parquet"
            if part_path.exists():
                curated_cache[local_d] = pd.read_parquet(part_path)
            else:
                found = False
                for p in sorted(curated_dir.glob("*.parquet")):
                    pdf = pd.read_parquet(p)
                    if "period_start_utc" in pdf.columns:
                        curated_cache[local_d] = pdf
                        found = True
                        break
                if not found:
                    curated_cache[local_d] = pd.DataFrame()

        cdf = curated_cache[local_d]
        if cdf.empty or "period_start_utc" not in cdf.columns or "actual" not in cdf.columns:
            return None, None
        matches = cdf[pd.to_datetime(cdf["period_start_utc"], utc=True) == target_ts]
        if matches.empty:
            return None, None
        actual_val = matches["actual"].iloc[0]
        if pd.isna(actual_val):
            return None, None

        operator_val = matches["forecast"].iloc[0] if "forecast" in matches.columns else None
        neso_f = (
            float(operator_val)
            if operator_val is not None and pd.notna(operator_val)
            else None
        )
        return float(actual_val), neso_f

    new_rows: list[dict[str, Any]] = []
    for _, row in forecasts_df.iterrows():
        it = pd.to_datetime(row["issue_time"], utc=True)
        h = int(row["horizon"])
        mv = str(row["model_version"])
        key = (it, h, mv)
        if key in existing_keys:
            continue

        tt = pd.to_datetime(row["target_time"], utc=True)
        actual_val, neso_forecast = get_target_observation(tt)
        if actual_val is None:
            continue

        f_val = float(row["value"])
        abs_err = abs(f_val - actual_val)
        neso_abs_err = abs(neso_forecast - actual_val) if neso_forecast is not None else None

        new_rows.append(
            {
                "issue_time": it.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "horizon": h,
                "target_time": tt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "model_version": mv,
                "forecast": f_val,
                "actual": actual_val,
                "neso_forecast": neso_forecast if neso_forecast is not None else "",
                "abs_error": abs_err,
                "neso_abs_error": neso_abs_err if neso_abs_err is not None else "",
            }
        )
        existing_keys.add(key)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if scores_path.exists() and scores_path.stat().st_size > 0:
            new_df.to_csv(scores_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(scores_path, mode="w", header=True, index=False)
        all_scores = pd.read_csv(scores_path)
    else:
        if not scores_path.exists() or scores_path.stat().st_size == 0:
            pd.DataFrame(columns=SCORE_COLUMNS).to_csv(
                scores_path, mode="w", header=True, index=False
            )
        all_scores = existing_scores

    generate_scoreboard_markdown(all_scores, scoreboard_md_path, start_date=start_date)

    points_scored = len(new_rows)
    if points_scored > 0:
        scored_df = pd.DataFrame(new_rows)
        with_neso = int((scored_df["neso_forecast"] != "").sum())
        model_mae = float(scored_df["abs_error"].astype(float).mean())
        neso_errs = scored_df.loc[
            scored_df["neso_abs_error"] != "", "neso_abs_error"
        ].astype(float)
        neso_mae = float(neso_errs.mean()) if not neso_errs.empty else None
    else:
        with_neso = 0
        model_mae = None
        neso_mae = None

    return ScoreSummary(
        points_scored=points_scored,
        with_operator_forecast=with_neso,
        model_mae=model_mae,
        operator_mae=neso_mae,
        total_points=len(all_scores),
    )


def run_forecast_cli(
    date_str: str | None = None,
    issue_time_str: str | None = None,
    curated_dir: Path = Path("data/curated"),
    output_path: Path = DEFAULT_FORECASTS_PATH,
) -> int:
    """CLI handler for gridcast forecast."""
    try:
        df, new_count = issue_forecast(
            issue_time=issue_time_str,
            date_val=date_str,
            curated_dir=curated_dir,
            output_path=output_path,
        )
        first_target = df["target_time"].min()
        last_target = df["target_time"].max()
        val = df["value"].iloc[0]
        print(
            f"Issued {len(df)} forecast points (model_version: {MODEL_VERSION}, value: {val:.1f})."
        )
        print(f"  First target: {first_target}")
        print(f"  Last target:  {last_target}")
        print(f"  New points written: {new_count}")
        print(f"  Store: {output_path}")
        return 0
    except Exception as exc:
        print(f"gridcast forecast error: {exc}", file=sys.stderr)
        return 1


def run_score_cli(
    curated_dir: Path = Path("data/curated"),
    forecasts_path: Path = DEFAULT_FORECASTS_PATH,
    scores_path: Path = DEFAULT_SCORES_PATH,
    scoreboard_md_path: Path = DEFAULT_SCOREBOARD_MD_PATH,
) -> int:
    """CLI handler for gridcast score."""
    try:
        summary = score_forecasts(
            curated_dir=curated_dir,
            forecasts_path=forecasts_path,
            scores_path=scores_path,
            scoreboard_md_path=scoreboard_md_path,
        )
        oper_mae_str = f"{summary.operator_mae:.2f}" if summary.operator_mae is not None else "N/A"
        mod_mae_str = f"{summary.model_mae:.2f}" if summary.model_mae is not None else "N/A"
        print("Score summary:")
        print(f"  points scored: {summary.points_scored}")
        print(f"  of which had operator forecast: {summary.with_operator_forecast}")
        print(f"  model MAE: {mod_mae_str}")
        print(f"  operator MAE: {oper_mae_str}")
        return 0
    except Exception as exc:
        print(f"gridcast score error: {exc}", file=sys.stderr)
        return 1
