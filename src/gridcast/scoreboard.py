"""Persistence and LightGBM live forecast issuing, scoring, and scoreboard generation."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from gridcast.features import ACTUAL_AVAILABILITY_LAG, LONDON_TZ, issue_time_for_date
from gridcast.holidays import UK_BANK_HOLIDAYS
from gridcast.models import load_model_booster, predict_lgbm

MODEL_VERSION: Final[str] = "persistence-v0"
LGBM_MODEL_VERSION: Final[str] = "lgbm-v2"

DEFAULT_FORECASTS_PATH: Final[Path] = Path("data/scoreboard/forecasts.parquet")
DEFAULT_SCORES_PATH: Final[Path] = Path("data/scoreboard/scores.csv")
DEFAULT_SCOREBOARD_MD_PATH: Final[Path] = Path("docs/scoreboard.md")
DEFAULT_SCOREBOARD_HTML_PATH: Final[Path] = Path("docs/index.html")
DEFAULT_MODEL_PATH: Final[Path] = Path("data/models/lgbm_model.txt")
DEFAULT_INTERVAL_PATH: Final[Path] = Path("data/models/interval_v2.json")
DEFAULT_MIX_DIR: Final[Path] = Path("data/mix")
DEFAULT_EVAL_FIXTURE_PATH: Final[Path] = Path("data/eval/frozen_split_v2.parquet")

FORECAST_COLUMNS: Final[list[str]] = [
    "issue_time",
    "horizon",
    "target_time",
    "model_version",
    "value",
    "issued_at",
    "neso_forecast_at_issue",
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
    "neso_source",
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


def get_operator_forecast_at_issue(
    curated_dir: Path, target_ts: pd.Timestamp, issue_ts: pd.Timestamp
) -> float | None:
    """Read the operator forecast published as of issue_ts for target_ts.

    Returns float forecast if published by issue_ts (forecast_first_seen_at <= issue_ts),
    otherwise None.
    """
    if not curated_dir.exists():
        return None

    ts_utc = pd.to_datetime(target_ts, utc=True)
    local_d = ts_utc.tz_convert(LONDON_TZ).date()
    target_partition = curated_dir / f"date={local_d}.parquet"

    df: pd.DataFrame | None = None
    if target_partition.exists():
        df = pd.read_parquet(target_partition)
    elif any(curated_dir.glob("date=*.parquet")):
        # Partitioned archive: missing date partition means target day is not yet ingested.
        return None
    else:
        for p in sorted(curated_dir.glob("*.parquet")):
            pdf = pd.read_parquet(p)
            if "period_start_utc" in pdf.columns:
                matches_p = pdf[pd.to_datetime(pdf["period_start_utc"], utc=True) == ts_utc]
                if not matches_p.empty:
                    df = pdf
                    break

    if df is None or df.empty:
        return None
    if "period_start_utc" not in df.columns or "forecast" not in df.columns:
        return None

    matches = df[pd.to_datetime(df["period_start_utc"], utc=True) == ts_utc]
    if matches.empty:
        return None

    val = matches["forecast"].iloc[0]
    if pd.isna(val) or val is None:
        return None

    if "forecast_first_seen_at" in matches.columns:
        seen = matches["forecast_first_seen_at"].iloc[0]
        if pd.isna(seen) or seen is None:
            return None
        seen_ts = pd.to_datetime(seen, utc=True)
        if seen_ts > issue_ts:
            return None
    else:
        return None

    return float(val)


def _resolve_issue_timestamp(
    issue_time: pd.Timestamp | datetime | str | None = None,
    date_val: date | str | None = None,
) -> pd.Timestamp:
    """Resolve an issue anchor to a UTC Timestamp (defaulting to today 00:00 Europe/London)."""
    if issue_time is not None:
        if isinstance(issue_time, (str, datetime, pd.Timestamp)):
            ts = pd.Timestamp(issue_time)
        else:
            raise ValueError(f"Unsupported issue_time type: {type(issue_time)}")
        if ts.tzinfo is None:
            return ts.tz_localize(LONDON_TZ).tz_convert("UTC")
        return ts.tz_convert("UTC")
    if date_val is not None:
        return issue_time_for_date(date_val)
    today_local = datetime.now(LONDON_TZ).date()
    return issue_time_for_date(today_local)


def _build_forecast_frame(
    resolved_issue: pd.Timestamp,
    values: list[float],
    model_version: str,
    curated_dir: Path,
) -> pd.DataFrame:
    """Construct a 96-row forecast DataFrame with operator snapshots as of issue_time."""
    if len(values) != 96:
        raise ValueError(f"Expected 96 forecast values, got {len(values)}")

    issued_at_now = pd.Timestamp.now(timezone.utc)
    horizons = list(range(1, 97))
    target_times = [resolved_issue + pd.Timedelta(minutes=30 * h) for h in horizons]

    neso_snapshots = [
        get_operator_forecast_at_issue(curated_dir, tt, resolved_issue) for tt in target_times
    ]

    forecast_df = pd.DataFrame(
        {
            "issue_time": [resolved_issue] * 96,
            "horizon": horizons,
            "target_time": target_times,
            "model_version": [model_version] * 96,
            "value": values,
            "issued_at": [issued_at_now] * 96,
            "neso_forecast_at_issue": neso_snapshots,
        }
    )

    forecast_df["issue_time"] = pd.to_datetime(forecast_df["issue_time"], utc=True)
    forecast_df["horizon"] = forecast_df["horizon"].astype("int64")
    forecast_df["target_time"] = pd.to_datetime(forecast_df["target_time"], utc=True)
    forecast_df["model_version"] = forecast_df["model_version"].astype("string")
    forecast_df["value"] = forecast_df["value"].astype("float64")
    forecast_df["issued_at"] = pd.to_datetime(forecast_df["issued_at"], utc=True)
    forecast_df["neso_forecast_at_issue"] = pd.to_numeric(
        forecast_df["neso_forecast_at_issue"], errors="coerce"
    )
    return forecast_df


def _append_forecasts_idempotently(forecast_df: pd.DataFrame, output_path: Path) -> int:
    """Append forecast rows keyed by (issue_time, horizon, model_version) without duplicates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_points_count = len(forecast_df)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        if "neso_forecast_at_issue" not in existing_df.columns:
            existing_df["neso_forecast_at_issue"] = float("nan")
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
    return new_points_count


def issue_forecast(
    issue_time: pd.Timestamp | datetime | str | None = None,
    date_val: date | str | None = None,
    curated_dir: Path = Path("data/curated"),
    output_path: Path = DEFAULT_FORECASTS_PATH,
    model_version: str = MODEL_VERSION,
) -> tuple[pd.DataFrame, int]:
    """Issue a 96-point half-hourly persistence baseline forecast for the next 48 hours."""
    resolved_issue = _resolve_issue_timestamp(issue_time=issue_time, date_val=date_val)
    cutoff_ts = resolved_issue - ACTUAL_AVAILABILITY_LAG
    actual_val = get_actual_at_timestamp(curated_dir, cutoff_ts)

    forecast_df = _build_forecast_frame(
        resolved_issue=resolved_issue,
        values=[actual_val] * 96,
        model_version=model_version,
        curated_dir=curated_dir,
    )
    new_points_count = _append_forecasts_idempotently(forecast_df, output_path)
    return forecast_df, new_points_count


def issue_lgbm_forecast(
    issue_time: pd.Timestamp | datetime | str | None = None,
    date_val: date | str | None = None,
    curated_dir: Path = Path("data/curated"),
    mix_dir: Path = DEFAULT_MIX_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    interval_path: Path = DEFAULT_INTERVAL_PATH,
    output_path: Path = DEFAULT_FORECASTS_PATH,
    model_version: str = LGBM_MODEL_VERSION,
) -> tuple[pd.DataFrame, int]:
    """Issue a 96-point half-hourly LightGBM forecast for the next 48 hours."""
    from gridcast.serve import resolve_features_for_issue

    resolved_issue = _resolve_issue_timestamp(issue_time=issue_time, date_val=date_val)
    booster = load_model_booster(model_path)

    i_path = Path(interval_path)
    if not i_path.exists():
        raise FileNotFoundError(f"Interval artifact not found at {i_path}")
    interval_data = json.loads(i_path.read_text(encoding="utf-8"))
    quantiles = interval_data.get("quantiles")
    if not isinstance(quantiles, dict):
        raise ValueError(f"Interval artifact {i_path} missing 'quantiles' mapping")

    X_issue = resolve_features_for_issue(
        issue_time=resolved_issue,
        booster=booster,
        model_path=model_path,
        curated_dir=curated_dir,
        mix_dir=mix_dir,
    )
    preds = np.asarray(
        predict_lgbm(booster, X_issue, feature_columns=booster.feature_name()),
        dtype=float,
    )
    if len(preds) != 96:
        raise ValueError(f"Expected 96 predictions for issue {resolved_issue}, got {len(preds)}")
    if not np.isfinite(preds).all():
        raise ValueError(f"Predictions for {resolved_issue} contain non-finite values")

    for h in range(1, 97):
        q_info = quantiles.get(str(h))
        if not isinstance(q_info, dict) or "q10" not in q_info or "q90" not in q_info:
            raise ValueError(f"Interval artifact {i_path} missing q10/q90 for horizon {h}")
        q10 = float(q_info["q10"])
        q90 = float(q_info["q90"])
        if not (q10 <= 0.0 <= q90 and q90 > q10):
            raise ValueError(f"Invalid interval quantiles at horizon {h}: q10={q10}, q90={q90}")

    forecast_df = _build_forecast_frame(
        resolved_issue=resolved_issue,
        values=[float(v) for v in preds],
        model_version=model_version,
        curated_dir=curated_dir,
    )
    new_points_count = _append_forecasts_idempotently(forecast_df, output_path)
    return forecast_df, new_points_count


def _compute_consecutive_days(scores_df: pd.DataFrame) -> int:
    """Count consecutive Europe/London issue dates starting from the first recorded issue."""
    if scores_df.empty or "issue_time" not in scores_df.columns:
        return 0
    issue_dates = sorted(
        set(
            pd.to_datetime(scores_df["issue_time"], utc=True)
            .dt.tz_convert(LONDON_TZ)
            .dt.date
        )
    )
    if not issue_dates:
        return 0
    consecutive = 1
    for i in range(1, len(issue_dates)):
        if issue_dates[i] == issue_dates[i - 1] + timedelta(days=1):
            consecutive += 1
        elif issue_dates[i] == issue_dates[i - 1]:
            continue
        else:
            break
    return consecutive


def _compute_macro_mae(df: pd.DataFrame, err_col: str = "abs_error") -> float | None:
    """Compute unweighted mean of per-horizon MAEs matching project scalar evaluation."""
    if df.empty or err_col not in df.columns:
        return None
    sub = df[df[err_col].notna() & (df[err_col] != "")].copy()
    if sub.empty:
        return None
    sub[err_col] = sub[err_col].astype(float)
    if "horizon" in sub.columns:
        return float(sub.groupby(sub["horizon"].astype(int))[err_col].mean().mean())
    return float(sub[err_col].mean())


def _render_bucket_rows(
    sub_df: pd.DataFrame,
    include_operator: bool = True,
    include_skill: bool = False,
) -> list[dict[str, Any]]:
    """Compute horizon-bucket metrics for 0-12h, 12-24h, 24-36h, and 36-48h."""
    rows: list[dict[str, Any]] = []
    for label, h_min, h_max in HORIZON_BUCKETS:
        if not sub_df.empty:
            b_df = sub_df[
                (sub_df["horizon"].astype(int) >= h_min)
                & (sub_df["horizon"].astype(int) <= h_max)
            ]
            b_count = len(b_df)
        else:
            b_df = sub_df
            b_count = 0

        if b_count > 0:
            m_mae_val: float | None = float(b_df["abs_error"].astype(float).mean())
            if include_operator and "neso_abs_error" in b_df.columns:
                b_neso = b_df[b_df["neso_abs_error"].notna() & (b_df["neso_abs_error"] != "")]
                op_mae_val: float | None = (
                    float(b_neso["neso_abs_error"].astype(float).mean())
                    if not b_neso.empty
                    else None
                )
            else:
                op_mae_val = None
            if (
                include_skill
                and m_mae_val is not None
                and op_mae_val is not None
                and op_mae_val > 0
            ):
                skill_val: float | None = (1.0 - m_mae_val / op_mae_val) * 100.0
            else:
                skill_val = None
        else:
            m_mae_val = None
            op_mae_val = None
            skill_val = None

        rows.append(
            {
                "label": label,
                "count": b_count,
                "model_mae": m_mae_val,
                "operator_mae": op_mae_val,
                "skill_pct": skill_val,
            }
        )
    return rows


def _format_markdown_buckets(
    sub_df: pd.DataFrame,
    include_operator: bool = True,
    include_skill: bool = False,
) -> str:
    """Format horizon bucket rows as Markdown table lines."""
    lines: list[str] = []
    for row in _render_bucket_rows(
        sub_df,
        include_operator=include_operator,
        include_skill=include_skill,
    ):
        m_str = f"{row['model_mae']:.2f}" if row["model_mae"] is not None else "N/A"
        if not include_operator:
            lines.append(f"| {row['label']} | {row['count']} | {m_str} |")
            continue
        o_str = f"{row['operator_mae']:.2f}" if row["operator_mae"] is not None else "N/A"
        if include_skill:
            sk_str = f"{row['skill_pct']:+.2f}%" if row["skill_pct"] is not None else "N/A"
            lines.append(f"| {row['label']} | {row['count']} | {m_str} | {o_str} | {sk_str} |")
        else:
            lines.append(f"| {row['label']} | {row['count']} | {m_str} | {o_str} |")
    return "\n".join(lines)


def _render_source_model_section(
    source_df: pd.DataFrame,
    *,
    heading: str,
    op_label: str,
    include_skill: bool,
    callout: str | None = None,
) -> str:
    """Render Snapshot or Archive markdown strictly separated per model_version."""
    parts: list[str] = [heading]
    if callout:
        parts.append(callout)

    table_header = (
        "| Horizon Bucket | Scored Points | Model MAE | Operator MAE | Skill vs Operator |\n"
        "|---|---|---|---|---|"
        if include_skill
        else "| Horizon Bucket | Scored Points | Model MAE | Operator MAE |\n|---|---|---|---|"
    )
    models = sorted(source_df["model_version"].astype(str).unique())
    for mv in models:
        mv_df = source_df[source_df["model_version"].astype(str) == mv]
        mv_td = pd.to_datetime(mv_df["target_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        mv_range = f"{mv_td.min()} to {mv_td.max()}"
        mv_points = len(mv_df)
        mv_model_mae_f = float(mv_df["abs_error"].astype(float).mean())
        mv_neso = mv_df[
            mv_df["neso_abs_error"].notna() & (mv_df["neso_abs_error"] != "")
        ]
        if not mv_neso.empty:
            mv_op_mae_f = float(mv_neso["neso_abs_error"].astype(float).mean())
            mv_op_str = f"{mv_op_mae_f:.2f}"
            skill_str = (
                f"{(1.0 - mv_model_mae_f / mv_op_mae_f) * 100.0:+.2f}%"
                if mv_op_mae_f > 0
                else "N/A"
            )
        else:
            mv_op_str = "N/A"
            skill_str = "N/A"

        bullets = [
            f"### {mv}",
            "",
            f"- **Date range covered:** {mv_range}",
            f"- **Scored points:** {mv_points}",
            f"- **Model MAE:** {mv_model_mae_f:.2f}",
            f"- **{op_label}:** {mv_op_str}",
        ]
        if include_skill:
            bullets.append(f"- **Skill vs Operator:** {skill_str}")
        buckets_str = _format_markdown_buckets(
            mv_df, include_operator=True, include_skill=include_skill
        )
        bullets.extend(["", "#### Horizon Performance", "", table_header, buckets_str])
        parts.append("\n".join(bullets))
    return "\n\n".join(parts)


def generate_scoreboard_markdown(
    scores_df: pd.DataFrame,
    output_path: Path = DEFAULT_SCOREBOARD_MD_PATH,
    start_date: date | str | None = None,
) -> str:
    """Generate docs/scoreboard.md from scores DataFrame.

    Reports all settled points per model_version (including neso_source == 'none')
    while strictly separating snapshot_at_issue and archive_at_scoring per ADR-022.
    """
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
        bucket_rows: list[str] = [
            f"| {label} | 0 | N/A | N/A |" for label, _, _ in HORIZON_BUCKETS
        ]
        bucket_table = "\n".join(bucket_rows)
        out_content = f"""# Scoreboard

scoreboard started {first_date}; consecutive days recorded: 0

## Overview

- **Date range covered:** None
- **Scored points:** 0
- **Model MAE:** N/A
- **Operator MAE:** N/A

## Horizon Performance

| Horizon Bucket | Scored Points | Model MAE | Operator MAE |
|---|---|---|---|
{bucket_table}
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out_content, encoding="utf-8")
        return out_content

    df = scores_df.copy()
    if "neso_source" not in df.columns:
        df["neso_source"] = "archive_at_scoring"
    else:
        df["neso_source"] = df["neso_source"].fillna("none").astype(str)
    if "model_version" not in df.columns:
        df["model_version"] = MODEL_VERSION

    consecutive_days = _compute_consecutive_days(df)

    header = (
        f"# Scoreboard\n\nscoreboard started {first_date}; "
        f"consecutive days recorded: {consecutive_days}"
    )
    sections: list[str] = [header]

    model_sections: list[str] = ["## Live Model Performance (All Settled Points)"]
    for mv in sorted(df["model_version"].astype(str).unique()):
        mv_df = df[df["model_version"].astype(str) == mv]
        mv_td = pd.to_datetime(mv_df["target_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        mv_range = f"{mv_td.min()} to {mv_td.max()}"
        mv_points = len(mv_df)
        mv_row_mae = float(mv_df["abs_error"].astype(float).mean())
        mv_macro_mae = _compute_macro_mae(mv_df, "abs_error")
        mv_macro_str = f"{mv_macro_mae:.2f}" if mv_macro_mae is not None else "N/A"
        src_counts = mv_df["neso_source"].value_counts().to_dict()
        src_summary = (
            f"snapshot_at_issue: {int(src_counts.get('snapshot_at_issue', 0))}, "
            f"archive_at_scoring: {int(src_counts.get('archive_at_scoring', 0))}, "
            f"none: {int(src_counts.get('none', 0))}"
        )
        mv_buckets = _format_markdown_buckets(mv_df, include_operator=False)
        model_sections.append(
            f"""### {mv}

- **Date range covered:** {mv_range}
- **Scored points:** {mv_points}
- **Model MAE (macro mean, h=1..96):** {mv_macro_str}
- **Model MAE (row mean):** {mv_row_mae:.2f}
- **Operator vintage breakdown:** {src_summary}

| Horizon Bucket | Scored Points | Model MAE |
|---|---|---|
{mv_buckets}"""
        )
    sections.append("\n\n".join(model_sections))

    snapshot_df = df[df["neso_source"] == "snapshot_at_issue"]
    archive_df = df[df["neso_source"] == "archive_at_scoring"]

    if not snapshot_df.empty:
        sections.append(
            _render_source_model_section(
                snapshot_df,
                heading="## Snapshot at Issue (Verified Vintage)",
                op_label="Operator MAE (at issue)",
                include_skill=True,
            )
        )
    elif not archive_df.empty:
        empty_snap_buckets = _format_markdown_buckets(
            snapshot_df, include_operator=True, include_skill=True
        )
        sections.append(
            f"""## Snapshot at Issue (Verified Vintage)

> No settled periods have a verified issue-time NESO snapshot yet (`snapshot_at_issue: 0`).
> Refusing to aggregate across sources; awaiting verified snapshots for skill comparison.

- **Date range covered:** None
- **Scored points:** 0
- **Model MAE:** N/A
- **Operator MAE (at issue):** N/A
- **Skill vs Operator:** N/A

### Horizon Performance

| Horizon Bucket | Scored Points | Model MAE | Operator MAE | Skill vs Operator |
|---|---|---|---|---|
{empty_snap_buckets}"""
        )

    if not archive_df.empty:
        sections.append(
            _render_source_model_section(
                archive_df,
                heading="## Archive at Scoring (Vintage Unverified)",
                op_label="Operator MAE (vintage unverified)",
                include_skill=False,
                callout=(
                    "> Published NESO forecast for the same periods, vintage unverified.\n"
                    "> Refusing to aggregate across sources; not eligible for skill comparison."
                ),
            )
        )

    out_content = "\n\n".join(sections) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out_content, encoding="utf-8")
    return out_content


def _compute_frozen_eval_context(
    eval_fixture_path: Path,
    mix_dir: Path,
) -> dict[str, Any] | None:
    """Compute frozen benchmark scalars, horizon buckets, and error decomposition."""
    if not eval_fixture_path.exists():
        return None
    df = pd.read_parquet(eval_fixture_path)
    required = {
        "issue_time",
        "horizon",
        "target_time",
        "actual",
        "neso_forecast",
        "persistence_pred",
        "seasonal_naive_pred",
        "candidate_pred",
    }
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["err_model"] = (df["candidate_pred"] - df["actual"]).abs()
    df["err_pers"] = (df["persistence_pred"] - df["actual"]).abs()
    df["err_sn"] = (df["seasonal_naive_pred"] - df["actual"]).abs()
    df["err_neso"] = (df["neso_forecast"] - df["actual"]).abs()

    fixture_sha = hashlib.sha256(eval_fixture_path.read_bytes()).hexdigest()

    per_h_model = df.groupby("horizon")["err_model"].mean()
    per_h_pers = df.groupby("horizon")["err_pers"].mean()
    per_h_sn = df.groupby("horizon")["err_sn"].mean()
    per_h_neso = df.groupby("horizon")["err_neso"].mean()

    model_gate = float(per_h_model.mean())
    pers_gate = float(per_h_pers.mean())
    sn_gate = float(per_h_sn.mean())
    neso_gate = float(per_h_neso.mean())

    buckets: list[dict[str, Any]] = []
    for label, lo, hi in HORIZON_BUCKETS:
        sub = df[(df["horizon"] >= lo) & (df["horizon"] <= hi)]
        buckets.append(
            {
                "label": label,
                "count": len(sub),
                "model": float(sub["err_model"].mean()),
                "persistence": float(sub["err_pers"].mean()),
                "seasonal_naive": float(sub["err_sn"].mean()),
                "neso": float(sub["err_neso"].mean()),
            }
        )

    local_target = df["target_time"].dt.tz_convert(LONDON_TZ)
    df["is_weekend"] = local_target.dt.weekday >= 5
    df["is_holiday"] = local_target.dt.date.isin(UK_BANK_HOLIDAYS)
    wkday = df[~df["is_weekend"]]
    wkend = df[df["is_weekend"]]
    hol_count = int(df["is_holiday"].sum())

    wind_regimes: list[dict[str, Any]] = []
    import_regimes: list[dict[str, Any]] = []
    if mix_dir.exists() and any(mix_dir.glob("date=*.parquet")):
        mix_df = pd.read_parquet(mix_dir)
        if "period_start_utc" in mix_df.columns and "wind" in mix_df.columns:
            mix_df = mix_df.copy()
            mix_df["period_start_utc"] = pd.to_datetime(mix_df["period_start_utc"], utc=True)
            mix_df = mix_df.sort_values("period_start_utc").drop_duplicates("period_start_utc")
            if "imports" in mix_df.columns:
                mix_df["d_imports_30m"] = mix_df["imports"].astype(float).diff().abs()
            merged = df.merge(
                mix_df,
                left_on="target_time",
                right_on="period_start_utc",
                how="left",
            )
            wind_bins = [
                ("Low wind / drought (<15%)", merged["wind"] < 15.0),
                ("Moderate wind (15–30%)", (merged["wind"] >= 15.0) & (merged["wind"] < 30.0)),
                ("High wind (>=30%)", merged["wind"] >= 30.0),
            ]
            for label, mask in wind_bins:
                sub = merged[mask.fillna(False)]
                if not sub.empty:
                    wind_regimes.append(
                        {
                            "label": label,
                            "count": len(sub),
                            "model": float(sub["err_model"].mean()),
                            "persistence": float(sub["err_pers"].mean()),
                            "neso": float(sub["err_neso"].mean()),
                        }
                    )
            if "d_imports_30m" in merged.columns:
                for label, mask in [
                    ("Interconnector ramp (|Δimports| >= 2.0 pp)", merged["d_imports_30m"] >= 2.0),
                    ("Calm imports (|Δimports| < 2.0 pp)", merged["d_imports_30m"] < 2.0),
                ]:
                    sub = merged[mask.fillna(False)]
                    if not sub.empty:
                        import_regimes.append(
                            {
                                "label": label,
                                "count": len(sub),
                                "model": float(sub["err_model"].mean()),
                                "persistence": float(sub["err_pers"].mean()),
                                "neso": float(sub["err_neso"].mean()),
                            }
                        )

    origin_rows: list[dict[str, Any]] = []
    df["origin_date"] = df["issue_time"].dt.tz_convert(LONDON_TZ).dt.date
    for od, grp in df.groupby("origin_date"):
        origin_rows.append(
            {
                "date": str(od),
                "model": float(grp["err_model"].mean()),
                "persistence": float(grp["err_pers"].mean()),
                "neso": float(grp["err_neso"].mean()),
            }
        )
    origin_rows.sort(key=lambda r: r["date"])

    target_end_date = (
        (df["target_time"] - pd.Timedelta(minutes=30)).dt.tz_convert(LONDON_TZ).dt.date.max()
    )
    window_end_date = max(df["origin_date"].max(), target_end_date)

    return {
        "row_count": len(df),
        "origin_count": int(df["issue_time"].nunique()),
        "window_start": str(df["origin_date"].min()),
        "window_end": str(window_end_date),
        "fixture_sha256": fixture_sha,
        "model_gate": model_gate,
        "persistence_gate": pers_gate,
        "seasonal_naive_gate": sn_gate,
        "neso_gate": neso_gate,
        "buckets": buckets,
        "wind_regimes": wind_regimes,
        "import_regimes": import_regimes,
        "weekday": {
            "count": len(wkday),
            "model": float(wkday["err_model"].mean()),
            "persistence": float(wkday["err_pers"].mean()),
            "neso": float(wkday["err_neso"].mean()),
        },
        "weekend": {
            "count": len(wkend),
            "model": float(wkend["err_model"].mean()),
            "persistence": float(wkend["err_pers"].mean()),
            "neso": float(wkend["err_neso"].mean()),
        },
        "holiday_count": hol_count,
        "origins": origin_rows,
    }


def _svg_grouped_bars(
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_unit: str = "MAE (gCO2/kWh)",
) -> str:
    """Render a deterministic inline SVG grouped bar chart."""
    w, h = 680, 250
    pad_l, pad_r, pad_t, pad_b = 54, 18, 28, 50
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    all_vals = [v for _, vals, _ in series for v in vals if np.isfinite(v)]
    max_val = max(all_vals) if all_vals else 10.0
    y_max = max(10.0, float(np.ceil(max_val / 10.0) * 10.0))

    n_groups = max(1, len(groups))
    n_series = max(1, len(series))
    group_w = plot_w / n_groups
    bar_w = min(26.0, (group_w * 0.76) / n_series)
    bars_span = bar_w * n_series

    parts: list[str] = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}" '
        f'class="chart-svg">'
    ]
    for tick_idx in range(5):
        y_val = y_max * tick_idx / 4.0
        y_pos = pad_t + plot_h - (y_val / y_max) * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{y_pos:.1f}" x2="{w - pad_r}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y_pos + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#475569">{y_val:.0f}</text>'
        )

    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#475569" transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">'
        f"{html.escape(y_unit)}</text>"
    )

    for g_idx, g_label in enumerate(groups):
        gx = pad_l + g_idx * group_w + (group_w - bars_span) / 2.0
        for s_idx, (_, vals, color) in enumerate(series):
            val = vals[g_idx] if g_idx < len(vals) else 0.0
            bh = (val / y_max) * plot_h if y_max > 0 else 0.0
            bx = gx + s_idx * bar_w
            by = pad_t + plot_h - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 2:.1f}" '
                f'height="{max(0.0, bh):.1f}" rx="2" fill="{color}">'
                f"<title>{html.escape(g_label)}: {val:.2f}</title></rect>"
            )
            if bh >= 14:
                parts.append(
                    f'<text x="{bx + (bar_w - 2) / 2:.1f}" y="{by - 4:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="#1e293b">{val:.1f}</text>'
                )
        parts.append(
            f'<text x="{pad_l + (g_idx + 0.5) * group_w:.1f}" y="{pad_t + plot_h + 18}" '
            f'text-anchor="middle" font-size="11" fill="#334155">{html.escape(g_label)}</text>'
        )

    lx: float = float(pad_l)
    ly = h - 12
    for name, _, color in series:
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{color}" />')
        parts.append(
            f'<text x="{lx + 14}" y="{ly}" font-size="11" fill="#334155">'
            f"{html.escape(name)}</text>"
        )
        lx += 16 + len(name) * 6.8 + 16

    parts.append("</svg>")
    return "".join(parts)


def _svg_line_series(
    x_labels: list[str],
    series: list[tuple[str, list[float | None], str]],
    title: str,
    y_unit: str = "gCO2/kWh",
    band: tuple[list[float], list[float]] | None = None,
) -> str:
    """Render a deterministic inline SVG line chart with optional shaded interval band."""
    w, h = 680, 240
    pad_l, pad_r, pad_t, pad_b = 54, 18, 22, 46
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    vals: list[float] = [
        float(v) for _, ys, _ in series for v in ys if v is not None and np.isfinite(v)
    ]
    if band is not None:
        vals.extend(float(v) for v in band[0] if np.isfinite(v))
        vals.extend(float(v) for v in band[1] if np.isfinite(v))

    y_min = min(0.0, min(vals)) if vals else 0.0
    y_max = max(vals) if vals else 100.0
    if y_max <= y_min:
        y_max = y_min + 10.0
    y_span = y_max - y_min

    n = max(1, len(x_labels))

    def _xy(idx: int, val: float) -> tuple[float, float]:
        xp = pad_l + (idx / max(1, n - 1)) * plot_w if n > 1 else pad_l + plot_w / 2.0
        yp = pad_t + plot_h - ((val - y_min) / y_span) * plot_h
        return xp, yp

    parts: list[str] = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}" '
        f'class="chart-svg">'
    ]
    for tick_idx in range(5):
        yv = y_min + y_span * tick_idx / 4.0
        yp = pad_t + plot_h - (tick_idx / 4.0) * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{w - pad_r}" y2="{yp:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{yp + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#475569">{yv:.0f}</text>'
        )

    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#475569" transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">'
        f"{html.escape(y_unit)}</text>"
    )

    if band is not None and len(band[0]) == n and len(band[1]) == n:
        lows, highs = band
        upper_pts = [_xy(i, float(highs[i])) for i in range(n)]
        lower_pts = [_xy(i, float(lows[i])) for i in range(n - 1, -1, -1)]
        poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in [*upper_pts, *lower_pts])
        parts.append(
            f'<polygon points="{poly_pts}" fill="#0f766e" fill-opacity="0.14" '
            f'stroke="none"><title>80% empirical interval (q10-q90)</title></polygon>'
        )

    for _name, ys, color in series:
        pts: list[tuple[float, float]] = []
        for idx, val in enumerate(ys):
            if val is not None and np.isfinite(val):
                pts.append(_xy(idx, float(val)))
        if len(pts) >= 2:
            d_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                f'points="{d_pts}" />'
            )
        elif len(pts) == 1:
            parts.append(
                f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="4" fill="{color}" />'
            )
        if len(pts) <= 20:
            for xp, yp in pts:
                parts.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="3" fill="{color}" />')

    step = max(1, n // 6)
    tick_indices = list(range(0, n, step))
    if (n - 1) not in tick_indices and n > 1:
        tick_indices.append(n - 1)
    for idx in tick_indices:
        xp, _ = _xy(idx, y_min)
        parts.append(
            f'<text x="{xp:.1f}" y="{pad_t + plot_h + 16}" text-anchor="middle" '
            f'font-size="10" fill="#475569">{html.escape(x_labels[idx])}</text>'
        )

    lx: float = float(pad_l)
    ly = h - 8
    for name, _, color in series:
        parts.append(
            f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 14}" y2="{ly - 4}" '
            f'stroke="{color}" stroke-width="2.5" />'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{ly}" font-size="11" fill="#334155">'
            f"{html.escape(name)}</text>"
        )
        lx += 20 + len(name) * 6.8 + 16

    parts.append("</svg>")
    return "".join(parts)


def _summarize_source_by_model(source_df: pd.DataFrame, *, include_skill: bool) -> str:
    """Format Snapshot or Archive summary per model_version without pooling across models."""
    if source_df.empty:
        return "<strong>0</strong> pts (Model MAE: N/A, Operator MAE: N/A)"
    items: list[str] = []
    for mv in sorted(source_df["model_version"].astype(str).unique()):
        mv_df = source_df[source_df["model_version"].astype(str) == mv]
        mv_td = pd.to_datetime(mv_df["target_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        mv_range = html.escape(f"{mv_td.min()} to {mv_td.max()}")
        m_mae = float(mv_df["abs_error"].astype(float).mean())
        neso_vals = mv_df[mv_df["neso_abs_error"].notna() & (mv_df["neso_abs_error"] != "")]
        if not neso_vals.empty:
            o_mae_f = float(neso_vals["neso_abs_error"].astype(float).mean())
            o_mae_str = f"{o_mae_f:.2f}"
            skill_part = (
                f", Skill: {(1.0 - m_mae / o_mae_f) * 100.0:+.2f}%"
                if include_skill and o_mae_f > 0
                else ""
            )
        else:
            o_mae_str = "N/A"
            skill_part = ""
        items.append(
            f"<code>{html.escape(mv)}</code>: <strong>{len(mv_df)}</strong> pts "
            f"({mv_range}, Model MAE: {m_mae:.2f}, Operator MAE: {o_mae_str}{skill_part})"
        )
    return "; ".join(items)


def generate_scoreboard_html(
    scores_df: pd.DataFrame,
    output_path: Path = DEFAULT_SCOREBOARD_HTML_PATH,
    forecasts_path: Path = DEFAULT_FORECASTS_PATH,
    eval_fixture_path: Path = DEFAULT_EVAL_FIXTURE_PATH,
    mix_dir: Path = DEFAULT_MIX_DIR,
    interval_path: Path = DEFAULT_INTERVAL_PATH,
    start_date: date | str | None = None,
) -> str:
    """Generate static self-contained GitHub Pages HTML scoreboard (docs/index.html)."""
    df = scores_df.copy()
    if not df.empty:
        if "neso_source" not in df.columns:
            df["neso_source"] = "archive_at_scoring"
        else:
            df["neso_source"] = df["neso_source"].fillna("none").astype(str)
        if "model_version" not in df.columns:
            df["model_version"] = MODEL_VERSION

    if not df.empty and "issue_time" in df.columns:
        first_date = str(
            pd.to_datetime(df["issue_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date.min()
        )
    elif start_date is not None:
        first_date = str(
            start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date))
        )
    else:
        first_date = "N/A"

    consecutive_days = _compute_consecutive_days(df)
    total_scored = len(df)
    if not df.empty and "target_time" in df.columns:
        td_series = pd.to_datetime(df["target_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        live_date_range = f"{td_series.min()} to {td_series.max()}"
    else:
        live_date_range = "None"

    frozen = _compute_frozen_eval_context(eval_fixture_path, mix_dir)

    latest_issue_str = "None"
    latest_models_str = "None"
    latest_curve_svg = '<p class="muted">No issued forecasts recorded yet.</p>'
    latest_curve_table_html = ""
    if forecasts_path.exists():
        fc_df = pd.read_parquet(forecasts_path)
        if not fc_df.empty and "issue_time" in fc_df.columns:
            fc_df = fc_df.copy()
            fc_df["issue_time"] = pd.to_datetime(fc_df["issue_time"], utc=True)
            latest_it = fc_df["issue_time"].max()
            latest_issue_str = latest_it.tz_convert(LONDON_TZ).strftime("%Y-%m-%d %H:%M %Z")
            latest_fc = fc_df[fc_df["issue_time"] == latest_it].sort_values(
                ["model_version", "horizon"]
            )
            models_present = sorted(latest_fc["model_version"].astype(str).unique())
            latest_models_str = ", ".join(models_present)

            x_labels = [
                f"{h * 0.5:.0f}h" if h % 2 == 0 else f"{h * 0.5:.1f}h"
                for h in range(1, 97)
            ]
            palette = {
                LGBM_MODEL_VERSION: "#0f766e",
                MODEL_VERSION: "#64748b",
            }
            line_series: list[tuple[str, list[float | None], str]] = []
            band_vals: tuple[list[float], list[float]] | None = None
            interval_q: dict[str, Any] = {}
            if interval_path.exists():
                try:
                    interval_q = json.loads(interval_path.read_text(encoding="utf-8")).get(
                        "quantiles", {}
                    )
                except Exception:
                    interval_q = {}

            table_rows: list[str] = []
            for mv in models_present:
                mv_fc = latest_fc[latest_fc["model_version"].astype(str) == mv].sort_values(
                    "horizon"
                )
                by_h = {int(r["horizon"]): float(r["value"]) for _, r in mv_fc.iterrows()}
                ys = [by_h.get(h) for h in range(1, 97)]
                color = palette.get(mv, "#1d4ed8")
                line_series.append((mv, ys, color))
                if mv == LGBM_MODEL_VERSION and interval_q:
                    lows: list[float] = []
                    highs: list[float] = []
                    for h in range(1, 97):
                        y_h = ys[h - 1]
                        q_h = interval_q.get(str(h))
                        if y_h is not None and isinstance(q_h, dict):
                            lows.append(float(y_h) + float(q_h["q10"]))
                            highs.append(float(y_h) + float(q_h["q90"]))
                    if len(lows) == 96 and len(highs) == 96:
                        band_vals = (lows, highs)

                vals_only = [float(v) for v in mv_fc["value"].tolist()]
                table_rows.append(
                    f"<tr><td><code>{html.escape(mv)}</code></td>"
                    f"<td>{len(vals_only)}</td>"
                    f"<td>{min(vals_only):.1f}</td>"
                    f"<td>{float(np.mean(vals_only)):.1f}</td>"
                    f"<td>{max(vals_only):.1f}</td></tr>"
                )
            latest_curve_svg = _svg_line_series(
                x_labels=x_labels,
                series=line_series,
                title="Latest 48h Issued Carbon Intensity Forecast",
                y_unit="gCO2/kWh",
                band=band_vals,
            )
            joined_rows = "\n".join(table_rows)
            latest_curve_table_html = (
                "<table><thead><tr>"
                '<th scope="col">Model</th><th scope="col">Horizons</th>'
                '<th scope="col">Min</th><th scope="col">Mean</th><th scope="col">Max</th>'
                f"</tr></thead><tbody>{joined_rows}</tbody></table>"
            )

    live_model_blocks: list[str] = []
    live_daily_svg = '<p class="muted">No settled daily trend yet.</p>'
    live_bucket_svg = '<p class="muted">No settled horizon buckets yet.</p>'
    pers_kpi_str = "N/A"
    pers_kpi_sub = "0 settled points"

    if not df.empty:
        df["issue_date_local"] = (
            pd.to_datetime(df["issue_time"], utc=True).dt.tz_convert(LONDON_TZ).dt.date
        )
        all_dates = sorted(df["issue_date_local"].unique())
        date_labels = [d.strftime("%m-%d") for d in all_dates]
        daily_series: list[tuple[str, list[float | None], str]] = []
        bucket_series: list[tuple[str, list[float], str]] = []
        palette = {LGBM_MODEL_VERSION: "#0f766e", MODEL_VERSION: "#64748b"}

        for mv in sorted(df["model_version"].astype(str).unique()):
            mv_df = df[df["model_version"].astype(str) == mv]
            macro = _compute_macro_mae(mv_df, "abs_error")
            row_mae = float(mv_df["abs_error"].astype(float).mean())
            macro_str = f"{macro:.2f}" if macro is not None else "N/A"
            if mv == MODEL_VERSION:
                pers_kpi_str = f"{macro_str} gCO2/kWh"
                pers_kpi_sub = f"n={len(mv_df)} settled (row mean {row_mae:.2f})"

            by_date = mv_df.groupby("issue_date_local")["abs_error"].mean().to_dict()
            mv_color = palette.get(mv, "#1d4ed8")
            daily_vals: list[float | None] = [
                float(by_date[d]) if d in by_date else None for d in all_dates
            ]
            daily_series.append((mv, daily_vals, mv_color))
            b_rows = _render_bucket_rows(mv_df, include_operator=False)
            bucket_series.append(
                (mv, [float(r["model_mae"] or 0.0) for r in b_rows], mv_color)
            )

            tr_parts: list[str] = []
            for r in b_rows:
                m_cell = f"{r['model_mae']:.2f}" if r["model_mae"] is not None else "N/A"
                lbl = html.escape(str(r["label"]))
                tr_parts.append(
                    f"<tr><td>{lbl}</td><td>{r['count']}</td><td>{m_cell}</td></tr>"
                )
            tr_html = "\n".join(tr_parts)
            live_model_blocks.append(
                "<div class=\"card\">\n"
                f"  <h3><code>{html.escape(mv)}</code> &mdash; {len(mv_df)} settled points</h3>\n"
                f"  <p class=\"meta\">Macro MAE (h=1..96): <strong>{macro_str}</strong>"
                f" &middot; Row MAE: <strong>{row_mae:.2f}</strong> gCO2/kWh</p>\n"
                "  <table>\n"
                '    <thead><tr><th scope="col">Horizon</th>'
                '<th scope="col">Points</th><th scope="col">Model MAE</th></tr></thead>\n'
                f"    <tbody>{tr_html}</tbody>\n"
                "  </table>\n"
                "</div>"
            )

        live_daily_svg = _svg_line_series(
            x_labels=date_labels,
            series=daily_series,
            title="Daily Live Settled MAE by Issue Date",
            y_unit="MAE (gCO2/kWh)",
        )
        live_bucket_svg = _svg_grouped_bars(
            groups=[label for label, _, _ in HORIZON_BUCKETS],
            series=bucket_series,
            title="Live Settled MAE by Horizon Bucket",
        )

    snap_df = df[df["neso_source"] == "snapshot_at_issue"] if not df.empty else pd.DataFrame()
    arch_df = df[df["neso_source"] == "archive_at_scoring"] if not df.empty else pd.DataFrame()
    snap_summary_html = _summarize_source_by_model(snap_df, include_skill=True)
    arch_summary_html = _summarize_source_by_model(arch_df, include_skill=False)

    if frozen is not None:
        frozen_kpi_str = f"{frozen['model_gate']:.3f} gCO2/kWh"
        frozen_kpi_sub = (
            f"PASS &le; 43.338 ({frozen['origin_count']} origins, n={frozen['row_count']})"
        )
        frozen_bucket_svg = _svg_grouped_bars(
            groups=[str(b["label"]) for b in frozen["buckets"]],
            series=[
                ("lgbm-v2", [float(b["model"]) for b in frozen["buckets"]], "#0f766e"),
                ("persistence", [float(b["persistence"]) for b in frozen["buckets"]], "#94a3b8"),
                (
                    "seasonal-naive",
                    [float(b["seasonal_naive"]) for b in frozen["buckets"]],
                    "#f59e0b",
                ),
                ("NESO*", [float(b["neso"]) for b in frozen["buckets"]], "#2563eb"),
            ],
            title="Frozen Evaluation Split MAE by Horizon Bucket",
        )
        decomp_rows: list[str] = []
        for reg in [*frozen["wind_regimes"], *frozen["import_regimes"]]:
            decomp_rows.append(
                f"<tr><td>{html.escape(str(reg['label']))}</td><td>{reg['count']}</td>"
                f"<td>{reg['model']:.2f}</td><td>{reg['persistence']:.2f}</td>"
                f"<td>{reg['neso']:.2f}</td></tr>"
            )
        for label, item in [
            ("Weekday targets", frozen["weekday"]),
            ("Weekend targets", frozen["weekend"]),
        ]:
            decomp_rows.append(
                f"<tr><td>{label}</td><td>{item['count']}</td>"
                f"<td>{item['model']:.2f}</td><td>{item['persistence']:.2f}</td>"
                f"<td>{item['neso']:.2f}</td></tr>"
            )
        decomp_rows.append(
            f"<tr><td>UK bank holidays in window</td><td>{frozen['holiday_count']}</td>"
            "<td>N/A (unmeasured)</td><td>N/A</td><td>N/A</td></tr>"
        )
        min_origin = min(frozen["origins"], key=lambda o: o["model"])
        max_origin = max(frozen["origins"], key=lambda o: o["model"])
        sha_short = html.escape(str(frozen["fixture_sha256"])[:12])
        win_start = html.escape(str(frozen["window_start"]))
        win_end = html.escape(str(frozen["window_end"]))
        min_o_date = html.escape(str(min_origin["date"]))
        max_o_date = html.escape(str(max_origin["date"]))
        frozen_html = (
            '<section id="frozen-eval">\n'
            "  <h2>Frozen 15-Origin Benchmark &amp; Error Decomposition</h2>\n"
            f'  <p class="muted">Evaluated on <code>frozen_split_v2.parquet</code> '
            f"(sha256 <code>{sha_short}...</code>, window {win_start} to {win_end}, "
            f"{frozen['origin_count']} origins, n={frozen['row_count']}).</p>\n"
            '  <div class="grid-2">\n'
            '    <div class="card">\n'
            "      <h3>Gate Scalars (Unweighted Mean MAE, h=1..96)</h3>\n"
            "      <table>\n"
            '        <thead><tr><th scope="col">Series</th><th scope="col">GATE_VALUE</th>'
            '<th scope="col">Notes</th></tr></thead>\n'
            "        <tbody>\n"
            f"          <tr><td><strong><code>lgbm-v2</code></strong></td>"
            f"<td><strong>{frozen['model_gate']:.3f}</strong></td>"
            "<td>PASS (&le; 43.3379 threshold)</td></tr>\n"
            f"          <tr><td><code>persistence</code></td>"
            f"<td>{frozen['persistence_gate']:.3f}</td>"
            "<td>Issue-anchored (T &minus; 3h)</td></tr>\n"
            f"          <tr><td><code>seasonal-naive</code></td>"
            f"<td>{frozen['seasonal_naive_gate']:.3f}</td>"
            "<td>Diurnal lag (&le; T &minus; 3h)</td></tr>\n"
            f"          <tr><td><code>NESO operator*</code></td>"
            f"<td>{frozen['neso_gate']:.3f}</td>"
            "<td>*Published NESO forecast, vintage unverified (ADR-022)</td></tr>\n"
            "        </tbody>\n"
            "      </table>\n"
            f'      <p class="meta">Origin range: best <code>{min_o_date}</code> '
            f"({min_origin['model']:.2f}) &middot; worst <code>{max_o_date}</code> "
            f"({max_origin['model']:.2f}, persistence {max_origin['persistence']:.2f}).</p>\n"
            "    </div>\n"
            '    <div class="card">\n'
            "      <h3>Horizon-Bucket MAE (Frozen Split)</h3>\n"
            f"      {frozen_bucket_svg}\n"
            "    </div>\n"
            "  </div>\n"
            '  <div class="card">\n'
            "    <h3>When This Model Is Wrong (Error Decomposition)</h3>\n"
            "    <table>\n"
            '      <thead><tr><th scope="col">Regime / Slice</th>'
            '<th scope="col">Periods</th><th scope="col">lgbm-v2 MAE</th>'
            '<th scope="col">Persistence MAE</th>'
            '<th scope="col">NESO MAE*</th></tr></thead>\n'
            f"      <tbody>{''.join(decomp_rows)}</tbody>\n"
            "    </table>\n"
            "  </div>\n"
            "</section>"
        )
    else:
        frozen_kpi_str = "Unavailable"
        frozen_kpi_sub = "Frozen fixture not present in workspace"
        frozen_html = (
            '<section id="frozen-eval">\n'
            "  <h2>Frozen 15-Origin Benchmark &amp; Error Decomposition</h2>\n"
            '  <div class="card"><p class="muted">'
            "Frozen evaluation fixture not present in this workspace.</p></div>\n"
            "</section>"
        )

    live_cards_html = (
        "\n".join(live_model_blocks)
        if live_model_blocks
        else '<div class="card"><p class="muted">No settled live forecasts recorded yet.</p></div>'
    )

    css_block = """    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #0f766e;
      --warn-bg: #fffbeb;
      --warn-border: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }
    header {
      background: #0f172a;
      color: #f8fafc;
      padding: 2rem 1.5rem;
      border-bottom: 4px solid var(--accent);
    }
    .wrap { max-width: 1100px; margin: 0 auto; }
    header h1 { margin: 0 0 0.35rem; font-size: 1.75rem; }
    header p { margin: 0; color: #cbd5e1; font-size: 0.95rem; }
    main { padding: 1.5rem; }
    section { margin-bottom: 2rem; }
    h2 {
      font-size: 1.25rem;
      margin: 0 0 0.75rem;
      padding-bottom: 0.35rem;
      border-bottom: 2px solid var(--border);
    }
    h3 { font-size: 1.05rem; margin: 0 0 0.5rem; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.75rem;
    }
    .kpi, .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.15rem;
    }
    .kpi-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .kpi-value { font-size: 1.45rem; font-weight: 700; margin: 0.2rem 0; color: var(--ink); }
    .kpi-sub, .meta, .muted { font-size: 0.875rem; color: var(--muted); margin: 0.25rem 0 0.6rem; }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
    th, td { text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid #e2e8f0; }
    th { background: #f1f5f9; font-weight: 600; color: #1e293b; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.86em; }
    .chart-svg { width: 100%; height: auto; display: block; margin-top: 0.35rem; }
    .callout {
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-border);
      padding: 0.75rem 1rem;
      border-radius: 4px;
      font-size: 0.9rem;
      margin-top: 1rem;
    }
    footer {
      border-top: 1px solid var(--border);
      padding: 1.25rem 1.5rem;
      font-size: 0.85rem;
      color: var(--muted);
    }
    a { color: var(--accent); }"""

    html_doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "  <title>gridcast — Live Forecast Scoreboard</title>\n"
        f"  <style>\n{css_block}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <header>\n"
        '    <div class="wrap">\n'
        "      <h1>gridcast &mdash; GB Grid Carbon Intensity Scoreboard</h1>\n"
        "      <p>48-hour half-hourly carbon intensity forecasting (gCO2/kWh) &middot; "
        "Anchor 00:00 Europe/London &middot; Static Pages report generated deterministically "
        "by <code>gridcast score</code></p>\n"
        "    </div>\n"
        "  </header>\n"
        '  <main class="wrap">\n'
        '    <div class="kpis">\n'
        '      <div class="kpi">\n'
        '        <div class="kpi-label">Live Record Streak</div>\n'
        f'        <div class="kpi-value">{consecutive_days} days</div>\n'
        f'        <div class="kpi-sub">Started {html.escape(first_date)} &middot; '
        f"{total_scored} settled rows</div>\n"
        "      </div>\n"
        '      <div class="kpi">\n'
        '        <div class="kpi-label">Live Persistence MAE</div>\n'
        f'        <div class="kpi-value">{pers_kpi_str}</div>\n'
        f'        <div class="kpi-sub">{html.escape(pers_kpi_sub)}</div>\n'
        "      </div>\n"
        '      <div class="kpi">\n'
        '        <div class="kpi-label">Frozen Gate (lgbm-v2)</div>\n'
        f'        <div class="kpi-value">{frozen_kpi_str}</div>\n'
        f'        <div class="kpi-sub">{frozen_kpi_sub}</div>\n'
        "      </div>\n"
        '      <div class="kpi">\n'
        '        <div class="kpi-label">Latest Issued Anchor</div>\n'
        f'        <div class="kpi-value">{html.escape(latest_issue_str)}</div>\n'
        f'        <div class="kpi-sub">Models: {html.escape(latest_models_str)}</div>\n'
        "      </div>\n"
        "    </div>\n\n"
        '    <section id="live-scoreboard">\n'
        "      <h2>Live Settled Scoreboard (<code>scores.csv</code>)</h2>\n"
        f'      <p class="muted">Target dates covered: '
        f"<strong>{html.escape(live_date_range)}</strong>. All settled rows are reported per "
        "model version (including periods without an issue-time NESO snapshot).</p>\n"
        '      <div class="grid-2">\n'
        f"        {live_cards_html}\n"
        "      </div>\n"
        '      <div class="grid-2">\n'
        '        <div class="card">\n'
        "          <h3>Daily Settled MAE Trend</h3>\n"
        f"          {live_daily_svg}\n"
        "        </div>\n"
        '        <div class="card">\n'
        "          <h3>Live MAE by Horizon Bucket</h3>\n"
        f"          {live_bucket_svg}\n"
        "        </div>\n"
        "      </div>\n"
        '      <div class="callout">\n'
        "        <strong>Operator Baseline Vintage Separation (ADR-022):</strong>\n"
        "        <code>snapshot_at_issue</code> (verified issue-time vintage, eligible for skill): "
        f"{snap_summary_html}. &middot;\n"
        "        <code>archive_at_scoring</code> (<em>Published NESO forecast for the same "
        f"periods, vintage unverified</em>): {arch_summary_html}.\n"
        "        Refusing to aggregate across sources or model versions.\n"
        "      </div>\n"
        "    </section>\n\n"
        f"    {frozen_html}\n\n"
        '    <section id="latest-forecast">\n'
        "      <h2>Latest Issued 48h Forecast Curve (<code>forecasts.parquet</code>)</h2>\n"
        '      <div class="card">\n'
        f'        <p class="meta">Latest issue anchor: '
        f"<strong>{html.escape(latest_issue_str)}</strong> &middot; Models: "
        f"<code>{html.escape(latest_models_str)}</code></p>\n"
        f"        {latest_curve_svg}\n"
        f"        {latest_curve_table_html}\n"
        "      </div>\n"
        "    </section>\n"
        "  </main>\n"
        '  <footer class="wrap">\n'
        "    Generated deterministically by <code>gridcast score</code>. "
        'Raw markdown record: <a href="scoreboard.md">docs/scoreboard.md</a> &middot; '
        'Source &amp; governance: <a href="https://github.com/Dash3210/gridcast">'
        "Dash3210/gridcast</a>.\n"
        "  </footer>\n"
        "</body>\n"
        "</html>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return html_doc


def score_forecasts(
    curated_dir: Path = Path("data/curated"),
    forecasts_path: Path = DEFAULT_FORECASTS_PATH,
    scores_path: Path = DEFAULT_SCORES_PATH,
    scoreboard_md_path: Path = DEFAULT_SCOREBOARD_MD_PATH,
    scoreboard_html_path: Path | None = None,
    eval_fixture_path: Path = DEFAULT_EVAL_FIXTURE_PATH,
    mix_dir: Path = DEFAULT_MIX_DIR,
    interval_path: Path = DEFAULT_INTERVAL_PATH,
) -> ScoreSummary:
    """Find issued forecasts whose target_time now has a settled actual, score them,

    and update data/scoreboard/scores.csv, docs/scoreboard.md, and docs/index.html.
    """
    resolved_html_path = (
        scoreboard_html_path
        if scoreboard_html_path is not None
        else (
            DEFAULT_SCOREBOARD_HTML_PATH
            if scoreboard_md_path == DEFAULT_SCOREBOARD_MD_PATH
            else scoreboard_md_path.with_name("index.html")
        )
    )

    if scores_path.exists() and scores_path.stat().st_size > 0:
        existing_scores = pd.read_csv(scores_path)
        if "neso_source" not in existing_scores.columns:
            existing_scores["neso_source"] = "archive_at_scoring"
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
        generate_scoreboard_html(
            existing_scores,
            output_path=resolved_html_path,
            forecasts_path=forecasts_path,
            eval_fixture_path=eval_fixture_path,
            mix_dir=mix_dir,
            interval_path=interval_path,
            start_date=start_date,
        )
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
        actual_val, _ = get_target_observation(tt)
        if actual_val is None:
            continue

        f_val = float(row["value"])
        abs_err = abs(f_val - actual_val)

        # Snapshot at issue instant
        neso_snap = row.get("neso_forecast_at_issue") if "neso_forecast_at_issue" in row else None
        if neso_snap is not None and pd.notna(neso_snap):
            snap_val = float(neso_snap)
            neso_forecast: float | str = snap_val
            neso_abs_err: float | str = abs(snap_val - actual_val)
            neso_source = "snapshot_at_issue"
        else:
            neso_forecast = ""
            neso_abs_err = ""
            neso_source = "none"

        new_rows.append(
            {
                "issue_time": it.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "horizon": h,
                "target_time": tt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "model_version": mv,
                "forecast": f_val,
                "actual": actual_val,
                "neso_forecast": neso_forecast,
                "abs_error": abs_err,
                "neso_abs_error": neso_abs_err,
                "neso_source": neso_source,
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
    generate_scoreboard_html(
        all_scores,
        output_path=resolved_html_path,
        forecasts_path=forecasts_path,
        eval_fixture_path=eval_fixture_path,
        mix_dir=mix_dir,
        interval_path=interval_path,
        start_date=start_date,
    )

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
    mix_dir: Path | None = None,
    model_path: Path | None = None,
    interval_path: Path | None = None,
) -> int:
    """CLI handler for gridcast forecast (issues persistence-v0 and lgbm-v2)."""
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
    except Exception as exc:
        print(f"gridcast forecast error: {exc}", file=sys.stderr)
        return 1

    resolved_mix_dir = DEFAULT_MIX_DIR if mix_dir is None else mix_dir
    resolved_model_path = Path(DEFAULT_MODEL_PATH if model_path is None else model_path)
    resolved_interval_path = DEFAULT_INTERVAL_PATH if interval_path is None else interval_path

    if not resolved_model_path.exists():
        print(
            f"gridcast forecast note ({LGBM_MODEL_VERSION} skipped: "
            f"model artifact missing at {resolved_model_path}).",
            file=sys.stderr,
        )
        return 0

    try:
        lgbm_df, lgbm_new_count = issue_lgbm_forecast(
            issue_time=issue_time_str,
            date_val=date_str,
            curated_dir=curated_dir,
            mix_dir=resolved_mix_dir,
            model_path=resolved_model_path,
            interval_path=resolved_interval_path,
            output_path=output_path,
        )
        mean_val = float(lgbm_df["value"].mean())
        print(
            f"Issued {len(lgbm_df)} forecast points "
            f"(model_version: {LGBM_MODEL_VERSION}, mean: {mean_val:.1f})."
        )
        print(f"  First target: {lgbm_df['target_time'].min()}")
        print(f"  Last target:  {lgbm_df['target_time'].max()}")
        print(f"  New points written: {lgbm_new_count}")
        print(f"  Store: {output_path}")
    except Exception as exc:
        print(f"gridcast forecast error ({LGBM_MODEL_VERSION}): {exc}", file=sys.stderr)
        return 1

    return 0


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
