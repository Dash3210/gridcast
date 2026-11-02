"""Command-line interface for gridcast."""

import argparse
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from gridcast import __version__
from gridcast._phase import todo
from gridcast.features import (
    build_example_frame,
    default_issue_schedule,
    load_feature_sources,
    select_training_arrays,
)
from gridcast.ingest import (
    ingest_mix_range,
    ingest_range,
    ingest_weather_range,
    prune_cache,
    reconcile_recent,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="Forecast Great Britain's half-hourly grid carbon intensity.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest half-hourly carbon intensity data.",
    )
    ingest_parser.add_argument(
        "--weather",
        action="store_true",
        default=False,
        help="Ingest weather forecast data (Open-Meteo previous-runs).",
    )
    ingest_parser.add_argument(
        "--mix",
        action="store_true",
        default=False,
        help="Ingest generation mix data (NESO).",
    )
    ingest_parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="The last N local days, inclusive of today.",
    )
    ingest_parser.add_argument(
        "--recent",
        type=int,
        default=None,
        help="Alias for --days, for the scheduled job.",
    )
    ingest_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start local date (YYYY-MM-DD), inclusive.",
    )
    ingest_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End local date (YYYY-MM-DD), inclusive.",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-fetch dates that look complete.",
    )
    ingest_parser.add_argument(
        "--prune-cache",
        action="store_true",
        default=False,
        help="Drop raw cache older than 90 days, then exit.",
    )
    ingest_parser.add_argument(
        "--reconcile-recent",
        action="store_true",
        default=False,
        help="Reconcile recent dates with pending null actuals or recorded gaps.",
    )
    ingest_parser.add_argument(
        "--window-days",
        type=int,
        default=3,
        help="Window size in days for --reconcile-recent (default: 3).",
    )

    features_parser = subparsers.add_parser(
        "features",
        help="Build and inspect the example feature frame.",
    )
    features_parser.add_argument(
        "--no-require-aux",
        dest="require_aux",
        action="store_false",
        default=True,
        help="Do not require auxiliary stores (weather and mix) to be present.",
    )

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Issue day-ahead half-hourly carbon intensity forecast curve.",
    )
    forecast_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Local date for issue time (YYYY-MM-DD, 00:00 Europe/London).",
    )
    forecast_parser.add_argument(
        "--issue-time",
        type=str,
        default=None,
        help="Specific UTC or ISO issue timestamp.",
    )
    forecast_parser.add_argument(
        "--curated-dir",
        type=str,
        default="data/curated",
        help="Path to curated archive directory.",
    )
    forecast_parser.add_argument(
        "--output",
        type=str,
        default="data/scoreboard/forecasts.parquet",
        help="Path to forecasts parquet store.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score settled forecasts against actuals and update scoreboard.",
    )
    score_parser.add_argument(
        "--curated-dir",
        type=str,
        default="data/curated",
        help="Path to curated archive directory.",
    )
    score_parser.add_argument(
        "--forecasts-path",
        type=str,
        default="data/scoreboard/forecasts.parquet",
        help="Path to forecasts parquet store.",
    )
    score_parser.add_argument(
        "--scores-path",
        type=str,
        default="data/scoreboard/scores.csv",
        help="Path to scores CSV store.",
    )
    score_parser.add_argument(
        "--scoreboard-md-path",
        type=str,
        default="docs/scoreboard.md",
        help="Path to scoreboard markdown documentation.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train carbon intensity forecast models.",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="data/models/lgbm_model.txt",
        help="Path to save trained model artifact.",
    )
    train_parser.add_argument(
        "--num-boost-round",
        type=int,
        default=100,
        help="Number of boosting iterations (default: 100).",
    )

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run backtest against NESO baseline.",
    )
    backtest_parser.add_argument(
        "--origins",
        nargs="*",
        default=None,
        help="Explicit list of origin dates or timestamps (default: fixed test split).",
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for test split origins derivation (default: 2026-09-05).",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for test split origins derivation (default: 2026-11-01).",
    )
    backtest_parser.add_argument(
        "--threshold",
        type=float,
        default=35.0,
        help="Gate threshold for GATE_VALUE in gCO2/kWh (default: 35.0).",
    )
    backtest_parser.add_argument(
        "--num-boost-round",
        type=int,
        default=100,
        help="Number of boosting iterations per fold (default: 100).",
    )
    backtest_parser.add_argument(
        "--no-fail-on-gate",
        dest="fail_on_gate",
        action="store_false",
        default=False,
        help="Do not return non-zero exit code if gate fails.",
    )
    subparsers.add_parser("serve", help="Run forecast API service.")
    subparsers.add_parser("version", help="Print package version.")

    return parser


def run_train_cli(
    output_path: Path | str = "data/models/lgbm_model.txt",
    num_boost_round: int = 100,
) -> int:
    """Train primary LightGBM model on curated training dataset and save artifact."""
    from pathlib import Path

    from gridcast.models import model_artifact_hash, train_lgbm

    print("Loading feature sources (intensity, weather, mix)...")
    intensity_df, weather_df, mix_df = load_feature_sources()
    latest_date = intensity_df["period_start_utc"].max().tz_convert("Europe/London").date()
    if not weather_df.empty:
        max_weather_date = weather_df["period_start_utc"].max().tz_convert("Europe/London").date()
        if max_weather_date < latest_date:
            latest_date = max_weather_date

    schedule = default_issue_schedule("2024-11-08", latest_date.isoformat())
    print(
        f"Building example frame across {len(schedule)} issue dates "
        f"({schedule[0]} to {schedule[-1]})..."
    )
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )
    print(f"Selecting non-null training arrays from {len(frame)} rows...")
    X_train, y_train = select_training_arrays(frame)
    print(f"Training LightGBM model on {len(X_train)} samples with {X_train.shape[1]} features...")
    booster = train_lgbm(X_train, y_train, num_boost_round=num_boost_round)
    model_hash = model_artifact_hash(booster)
    print(f"Model trained successfully. Artifact SHA-256: {model_hash}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    print(f"Saved model artifact to {out}")
    return 0


def run_backtest_cli(
    origins: Sequence[pd.Timestamp | str] | None = None,
    threshold: float | None = 35.0,
    start_date: str | None = None,
    end_date: str | None = None,
    num_boost_round: int = 100,
    verbose: bool = True,
    fail_on_gate: bool = False,
) -> int:
    """Run rolling-origin backtest and evaluate accuracy gate."""
    import pandas as pd

    from gridcast.backtest import (
        TEST_SPLIT_END_DATE,
        TEST_SPLIT_START_DATE,
        derive_test_split_origins,
        evaluate_gate,
        run_rolling_origin_backtest,
    )

    resolved_start = start_date or TEST_SPLIT_START_DATE
    resolved_end = end_date or TEST_SPLIT_END_DATE

    if origins is None:
        origins = derive_test_split_origins(resolved_start, resolved_end)

    if verbose:
        print("Loading feature sources for backtest...")
    intensity_df, weather_df, mix_df = load_feature_sources()

    frame_start = "2024-11-08"
    schedule = default_issue_schedule(frame_start, resolved_end)
    if verbose:
        print(
            f"Calling build_example_frame for {len(schedule)} issues "
            f"({frame_start} to {resolved_end})..."
        )
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )

    operator_forecasts = pd.Series(
        intensity_df["forecast"].values,
        index=pd.to_datetime(intensity_df["period_start_utc"], utc=True),
    )

    results = run_rolling_origin_backtest(
        frame=frame,
        origins=origins,
        operator_forecasts=operator_forecasts,
        num_boost_round=num_boost_round,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 94)
        print("PER-HORIZON PERFORMANCE (h=1..96)")
        print("=" * 94)
        print(
            f"{'Horizon':<8}{'Periods':<10}{'Model MAE':<14}{'Operator MAE':<16}"
            f"{'Persistence':<14}{'Seasonal-Naive':<16}{'Skill vs NESO':<14}"
        )
        print("-" * 94)
        for _, r in results.per_horizon_table.iterrows():
            h_int = int(r["horizon"])
            cnt = int(r["count"])
            m_mae = f"{r['model_mae']:.2f}" if pd.notna(r["model_mae"]) else "N/A"
            o_mae = f"{r['operator_mae']:.2f}" if pd.notna(r["operator_mae"]) else "N/A"
            p_mae = f"{r['persistence_mae']:.2f}" if pd.notna(r["persistence_mae"]) else "N/A"
            s_mae = f"{r['seasonal_naive_mae']:.2f}" if pd.notna(r["seasonal_naive_mae"]) else "N/A"
            sk = f"{r['skill_vs_operator']:+.2%}" if pd.notna(r["skill_vs_operator"]) else "N/A"
            print(f"{h_int:<8}{cnt:<10}{m_mae:<14}{o_mae:<16}{p_mae:<14}{s_mae:<16}{sk:<14}")
        print("=" * 94)

    evaluate_gate(results, threshold=threshold, verbose=verbose)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for gridcast."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.subcommand == "version":
        print(f"gridcast {__version__}")
        return 0

    if args.subcommand == "train":
        from pathlib import Path

        return run_train_cli(
            output_path=Path(args.output),
            num_boost_round=args.num_boost_round,
        )

    if args.subcommand == "backtest":
        return run_backtest_cli(
            origins=getattr(args, "origins", None),
            threshold=args.threshold,
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            num_boost_round=args.num_boost_round,
            fail_on_gate=args.fail_on_gate,
        )

    if args.subcommand == "ingest":
        if args.prune_cache:
            pruned = prune_cache(keep_days=90)
            print(f"Pruned {len(pruned)} cache file(s) older than 90 days.")
            return 0

        today = datetime.now(ZoneInfo("Europe/London")).date()

        if (
            args.reconcile_recent
            and args.days is None
            and args.recent is None
            and args.start is None
            and args.end is None
        ):
            reconcile_recent(window_days=args.window_days, today=today)
            return 0

        if args.start is not None or args.end is not None:
            if args.start is None or args.end is None:
                print(
                    "gridcast: both --start and --end must be specified together.",
                    file=sys.stderr,
                )
                return 2
            try:
                start_date = date.fromisoformat(args.start)
                end_date = date.fromisoformat(args.end)
            except ValueError as exc:
                print(f"gridcast: invalid date format: {exc}", file=sys.stderr)
                return 2

            if start_date > end_date:
                print(
                    f"gridcast: start date ({start_date}) cannot be after end date ({end_date}).",
                    file=sys.stderr,
                )
                return 2
        else:
            days = (
                args.recent
                if args.recent is not None
                else (args.days if args.days is not None else 2)
            )
            if days < 1:
                print("gridcast: --days/--recent must be at least 1.", file=sys.stderr)
                return 2
            start_date = today - timedelta(days=days - 1)
            end_date = today

        if args.weather:
            summary = ingest_weather_range(start=start_date, end=end_date, force=args.force)
            return 1 if summary.failures > 0 else 0

        if args.mix:
            summary = ingest_mix_range(start=start_date, end=end_date, force=args.force)
            return 1 if summary.failures > 0 else 0

        summary = ingest_range(start=start_date, end=end_date, force=args.force)

        if args.reconcile_recent:
            reconcile_recent(window_days=args.window_days, today=today)

        return 1 if summary.failures > 0 else 0

    if args.subcommand == "features":
        import pandas as pd

        from gridcast.features import (
            FEATURE_COLUMNS,
            build_example_frame,
            load_feature_sources,
        )

        print("Loading feature sources (curated intensity, weather, mix)...")
        intensity_df, weather_df, mix_df = load_feature_sources()
        print("Building example frame keyed by (issue_time, horizon)...")
        df = build_example_frame(
            intensity_df=intensity_df,
            weather_df=weather_df,
            mix_df=mix_df,
            require_aux=args.require_aux,
        )
        n_issues = df["issue_time"].nunique()
        n_horizons = df["horizon"].nunique()
        print("\nFeature frame constructed successfully:")
        print(f"  Shape:             {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Distinct issues:   {n_issues:,} (daily at 00:00 Europe/London)")
        total_schedule_dates = (
            pd.to_datetime(intensity_df["period_start_utc"], utc=True)
            .dt.tz_convert(ZoneInfo("Europe/London"))
            .dt.date.nunique()
        )
        dropped_count = total_schedule_dates - n_issues
        if dropped_count > 0:
            first_issue = df["issue_time"].dt.tz_convert(ZoneInfo("Europe/London")).dt.date.min()
            print(
                f"  Leading dates dropped: {dropped_count} "
                f"(warmup for 168h actual lag and 48h mix join; frame begins at {first_issue})"
            )
        print(f"  Distinct horizons: {n_horizons} (range 1..96 half-hours ahead)")
        print(f"  Feature count (X): {len(FEATURE_COLUMNS)} columns")
        print(f"  Target column (y): 'y' ({df['y'].notna().sum():,} non-null)")
        print("\nNull counts per column:")
        null_counts = df.isna().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col:<26}: {count:>6} nulls ({count / len(df) * 100:.2f}%)")
            else:
                print(f"  {col:<26}:      0 nulls")
        print("\nTo extract feature matrix X and target y:")
        print("  X = select_model_inputs(df)")
        print("  y = df['y']")
        return 0

    if args.subcommand == "forecast":
        from pathlib import Path

        from gridcast.scoreboard import run_forecast_cli

        return run_forecast_cli(
            date_str=args.date,
            issue_time_str=args.issue_time,
            curated_dir=Path(args.curated_dir),
            output_path=Path(args.output),
        )

    if args.subcommand == "score":
        from pathlib import Path

        from gridcast.scoreboard import run_score_cli

        return run_score_cli(
            curated_dir=Path(args.curated_dir),
            forecasts_path=Path(args.forecasts_path),
            scores_path=Path(args.scores_path),
            scoreboard_md_path=Path(args.scoreboard_md_path),
        )

    if args.subcommand == "serve":
        todo("P4", "serve")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
