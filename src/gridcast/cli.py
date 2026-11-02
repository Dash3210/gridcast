"""Command-line interface for gridcast."""

import argparse
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from gridcast import __version__
from gridcast._phase import todo
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

    subparsers.add_parser("train", help="Train carbon intensity forecast models.")
    subparsers.add_parser("backtest", help="Run backtest against NESO baseline.")
    subparsers.add_parser("serve", help="Run forecast API service.")
    subparsers.add_parser("version", help="Print package version.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for gridcast."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.subcommand == "version":
        print(f"gridcast {__version__}")
        return 0

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

    if args.subcommand == "train":
        todo("P3", "train")
    elif args.subcommand == "backtest":
        todo("P3", "backtest")
    elif args.subcommand == "serve":
        todo("P4", "serve")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
