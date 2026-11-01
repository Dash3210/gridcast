"""Command-line interface for gridcast."""

import argparse
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from gridcast import __version__
from gridcast._phase import todo
from gridcast.ingest import ingest_range, prune_cache


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

        summary = ingest_range(start=start_date, end=end_date, force=args.force)
        return 1 if summary.failures > 0 else 0

    if args.subcommand == "train":
        todo("P3", "train")
    elif args.subcommand == "backtest":
        todo("P3", "backtest")
    elif args.subcommand == "serve":
        todo("P4", "serve")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
