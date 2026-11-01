"""Command-line interface for gridcast."""

import argparse
import sys

from gridcast import __version__
from gridcast._phase import todo


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="Forecast Great Britain's half-hourly grid carbon intensity.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("ingest", help="Ingest half-hourly carbon intensity data.")
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
        todo("P1", "ingest")
    elif args.subcommand == "train":
        todo("P3", "train")
    elif args.subcommand == "backtest":
        todo("P3", "backtest")
    elif args.subcommand == "serve":
        todo("P4", "serve")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
