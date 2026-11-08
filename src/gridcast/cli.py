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

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run model evaluation benchmark from frozen fixture.",
    )
    eval_parser.add_argument(
        "--fixture",
        type=str,
        default="data/eval/frozen_split_v2.parquet",
        help="Path to frozen split parquet fixture (default: data/eval/frozen_split_v2.parquet).",
    )
    eval_parser.add_argument(
        "--sidecar",
        type=str,
        default="data/eval/frozen_split_v2.json",
        help="Path to fixture sidecar metadata JSON (default: data/eval/frozen_split_v2.json).",
    )
    eval_parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        default=True,
        help="Disable detailed table output.",
    )

    eval_refresh_parser = subparsers.add_parser(
        "eval-refresh",
        help="Rebuild frame and refresh evaluation fixture (slow, manual).",
    )
    eval_refresh_parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="Fixture version identifier (default: v2).",
    )
    eval_refresh_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval",
        help="Output directory for fixture files (default: data/eval).",
    )
    eval_refresh_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for test split (default: 2026-09-05).",
    )
    eval_refresh_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for test split (default: 2026-11-01).",
    )
    eval_refresh_parser.add_argument(
        "--num-boost-round",
        type=int,
        default=100,
        help="Number of boosting iterations per fold (default: 100).",
    )
    eval_refresh_parser.add_argument(
        "--reconcile-window-days",
        type=int,
        default=3,
        help="Required age buffer in days behind newest archive period (default: 3).",
    )
    eval_refresh_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing fixture.",
    )

    eval_reproduce_parser = subparsers.add_parser(
        "eval-reproduce",
        help="Retrain 15 folds and verify candidate predictions against fixture (slow, manual).",
    )
    eval_reproduce_parser.add_argument(
        "--fixture",
        type=str,
        default="data/eval/frozen_split_v2.parquet",
        help="Path to frozen split parquet fixture (default: data/eval/frozen_split_v2.parquet).",
    )
    eval_reproduce_parser.add_argument(
        "--sidecar",
        type=str,
        default="data/eval/frozen_split_v2.json",
        help="Path to fixture sidecar metadata JSON (default: data/eval/frozen_split_v2.json).",
    )
    eval_reproduce_parser.add_argument(
        "--max-pred-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed per-row absolute difference for predictions (default: 0.0).",
    )
    eval_reproduce_parser.add_argument(
        "--gate-value-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed absolute difference for GATE_VALUE (default: 0.0).",
    )
    eval_reproduce_parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        default=True,
        help="Disable detailed progress and table output.",
    )
    eval_reproduce_parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Suppress progress and table output.",
    )

    ablation_weather_parser = subparsers.add_parser(
        "ablation-weather",
        help="Run pre-registered weather removal ablation against frozen fixture.",
    )
    ablation_weather_parser.add_argument(
        "--fixture",
        type=str,
        default="data/eval/frozen_split_v1.parquet",
        help="Path to frozen split parquet fixture (default: data/eval/frozen_split_v1.parquet).",
    )
    ablation_weather_parser.add_argument(
        "--sidecar",
        type=str,
        default="data/eval/frozen_split_v1.json",
        help="Path to fixture sidecar metadata JSON (default: data/eval/frozen_split_v1.json).",
    )
    ablation_weather_parser.add_argument(
        "--output-doc",
        type=str,
        default="docs/ablations/weather_removal.md",
        help="Path to markdown document to update generated table.",
    )
    ablation_weather_parser.add_argument(
        "--max-control-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed absolute difference for control arm reproduction (default: 0.0).",
    )
    ablation_weather_parser.add_argument(
        "--threshold-percent",
        type=float,
        default=2.0,
        help="Decision threshold percentage for KEEP vs DROP (default: 2.0).",
    )
    ablation_weather_parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        default=True,
        help="Disable detailed progress and table output.",
    )
    ablation_weather_parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Suppress progress and table output.",
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
    if weather_df is not None and not weather_df.empty:
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


def run_eval_cli(
    fixture_path: Path | str = "data/eval/frozen_split_v2.parquet",
    sidecar_path: Path | str = "data/eval/frozen_split_v2.json",
    verbose: bool = True,
    pinned_sha256: str | None = None,
) -> int:
    """Run model evaluation benchmark reading only the frozen fixture."""
    from gridcast.backtest import PINNED_EVAL_FIXTURE_SHA256, evaluate_from_fixture

    resolved_pin = PINNED_EVAL_FIXTURE_SHA256 if pinned_sha256 is None else pinned_sha256

    try:
        evaluate_from_fixture(
            parquet_path=fixture_path,
            sidecar_path=sidecar_path,
            verbose=verbose,
            pinned_sha256=resolved_pin,
        )
        return 0
    except ValueError as e:
        print(f"Evaluation refused: {e}", file=sys.stderr)
        return 1


def run_eval_reproduce_cli(
    fixture_path: Path | str = "data/eval/frozen_split_v2.parquet",
    sidecar_path: Path | str = "data/eval/frozen_split_v2.json",
    max_pred_tolerance: float = 0.0,
    gate_value_tolerance: float = 0.0,
    verbose: bool = True,
) -> int:
    """Retrain 15 folds on frozen origins and verify candidate predictions against fixture."""
    from gridcast.backtest import reproduce_eval_fixture

    try:
        reproduce_eval_fixture(
            parquet_path=fixture_path,
            sidecar_path=sidecar_path,
            max_pred_tolerance=max_pred_tolerance,
            gate_value_tolerance=gate_value_tolerance,
            verbose=verbose,
        )
        return 0
    except Exception as e:
        print(f"Reproduction check failed: {e}", file=sys.stderr)
        return 1



def run_eval_refresh_cli(
    version: str = "v1",
    output_dir: Path | str = "data/eval",
    start_date: str | None = None,
    end_date: str | None = None,
    num_boost_round: int = 100,
    reconcile_window_days: int = 3,
    verbose: bool = True,
    force: bool = False,
) -> int:
    """Rebuild evaluation frame and refresh frozen fixture."""
    from gridcast.backtest import (
        TEST_SPLIT_END_DATE,
        TEST_SPLIT_START_DATE,
        refresh_eval_fixture,
    )

    resolved_start = start_date or TEST_SPLIT_START_DATE
    resolved_end = end_date or TEST_SPLIT_END_DATE

    refresh_eval_fixture(
        version=version,
        output_dir=output_dir,
        start_date=resolved_start,
        end_date=resolved_end,
        num_boost_round=num_boost_round,
        reconcile_window_days=reconcile_window_days,
        verbose=verbose,
        force=force,
    )
    return 0


def run_ablation_weather_cli(
    fixture_path: Path | str = "data/eval/frozen_split_v1.parquet",
    sidecar_path: Path | str = "data/eval/frozen_split_v1.json",
    output_doc: Path | str | None = "docs/ablations/weather_removal.md",
    max_control_tolerance: float = 0.0,
    threshold_percent: float = 2.0,
    verbose: bool = True,
) -> int:
    """Run pre-registered weather removal ablation benchmark against frozen fixture."""
    from gridcast.backtest import run_weather_ablation

    try:
        run_weather_ablation(
            parquet_path=fixture_path,
            sidecar_path=sidecar_path,
            output_doc=output_doc,
            max_control_tolerance=max_control_tolerance,
            decision_threshold_percent=threshold_percent,
            verbose=verbose,
        )
        return 0
    except Exception as e:
        print(f"Weather removal ablation failed: {e}", file=sys.stderr)
        return 1


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

    if args.subcommand == "eval":
        return run_eval_cli(
            fixture_path=args.fixture,
            sidecar_path=args.sidecar,
            verbose=args.verbose,
        )

    if args.subcommand == "eval-reproduce":
        return run_eval_reproduce_cli(
            fixture_path=args.fixture,
            sidecar_path=args.sidecar,
            max_pred_tolerance=args.max_pred_tolerance,
            gate_value_tolerance=args.gate_value_tolerance,
            verbose=args.verbose,
        )

    if args.subcommand == "eval-refresh":
        return run_eval_refresh_cli(
            version=args.version,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            num_boost_round=args.num_boost_round,
            reconcile_window_days=args.reconcile_window_days,
            force=args.force,
        )

    if args.subcommand == "ablation-weather":
        return run_ablation_weather_cli(
            fixture_path=args.fixture,
            sidecar_path=args.sidecar,
            output_doc=args.output_doc,
            max_control_tolerance=args.max_control_tolerance,
            threshold_percent=args.threshold_percent,
            verbose=args.verbose,
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
