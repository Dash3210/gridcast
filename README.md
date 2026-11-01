# gridcast

**Target.** gridcast is a containerised forecasting service for Great Britain's half-hourly grid carbon intensity, forecasting 48 hours ahead and gated in CI on forecast accuracy. It aims to answer whether a solo open-source model can beat the National Energy System Operator's (NESO) official day-ahead forecast on a permanent, public scoreboard.

> Phase P0 is a skeleton: the ingest, model, API, container and CI stages do not exist yet, and every one of them says so when you invoke it. The status table below is the honest state.

> Note: Upon completion of Phase P7, this README will open with the live forecast-vs-actual scoreboard table comparing gridcast models against the NESO day-ahead baseline, positioned directly above the Quickstart.

## Status

| Phase | Description | Status |
|---|---|---|
| P0 | Setup: repository skeleton, environment, CLI gating, and smoke tests | done |
| P1 | Ingest: half-hourly API fetch to append-only Parquet daily partitions | next |
| P2 | Features: feature extraction and temporal leakage test suite | not started |
| P3 | Models: baseline models, backtesting engine, and accuracy gate vs NESO | not started |
| P4 | Service: FastAPI prediction API and Prometheus metrics endpoint | not started |
| P5 | Packaging: Docker containerisation and GitHub Actions CI workflow | not started |
| P6 | Verification: deliberately failing pull request demonstrating accuracy gate | not started |
| P7 | Deployment: automated schedule, documentation, and scoreboard publication | not started |

## Quickstart

### Prerequisites

- Python 3.10+
- GNU Make
  - Windows: `winget install ezwinports.make`
  - Debian / Ubuntu: `sudo apt-get install make python3-venv`

### Clone and setup

```bash
git clone https://github.com/Dash3210/gridcast.git
cd gridcast
make install
```

If your Python 3.10+ interpreter is not the default `python` on PATH, pass `PYTHON` to override it:

```bash
make install PYTHON=/path/to/python3.10
```

### Verification targets

Run the test suite, linter, and type checker:

```bash
make test
make lint
make typecheck
```

`make test` runs pytest with coverage reporting and strictly enforces the 85% code coverage gate.

### Running the CLI

Run subcommands using the virtual environment interpreter without needing manual shell activation:

- Windows:
  ```cmd
  .venv\Scripts\python.exe -m gridcast version
  ```
- Linux / macOS:
  ```bash
  .venv/bin/python -m gridcast version
  ```

Alternatively, if the virtual environment is activated in your shell:

```bash
python -m gridcast version
```

## What works today

Today (Phase P0), the working automation targets are:
- `make install` — creates the virtual environment and installs dependencies in editable mode
- `make test` — runs the test suite with coverage enforcement (>= 85%)
- `make lint` — checks codebase formatting and linting rules with ruff
- `make typecheck` — validates static typing with mypy across `src` and `tests`
- `make clean` — removes virtual environment, caches, and build artefacts

Everything else fails on purpose:
- `make ingest`, `make train`, `make eval`, `make serve`, and `make docker` exit non-zero with explicit messages indicating the respective future phase in which they arrive.
- Note: `make eval` maps to the `gridcast backtest` CLI subcommand (scheduled for P3).

## Layout

```
gridcast/
|-- Makefile              # Automation recipes restricted to cmd/sh common subset
|-- pyproject.toml        # PEP 621 package metadata and tool configurations
|-- README.md             # Project overview, status tracking, and module guide
|-- docs/
|   `-- decisions.md      # Architecture Decision Records (ADRs)
|-- src/
|   `-- gridcast/
|       |-- py.typed      # PEP 561 marker for inline type annotations
|       |-- __init__.py   # Package version resolution from metadata
|       |-- _phase.py     # Loud exit helper for unimplemented phase milestones
|       |-- cli.py        # CLI entry point routing subcommands to modules or phase gates
|       |-- ingest.py     # Half-hourly API data ingest to Parquet partitions (P1)
|       |-- features.py   # Feature engineering and temporal leakage prevention tests (P2)
|       |-- models.py     # Machine learning forecast models and baseline comparison (P3)
|       |-- backtest.py   # Backtesting engine and historical evaluation (P3)
|       `-- serve.py      # FastAPI service and prediction endpoints (P4)
|-- tests/
|   `-- test_smoke.py     # Smoke tests verifying package metadata, CLI routing, and gating
`-- data/
    `-- raw/              # Raw data storage for append-only daily Parquet partitions
```
