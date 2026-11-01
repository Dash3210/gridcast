# Decisions

## ADR-001 — Storage: append-only Parquet partitions
- Status: accepted
- Date: 2026-11-01
- Context: A nightly GitHub Action must append half-hourly carbon intensity rows and commit them to a public repository so reviewers can inspect the data in git diffs.
- Decision: Store data as append-only Parquet partitions, one file per day: `data/raw/date=YYYY-MM-DD.parquet`, deduplicated on the period start timestamp in UTC. No DuckDB, no SQLite, no database file committed to git.
- Consequences: Each daily append creates an isolated partition file easily versioned and inspected; avoids binary database corruption or unreviewable blob diffs.
- Alternatives rejected: DuckDB database file (rejected: a binary file the nightly Action rewrites and commits whole, which no reviewer can diff), SQLite (same), Parquet plus a DuckDB query layer (an extra dependency and a second system to explain, for no benefit at ~35k rows).

## ADR-002 — Makefile: POSIX-first common subset
- Status: accepted
- Date: 2026-11-01
- Context: Local development runs on Windows 11 where Make recipes execute under cmd.exe, while CI executes under /bin/sh on ubuntu-latest. The build system must operate reliably across both without platform divergence.
- Decision: A single POSIX-first Makefile with an `ifeq ($(OS),Windows_NT)` switch for the interpreter path, and recipe bodies strictly restricted to the cmd/sh common syntax subset. Conditional or destructive logic is handled exclusively in Python.
- Consequences: Commands run identically on local Windows workstations and Linux CI runners without PowerShell recipes, wrapper scripts, Docker overhead, or task-runner indirection.
- Alternatives rejected: PowerShell-based build scripts (non-portable to Linux CI), Docker-wrapping every target (slow local development feedback loop, Docker deferred to P5), Python task runners like Invoke/Taskfile (extra indirection layer when Make is standard).

### Amendment: Quoting and forward-slash path convention
- Date: 2026-11-01
- Context: On Windows workstations, Make may invoke recipes under `cmd.exe` or Git Bash / MSYS2 `sh.exe`, while paths may contain spaces (e.g. `C:\Program Files\Python310\python.exe`). Testing verified that:
  - `cmd /c ".venv/Scripts/python.exe" -V` -> `Python 3.10.11` (quoted forward-slash path succeeds in cmd.exe)
  - `cmd /c .venv/Scripts/python.exe -V` -> `'.venv' is not recognized` (unquoted forward slash fails in cmd.exe)
  - `sh -c 'echo .venv\Scripts\python.exe'` -> `.venvScriptspython.exe` (POSIX shells strip unquoted backslashes)
- Decision: Use forward slashes in interpreter paths (`PY ?= .venv/Scripts/python.exe`) and double-quote all expansions of `$(PY)` and `$(PYTHON)` across all recipes (e.g., `"$(PY)" -m pytest`). Shell detection via `$(SHELL)` is avoided because `$(SHELL)` reports `sh.exe` even when Make is actually driving `cmd.exe`.
