# PYTHON is the interpreter used once, to create the venv. Override it if your
# python 3.10+ is not the one on PATH:  make install PYTHON=/path/to/python3.10
ifeq ($(OS),Windows_NT)
PYTHON ?= python
PY ?= .venv/Scripts/python.exe
else
PYTHON ?= python3
PY ?= .venv/bin/python
endif

.PHONY: help install test clean ingest train eval serve docker lint typecheck
.DEFAULT_GOAL := help

help:
	@echo gridcast targets:
	@echo   install    - Create virtual environment and install dependencies
	@echo   test       - Run test suite
	@echo   clean      - Remove virtual environment, caches, and build artefacts
	@echo   ingest     - Run ingest CLI (arrives in P1)
	@echo   train      - Run model training CLI (arrives in P3)
	@echo   eval       - Run model evaluation backtest (arrives in P3)
	@echo   serve      - Run service API (arrives in P4)
	@echo   docker     - Build Docker image (arrives in P5)
	@echo   lint       - Run ruff linter
	@echo   typecheck  - Run mypy type checker

install:
	"$(PYTHON)" -c "import os, subprocess, sys; os.path.exists('.venv') or subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)"
	"$(PY)" -m pip install --upgrade pip setuptools
	"$(PY)" -m pip install -e ".[dev]"

test:
	"$(PY)" -m pytest --cov --cov-report=term-missing

clean:
	"$(PYTHON)" -c "import glob, os, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist', *glob.glob('src/*.egg-info'), *glob.glob('*.egg-info')]]; [os.remove(f) for f in glob.glob('**/*.pyc', recursive=True)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True)]; os.remove('.coverage') if os.path.exists('.coverage') else None"

ingest:
	"$(PY)" -m gridcast.cli ingest

train:
	"$(PY)" -m gridcast.cli train

eval:
	"$(PY)" -m gridcast.cli backtest

serve:
	"$(PY)" -m gridcast.cli serve

docker:
	"$(PY)" -c "from gridcast._phase import todo; todo('P5', 'Dockerfile')"

lint:
	"$(PY)" -m ruff check src tests

typecheck:
	"$(PY)" -m mypy
