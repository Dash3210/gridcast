# PYTHON is the interpreter used once, to create the venv. Override it if your
# python 3.10+ is not the one on PATH:  make install PYTHON=/path/to/python3.10
ifeq ($(OS),Windows_NT)
PYTHON ?= python
PY ?= .venv/Scripts/python.exe
else
PYTHON ?= python3
PY ?= .venv/bin/python
endif

.PHONY: help install test data-check clean ingest features train eval eval-refresh eval-reproduce ablation-weather serve docker lint typecheck check metrics-fixture
.DEFAULT_GOAL := help

help:
	@echo gridcast targets:
	@echo   install         - Create virtual environment and install dependencies
	@echo   test            - Run test suite
	@echo   data-check      - Assert integrity of committed curated dataset
	@echo   clean           - Remove virtual environment, caches, and build artefacts
	@echo   ingest          - Run ingest CLI
	@echo   features        - Build and inspect the example feature frame
	@echo   train           - Run model training CLI
	@echo   eval            - Run model evaluation benchmark from frozen fixture
	@echo   eval-refresh    - Rebuild and refresh frozen evaluation fixture (slow, manual)
	@echo   eval-reproduce  - Retrain 15 folds and verify candidate predictions against fixture (slow, manual)
	@echo   ablation-weather - Run pre-registered weather removal ablation against fixture
	@echo   metrics-fixture - Regenerate frozen split metrics fixture from eval run
	@echo   serve           - Run service API (arrives in P4)
	@echo   docker          - Build Docker image (arrives in P5)
	@echo   lint            - Run ruff linter
	@echo   typecheck       - Run mypy type checker

install:
	"$(PYTHON)" -c "import os, subprocess, sys; os.path.exists('.venv') or subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)"
	"$(PY)" -m pip install --upgrade pip setuptools
	"$(PY)" -m pip install -e ".[dev]"

test:
	"$(PY)" -m pytest --cov --cov-report=term-missing

data-check:
	"$(PY)" -m pytest tests/test_data_integrity.py

clean:
	"$(PYTHON)" -c "import glob, os, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist', *glob.glob('src/*.egg-info'), *glob.glob('*.egg-info')]]; [os.remove(f) for f in glob.glob('**/*.pyc', recursive=True)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True)]; os.remove('.coverage') if os.path.exists('.coverage') else None"

ingest:
	"$(PY)" -m gridcast ingest --days 2
	"$(PY)" -m gridcast ingest --reconcile-recent
	"$(PY)" -m gridcast ingest --weather --days 2
	"$(PY)" -m gridcast ingest --mix --days 2

features:
	"$(PY)" -m gridcast.cli features

train:
	"$(PY)" -m gridcast.cli train

eval:
	"$(PY)" -m gridcast.cli eval

eval-refresh:
	"$(PY)" -m gridcast.cli eval-refresh

eval-reproduce:
	"$(PY)" -m gridcast.cli eval-reproduce

ablation-weather:
	"$(PY)" -m gridcast.cli ablation-weather

metrics-fixture:
	"$(PY)" -c "from tests.test_model_card_consistency import write_metrics_fixture; write_metrics_fixture()"


serve:
	"$(PY)" -m gridcast.cli serve

docker:
	"$(PY)" -c "from gridcast._phase import todo; todo('P5', 'Dockerfile')"

lint:
	"$(PY)" -m ruff check src tests

typecheck:
	"$(PY)" -m mypy

check: lint typecheck test data-check
