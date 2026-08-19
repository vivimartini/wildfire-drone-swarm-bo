#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-repro.txt
.venv/bin/python -m pip install --no-deps -e .

MPLBACKEND=Agg .venv/bin/python -m pytest -q
MPLBACKEND=Agg .venv/bin/python -m fire_model.demo --quick --output-dir artifacts/reproduction
MPLBACKEND=Agg .venv/bin/python -m fire_model.finsler_validation --quick --output-dir artifacts/finsler
