#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=16G time=00:30:00
# HYPOTHESIS: none -- round-32 dev provenance, superseded a1 vs retained a2 (read-only; owner request 2026-11-08).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase a "false" '
    "$VENV/bin/python" -m qis.analysis.r32_provenance "$PROJ_DATA" "$AE_RESULTS/r32_provenance.json" > /dev/null
    "$VENV/bin/python" -c "import json; d=json.load(open(\"$AE_RESULTS/r32_provenance.json\")); print(json.dumps(d)[:2000])"
'
