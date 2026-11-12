#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=00:45:00
# HYPOTHESIS: none -- EXPLORATORY (A15.2, D122): full CPU test suite, then the interference trace over the training
# records of all main runs (shared hard prompts, 16-round windows, dev comparison). Training records only; no test input.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q ' passed' $AE_RESULTS/pytest.txt && ! grep -qE 'failed|error' $AE_RESULTS/pytest.txt" '
    rc=0; "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/ > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -15 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase a "false" '
    "$VENV/bin/python" -m qis.analysis.interference "$PROJ_DATA/runs" "$AE_RESULTS/interference.json"
'
