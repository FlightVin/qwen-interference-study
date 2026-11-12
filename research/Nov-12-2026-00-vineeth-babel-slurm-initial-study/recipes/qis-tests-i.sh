#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=00:20:00
# HYPOTHESIS: none -- CPU unit tests for GRPO math, schedules, verifier, splits, analysis, diagnostic, A10 cap schedule; diag_local resume (D114).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q ' passed' $AE_RESULTS/pytest.txt && ! grep -qE 'failed|error' $AE_RESULTS/pytest.txt" '
    rc=0; "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/ > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -40 "$AE_RESULTS/pytest.txt"; exit $rc
'
