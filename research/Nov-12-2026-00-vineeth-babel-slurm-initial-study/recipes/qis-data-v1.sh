#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=01:00:00 after=qis-env-v1
# HYPOTHESIS: none -- engineering. Runs the verifier/split unit tests, then builds the split
# manifest (dedupe, 60/20/20 seeded splits, gold eligibility) and the annotation exports.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/splits-v1"

phase precondition "false" '
    test -f "$VENV/.complete" || { echo "PRECONDITION FAILED: env venv-v1 incomplete"; exit 1; }
'
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q passed $AE_RESULTS/pytest.txt" '
    rc=0; "$VENV/bin/python" -m pytest -q tests/test_verify.py tests/test_splits.py > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -30 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase build "test -s $AE_RESULTS/build_summary.json" '
    "$VENV/bin/python" -m qis.data.build --raw "$PROJ_DATA/data/raw-21a5633" --out "$OUT" --export "$AE_RESULTS"
'
