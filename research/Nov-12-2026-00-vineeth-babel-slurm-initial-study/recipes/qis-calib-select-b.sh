#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:20:00
# HYPOTHESIS: none -- runs tests/test_analysis.py, then applies the frozen §7 calibration selection rule to the
# committed calibration metrics and records the result with input checksums (statistics on a CPU allocation).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
phase test "test -s $AE_RESULTS/pytest.txt && grep -q ' passed' $AE_RESULTS/pytest.txt && ! grep -qE 'failed|error' $AE_RESULTS/pytest.txt" '
    rc=0; "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_analysis.py > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -5 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase select "false" '
    "$VENV/bin/python" -m qis.analysis.calib_select research/evidence/calibration > "$AE_RESULTS/selection.json"
    sha256sum research/evidence/calibration/*.metrics.jsonl > "$AE_RESULTS/inputs.sha256"
    tail -40 "$AE_RESULTS/selection.json"
'
