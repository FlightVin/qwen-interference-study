#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00 after=qis-rescreen-k8-h
# HYPOTHESIS: none -- calibration control panels at the 8,192 cap (D45), from calib16k8 screening.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/calib-panels-k8"
phase panels "test -s $OUT/panels_summary.json" '
    test -f "$PROJ_DATA/screen/calib16k8/DONE" || { echo "PRECONDITION FAILED: calib16k8 incomplete"; exit 1; }
    "$VENV/bin/python" -m qis.data.calib_panels --cand "$PROJ_DATA/data/candidates-v1-calib" --screen "$PROJ_DATA/screen/calib16k8" --out "$OUT" --stage calib16k8
'
phase export "test -s $AE_RESULTS/panels_summary.json" '
    cp "$OUT/panels_summary.json" "$AE_RESULTS/"; sha256sum "$OUT/bundle.json" > "$AE_RESULTS/bundle.sha256"
'
