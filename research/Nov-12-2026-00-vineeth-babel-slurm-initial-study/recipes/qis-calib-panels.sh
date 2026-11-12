#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00 after=qis-screen1-calib
# HYPOTHESIS: none -- freeze calibration control panels from the calib16 baseline screen.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/calib-panels-v1"
phase panels "test -s $OUT/panels_summary.json" '
    test -f "$PROJ_DATA/screen/calib16/DONE_calib" || { echo "PRECONDITION FAILED: calib16 screening incomplete"; exit 1; }
    "$VENV/bin/python" -m qis.data.calib_panels --cand "$PROJ_DATA/data/candidates-v1-calib" --screen "$PROJ_DATA/screen/calib16" --out "$OUT"
'
phase export "test -s $AE_RESULTS/panels_summary.json" '
    cp "$OUT/panels_summary.json" "$AE_RESULTS/"
    sha256sum "$OUT/bundle.json" > "$AE_RESULTS/bundle.sha256"
    for sp in calib_train calib_val; do cp "$PROJ_DATA/screen/calib16/$sp/calib16_counts.jsonl" "$AE_RESULTS/${sp}_calib16_counts.jsonl"; done
'
