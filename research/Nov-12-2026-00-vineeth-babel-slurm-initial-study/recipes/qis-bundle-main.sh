#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=8G time=00:15:00
# HYPOTHESIS: none -- build the frozen main-study bundle + sealed test panel + manifest (D45-D48).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/main-v1"
phase build "test -s $OUT/manifest.json" '
    "$VENV/bin/python" -m qis.data.bundle_main "$PROJ_DATA" "$OUT"
'
phase export "test -s $AE_RESULTS/manifest.json" '
    cp "$OUT/manifest.json" "$AE_RESULTS/"
'
