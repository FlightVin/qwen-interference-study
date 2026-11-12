#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=16G time=00:45:00
# HYPOTHESIS: none -- A9.3 endpoint validation (+ attempt id a2, D99) for every main run with a round_0128 checkpoint (read-only).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase a "false" '
    for d in "$PROJ_DATA"/runs/qis-main-*; do
        r=$(basename "$d"); c="$PROJ_DATA/ckpt/$r"
        [ -f "$c/round_0128/state.json" ] || { echo "$r: no round_0128"; continue; }
        echo "== $r"; "$VENV/bin/python" -m qis.analysis.validate_endpoint "$d" "$c" "$AE_RESULTS/endpoint_$r.json" a2
    done
'
