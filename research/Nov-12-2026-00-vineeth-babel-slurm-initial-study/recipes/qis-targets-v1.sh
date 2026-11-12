#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- freeze held-out target and probe panels by rubric only (no model outcomes used).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/targets-v1"
phase select "test -s $OUT/panels.json" '
    "$VENV/bin/python" -m qis.data.select_targets --cand "$PROJ_DATA/data/candidates-v1-study" --ann data/annotations --out "$OUT"
'
phase export "test -s $AE_RESULTS/panels.json" '
    cp "$OUT/panels.json" "$AE_RESULTS/"; sha256sum "$OUT"/*.jsonl > "$AE_RESULTS/panels.sha256"
'
