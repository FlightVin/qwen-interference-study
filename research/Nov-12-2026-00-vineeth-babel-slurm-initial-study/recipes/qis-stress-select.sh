#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=01:00:00
# HYPOTHESIS: none -- A17 §5: select, verify and freeze Harder-128 / SuperHard-128 (rubric v1, qis.data.stress_panels).
# Reads only candidates + agent annotations + study item texts; no model output. Writes $PROJ_DATA/data/stress/.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
export S="$PROJ_DATA/data/stress"
phase select "test -s $S/selection_report.json" '
    mkdir -p "$S.tmp"
    "$VENV/bin/python" -m qis.data.stress_panels data/external/a17/candidates_v1.jsonl research/evidence/a17/panels/annotations "$S.tmp" "$PROJ_DATA/data" "$PWD/data" > "$S.tmp/select.log"
    tail -5 "$S.tmp/select.log"
    rm -rf "$S"; mv "$S.tmp" "$S"
'
phase export "test -s $AE_RESULTS/selection_report.json" '
    cp "$S"/*.json "$S/select.log" "$AE_RESULTS/"
    sha256sum "$S"/*.json | tee "$AE_RESULTS/sha256.txt"
'
