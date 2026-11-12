#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- frozen H selection rule applied to the 8,192 re-screen (D45).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase select "false" '
    "$VENV/bin/python" -m qis.data.select_hard_k8 "$PROJ_DATA/data/rescreen-k8-h/h_k8.jsonl" "$PROJ_DATA/screen/s2k8/h/s2k8_counts.jsonl" data/annotations "$AE_RESULTS/selection.json" | tee "$AE_RESULTS/summary.txt"
    cp "$PROJ_DATA/data/rescreen-k8-h/h_k8.jsonl" "$AE_RESULTS/"
'
