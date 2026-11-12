#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- read-only per-run training summary in 8-round blocks (interim; no decisions).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase summ "false" '
    "$VENV/bin/python" -m qis.analysis.train_summary "$PROJ_DATA/runs" "$AE_RESULTS/train_summary.json"
'
