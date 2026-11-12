#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:05:00
# HYPOTHESIS: none -- monitoring. Read-only summary of all runs' metrics (no effect on runs).
export PROJ_DATA="/data/user_data/$USER/qis"
export PYTHONPATH="$PWD/src"
phase peek "false" '
    python3 -m qis.peek "$PROJ_DATA/runs" "$AE_RESULTS/peek.json"
    du -sh "$PROJ_DATA/ckpt"/* > "$AE_RESULTS/ckpt_du.txt" 2>&1 || true
    df -h /data/user_data/$USER | tail -1 >> "$AE_RESULTS/ckpt_du.txt"
'
