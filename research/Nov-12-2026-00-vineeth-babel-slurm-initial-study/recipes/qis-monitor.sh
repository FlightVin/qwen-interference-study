#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:30:00
# HYPOTHESIS: none -- hourly read-only monitor (D66, D67): truncation by arm/role, eval truncation, free space.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export WANDB_MODE=online
phase mon "false" '
    "$VENV/bin/python" -m qis.analysis.trunc_monitor "$PROJ_DATA/runs" "$AE_RESULTS/trunc_monitor.json" > /dev/null
    "$VENV/bin/python" -m qis.peek "$PROJ_DATA/runs" "$AE_RESULTS/peek.json"
    "$VENV/bin/python" -m qis.analysis.curves "$PROJ_DATA/runs" "$AE_RESULTS/curves.json"
    "$VENV/bin/python" -m qis.analysis.train_summary "$PROJ_DATA/runs" "$AE_RESULTS/train_summary.json"
    "$VENV/bin/python" -m qis.analysis.wandb_companion "$PROJ_DATA/runs" "$PROJ_DATA/data/main-v1/bundle_main_v1.json" "$PROJ_DATA/wandb_companion_state.json" qis-qwen-interference > "$AE_RESULTS/wandb_companion.txt" 2>&1 || echo "wandb companion failed (see wandb_companion.txt); local JSONL unaffected"
    { df -h /data/user_data/$USER | tail -1; du -s --block-size=1G "$PROJ_DATA"/ckpt 2>/dev/null; } > "$AE_RESULTS/storage.txt"
    cat "$AE_RESULTS/storage.txt"
'
