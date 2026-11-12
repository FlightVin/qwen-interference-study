#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- A11.3: move superseded attempt-a1 run and checkpoint dirs of the s101 runs to superseded/
# (provenance kept, nothing deleted); tag superseded W&B runs (no deletion).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export WANDB_MODE=online
phase move "test -s $AE_RESULTS/moved.txt" '
    mkdir -p "$PROJ_DATA/runs/superseded" "$PROJ_DATA/ckpt/superseded"
    for r in qis-main-R-s101-sym qis-main-U-s101-sym qis-main-H-s101-sym; do
        [ -d "$PROJ_DATA/runs/$r" ] && mv "$PROJ_DATA/runs/$r" "$PROJ_DATA/runs/superseded/$r-a1"
        [ -d "$PROJ_DATA/ckpt/$r" ] && mv "$PROJ_DATA/ckpt/$r" "$PROJ_DATA/ckpt/superseded/$r-a1"
    done
    { ls -la "$PROJ_DATA/runs/superseded" "$PROJ_DATA/ckpt/superseded"; ls "$PROJ_DATA/runs" "$PROJ_DATA/ckpt"; } > "$AE_RESULTS/moved.txt"; cat "$AE_RESULTS/moved.txt"
'
phase tag "false" '
    "$VENV/bin/python" -m qis.analysis.wandb_tag_superseded RL-experiments/qis-qwen-interference \
      qis-main-R-s101-sym-a1 qis-main-U-s101-sym-a1 qis-main-H-s101-sym-a1 \
      qis-main-R-s101-sym-curves qis-main-U-s101-sym-curves qis-main-H-s101-sym-curves qis-RvU-s101-sym-curves 2>&1 | tail -10
    rm -f "$PROJ_DATA/wandb_companion_state.json"
'
