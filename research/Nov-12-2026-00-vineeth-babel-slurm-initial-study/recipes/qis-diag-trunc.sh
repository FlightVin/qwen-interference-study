#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:15:00
# HYPOTHESIS: diagnosis (D44) — are truncated calib_val responses after 32 rounds repetition loops or long reasoning?
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export R="$PROJ_DATA/runs/qis-cal-lr1e-6-sym-s11/records"
phase diag "false" '
    "$VENV/bin/python" -m qis.diag_trunc "$MODEL" "$AE_RESULTS/diag.json" "$R/eval_calib_val_r0000.jsonl.gz" "$R/eval_calib_val_r0016.jsonl.gz" "$R/eval_calib_val_r0032.jsonl.gz"
'
