#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=10:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- A26 pass A (CHECKLIST 2.6): fresh hash, load and integrity check of every retained checkpoint under
# $PROJ_DATA, shard 3 of 4 (directories sorted, index mod 4). Read-only on project data. Tool: research/audit-a26/tools/ckpt_audit.py.
# General pool (D160): the CPU queue had 111 pending jobs; A26 authorizes 8 general GPUs for verification.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export BASE="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export K=3

phase audit "test -s $AE_RESULTS/ckpt_k$K.jsonl" '
    "$VENV/bin/python" research/audit-a26/tools/ckpt_audit.py "$PROJ_DATA" "$BASE" "$AE_RESULTS/ckpt_k$K.jsonl.part" --shard $K --of 4
    mv "$AE_RESULTS/ckpt_k$K.jsonl.part" "$AE_RESULTS/ckpt_k$K.jsonl"
'
