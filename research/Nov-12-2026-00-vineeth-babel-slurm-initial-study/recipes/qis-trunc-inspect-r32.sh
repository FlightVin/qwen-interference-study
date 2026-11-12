#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:20:00
# HYPOTHESIS: none -- A8.4 inspection after a 5% crossing: dev truncation 11.6% (R-s101) and 16.7% (U-s101) at
# round 32; loop score and tail excerpts of truncated responses at round 0 vs 32 (read-only; no cap change).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
phase inspect "false" '
    R="$PROJ_DATA/runs"
    "$VENV/bin/python" -m qis.diag_trunc "$MODEL" "$AE_RESULTS/trunc_r32.json" \
      $R/qis-main-R-s101-sym/records/eval_dev_r0032.jsonl.gz $R/qis-main-U-s101-sym/records/eval_dev_r0032.jsonl.gz \
      $R/qis-main-R-s101-sym/records/eval_dev_r0000.jsonl.gz
    "$VENV/bin/python" -m qis.diag_trunc "$MODEL" "$AE_RESULTS/trunc_train.json" \
      $(ls $R/qis-main-U-s101-sym/records/train_r00[23]*.jsonl.gz | tail -4) $(ls $R/qis-main-R-s101-sym/records/train_r00[23]*.jsonl.gz | tail -4)
'
