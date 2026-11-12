#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- R-s101 retry prep (D103): preserve records of the failed rounds 125-128 trajectory of job 11081724
# (train_r0124..r0127, eval_*_r0128) by moving them into records/superseded-11081724/, so the resumed run (from round_0124)
# regenerates them and the round-128 evaluation is of the new policy. Nothing is deleted. Read-only otherwise.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/moved.txt" '
    R="$PROJ_DATA/runs/qis-main-R-s101-sym/records"; S="$R/superseded-11081724"
    mkdir -p "$S"
    for f in train_r0124 train_r0125 train_r0126 train_r0127 eval_dev_r0128 eval_probe_J_R_r0128 eval_probe_J_U_r0128; do
        [ -f "$R/$f.jsonl.gz" ] && mv "$R/$f.jsonl.gz" "$S/"
    done
    cat "$PROJ_DATA/ckpt/qis-main-R-s101-sym/LATEST.json"; echo
    ls -la "$S" | tee "$AE_RESULTS/moved.txt"; ls "$R" | grep -E "r012[4-8]" || echo "no r0124-r0128 records left in main dir"
'
