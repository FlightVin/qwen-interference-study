#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- R-s202 restart prep (D105): preserve records written by cancelled job 11081729 after its last committed
# checkpoint (LATEST round 16): train_r0016 and later and any eval_*_r0017+ move to records/superseded-11081729/, so the
# resumed run regenerates them. Nothing is deleted.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/moved.txt" '
    C="$PROJ_DATA/ckpt/qis-main-R-s202-sym"; R="$PROJ_DATA/runs/qis-main-R-s202-sym/records"; S="$R/superseded-11081729"
    L=$(grep -o "\"round\": [0-9]*" "$C/LATEST.json" | grep -o "[0-9]*$"); echo "LATEST round=$L"
    [ "$L" = 16 ] || { echo "UNEXPECTED LATEST $L; stopping"; exit 1; }
    mkdir -p "$S"
    for f in "$R"/train_r*.jsonl.gz "$R"/eval_*_r*.jsonl.gz; do
        b=$(basename "$f"); n=$(echo "$b" | grep -o "r[0-9]\{4\}" | tr -d r); n=$((10#$n))
        case "$b" in train_*) [ "$n" -ge "$L" ] && mv "$f" "$S/";; eval_*) [ "$n" -gt "$L" ] && mv "$f" "$S/";; esac
    done
    ls -la "$S" | tee "$AE_RESULTS/moved.txt"; ls "$R" | tail -3
'
