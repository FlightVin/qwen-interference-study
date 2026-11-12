#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- storage check after repeated Errno 512 save failures (D116): filesystem free space, project usage,
# checkpoint trees incl. orphaned .tmp directories. Read-only.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/df.txt" '
    { df -h /data/user_data "$PROJ_DATA" 2>&1; echo; timeout 300 du -sh "$PROJ_DATA" 2>&1; echo;
      for d in "$PROJ_DATA"/ckpt/qis-main-*; do echo "== $d"; ls -la "$d"; timeout 120 du -sh "$d"/* 2>&1; done;
      echo; mount | grep -E "/data" ; } | tee "$AE_RESULTS/df.txt"
'
