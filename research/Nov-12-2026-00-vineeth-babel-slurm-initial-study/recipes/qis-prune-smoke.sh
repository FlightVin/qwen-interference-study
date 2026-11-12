#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- retention. Delete disposable smoke checkpoints (G2 evidence is tracked in git),
# then report project disk use and account free space.
export PROJ_DATA="/data/user_data/$USER/qis"
phase prune "test -s $AE_RESULTS/du.txt" '
    ls -la "$PROJ_DATA/ckpt/smoke-v1" > "$AE_RESULTS/deleted_listing.txt" 2>&1 || true
    du -sh "$PROJ_DATA/ckpt/smoke-v1" >> "$AE_RESULTS/deleted_listing.txt" 2>&1 || true
    rm -rf "$PROJ_DATA/ckpt/smoke-v1"
    du -sh "$PROJ_DATA"/* > "$AE_RESULTS/du.txt" 2>&1
    df -h /data/user_data/$USER | tail -1 >> "$AE_RESULTS/du.txt"
'
