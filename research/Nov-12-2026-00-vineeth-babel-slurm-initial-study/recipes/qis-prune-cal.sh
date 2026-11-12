#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:30:00
# HYPOTHESIS: none -- safe retention cleanup (project.md line 240/268; D67): delete project-owned calibration restart
# checkpoints after the extension decision (D57: gate passed, no extension). Exact named directories only; evidence
# (metrics) is in git under research/evidence/calibration/. Sizes and file lists logged before deletion.
export PROJ_DATA="/data/user_data/$USER/qis"
phase prune "test -s $AE_RESULTS/deleted.txt" '
    set -u
    for r in qis-cal8k-lr1e-6-asym-s11 qis-cal8k-lr1e-6-sym-s11 qis-cal8k-lr1e-6-sym-s22 qis-cal8k-lr2e-6-asym-s11 qis-cal8k-lr2e-6-sym-s11 qis-cal8k-lr2e-6-sym-s22 qis-cal8k-lr5e-7-asym-s11 qis-cal8k-lr5e-7-sym-s11 qis-cal-lr1e-6-asym-s11 qis-cal-lr1e-6-sym-s11; do
        d="$PROJ_DATA/ckpt/$r"
        [ -d "$d" ] || { echo "absent $d" >> "$AE_RESULTS/deleted.txt"; continue; }
        echo "$(du -s --block-size=1M "$d" | cut -f1) MiB $d" >> "$AE_RESULTS/deleted.txt"
        find "$d" -maxdepth 2 | sed "s|^|  |" >> "$AE_RESULTS/deleted.txt"
        rm -rf -- "$d"
    done
    df -h /data/user_data/$USER | tail -1 >> "$AE_RESULTS/deleted.txt"
    cat "$AE_RESULTS/deleted.txt" | grep -v "^  "
'
