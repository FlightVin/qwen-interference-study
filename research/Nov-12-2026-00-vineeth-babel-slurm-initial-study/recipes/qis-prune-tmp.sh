#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:30:00
# HYPOTHESIS: none -- storage floor breach (D116; project.md §11: keep >= 50 GiB free; prune superseded project-owned
# copies after verifying references and recording deletions). Account quota 495/500 GB used at 21:16Z. Deletes ONLY:
# (1) failed partial checkpoint writes `round_*.tmp` (never valid checkpoints) under the completed s101 runs and R-s202's
# dead round_0080.tmp / round_0096.tmp (U-s202 is skipped: a save is in progress); (2) superseded attempt-a1 checkpoint
# trees under ckpt/superseded/ (A11, D82; no active or pending recipe references them). Every path is checked to be under
# $PROJ_DATA/ckpt and logged with its size before removal. Logs, metrics, records and valid checkpoints are not touched.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/deleted.txt" '
    C="$PROJ_DATA/ckpt"; df -h "$PROJ_DATA" | tee "$AE_RESULTS/df_before.txt"
    find "$C" -maxdepth 3 -printf "%y %TY-%Tm-%Td %TH:%TM %s %p\n" | sort -k5 > "$AE_RESULTS/inventory.txt"
    T=""
    for r in H-s101 R-s101 U-s101; do for d in "$C"/qis-main-$r-sym/round_*.tmp; do [ -d "$d" ] && T="$T $d"; done; done
    for d in "$C"/qis-main-R-s202-sym/round_0080.tmp "$C"/qis-main-R-s202-sym/round_0096.tmp; do [ -d "$d" ] && T="$T $d"; done
    for d in "$C"/superseded/*; do [ -d "$d" ] && T="$T $d"; done
    : > "$AE_RESULTS/deleted.txt"
    for d in $T; do
        p=$(realpath "$d"); case "$p" in "$C"/*) ;; *) echo "REFUSE $p"; exit 1;; esac
        s=$(find "$p" -type f -printf "%s\n" | awk "{t+=\$1} END {printf \"%.1f\", t/1e9}")
        echo "$(date -u +%FT%TZ) rm -rf $p  ${s} GB" | tee -a "$AE_RESULTS/deleted.txt"
        rm -rf "$p"
    done
    echo "done" >> "$AE_RESULTS/deleted.txt"
    df -h "$PROJ_DATA" | tee "$AE_RESULTS/df_after.txt"
'
