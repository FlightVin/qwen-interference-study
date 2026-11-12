#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- read-only storage scan of this project's checkpoint tree (A8.5 peak-storage check).
export PROJ_DATA="/data/user_data/$USER/qis"
phase scan "false" '
    { df -h /data/user_data/$USER | tail -1; echo "--- per run dir"; du -s --block-size=1G "$PROJ_DATA"/ckpt/* 2>/dev/null | sort -k2;
      echo "--- per checkpoint subdir"; for d in "$PROJ_DATA"/ckpt/*/; do for s in "$d"*/; do [ -d "$s" ] && echo "$(du -s --block-size=1G "$s" | cut -f1)G $s"; done; ls "$d" | grep -v / | sed "s|^|  file: $d|"; done;
      echo "--- other top-level in PROJ_DATA"; du -s --block-size=1G "$PROJ_DATA"/* 2>/dev/null | sort -k2; } > "$AE_RESULTS/storage_scan.txt" 2>&1
    cat "$AE_RESULTS/storage_scan.txt" | head -120
'
