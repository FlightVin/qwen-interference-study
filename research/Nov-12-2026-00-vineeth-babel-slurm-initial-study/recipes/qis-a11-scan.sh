#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:10:00
# HYPOTHESIS: none -- A11 pause record: per s101 run, LATEST checkpoint, checkpoint dirs on disk, last completed round
# in metrics, evaluations done (read-only).
export PROJ_DATA="/data/user_data/$USER/qis"
phase scan "false" '
    for r in qis-main-R-s101-sym qis-main-U-s101-sym qis-main-H-s101-sym; do
        echo "== $r"; cat "$PROJ_DATA/ckpt/$r/LATEST.json" 2>/dev/null; echo; ls "$PROJ_DATA/ckpt/$r"
        for d in "$PROJ_DATA/ckpt/$r"/round_*; do [ -f "$d/state.json" ] && echo "valid $(basename $d) $(du -s --block-size=1M $d | cut -f1)MiB"; done
        grep -c "\"kind\": \"round\"" "$PROJ_DATA/runs/$r/metrics.jsonl"; grep "\"kind\": \"round\"" "$PROJ_DATA/runs/$r/metrics.jsonl" | tail -1 | grep -o "\"round\": [0-9]*"
        ls "$PROJ_DATA/runs/$r/records" | grep -c train_r; ls "$PROJ_DATA/runs/$r/records" | grep eval_ | tr "\n" " "; echo
    done > "$AE_RESULTS/scan.txt" 2>&1; cat "$AE_RESULTS/scan.txt"
'
