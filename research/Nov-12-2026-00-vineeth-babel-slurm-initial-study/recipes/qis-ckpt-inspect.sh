#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:10:00
# HYPOTHESIS: none -- read-only listing of main-run checkpoint trees (sizes, mtimes, .tmp dirs) and last checkpoint events, to diagnose slow/failed saves (Errno 512).
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "false" '
    date -u
    for d in "$PROJ_DATA"/ckpt/qis-main-*; do echo "== $d"; cat "$d/LATEST.json" 2>/dev/null; echo; for s in "$d"/round_*; do echo "  $s"; ls -la --time-style=+%FT%T "$s" | tail -n +4; done; done
    for r in "$PROJ_DATA"/runs/qis-main-*; do echo "== events $(basename $r)"; grep "\"kind\": \"event\"" "$r/metrics.jsonl" | tail -4 | cut -c1-400; done
'
