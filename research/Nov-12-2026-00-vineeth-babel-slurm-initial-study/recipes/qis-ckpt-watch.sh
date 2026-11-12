#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=1G time=01:30:00
# HYPOTHESIS: none -- deterministic wait (read-only): report when R-s202's round-12 checkpoint is committed (LATEST.json) or 75 min pass; prints mtimes of the round_0012 files to measure /data write time (D104).
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "false" '
    C="$PROJ_DATA/ckpt/qis-main-R-s202-sym"; t0=$(date +%s)
    while :; do
        r=$(grep -o "\"round\": [0-9]*" "$C/LATEST.json" 2>/dev/null | grep -o "[0-9]*$")
        [ "${r:-0}" -ge 12 ] && break
        [ $(( $(date +%s) - t0 )) -ge 4500 ] && { echo "TIMEOUT: R-s202 LATEST round=${r:-none} after 75 min"; break; }
        sleep 60
    done
    date -u; cat "$C/LATEST.json" 2>/dev/null; echo
    for s in "$C"/round_*; do echo "  $s"; ls -la --time-style=+%FT%T "$s" | tail -n +4; done
    for c in "$PROJ_DATA"/ckpt/qis-main-H-s101-sym "$PROJ_DATA"/ckpt/qis-main-R-s101-sym; do echo "== $c"; cat "$c/LATEST.json" 2>/dev/null; echo; done
'
