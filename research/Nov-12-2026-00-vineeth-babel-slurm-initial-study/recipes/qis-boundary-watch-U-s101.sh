#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=1G time=10:00:00
# HYPOTHESIS: none -- A10 boundary watcher (D73/D74) for qis-main-U-s101-sym: waits until round_0064 is complete
# (state.json present, LATEST >= 64), hard-links it to boundary_0064 (protects it from rolling deletion), then exits
# so the orchestrator can cancel the old job and release its replacement. Takes no other action.
export CK="/data/user_data/$USER/qis/ckpt/qis-main-U-s101-sym"
phase watch "test -s $AE_RESULTS/boundary.txt" '
    while ! { test -f "$CK/round_0064/state.json" && grep -qE "\"round\": (6[4-9]|[7-9][0-9]|1[0-9][0-9])" "$CK/LATEST.json"; }; do sleep 30; done
    [ -d "$CK/boundary_0064" ] || cp -al "$CK/round_0064" "$CK/boundary_0064"
    { date -u +%FT%TZ; cat "$CK/LATEST.json"; ls -la "$CK/boundary_0064"; } > "$AE_RESULTS/boundary.txt"; cat "$AE_RESULTS/boundary.txt"
'
