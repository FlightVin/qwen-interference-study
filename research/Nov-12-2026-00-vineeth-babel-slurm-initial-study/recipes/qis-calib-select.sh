#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:05:00
# HYPOTHESIS: none -- applies the frozen §7 calibration selection rule to the committed calibration metrics
# (research/evidence/calibration/*.metrics.jsonl) and records the result. Statistics on a CPU allocation (§3).
export PYTHONPATH="$PWD/src"
phase select "false" '
    python3 -m qis.analysis.calib_select research/evidence/calibration > "$AE_RESULTS/selection.json"
    sha256sum research/evidence/calibration/*.metrics.jsonl > "$AE_RESULTS/inputs.sha256"
    cat "$AE_RESULTS/selection.json" | tail -40
'
