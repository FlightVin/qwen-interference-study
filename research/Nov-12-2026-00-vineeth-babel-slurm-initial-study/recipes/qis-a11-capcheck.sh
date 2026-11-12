#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=02:30:00
# A11.6: verify from run logs that round 33 uses the 12,288 cap in rollout (record lengths), training (round metric
# max_tokens) and grading (capped responses graded truncated, reward 0; eval max_tokens). Read-only.
# Waits (deterministically, polling every 2 min) until R-s101 has logged round 34, so rounds 31-34 are complete.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export M="$PROJ_DATA/runs/qis-main-R-s101-sym/metrics.jsonl"
phase a "false" '
    until grep -q "\"kind\": \"round\".*\"round\": 34[,}]\|\"round\": 34, .*\"kind\": \"round\"" "$M"; do sleep 120; done
    "$VENV/bin/python" -m qis.analysis.capcheck "$PROJ_DATA/runs" "$AE_RESULTS/capcheck.json" > /dev/null
    cat "$AE_RESULTS/capcheck.json"
'
