#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=03:00:00
# A11.6: verify from run logs that round 33 uses the 12,288 cap in rollout (record lengths), training (round metric
# max_tokens) and grading (capped responses graded truncated, reward 0; eval max_tokens). Read-only.
# Second pass: waits (polling every 2 min) until R-, U- and H-s101 have all logged round 36, so each has rounds 31-35
# (The first evaluation at 12,288 is at round 64; checked separately.)
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase a "false" '
    for a in R U H; do
        until grep -q "\"kind\": \"round\".*\"round\": 36[,}]" "$PROJ_DATA/runs/qis-main-$a-s101-sym/metrics.jsonl"; do sleep 120; done
    done
    "$VENV/bin/python" -m qis.analysis.capcheck "$PROJ_DATA/runs" "$AE_RESULTS/capcheck.json" > /dev/null
    cat "$AE_RESULTS/capcheck.json"
'
