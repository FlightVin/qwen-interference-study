#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=03:00:00
# HYPOTHESIS: none -- A11 cap-transition check for the seed-202 runs (read-only): waits until R-s202 and U-s202 have logged round 36, then runs capcheck on all main runs.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase a "false" '
    for r in R U; do M="$PROJ_DATA/runs/qis-main-$r-s202-sym/metrics.jsonl"
        until grep -q "\"kind\": \"round\".*\"round\": 36[,}]" "$M"; do sleep 120; done; done
    "$VENV/bin/python" -m qis.analysis.capcheck "$PROJ_DATA/runs" "$AE_RESULTS/capcheck.json" > /dev/null
    "$VENV/bin/python" -c "import json; d=json.load(open(\"$AE_RESULTS/capcheck.json\")); [print(k, json.dumps(d[k][\"rounds\"])) for k in d if \"s202\" in k]"
'
