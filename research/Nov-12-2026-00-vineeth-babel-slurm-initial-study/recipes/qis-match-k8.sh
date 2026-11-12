#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:15:00
# HYPOTHESIS: none -- build R/U matching inputs at 8,192 and run the frozen matching + A5 coverage (D39).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/match-k8"
phase match "false" '
    "$VENV/bin/python" -m qis.data.aux_inputs "$PROJ_DATA" data/annotations "$OUT" | tee "$AE_RESULTS/inputs_summary.txt"
    "$VENV/bin/python" -c "import json; s=json.load(open(\"data/screening/hsel_k8.json\")); json.dump([x[\"family\"] for x in s[\"selected\"]], open(\"$OUT/h_primaries.json\",\"w\"))"
    "$VENV/bin/python" -m qis.data.match_aux --related "$OUT/related.jsonl" --unrelated "$OUT/unrelated.jsonl" --h-primaries "$OUT/h_primaries.json" --out "$OUT/match.json" | tee "$AE_RESULTS/match_summary.txt"
    cp "$OUT/match.json" "$OUT/related.jsonl" "$OUT/unrelated.jsonl" "$OUT/summary.json" "$AE_RESULTS/"
'
