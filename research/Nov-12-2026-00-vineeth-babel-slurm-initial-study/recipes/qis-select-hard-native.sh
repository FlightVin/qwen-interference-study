#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- apply the frozen H_train selection rule (D28/D31) to the native pool (A4.1).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export C="$PROJ_DATA/data/candidates-v1-study"
export S2="$PROJ_DATA/screen/s2"
export OUT="$PROJ_DATA/data/hsel-native-v1"
phase select "test -s $OUT/selection.json" '
    mkdir -p "$OUT"
    "$VENV/bin/python" - <<PY
import json
with open("$OUT/cands.jsonl", "w") as f:
    for src in ("hard", "hard_reserve"):
        for l in open("$C/" + src + ".jsonl"):
            r = json.loads(l); f.write(json.dumps({"id": r["id"], "source": src, "level": r["level"]}) + "\n")
with open("$OUT/s2.jsonl", "w") as f:
    for src in ("hard", "hard_reserve"):
        for l in open("$S2/" + src + "/s2_counts.jsonl"): f.write(l)
PY
    "$VENV/bin/python" -m qis.data.select_hard --candidates "$OUT/cands.jsonl" --s2 "$OUT/s2.jsonl" \
        --pass1 data/annotations/pass1.jsonl --pass2 data/annotations/pass2_hard.jsonl \
        --third data/annotations/third_read.jsonl --tags data/annotations/tags_v13.jsonl --out "$OUT/selection.json" | tee "$AE_RESULTS/summary.txt"
    cp "$OUT/selection.json" "$AE_RESULTS/"
'
