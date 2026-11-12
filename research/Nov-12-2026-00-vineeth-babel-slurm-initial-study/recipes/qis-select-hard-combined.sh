#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- apply the frozen combined H selection rule (D38) to native + Omni-MATH.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export OUT="$PROJ_DATA/data/hsel-combined-v1"
phase select "test -s $OUT/selection.json" '
    mkdir -p "$OUT"
    "$VENV/bin/python" -m qis.data.select_hard_combined --native "$PROJ_DATA/data/hsel-native-v1/selection.json" \
        --omni-s2 data/screening/omni_hard_s2_counts.jsonl --omni-p1 data/annotations/omni_pass1.jsonl \
        --omni-p2 data/annotations/omni_pass2.jsonl --omni-third data/annotations/omni_third_read.jsonl \
        --omni-intake data/screening/omni_intake_report.jsonl --out "$OUT/selection.json" | tee "$AE_RESULTS/summary.txt"
    cp "$OUT/selection.json" "$AE_RESULTS/"
'
