#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=00:40:00
# HYPOTHESIS: none -- intake for authored tranche 2 (D40): verifier gold check, cap, dedupe vs all MATH and tranche 1.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export OUT="$PROJ_DATA/data/authored-v2"
phase intake "test -s $OUT/intake_summary.json" '
    "$VENV/bin/python" -m qis.data.authored --authored data/authored/R3.jsonl data/authored/R4.jsonl data/authored/R5.jsonl data/authored/U3.jsonl \
        --manifest "$PROJ_DATA/data/splits-v1/manifest.jsonl" --model "$MODEL" --out "$OUT" \
        --also-dedupe data/authored/R1.jsonl data/authored/R2.jsonl data/authored/U1.jsonl data/authored/U2.jsonl
'
phase export "test -s $AE_RESULTS/intake_summary.json" '
    cp "$OUT/intake_summary.json" "$OUT/intake_report.jsonl" "$AE_RESULTS/"
'
