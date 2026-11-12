#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=00:30:00
# HYPOTHESIS: none -- intake checks for the 400 authored auxiliary candidates (D32): verifier
# gold self-check, prompt cap, near-duplicate check against all MATH splits.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export OUT="$PROJ_DATA/data/authored-v1"
phase intake "test -s $OUT/intake_summary.json" '
    "$VENV/bin/python" -m qis.data.authored --authored data/authored/R1.jsonl data/authored/R2.jsonl data/authored/U1.jsonl data/authored/U2.jsonl \
        --manifest "$PROJ_DATA/data/splits-v1/manifest.jsonl" --model "$MODEL" --out "$OUT"
'
phase export "test -s $AE_RESULTS/intake_summary.json" '
    cp "$OUT/intake_summary.json" "$OUT/intake_report.jsonl" "$AE_RESULTS/"
'
