#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS (domain study): the "no initial success" band for Qwen3-1.7B starts at products of 7- to 10-digit integers
# and at 12-digit base-9 addition. qis-ui-probe measured pass@1 of 5.5% at 6 digits; this job measures the next steps.
# 32 base answers per problem, 100 problems per pool. Sampling only; no gradients, no weight changes.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export UI="$PROJ_DATA/ui"
export PANEL="$UI/probe/panel_probe_hard_v1.json"
export S="$UI/probe/samples-hard"
export POOLS="mult7 mult8 mult10 add9x12"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$S"
'
phase panel "test -s $PANEL" '
    python -m qis.interf.build_probe_panel --hard --out "$PANEL.tmp" --n 100
    mv "$PANEL.tmp" "$PANEL"
'
for pool in $POOLS; do
    export POOL=$pool
    phase "sample-$pool" "test -s $S/$pool/summary.json" '
        export HF_HUB_OFFLINE=1
        python -m qis.evaluate_ckpt --model "$MODEL" --ckpt base --panel "$PANEL" --panel-key "$POOL" --n 32 --out "$S/$POOL" --seed-label ui-probe-hard-v1
    '
done
phase export "false" '
    for p in $POOLS; do cp "$S/$p/summary.json" "$AE_RESULTS/sample_$p.json"; done
'
