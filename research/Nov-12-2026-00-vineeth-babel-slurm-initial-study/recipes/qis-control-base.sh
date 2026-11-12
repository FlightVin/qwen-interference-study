#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: A26 x-transfer control panel (research/audit-a26/transfer/PREREG.md): do the J_U/J_R gains of every arm also appear
# on MATH test subjects no arm practises? Policy base (Qwen3-1.7B 70d244cc, not trained); control_v1 (64 problems), n 8, cap 12,288, seed label control-v1.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export PANEL="$PWD/data/control/control_v1.json"
export PANEL_SHA="fc2ceb70ef920ab3c5a33baa78b7221f9a561db734a04942eba20ba48126e112"
export OUT="$PROJ_DATA/control-eval/base"
export CKPT="base"

phase precheck "false" '
    echo "$PANEL_SHA  $PANEL" | sha256sum -c - || { echo "CONTROL PANEL HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key control --n 8 --out "$OUT" --seed-label control-v1 --max-tokens 12288
'
phase export "test -s $AE_RESULTS/summary.json" '
    cp "$OUT/summary.json" "$AE_RESULTS/"
'
