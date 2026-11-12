#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- A11 truncation study: target_dev (53 x 32) at cap 16,384 from the latest (superseded) U-s101
# checkpoint round_0048; both budgets, crossing 8,192, crossing-then-correct, throughput and cost. Sealed test unused.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export CKPT="$PROJ_DATA/ckpt/qis-main-U-s101-sym/round_0048"
export OUT="$PROJ_DATA/a11/dev16k/U-s101-r48-v2"
phase precheck "false" '
    test -f "$CKPT/state.json" || { echo "PRECONDITION FAILED: $CKPT"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"; mkdir -p "$OUT"
'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$BUNDLE" --panel-key dev --n 32 --out "$OUT" --seed-label a11-dev16k --max-tokens 16384
'
phase export "test -s $AE_RESULTS/summary.json" '
    cp "$OUT/summary.json" "$AE_RESULTS/"
'
