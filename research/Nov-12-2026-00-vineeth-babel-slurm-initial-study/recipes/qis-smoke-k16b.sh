#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=02:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: A11 smoke test (rerun of 11079543 after the LM-head recompute fix) -- one complete training round (16x8, 4 optimizer steps) at a 16,384 cap on one 48 GB
# GPU with forced long responses; engine context >= 17,408; no clipping at 8,192; peak memory; phase times.
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
phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase smoke "test -s $AE_RESULTS/smoke_k16.json" '
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$AE_RESULTS/nvsmi_mem.txt"; sleep 5; done ) & mon=$!
    "$VENV/bin/python" -m qis.smoke_long --bundle "$BUNDLE" --model "$MODEL" --cap 16384 --out "$AE_RESULTS/smoke_k16.json"; rc=$?
    kill $mon; sort -n "$AE_RESULTS/nvsmi_mem.txt" | tail -1 | sed "s/^/nvidia-smi peak MiB: /"; exit $rc
'
