#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=06:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- D45 re-screen at the 8,192 cap: H candidates (audit-eligible, 4k c <= 12/32;
# 32 attempts, stream s2k8) and calibration candidates (16 attempts, stream calib16k8).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export CAP=8192
export SET="$PROJ_DATA/data/rescreen-k8-h"
phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase sets "test -s $SET/summary.json" '
    "$VENV/bin/python" -m qis.data.rescreen_set "$PROJ_DATA" data/annotations "$SET"
'
phase h "test -f $PROJ_DATA/screen/s2k8/DONE_h" '
    "$VENV/bin/python" -m qis.screen --items "$SET/h_k8.jsonl" --stage s2k8 --n 32 --out "$PROJ_DATA/screen/s2k8/h" --model "$MODEL" --max-tokens $CAP
    touch "$PROJ_DATA/screen/s2k8/DONE_h"
'
phase calib "test -f $PROJ_DATA/screen/calib16k8/DONE" '
    for sp in calib_train calib_val; do
        "$VENV/bin/python" -m qis.screen --items "$PROJ_DATA/data/candidates-v1-calib/$sp.jsonl" --stage calib16k8 --n 16 --out "$PROJ_DATA/screen/calib16k8/$sp" --model "$MODEL" --max-tokens $CAP
    done
    touch "$PROJ_DATA/screen/calib16k8/DONE"
'
phase export "test -s $AE_RESULTS/exported" '
    cp "$SET/summary.json" "$AE_RESULTS/sets_summary.json"
    cp "$PROJ_DATA/screen/s2k8/h/s2k8_counts.jsonl" "$AE_RESULTS/h_s2k8_counts.jsonl"
    for sp in calib_train calib_val; do cp "$PROJ_DATA/screen/calib16k8/$sp/calib16k8_counts.jsonl" "$AE_RESULTS/${sp}_calib16k8_counts.jsonl"; done
    touch "$AE_RESULTS/exported"
'
