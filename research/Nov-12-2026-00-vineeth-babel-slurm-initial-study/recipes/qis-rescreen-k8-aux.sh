#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=08:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- D45 re-screen at the 8,192 cap: auxiliary candidates (all provenances) with
# 4k stage-1 <= 14/16 -> stage 1 (s1k8, 16) -> shortlist 4-14/16 -> stage 2 (s2k8, 32).
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
export SET="$PROJ_DATA/data/rescreen-k8-aux"
export S1="$PROJ_DATA/screen/s1k8"
export S2="$PROJ_DATA/screen/s2k8"
export SL="$PROJ_DATA/data/shortlist-k8"
phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase sets "test -s $SET/summary.json" '
    "$VENV/bin/python" -m qis.data.rescreen_set "$PROJ_DATA" data/annotations "$SET"
'
phase stage1 "test -f $S1/DONE_aux" '
    for p in related_k8 unrelated_k8; do
        "$VENV/bin/python" -m qis.screen --items "$SET/$p.jsonl" --stage s1k8 --n 16 --out "$S1/$p" --model "$MODEL" --max-tokens $CAP
    done
    touch "$S1/DONE_aux"
'
phase shortlist "test -s $SL/shortlist_summary.json" '
    "$VENV/bin/python" -m qis.data.shortlist --cand "$SET" --s1 "$S1" --out "$SL" --pools related_k8 unrelated_k8 --stage s1k8
'
phase stage2 "test -f $S2/DONE_aux" '
    for p in related_k8 unrelated_k8; do
        "$VENV/bin/python" -m qis.screen --items "$SL/$p.jsonl" --stage s2k8 --n 32 --out "$S2/$p" --model "$MODEL" --max-tokens $CAP
    done
    touch "$S2/DONE_aux"
'
phase export "test -s $AE_RESULTS/exported" '
    cp "$SET/summary.json" "$AE_RESULTS/sets_summary.json"; cp "$SL/shortlist_summary.json" "$AE_RESULTS/"
    for p in related_k8 unrelated_k8; do cp "$S1/$p/s1k8_counts.jsonl" "$AE_RESULTS/${p}_s1k8_counts.jsonl"; cp "$S2/$p/s2k8_counts.jsonl" "$AE_RESULTS/${p}_s2k8_counts.jsonl"; done
    touch "$AE_RESULTS/exported"
'
