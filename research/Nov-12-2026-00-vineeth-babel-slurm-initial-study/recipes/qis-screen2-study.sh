#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB after=qis-screen1-study
# HYPOTHESIS: none -- dataset construction. Stage-1 shortlist (H 0-5/16, aux 4-14/16, fixed order),
# then stage 2: 32 independent samples per shortlisted item (stream s2), cap 4096, frozen decoding.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export C="$PROJ_DATA/data/candidates-v1-study"
export S1="$PROJ_DATA/screen/s1"
export SL="$PROJ_DATA/data/shortlist-v1"
export S2="$PROJ_DATA/screen/s2"

phase precheck "false" '
    test -f "$S1/DONE_study" || { echo "PRECONDITION FAILED: stage 1 not complete"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase shortlist "test -s $SL/shortlist_summary.json" '
    "$VENV/bin/python" -m qis.data.shortlist --cand "$C" --s1 "$S1" --out "$SL"
    cp "$SL/shortlist_summary.json" "$AE_RESULTS/"
'
phase screen "test -f $S2/DONE" '
    mkdir -p "$S2"
    for p in hard hard_reserve related unrelated; do
        "$VENV/bin/python" -m qis.screen --items "$SL/$p.jsonl" --stage s2 --n 32 --out "$S2/$p" --model "$MODEL" --max-tokens 4096
    done
    touch "$S2/DONE"
'
phase export "test -s $AE_RESULTS/exported" '
    for p in hard hard_reserve related unrelated; do cp "$S2/$p/s2_counts.jsonl" "$AE_RESULTS/${p}_s2_counts.jsonl"; done
    touch "$AE_RESULTS/exported"
'
