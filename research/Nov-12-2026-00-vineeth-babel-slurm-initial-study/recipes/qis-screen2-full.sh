#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- dataset construction (A4.1). Stage 2 (stream s2, 32 attempts, cap 4096) on ALL
# 154 hard + 30 reserve candidates. Resumes the s2 output of qis-screen2-study, so items already
# screened there are reused, not redrawn. Budget <= 5,888 attempts in total.
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
export S2="$PROJ_DATA/screen/s2"

phase precheck "false" '
    test -f "$S2/DONE" || { echo "PRECONDITION FAILED: qis-screen2-study incomplete"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    for p in hard hard_reserve; do wc -l < "$S2/$p/s2_counts.jsonl" > "$AE_RESULTS/${p}_reused_count.txt"; done
'
phase screen "test -f $S2/DONE_full" '
    for p in hard hard_reserve; do
        "$VENV/bin/python" -m qis.screen --items "$C/$p.jsonl" --stage s2 --n 32 --out "$S2/$p" --model "$MODEL" --max-tokens 4096
    done
    touch "$S2/DONE_full"
'
phase export "test -s $AE_RESULTS/exported" '
    for p in hard hard_reserve; do cp "$S2/$p/s2_counts.jsonl" "$AE_RESULTS/${p}_s2_counts.jsonl"; done
    cp "$C/hard.jsonl" "$AE_RESULTS/hard_candidates.jsonl"; cp "$C/hard_reserve.jsonl" "$AE_RESULTS/hard_reserve_candidates.jsonl"
    touch "$AE_RESULTS/exported"
'
