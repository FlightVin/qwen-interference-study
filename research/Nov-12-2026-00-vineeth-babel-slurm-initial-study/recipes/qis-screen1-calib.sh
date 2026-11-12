#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- dataset construction. Stage-1 screening (calib16, 16 samples/item, cap 4096,
# frozen decoding) of pools: calib_train calib_val. Candidates from pass-1 annotations under rubric v1.1 (+D19).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export MANIFEST="$PROJ_DATA/data/splits-v1/manifest.jsonl"
export C="$PROJ_DATA/data/candidates-v1-calib"
export S="$PROJ_DATA/screen/calib16"
export POOLS="calib_train calib_val"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase candidates "test -s $C/candidates_summary.json" '
    "$VENV/bin/python" -m qis.data.candidates --manifest "$MANIFEST" --annotations data/annotations/pass1.jsonl --out "$C"
    cp "$C/candidates_summary.json" "$AE_RESULTS/"
'
phase screen "test -f $S/DONE_calib" '
    mkdir -p "$S"
    for p in $POOLS; do
        "$VENV/bin/python" -m qis.screen --items "$C/$p.jsonl" --stage calib16 --n 16 --out "$S/$p" --model "$MODEL" --max-tokens 4096
    done
    touch "$S/DONE_calib"
'
phase export "test -s $AE_RESULTS/exported" '
    for p in $POOLS; do cp "$S/$p/calib16_counts.jsonl" "$AE_RESULTS/${p}_counts.jsonl"; cp "$S/$p/calib16_over_cap.json" "$AE_RESULTS/${p}_over_cap.json"; done
    touch "$AE_RESULTS/exported"
'
