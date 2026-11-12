#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=08:00:00 part=preempt constraint=L40S
# HYPOTHESIS: none -- A17 §6 EXPLORATORY stress-panel evaluation (Harder-128 + SuperHard-128, frozen stress-v1, D142):
# policy U-s202 round 128 (endpoint); 256 problems x 8 attempts, uncued, cap 12,288, frozen verifier, seed label stress-v1, L40S (D141 recipe).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export PANEL="$PROJ_DATA/data/stress/stress_v1.json"
export PANEL_SHA="757393bef80d5cf017d53bddbfcab84e049c77dee3b07b74898125379f76405d"
export OUT="$PROJ_DATA/stress-eval/U-s202-r128"
export CKPT="$PROJ_DATA/ckpt/qis-main-U-s202-sym/round_0128"

phase precheck "false" '
    echo "$PANEL_SHA  $PANEL" | sha256sum -c - || { echo "STRESS PANEL HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -q "L40S" || { echo "NOT AN L40S GPU"; exit 1; }
    mkdir -p "$OUT"
    test -f "$CKPT/model.pt" || { echo "PRECONDITION FAILED: no $CKPT/model.pt"; exit 1; }
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CKPT/state.json")
    have=$(sha256sum "$CKPT/model.pt" | cut -d" " -f1)
    echo "model.pt sha256 recorded=$want on-disk=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH"; exit 1; }
'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key stress --n 8 --out "$OUT" --seed-label stress-v1 --max-tokens 12288
'
phase export "test -s $AE_RESULTS/summary.json" '
    cp "$OUT/summary.json" "$AE_RESULTS/"
'
