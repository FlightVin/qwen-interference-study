#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=preempt constraint=L40S
# HYPOTHESIS: none -- A19 §2 / D154 EXPLORATORY length diagnostic: policy U-s101 round 128. Continue the capped attempts of the
# frozen stress-v1 evaluation (seed label stress-v1, cap 12,288) to 16,384 response tokens with qis.eval_extend.
# Phase t1: fixed stratified sample (64 problems, 32 per tier, samples 0-3). Phase rest: every other attempt of the panel.
# The original stress records are read only; outputs go to $PROJ_DATA/length-diag/U-s101-r128/{t1,rest}.
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
export REC="$PROJ_DATA/stress-eval/U-s101-r128/records.jsonl.gz"
export OUT="$PROJ_DATA/length-diag/U-s101-r128"
export CKPT="$PROJ_DATA/ckpt/qis-main-U-s101-sym/round_0128"

phase precheck "false" '
    echo "$PANEL_SHA  $PANEL" | sha256sum -c - || { echo "STRESS PANEL HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -q "L40S" || { echo "NOT AN L40S GPU"; exit 1; }
    test -s "$REC" || { echo "PRECONDITION FAILED: no $REC"; exit 1; }
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CKPT/state.json")
    have=$(sha256sum "$CKPT/model.pt" | cut -d" " -f1)
    echo "model.pt sha256 recorded=$want on-disk=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH"; exit 1; }
    env CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -m pytest -q tests/test_eval_extend.py 2>&1 | tail -3 | tee "$AE_RESULTS/tests.txt"
    grep -q "3 passed" "$AE_RESULTS/tests.txt" || { echo "eval_extend tests failed"; exit 1; }
    mkdir -p "$OUT"
'
phase t1 "test -s $OUT/t1/summary.json" '
    "$VENV/bin/python" -m qis.eval_extend --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key stress --records "$REC" --out "$OUT/t1" --select tranche1
    cp "$OUT/t1/summary.json" "$AE_RESULTS/t1_summary.json"
'
phase rest "test -s $OUT/rest/summary.json" '
    "$VENV/bin/python" -m qis.eval_extend --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key stress --records "$REC" --out "$OUT/rest" --select rest --exclude "$OUT/t1"
    cp "$OUT/rest/summary.json" "$AE_RESULTS/rest_summary.json"
'
