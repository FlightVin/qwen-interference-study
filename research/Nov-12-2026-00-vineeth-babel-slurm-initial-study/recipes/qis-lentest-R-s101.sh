#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=L40S
# HYPOTHESIS: none -- A19 §2 / A20 / D155 EXPLORATORY length diagnostic on the SEALED TEST: policy R-s101 round 128. Continue the capped
# attempts of the ORIGINAL sealed-test evaluation (seed label test-final, cap 12,288; scores unchanged) to 16,384 tokens (qis.eval_extend).
# One phase: every attempt of the 128 x 32 panel (--select all; seed label extend-test-v1).
# The original test records are read only; outputs go to $PROJ_DATA/length-test/R-s101-r128/all.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export PANEL="$PROJ_DATA/data/main-v1/target_test_sealed.json"
export PANEL_SHA="2ffe70ddd5f81bee5c9feadee6f08335d8b189b33f38ccab474263182dec29ca"
export REC="$PROJ_DATA/test-eval/qis-test-R-s101-sym/records.jsonl.gz"
export OUT="$PROJ_DATA/length-test/R-s101-r128"
export CKPT="$PROJ_DATA/ckpt/qis-main-R-s101-sym/round_0128"

phase precheck "false" '
    echo "$PANEL_SHA  $PANEL" | sha256sum -c - || { echo "SEALED TEST HASH MISMATCH"; exit 1; }
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
phase all "test -s $OUT/all/summary.json" '
    "$VENV/bin/python" -m qis.eval_extend --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key test --records "$REC" --out "$OUT/all" --select all --seed-label extend-test-v1
    cp "$OUT/all/summary.json" "$AE_RESULTS/all_summary.json"
'
