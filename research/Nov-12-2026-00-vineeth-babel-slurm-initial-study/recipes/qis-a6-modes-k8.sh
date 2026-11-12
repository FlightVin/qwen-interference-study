#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=06:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: A6 diagnostic (project.md §20; D56) — thinking vs non-thinking baseline at the same 8,192 cap on a fixed
# seeded 80-item panel (32 H, 16 R + 16 matched U, 16 dev), 8 attempts per item per mode; only enable_thinking changes.
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
export OUT="$PROJ_DATA/a6/modes-cap8192"

phase precheck "false" '
    echo "0e255243d8cca981a653c4fd42b38beeeb163cee924886fa166b972d585d8c51  $BUNDLE" | sha256sum -c - || { echo "BUNDLE HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
'
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q \" passed\" $AE_RESULTS/pytest.txt && ! grep -qE \"failed|error\" $AE_RESULTS/pytest.txt" '
    rc=0; env CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_mode_diag.py > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -5 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase run "test -s $OUT/summary_cap8192.json" '
    "$VENV/bin/python" -m qis.mode_diag --bundle "$BUNDLE" --model "$MODEL" --out "$OUT" --modes off,on --max-tokens 8192 --gpu-mem-util 0.85
'
phase export "test -s $AE_RESULTS/summary_cap8192.json" '
    cp "$OUT"/panel.json "$OUT"/template_check.json "$OUT"/summary_cap8192.json "$OUT"/per_item_cap8192.json "$OUT"/failure_sample_cap8192.json "$AE_RESULTS/"
'
