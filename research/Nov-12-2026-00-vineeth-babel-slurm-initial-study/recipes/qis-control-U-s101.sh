#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: A26 x-transfer control panel (research/audit-a26/transfer/PREREG.md): do the J_U/J_R gains of every arm also appear
# on MATH test subjects no arm practises? Policy U-s101 round 128 (endpoint of qis-main-U-s101-sym); control_v1 (64 problems), n 8, cap 12,288, seed label control-v1.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export PANEL="$PWD/data/control/control_v1.json"
export PANEL_SHA="fc2ceb70ef920ab3c5a33baa78b7221f9a561db734a04942eba20ba48126e112"
export OUT="$PROJ_DATA/control-eval/U-s101-r128"
export CKPT="$PROJ_DATA/ckpt/qis-main-U-s101-sym/round_0128"

phase precheck "false" '
    echo "$PANEL_SHA  $PANEL" | sha256sum -c - || { echo "CONTROL PANEL HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
    test -f "$PROJ_DATA/runs/qis-main-U-s101-sym/RUN_COMPLETE" || { echo "PRECONDITION FAILED: qis-main-U-s101-sym not complete"; exit 1; }
    test -f "$CKPT/model.pt" || { echo "PRECONDITION FAILED: no $CKPT/model.pt"; exit 1; }
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CKPT/state.json")
    have=$(sha256sum "$CKPT/model.pt" | cut -d" " -f1)
    echo "model.pt sha256 recorded=$want on-disk=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH"; exit 1; }
    "$VENV/bin/python" -c "import json,sys; st=json.load(open(sys.argv[1])); ok=(st[\"round\"],st[\"attempt_id\"])==(128,\"a2\"); print(\"provenance round\",st[\"round\"],\"attempt\",st[\"attempt_id\"],\"ok\" if ok else \"FAIL\")" "$CKPT/state.json" > "$AE_RESULTS/provenance.txt"
    cat "$AE_RESULTS/provenance.txt"; grep -q " ok$" "$AE_RESULTS/provenance.txt" || { echo "PROVENANCE CHECK FAILED (expected round 128, a2)"; exit 1; }
'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key control --n 8 --out "$OUT" --seed-label control-v1 --max-tokens 12288
'
phase export "test -s $AE_RESULTS/summary.json" '
    cp "$OUT/summary.json" "$AE_RESULTS/"
'
