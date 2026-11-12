#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=14:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: local update diagnostic (§7; PROTOCOL §8; A9.2), source r64 of the selected-clip R trajectory (seed 101):
# branches none/hard/related/unrelated x minibatch seeds 4101-4104 (seed-major, resumable: D114); one GRPO step each; J_H/J_R/J_U at 16 responses.
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
export SRC="$PROJ_DATA/ckpt/qis-main-R-s101-sym/round_0064"
export OUT="$PROJ_DATA/diag-local/r64"

phase precheck "false" '
    echo "0e255243d8cca981a653c4fd42b38beeeb163cee924886fa166b972d585d8c51  $BUNDLE" | sha256sum -c - || { echo "BUNDLE HASH MISMATCH"; exit 1; }
    test -f "$SRC/optim.pt" -a -f "$SRC/state.json" || { echo "PRECONDITION FAILED: R-s101 round-64 checkpoint missing"; exit 1; }
    "$VENV/bin/python" -c "import json,sys; a=json.load(open(sys.argv[1])).get(\"attempt_id\"); print(\"source attempt\", a); sys.exit(a != \"a2\")" "$SRC/state.json" || { echo "PRECONDITION FAILED: source is not attempt a2"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
'
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q \" passed\" $AE_RESULTS/pytest.txt && ! grep -qE \"failed|error\" $AE_RESULTS/pytest.txt" '
    rc=0; env CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_diag_local.py > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -3 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase run "test -f $OUT/DONE" '
    "$VENV/bin/python" -m qis.diag_local --bundle "$BUNDLE" --model "$MODEL" --source "$SRC" --label r64 --out "$OUT" --lr 2e-6 --eps-low 0.2 --eps-high 0.2 && touch "$OUT/DONE"
'
phase export "test -s $AE_RESULTS/diag_r64.jsonl" '
    cp "$OUT/diag_r64.jsonl" "$AE_RESULTS/"
'
