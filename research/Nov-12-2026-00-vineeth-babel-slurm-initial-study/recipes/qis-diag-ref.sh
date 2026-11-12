#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=04:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: reference check for the local update diagnostic (D131, EXPLORATORY). In r0, all 12 update rows sit below
# "none" on J_R by 0.020-0.040. Test whether the sleep -> restore -> reload cycle (or evaluation nondeterminism) explains
# a uniform shift: evaluate each source twice without an update (none_rerun fresh; none_cycled after the cycle), r0 then r64.
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
export OUT0="$PROJ_DATA/diag-local/r0"
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
phase ref0 "test -f $OUT0/REF_DONE" '
    "$VENV/bin/python" -m qis.diag_local --bundle "$BUNDLE" --model "$MODEL" --source base --label r0 --out "$OUT0" --reference-only && touch "$OUT0/REF_DONE"
'
phase ref64 "test -f $OUT/REF_DONE" '
    "$VENV/bin/python" -m qis.diag_local --bundle "$BUNDLE" --model "$MODEL" --source "$SRC" --label r64 --out "$OUT" --reference-only && touch "$OUT/REF_DONE"
'
phase export "test -s $AE_RESULTS/diag_r64_ref.jsonl" '
    cp "$OUT0/diag_r0_ref.jsonl" "$OUT/diag_r64_ref.jsonl" "$AE_RESULTS/"
'
