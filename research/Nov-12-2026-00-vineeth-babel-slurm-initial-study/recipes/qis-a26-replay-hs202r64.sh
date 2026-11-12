#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- A26 pass A (a-ckpt) audit. Restore H-s202 round_0064 with the production loader (qis.train.ckpt.load
# into qis.train.learner.Learner), check the entropy fingerprint against the logged round-64 rows of trajectory a2-r60 and
# a2-orig, replay round 65 (0-based 64) from the canonical records and compare with the logged step rows; displaced-state
# control; diag_local snapshot/restore. The result is copied to $AE_RESULTS only after the script exits 0 (done-predicate). Disposable: nothing is written under $P/ckpt or $P/runs.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export OUT="$PROJ_DATA/audit-a26/a-ckpt/replay"
export RUN="$PROJ_DATA/runs/qis-main-H-s202-sym"
export CK="$PROJ_DATA/ckpt/qis-main-H-s202-sym/round_0064"
export TOOL="research/audit-a26/tools/ckpt_replay.py"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    cat "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
    test -f "$CK/state.json" || { echo "PRECONDITION FAILED: no $CK/state.json"; exit 1; }
'
phase hs202r64 "test -s $AE_RESULTS/hs202r64.json" '
    "$VENV/bin/python" "$TOOL" --mode restore --ckpt "$CK" --run-config "$RUN/config.json" --out "$OUT/hs202r64.json" \
        --fp "a2-r60|$RUN/records/train_r0063.jsonl.gz|$RUN/metrics.jsonl|64|last" \
        --fp "a2-orig|$RUN/records_a2orig_r60-64/train_r0063.jsonl.gz|$RUN/metrics.pre_resume_a17.jsonl|64|last" \
        --replay-round 64 --replay-records "$RUN/records/train_r0064.jsonl.gz" --ref-metrics "$RUN/metrics.jsonl" \
        --control --diag-restore
    cp "$OUT/hs202r64.json" "$AE_RESULTS/"
'
