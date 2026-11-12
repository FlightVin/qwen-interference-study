#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- A26 pass A (a-ckpt) audit. Round-128 checkpoints with optim.pt (R-s202, U-s202): restore with the
# production loader, entropy fingerprint vs the logged round-128 row, then ONE disposable optimizer step on minibatch 0 of
# the round-128 records (0-based 127). Mechanics only (AdamW step 512 -> 513, finite loss and weights): no logged reference
# exists for a step after round 128. Nothing is written under $P/ckpt or $P/runs. Results copied to $AE_RESULTS on exit 0.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export OUT="$PROJ_DATA/audit-a26/a-ckpt/replay"
export TOOL="research/audit-a26/tools/ckpt_replay.py"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    cat "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
'
phase rs202r128 "test -s $AE_RESULTS/rs202r128.json" '
    RUN="$PROJ_DATA/runs/qis-main-R-s202-sym"
    "$VENV/bin/python" "$TOOL" --mode restore --ckpt "$PROJ_DATA/ckpt/qis-main-R-s202-sym/round_0128" --run-config "$RUN/config.json" \
        --out "$OUT/rs202r128.json" --fp "r128|$RUN/records/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|last" \
        --replay-round 127 --replay-records "$RUN/records/train_r0127.jsonl.gz" --n-mb 1
    cp "$OUT/rs202r128.json" "$AE_RESULTS/"
'
phase us202r128 "test -s $AE_RESULTS/us202r128.json" '
    RUN="$PROJ_DATA/runs/qis-main-U-s202-sym"
    "$VENV/bin/python" "$TOOL" --mode restore --ckpt "$PROJ_DATA/ckpt/qis-main-U-s202-sym/round_0128" --run-config "$RUN/config.json" \
        --out "$OUT/us202r128.json" --fp "r128|$RUN/records/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|last" \
        --replay-round 127 --replay-records "$RUN/records/train_r0127.jsonl.gz" --n-mb 1
    cp "$OUT/us202r128.json" "$AE_RESULTS/"
'
