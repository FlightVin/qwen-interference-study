#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=01:30:00 part=general constraint=L40
# HYPOTHESIS: none -- A26 pass A (a-ckpt) audit, GPU-matched entropy fingerprints. Replays on the logging GPU type gave
# bit-identical fingerprints (5/5); across L40 <-> L40S they differed by 1.3e-3 to 1.9e-3 relative, more than a one-round
# state displacement (8.4e-4 to 3.9e-3). This job repeats the weights-only fingerprint of H-s101 r128 (and H-s202 r64 vs a2-orig, logged on L40) on L40, the GPU type that logged
# the round-128 row. Read-only on $P/ckpt and $P/runs; output under $P/audit-a26/a-ckpt/replay-l40.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export OUT="$PROJ_DATA/audit-a26/a-ckpt/replay-l40"
export TOOL="research/audit-a26/tools/ckpt_replay.py"


phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    cat "$AE_RESULTS/gpu.txt"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -qx "NVIDIA L40" || { echo "NOT AN L40 GPU"; exit 1; }
    mkdir -p "$OUT"
'
phase hs101r128 "test -s $AE_RESULTS/hs101r128.json" '
    RUN="$PROJ_DATA/runs/qis-main-H-s101-sym"
    "$VENV/bin/python" "$TOOL" --mode weights --ckpt "$PROJ_DATA/ckpt/qis-main-H-s101-sym/round_0128" --run-config "$RUN/config.json" \
        --out "$OUT/hs101r128.json" --fp "r128|$RUN/records/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|last"
    cp "$OUT/hs101r128.json" "$AE_RESULTS/"
'
phase hs202r64orig "test -s $AE_RESULTS/hs202r64_l40.json" '
    RUN="$PROJ_DATA/runs/qis-main-H-s202-sym"
    "$VENV/bin/python" "$TOOL" --mode weights --ckpt "$PROJ_DATA/ckpt/qis-main-H-s202-sym/round_0064" --run-config "$RUN/config.json" \
        --out "$OUT/hs202r64_l40.json" \
        --fp "a2-orig|$RUN/records_a2orig_r60-64/train_r0063.jsonl.gz|$RUN/metrics.pre_resume_a17.jsonl|64|last" \
        --fp "a2-r60|$RUN/records/train_r0063.jsonl.gz|$RUN/metrics.jsonl|64|last"
    cp "$OUT/hs202r64_l40.json" "$AE_RESULTS/"
'
