#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=01:30:00 part=general constraint=L40S
# HYPOTHESIS: none -- A26 pass A (a-ckpt) audit, GPU-matched entropy fingerprints. Replays on the logging GPU type gave
# bit-identical fingerprints (5/5); across L40 <-> L40S they differed by 1.3e-3 to 1.9e-3 relative, more than a one-round
# state displacement (8.4e-4 to 3.9e-3). This job repeats the weights-only fingerprint of R-s101 r128 and U-s101 r128 on L40S, the GPU type that logged
# the round-128 row. Read-only on $P/ckpt and $P/runs; output under $P/audit-a26/a-ckpt/replay-l40s.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export OUT="$PROJ_DATA/audit-a26/a-ckpt/replay-l40s"
export TOOL="research/audit-a26/tools/ckpt_replay.py"


phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    cat "$AE_RESULTS/gpu.txt"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -qx "NVIDIA L40S" || { echo "NOT AN L40S GPU"; exit 1; }
    mkdir -p "$OUT"
'
phase rs101r128 "test -s $AE_RESULTS/rs101r128.json" '
    RUN="$PROJ_DATA/runs/qis-main-R-s101-sym"
    "$VENV/bin/python" "$TOOL" --mode weights --ckpt "$PROJ_DATA/ckpt/qis-main-R-s101-sym/round_0128" --run-config "$RUN/config.json" \
        --out "$OUT/rs101r128.json" --fp "r128-retry|$RUN/records/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|last" \
        --fp "r128-superseded-11081724|$RUN/records/superseded-11081724/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|first"
    cp "$OUT/rs101r128.json" "$AE_RESULTS/"
'
phase us101r128 "test -s $AE_RESULTS/us101r128.json" '
    RUN="$PROJ_DATA/runs/qis-main-U-s101-sym"
    "$VENV/bin/python" "$TOOL" --mode weights --ckpt "$PROJ_DATA/ckpt/qis-main-U-s101-sym/round_0128" --run-config "$RUN/config.json" \
        --out "$OUT/us101r128.json" --fp "r128|$RUN/records/train_r0127.jsonl.gz|$RUN/metrics.jsonl|128|last"
    cp "$OUT/us101r128.json" "$AE_RESULTS/"
'
