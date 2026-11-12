#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- A26 pass A (a-ckpt) audit. Weights-only saves: R-s101 r128 and U-s101 r128 (entropy fingerprint vs the
# logged round-128 rows; R-s101 also vs its superseded first attempt), the H-s202 round-116 fallback copy (gradient norm of
# round 117 minibatch 0, no step, vs the logged step row opt_step 465); tied-embedding check of the base model and of the
# R-s202 partial .tmp saves; the evaluate_ckpt BF16 vLLM load path with a full read-back (U-s101 r128).
# Nothing is written under $P/ckpt or $P/runs. Results copied to $AE_RESULTS on exit 0.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export OUT="$PROJ_DATA/audit-a26/a-ckpt/replay"
export TOOL="research/audit-a26/tools/ckpt_replay.py"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    cat "$AE_RESULTS/gpu.txt"
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
phase hs202fb "test -s $AE_RESULTS/hs202fb.json" '
    RUN="$PROJ_DATA/runs/qis-main-H-s202-sym"
    "$VENV/bin/python" "$TOOL" --mode weights --ckpt "$PROJ_DATA/ckpt/qis-main-H-s202-sym-fallback/ckpt" --run-config "$RUN/config.json" \
        --out "$OUT/hs202fb.json" --replay-round 116 --replay-records "$RUN/records/train_r0116.jsonl.gz" \
        --ref-metrics "$RUN/metrics.jsonl" --n-mb 1 --no-step \
        --tied-extra "$MODEL" --tied-extra "$PROJ_DATA/ckpt/qis-main-R-s202-sym/round_0004.tmp/model.pt" \
        --tied-extra "$PROJ_DATA/ckpt/qis-main-R-s202-sym/round_0012.tmp/model.pt"
    cp "$OUT/hs202fb.json" "$AE_RESULTS/"
'
phase vllm "test -s $AE_RESULTS/vllm_us101r128.json" '
    "$VENV/bin/python" "$TOOL" --mode vllm --ckpt "$PROJ_DATA/ckpt/qis-main-U-s101-sym/round_0128" \
        --run-config "$PROJ_DATA/runs/qis-main-U-s101-sym/config.json" --out "$OUT/vllm_us101r128.json"
    cp "$OUT/vllm_us101r128.json" "$AE_RESULTS/"
'
