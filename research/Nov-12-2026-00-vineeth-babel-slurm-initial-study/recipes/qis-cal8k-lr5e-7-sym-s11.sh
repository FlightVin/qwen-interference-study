#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=12:00:00 part=preempt constraint=VRAM_48GB after=qis-calib-panels-k8
# HYPOTHESIS: optimizer calibration (project.md §7) — lr=5e-7, clip=(0.2,0.2), seed 11, 32 rounds (cap 8,192; D45) on the
# frozen calibration-train panel; endpoint calib_val pass@1 gain (16 resp/problem) ranks configs.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export WANDB_MODE=online
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/calib-panels-k8/bundle.json"
export RUN="qis-cal8k-lr5e-7-sym-s11"
export W="$PROJ_DATA/runs/$RUN"
export CK="$PROJ_DATA/ckpt/$RUN"

phase precheck "false" '
    test -s "$BUNDLE" || { echo "PRECONDITION FAILED: calibration bundle missing"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$W" "$CK"
'
phase config "test -s $W/config.json" '
    cat > "$W/config.json" <<JSON
{"run_id": "$RUN", "attempt_id": "a1", "phase": "calibration", "arm": "C", "seed": 11,
 "lr": 5e-7, "eps_low": 0.2, "eps_high": 0.2, "rounds": 32, "ckpt_every": 4,
 "eval_rounds": [0, 16, 32], "eval_panels": {"calib_val": 16},
 "bundle": "$BUNDLE", "model": "$MODEL", "max_tokens": 8192, "gpu_mem_util": 0.30,
 "out_dir": "$W", "ckpt_dir": "$CK", "ckpt_lock_root": "$PROJ_DATA/ckpt",
 "wandb": {"enabled": true, "project": "qis-qwen-interference"}}
JSON
'
phase train "test -f $W/RUN_COMPLETE" '
    rc=0; "$VENV/bin/python" -m qis.train.run --config "$W/config.json" > "$W/stdout.log" 2>&1 || rc=$?
    tail -20 "$W/stdout.log"
    if [ $rc -eq 143 ]; then scontrol requeue $SLURM_JOB_ID; sleep 900; fi
    exit $rc
'
phase export "test -s $AE_RESULTS/metrics.jsonl" '
    cp "$W/metrics.jsonl" "$AE_RESULTS/metrics.jsonl"
    cp "$W/config.json" "$AE_RESULTS/config.json"
'
