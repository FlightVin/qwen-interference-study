#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=40:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: main study (primary) — arm R, seed 202, lr 2e-6, clip (0.2,0.2), 128 rounds,
# cap schedule 0:8192,32:12288 (A10), bundle main-v1 (sha 0e255243…). Primary contrast R−U on target_test (separate job).
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
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export RUN="qis-main-R-s202-sym"
export W="$PROJ_DATA/runs/$RUN"
export CK="$PROJ_DATA/ckpt/$RUN"

phase precheck "false" '
    test -s "$BUNDLE" || { echo "PRECONDITION FAILED: bundle missing"; exit 1; }
    echo "0e255243d8cca981a653c4fd42b38beeeb163cee924886fa166b972d585d8c51  $BUNDLE" | sha256sum -c - || { echo "BUNDLE HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$W" "$CK"
    if [ -f "$W/metrics.jsonl" ] && ! head -1 "$W/metrics.jsonl" | grep -q "\"attempt_id\": \"a2\""; then echo "STALE RUN DIR (other attempt) in $W"; exit 1; fi
    if [ -f "$CK/LATEST.json" ] && [ ! -f "$W/metrics.jsonl" ]; then echo "STALE CHECKPOINTS in $CK"; exit 1; fi
'
phase config "false" '
    cat > "$W/config.json" <<JSON
{"run_id": "$RUN", "attempt_id": "a2", "phase": "primary", "arm": "R", "seed": 202,
 "lr": 2e-6, "eps_low": 0.2, "eps_high": 0.2, "rounds": 128, "ckpt_every": 16,
 "eval_rounds": [0, 32, 64, 96, 128], "eval_panels": {"dev": 8, "probe_J_R": 8, "probe_J_U": 8},
 "retain_rounds": [],
 "bundle": "$BUNDLE", "model": "$MODEL", "max_tokens": 12288, "cap_schedule": [[0, 8192], [32, 12288]], "gpu_mem_util": 0.30,
 "out_dir": "$W", "ckpt_dir": "$CK", "ckpt_lock_root": "$PROJ_DATA/ckpt",
 "wandb": {"enabled": true, "project": "qis-qwen-interference"}}
JSON
'
phase boundary_eval "test ! -d $CK/boundary_0064 -o -s $W/boundary_eval/summary.json" '
    # A10: dev panel at the amended cap from the pre-switch checkpoint (both budgets + throughput); no training effect
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CK/boundary_0064" --panel "$BUNDLE" --panel-key dev --n 32 --out "$W/boundary_eval" --seed-label boundary-r64 --max-tokens 16384
'
phase train "test -f $W/RUN_COMPLETE" '
    rc=0; "$VENV/bin/python" -m qis.train.run --config "$W/config.json" >> "$W/stdout.log" 2>&1 || rc=$?
    tail -20 "$W/stdout.log"
    if [ $rc -eq 143 ]; then scontrol requeue $SLURM_JOB_ID; sleep 900; fi
    exit $rc
'
phase export "test -s $AE_RESULTS/metrics.jsonl" '
    cp "$W/metrics.jsonl" "$W/config.json" "$AE_RESULTS/"; [ -s "$W/boundary_eval/summary.json" ] && cp "$W/boundary_eval/summary.json" "$AE_RESULTS/boundary_eval_summary.json"; true
'
