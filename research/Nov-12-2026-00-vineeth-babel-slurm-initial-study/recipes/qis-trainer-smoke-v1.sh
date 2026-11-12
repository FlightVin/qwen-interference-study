#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- engineering gate G2. (1) precision audit (addendum A1); (2) 4-round R-arm
# run, uninterrupted (A); (3) same run with forced exit after round 2 and resume (B); compare.
# Development-only smoke bundle from the construction split. Disposable; not a study run.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export MANIFEST="$PROJ_DATA/data/splits-v1/manifest.jsonl"
export W="$PROJ_DATA/runs/smoke-v1"
export CK="$PROJ_DATA/ckpt/smoke-v1"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$W" "$CK"
'
phase bundle "test -s $W/bundle.json" '
    "$VENV/bin/python" -m qis.data.smoke_bundle --manifest "$MANIFEST" --out "$W/bundle.json" --n 16
    "$VENV/bin/python" -c "import json; b=json.load(open(\"$W/bundle.json\")); [print(json.dumps(x)) for x in b[\"pools\"][\"hard\"][:8]]" > "$W/audit_problems.jsonl"
'
phase audit "test -s $AE_RESULTS/precision_audit.json" '
    "$VENV/bin/python" -m qis.train.precision_audit --model "$MODEL" --problems "$W/audit_problems.jsonl" \
        --out "$AE_RESULTS" --ckpt-dir "$CK/audit" 2>&1 | tail -60
    rm -rf "$CK/audit"
'
phase configs "test -s $W/cfgB.json" '
    for x in A B; do cat > "$W/cfg$x.json" <<JSON
{"run_id": "smoke-v1-$x", "attempt_id": "a1", "phase": "smoke", "arm": "R", "seed": 101,
 "lr": 1e-6, "eps_low": 0.2, "eps_high": 0.28, "rounds": 4, "ckpt_every": 2,
 "eval_rounds": [0, 4], "eval_panels": {"dev_smoke": 8},
 "bundle": "$W/bundle.json", "model": "$MODEL", "max_tokens": 4096, "gpu_mem_util": 0.30,
 "out_dir": "$W/run$x", "ckpt_dir": "$CK/run$x", "wandb": {"enabled": false}}
JSON
    done
'
phase runA "test -f $W/runA/RUN_COMPLETE" '
    "$VENV/bin/python" -m qis.train.run --config "$W/cfgA.json" 2>&1 | tail -30
'
phase runB "test -f $W/runB/RUN_COMPLETE" '
    rc=0; QIS_EXIT_AFTER_ROUND=2 "$VENV/bin/python" -m qis.train.run --config "$W/cfgB.json" 2>&1 | tail -5 || rc=$?
    echo "first invocation rc=$rc (expect 143)"
    test -f "$W/runB/FORCED_EXIT_DONE" || { echo "forced exit did not happen"; exit 1; }
    QIS_EXIT_AFTER_ROUND=2 "$VENV/bin/python" -m qis.train.run --config "$W/cfgB.json" 2>&1 | tail -10
'
phase compare "test -s $AE_RESULTS/resume_compare.json" '
    "$VENV/bin/python" -m qis.train.compare_runs "$W/runA" "$W/runB" "$CK/runA/round_0004" "$CK/runB/round_0004" "$AE_RESULTS/resume_compare.json"
    cp "$W/runA/metrics.jsonl" "$AE_RESULTS/metrics_A.jsonl"; cp "$W/runB/metrics.jsonl" "$AE_RESULTS/metrics_B.jsonl"
    du -sh "$CK/runA/round_0004" > "$AE_RESULTS/ckpt_size.txt"
'
