#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=01:00:00 part=general constraint=VRAM_48GB after=qis-env-v1b
# HYPOTHESIS: none -- engineering probe. Non-thinking template fixture, vLLM generation under the
# study decoding, rollout vs teacher-forced logprob agreement, one full-parameter AdamW step
# (32 responses, micro-batch 1) with peak memory, weight sync to vLLM, checkpoint save/reload.
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

phase precheck "false" '
    test -s "$MANIFEST" || { echo "PRECONDITION FAILED: no manifest"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase probe "test -s $AE_RESULTS/probe.json" '
    "$VENV/bin/python" - <<PY > "$AE_RUNDIR/probe_problems.jsonl"
import json
n = 0
for line in open("$MANIFEST"):
    r = json.loads(line)
    if r["split"] == "construction" and r["subject"] == "intermediate_algebra" and r["level"] == "Level 4" and r.get("gold_eligible"):
        print(json.dumps({"id": r["id"], "problem": r["problem"], "gold": r["gold"]})); n += 1
    if n == 4: break
PY
    ls -laL "$MODEL" > "$AE_RESULTS/model_files.txt"
    "$VENV/bin/python" -m qis.probe_gpu --model "$MODEL" --problems "$AE_RUNDIR/probe_problems.jsonl" \
        --out "$AE_RESULTS" --ckpt-dir "$PROJ_DATA/ckpt/probe" 2>&1 | tail -80
    rm -f "$PROJ_DATA/ckpt/probe/probe.pt"
'
