#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=06:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: none -- external H tranche (D36, A4.3). Intake checks (verifier gold, prompt cap,
# near-duplicates vs all MATH) then 32 attempts per eligible item (stream s2), frozen decoding, cap 4096.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export O="$PROJ_DATA/data/omni-v1"
export S2="$PROJ_DATA/screen/s2"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$O"
'
phase intake "test -s $O/intake_summary.json" '
    "$VENV/bin/python" - <<PY
import json
with open("$O/as_authored.jsonl", "w") as f:
    for l in open("data/external/omni_tranche_v1.jsonl"):
        r = json.loads(l)
        f.write(json.dumps({"id": r["id"], "arm": "omni_hard", "template_family": r["source"], "problem": r["problem"], "answer": r["gold"]}) + "\n")
PY
    "$VENV/bin/python" -m qis.data.authored --authored "$O/as_authored.jsonl" --manifest "$PROJ_DATA/data/splits-v1/manifest.jsonl" --model "$MODEL" --out "$O"
'
phase screen "test -f $S2/DONE_omni" '
    "$VENV/bin/python" -m qis.screen --items "$O/omni_hard.jsonl" --stage s2 --n 32 --out "$S2/omni_hard" --model "$MODEL" --max-tokens 4096
    touch "$S2/DONE_omni"
'
phase export "test -s $AE_RESULTS/exported" '
    cp "$O/intake_summary.json" "$O/intake_report.jsonl" "$AE_RESULTS/"
    cp "$S2/omni_hard/s2_counts.jsonl" "$AE_RESULTS/omni_hard_s2_counts.jsonl"
    touch "$AE_RESULTS/exported"
'
