#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- dataset construction (D32). Authored auxiliary candidates: stage 1 (s1, 16),
# shortlist 4-14/16, stage 2 (s2, 32); same model, prompt, decoding, cap 4096 and verifier.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export A="$PROJ_DATA/data/authored-v1"
export C="$PROJ_DATA/data/candidates-authored-v1"
export S1="$PROJ_DATA/screen/s1"
export SL="$PROJ_DATA/data/shortlist-authored-v1"
export S2="$PROJ_DATA/screen/s2"

phase precheck "false" '
    test -s "$A/intake_summary.json" || { echo "PRECONDITION FAILED: intake incomplete"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$C"
'
phase order "test -s $C/auth_unrelated.jsonl" '
    "$VENV/bin/python" - <<PY
import json
from qis import seeds
for arm, name in (("related", "auth_related"), ("unrelated", "auth_unrelated")):
    rows = sorted((json.loads(l) for l in open("$A/" + arm + ".jsonl")), key=lambda r: r["id"])
    perm = seeds.rng(seeds.CONSTRUCTION_SEED, "candidates/" + name).permutation(len(rows))
    with open("$C/" + name + ".jsonl", "w") as f:
        for i in perm:
            f.write(json.dumps(rows[i]) + "\n")
    print(name, len(rows))
PY
'
phase stage1 "test -f $S1/DONE_authored" '
    for p in auth_related auth_unrelated; do
        "$VENV/bin/python" -m qis.screen --items "$C/$p.jsonl" --stage s1 --n 16 --out "$S1/$p" --model "$MODEL" --max-tokens 4096
    done
    touch "$S1/DONE_authored"
'
phase shortlist "test -s $SL/shortlist_summary.json" '
    "$VENV/bin/python" -m qis.data.shortlist --cand "$C" --s1 "$S1" --out "$SL" --pools auth_related auth_unrelated
'
phase stage2 "test -f $S2/DONE_authored" '
    for p in auth_related auth_unrelated; do
        "$VENV/bin/python" -m qis.screen --items "$SL/$p.jsonl" --stage s2 --n 32 --out "$S2/$p" --model "$MODEL" --max-tokens 4096
    done
    touch "$S2/DONE_authored"
'
phase export "test -s $AE_RESULTS/exported" '
    cp "$SL/shortlist_summary.json" "$AE_RESULTS/"
    for p in auth_related auth_unrelated; do cp "$S1/$p/s1_counts.jsonl" "$AE_RESULTS/${p}_s1_counts.jsonl"; cp "$S2/$p/s2_counts.jsonl" "$AE_RESULTS/${p}_s2_counts.jsonl"; done
    touch "$AE_RESULTS/exported"
'
