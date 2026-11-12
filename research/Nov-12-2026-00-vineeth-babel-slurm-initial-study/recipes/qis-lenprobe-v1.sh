#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=02:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- length probe (project.md Section 6). Cap-hit rate at 4096 tokens on 64
# construction hard-like items (16 samples each); the same items at 8192 to see whether the
# larger cap resolves incomplete attempts. Development only: separate stream names from screening.
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
export W="$PROJ_DATA/runs/lenprobe-v1"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$W"
'
phase items "test -s $W/items.jsonl" '
    "$VENV/bin/python" - <<PY
import json, re
from qis import seeds
from qis.data.build import FACTOR_RE
rows = []
for line in open("$MANIFEST"):
    r = json.loads(line)
    if r["split"] == "construction" and r.get("gold_eligible") and r["subject"] in ("algebra", "intermediate_algebra") \
       and r["level"] in ("Level 4", "Level 5") and FACTOR_RE.search(r["solution"]):
        rows.append({"id": r["id"], "problem": r["problem"], "gold": r["gold"]})
rows.sort(key=lambda x: x["id"])
perm = seeds.rng(1701, "lenprobe/order").permutation(len(rows))
with open("$W/items.jsonl", "w") as f:
    for i in perm[:64]:
        f.write(json.dumps(rows[i]) + "\n")
print(len(rows), "eligible; wrote 64")
PY
'
phase cap4k "test -s $W/cap4096/lenprobe_over_cap.json" '
    "$VENV/bin/python" -m qis.screen --items "$W/items.jsonl" --stage lenprobe --n 16 --out "$W/cap4096" --model "$MODEL" --max-tokens 4096
'
phase cap8k "test -s $W/cap8192/lenprobe_over_cap.json" '
    "$VENV/bin/python" -m qis.screen --items "$W/items.jsonl" --stage lenprobe --n 16 --out "$W/cap8192" --model "$MODEL" --max-tokens 8192
'
phase summary "test -s $AE_RESULTS/lenprobe_summary.json" '
    cp "$W/cap4096/lenprobe_counts.jsonl" "$AE_RESULTS/counts_4096.jsonl"
    cp "$W/cap8192/lenprobe_counts.jsonl" "$AE_RESULTS/counts_8192.jsonl"
    "$VENV/bin/python" - <<PY
import json
out = {}
for cap in (4096, 8192):
    rows = [json.loads(l) for l in open(f"$W/cap{cap}/lenprobe_counts.jsonl")]
    n = sum(r["n"] for r in rows)
    out[cap] = {"items": len(rows), "attempts": n, "success": sum(r["c"] for r in rows) / n,
                "truncated_frac": sum(r["truncated"] for r in rows) / n,
                "truncated_would_match": sum(r["truncated_would_match"] for r in rows),
                "no_answer_frac": sum(r["no_answer"] for r in rows) / n,
                "unknown": sum(r["unknown"] for r in rows),
                "mean_len": sum(r["resp_len_mean"] for r in rows) / len(rows)}
json.dump(out, open("$AE_RESULTS/lenprobe_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
'
