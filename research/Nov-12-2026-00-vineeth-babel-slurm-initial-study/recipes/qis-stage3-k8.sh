#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=preempt constraint=VRAM_48GB after=qis-bundle-main
# HYPOTHESIS: none -- independent stage-3 baseline (16 fresh samples, stream s3k8, cap 8,192) for the
# selected H_train, E_related, E_unrelated; reported, never used to reselect (A4.2, project.md §5).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export B="$PROJ_DATA/data/main-v1"
export S3="$PROJ_DATA/screen/s3k8"
phase precheck "false" '
    test -s "$B/bundle_main_v1.json" || { echo "PRECONDITION FAILED: bundle missing"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
'
phase items "test -s $S3/unrelated.jsonl" '
    mkdir -p "$S3"
    "$VENV/bin/python" -c "
import json
b = json.load(open(\"$B/bundle_main_v1.json\"))[\"pools\"]
for k in (\"hard\", \"related\", \"unrelated\"):
    with open(\"$S3/\" + k + \".jsonl\", \"w\") as f:
        for r in b[k]: f.write(json.dumps({\"id\": r[\"id\"], \"problem\": r[\"problem\"], \"gold\": r[\"gold\"]}) + \"\n\")
"
'
phase screen "test -f $S3/DONE" '
    for p in hard related unrelated; do
        "$VENV/bin/python" -m qis.screen --items "$S3/$p.jsonl" --stage s3k8 --n 16 --out "$S3/$p" --model "$MODEL" --max-tokens 8192
    done
    touch "$S3/DONE"
'
phase export "test -s $AE_RESULTS/exported" '
    for p in hard related unrelated; do cp "$S3/$p/s3k8_counts.jsonl" "$AE_RESULTS/${p}_s3k8_counts.jsonl"; done
    "$VENV/bin/python" - <<PY > "$AE_RESULTS/mixed_groups.json"
import gzip, json, glob
out = {}
for p in ("hard", "related", "unrelated"):
    groups = mixed = 0
    for f in sorted(glob.glob("$S3/" + p + "/s3k8_records_*.jsonl.gz")):
        by = {}
        for l in gzip.open(f, "rt"):
            r = json.loads(l); by.setdefault(r["id"], []).append((r["sample"], r["reward"]))
        for i, rs in by.items():
            rs = [x[1] for x in sorted(rs)]
            for g in (rs[:8], rs[8:16]):
                groups += 1; mixed += (0 < sum(g) < 8)
    out[p] = {"groups": groups, "mixed": mixed, "fraction": mixed / max(1, groups)}
print(json.dumps(out, indent=1))
PY
    touch "$AE_RESULTS/exported"
'
