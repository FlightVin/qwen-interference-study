#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=120G time=07:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS (domain study, negative coupling): a brevity reward (1 for the shorter half of the 8 answers of a group) pulls
# the base model against the correctness reward. If so, the cosine between the brevity gradient and the correctness
# gradient is negative, for the same problem (independent samples) and across problems. If the two rewards are unrelated,
# it is zero within its interval. Uses the stored 64-answer base samples and the stored correctness sketches of
# qis-ui-geom-base64 (same sketch seed). Backward passes only; no weight changes.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export CONTROL="$PWD/data/Nov-12-2026-00-vineeth-babel-slurm-initial-study/control/control_v1.json"
export UI="$PROJ_DATA/ui"
export S="$UI/samples/base64"
export A="$UI/grad/base64"
export O="$UI/grad/base64-short"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    for p in hard related unrelated control; do test -s "$S/$p/records.jsonl.gz"; done
    test -f "$A/DONE"; test -s "$A/sk_global.npy"; test -s "$A/groups.jsonl"
    mkdir -p "$O"
'
phase selftest "false" '
    T="$AE_RUNDIR/selftest"; rm -rf "$T"; mkdir -p "$T/a" "$T/b"
    python - "$T" <<PY
import json, sys, numpy as np
T = sys.argv[1]; rng = np.random.default_rng(1); d = 4096
# planted structure: b gradient = -0.5 * (problem signal of a) + noise, so cross cosines must be negative
sig = rng.normal(size=(30, d))
for name, sign in (("a", 1.0), ("b", -0.5)):
    rows, X = [], []
    for p in range(30):
        for h in range(4):
            X.append(sign * sig[p] + 3.0 * rng.normal(size=d)); rows.append({"label": "hard" if name == "a" else "hard~short", "prompt_id": f"p{p}", "half": h, "row": len(X) - 1})
    np.save(f"{T}/{name}/sk_global.npy", np.array(X, dtype=np.float32))
    open(f"{T}/{name}/groups.jsonl", "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
PY
    python -m qis.interf.summ_reward --a "$T/a" --b "$T/b" --out "$T/out" --boot 100
    python - "$T/out/summary.json" <<PY
import json, sys
r = json.load(open(sys.argv[1])); c = r["cross"]["hard|hard~short"]
sp, dp = c["same_problem"]["mean_cos"], c["diff_problem"]["mean_cos"]
exp = -0.5 * 4096 / ((4096 + 9 * 4096) ** 0.5 * (0.25 * 4096 + 9 * 4096) ** 0.5)
print("selftest same_problem", sp, "expected", exp, "diff_problem", dp)
assert abs(sp - exp) < 0.01 and abs(dp) < 0.005 and c["ci95_mean_cos"]["same_problem"][1] < 0, "summ_reward selftest failed"
assert c["same_problem"]["n_pairs"] == 30 * 12 and c["same_group"]["n_pairs"] == 30 * 4
PY
'
phase grad "test -f $O/DONE" '
    python -m qis.interf.grad_geom --model "$MODEL" --ckpt base --out "$O" \
        --set "hard~short=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related~short=$S/related/records.jsonl.gz:$BUNDLE:related" \
        --set "unrelated~short=$S/unrelated/records.jsonl.gz:$BUNDLE:unrelated" --set "control~short=$S/control/records.jsonl.gz:$CONTROL:control"
'
phase summ "test -s $AE_RESULTS/summary.json" '
    python -m qis.interf.summ_reward --a "$A" --b "$O" --out "$O/summ"
    cp "$O/summ/summary.json" "$AE_RESULTS/"
'
