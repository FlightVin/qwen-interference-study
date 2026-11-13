#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=03:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: none -- canary of the study Nov-12-2026-01 measurement pipeline. Sample 16 answers per problem of the
# related pool from the base model, then run grad_geom on 6 problems and logp_path on 3 problems with one checkpoint.
# Pass: finite gradients, sketch inner products agree with exact norms, lp.npz has 2 conditions.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export CONTROL="$PWD/data/Nov-12-2026-00-vineeth-babel-slurm-initial-study/control/control_v1.json"
export CK="$PROJ_DATA/ckpt"
export UI="$PROJ_DATA/ui"
export S="$UI/samples/base/related"
export O="$UI/canary"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    test -s "$BUNDLE"; test -s "$CONTROL"; mkdir -p "$S" "$O"
'
phase sample "test -s $S/summary.json" '
    python -m qis.evaluate_ckpt --model "$MODEL" --ckpt base --panel "$BUNDLE" --panel-key related --n 16 --out "$S" --seed-label ui-grad-v1
'
phase grad "test -f $O/grad/DONE" '
    python -m qis.interf.grad_geom --model "$MODEL" --ckpt base --out "$O/grad" --set "related=$S/records.jsonl.gz:$BUNDLE:related" --max-prompts 6
'
phase lp "test -f $O/lp/DONE" '
    python -m qis.interf.logp_path --model "$MODEL" --out "$O/lp" --set "related=$S/records.jsonl.gz:$BUNDLE:related" --ckpt "R-s101-r128=$CK/qis-main-R-s101-sym/round_0128" --alphas 1 --max-prompts 3
'
phase export "false" '
    cp "$S/summary.json" "$AE_RESULTS/sample_summary.json"
    cp "$O/grad/groups.jsonl" "$O/lp/meta.json" "$AE_RESULTS/"
    python - <<PY
import numpy as np, json, os
o=os.environ["O"]; g=np.load(o+"/grad/sk_global.npy"); q=np.load(o+"/grad/sq.npy"); m=json.load(open(o+"/grad/meta.json"))
nt=len(m["types"]); ex=q[:,:nt].sum(1); sk=(g.astype(np.float64)**2).sum(1)
G=g.astype(np.float64)@g.astype(np.float64).T; d=np.sqrt(np.diag(G))
z=np.load(o+"/lp/lp.npz")
json.dump({"exact_sq":ex.tolist(),"sketch_sq":sk.tolist(),"cos":(G/np.outer(d,d)).round(4).tolist(),
           "sq_type_mean":dict(zip(m["types"],q[:,:nt].mean(0).tolist())),
           "conds":[str(c) for c in z["conds"]],"lp_mean":z["lp"].mean(1).tolist(),"lp":z["lp"].round(5).tolist()},
          open(os.environ["AE_RESULTS"]+"/canary.json","w"))
PY
'
