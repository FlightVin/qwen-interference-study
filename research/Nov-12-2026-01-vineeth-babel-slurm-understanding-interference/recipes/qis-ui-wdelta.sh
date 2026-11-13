#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=120G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: weight updates of the six runs (round 128) and two round-64 checkpoints: how similar are their directions,
# where in the network do they sit, what rank do they have, how many BF16 values changed. Read-only on checkpoints.
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
export O="$UI/wdelta"

phase precheck "false" '
    for r in H-s101 H-s202 R-s101 R-s202 U-s101 U-s202; do test -f "$CK/qis-main-$r-sym/round_0128/model.pt"; done
    test -f "$CK/qis-main-R-s101-sym/round_0064/model.pt"; test -f "$CK/qis-main-H-s202-sym/round_0064/model.pt"
    mkdir -p "$O"; free -g | head -2
'
phase wdelta "test -s $O/wdelta.json" '
    python -m qis.interf.wdelta --model "$MODEL" --out "$O" \
        --ckpt "H-s101-r128=$CK/qis-main-H-s101-sym/round_0128" --ckpt "R-s101-r128=$CK/qis-main-R-s101-sym/round_0128" \
        --ckpt "U-s101-r128=$CK/qis-main-U-s101-sym/round_0128" --ckpt "H-s202-r128=$CK/qis-main-H-s202-sym/round_0128" \
        --ckpt "R-s202-r128=$CK/qis-main-R-s202-sym/round_0128" --ckpt "U-s202-r128=$CK/qis-main-U-s202-sym/round_0128" \
        --ckpt "R-s101-r64=$CK/qis-main-R-s101-sym/round_0064" --ckpt "H-s202-r64=$CK/qis-main-H-s202-sym/round_0064"
'
phase export "test -s $AE_RESULTS/wdelta.json.gz" '
    python - <<PY
import json, gzip, os
d=json.load(open(os.environ["O"]+"/wdelta.json"))
def rnd(x):
    if isinstance(x,float): return float("%.5g"%x)
    if isinstance(x,list): return [rnd(v) for v in x]
    if isinstance(x,dict): return {k:rnd(v) for k,v in x.items()}
    return x
for t in d["tensors"]:
    t.pop("sv_top",None) if t.get("layer") not in ("0","13","27") else None
gzip.open(os.environ["AE_RESULTS"]+"/wdelta.json.gz","wt").write(json.dumps(rnd(d)))
PY
    ls -la "$AE_RESULTS"
'
