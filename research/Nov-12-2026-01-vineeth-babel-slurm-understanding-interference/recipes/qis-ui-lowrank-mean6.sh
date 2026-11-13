#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS (capacity demand): if the learned effect of an update sits in a few directions per weight matrix, the rank-r
# part of the update (r = 1, 8, 64) reproduces the surrogate transfer S_D of the full update and the remainder has none.
# If the effect needs the full-rank update, the rank-r part gives only its norm share. Forward passes only, FP32 weights
# and FP32 compute, on the base samples of qis-ui-geom-base. Update: mean6.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export CONTROL="$PWD/data/Nov-12-2026-00-vineeth-babel-slurm-initial-study/control/control_v1.json"
export CK="$PROJ_DATA/ckpt"
export S="$PROJ_DATA/ui/samples/base"
export O="$PROJ_DATA/ui/lowrank/mean6"
export SO="$PROJ_DATA/ui/soup"

phase precheck "false" '
    for p in hard related unrelated control; do test -s "$S/$p/summary.json"; done
    test -s "$SO/soups.json"
    mkdir -p "$O"; nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase lowrank "test -f $O/DONE" '
    python -m qis.interf.lowrank_path --model "$MODEL" --out "$O" --ranks 1,8,64 --samples 8 \
        --set "hard=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related=$S/related/records.jsonl.gz:$BUNDLE:related" \
        --set "unrelated=$S/unrelated/records.jsonl.gz:$BUNDLE:unrelated" --set "control=$S/control/records.jsonl.gz:$CONTROL:control" \
        --ckpt "mean6=$SO/mean6"
'
phase export "false" '
    cp "$O/lp.npz" "$O/meta.json" "$O/shares.json" "$AE_RESULTS/"
'
