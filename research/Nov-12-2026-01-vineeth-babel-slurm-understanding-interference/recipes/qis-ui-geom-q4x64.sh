#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=120G time=08:00:00 part=general constraint=VRAM_80GB|VRAM_96GB
# HYPOTHESIS (scale): per-domain GRPO gradients of the untrained Qwen/Qwen3-4B on the same four problem pools, with its own
# on-policy answers. Does the cross-domain gradient cosine change with model size at initialisation? No training.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export CONTROL="$PWD/data/Nov-12-2026-00-vineeth-babel-slurm-initial-study/control/control_v1.json"
export UI="$PROJ_DATA/ui"
export REPO="Qwen/Qwen3-4B"
export MP="$UI/models/q4.path"
export S="$UI/samples/q4x64"
export O="$UI/grad/q4x64"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    test -s "$BUNDLE"; test -s "$CONTROL"; mkdir -p "$S" "$O" "$UI/models"
'
phase download "test -s $MP" '
    HF_HUB_OFFLINE=0 python -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download(os.environ[\"REPO\"], allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"*.jinja\"])
open(os.environ[\"MP\"] + \".tmp\", \"w\").write(p)
"
    mv "$MP.tmp" "$MP"; cat "$MP"; du -sh "$(cat "$MP")/" ; cp "$MP" "$AE_RESULTS/model_path.txt"
'
export HF_HUB_OFFLINE=1
for pool in hard related unrelated control; do
    export POOL=$pool
    phase "sample-$pool" "test -s $S/$pool/summary.json" '
        MODEL=$(cat "$MP")
        if [ "$POOL" = control ]; then PANEL="$CONTROL"; else PANEL="$BUNDLE"; fi
        python -m qis.evaluate_ckpt --model "$MODEL" --ckpt base --panel "$PANEL" --panel-key "$POOL" --n 64 --out "$S/$POOL" --seed-label ui-grad-v2
    '
done
phase grad "test -f $O/DONE" '
    MODEL=$(cat "$MP")
    python -m qis.interf.grad_geom --model "$MODEL" --ckpt base --out "$O" \
        --set "hard=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related=$S/related/records.jsonl.gz:$BUNDLE:related" \
        --set "unrelated=$S/unrelated/records.jsonl.gz:$BUNDLE:unrelated" --set "control=$S/control/records.jsonl.gz:$CONTROL:control"
'
phase summ "test -s $AE_RESULTS/summary.json" '
    python -m qis.interf.summ --grad "$O" --out "$O/summ"
    cp "$O/summ/summary.json" "$O/summ/small.npz" "$O/summ/groups_slim.json" "$AE_RESULTS/"
    for p in hard related unrelated control; do cp "$S/$p/summary.json" "$AE_RESULTS/sample_$p.json"; done
'
