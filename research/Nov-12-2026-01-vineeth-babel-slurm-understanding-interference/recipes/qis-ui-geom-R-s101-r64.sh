#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=08:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS: per-domain GRPO gradients at checkpoint R-s101-r64. Do gradients of the hard, related, unrelated and control domains
# conflict (negative cosine), align, or look independent, compared with gradients of one domain and of one problem?
# 16 on-policy answers per problem (two groups of 8); count sketches of every mixed group's gradient. No training.
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
export UI="$PROJ_DATA/ui"
export LABEL="R-s101-r64"
export CKPT="$PROJ_DATA/ckpt/qis-main-R-s101-sym/round_0064"
export OPTIM="yes"
export S="$UI/samples/R-s101-r64"
export O="$UI/grad/R-s101-r64"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    test -s "$BUNDLE"; test -s "$CONTROL"
    [ "$CKPT" = base ] || test -f "$CKPT/model.pt"
    [ "$OPTIM" = no ] || test -f "$CKPT/optim.pt"
    mkdir -p "$S" "$O"
'
for pool in hard related unrelated control; do
    export POOL=$pool
    phase "sample-$pool" "test -s $S/$pool/summary.json" '
        if [ "$POOL" = control ]; then PANEL="$CONTROL"; else PANEL="$BUNDLE"; fi
        python -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$PANEL" --panel-key "$POOL" --n 16 --out "$S/$POOL" --seed-label ui-grad-v1
    '
done
phase grad "test -f $O/DONE" '
    OPT=""; [ "$OPTIM" = no ] || OPT="--optim $CKPT/optim.pt"
    python -m qis.interf.grad_geom --model "$MODEL" --ckpt "$CKPT" --out "$O" $OPT \
        --set "hard=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related=$S/related/records.jsonl.gz:$BUNDLE:related" \
        --set "unrelated=$S/unrelated/records.jsonl.gz:$BUNDLE:unrelated" --set "control=$S/control/records.jsonl.gz:$CONTROL:control"
'
phase summ "test -s $AE_RESULTS/summary.json" '
    DS=""; [ -f "$UI/wdelta/sk_delta.npy" ] && DS="--delta-sketch $UI/wdelta/sk_delta.npy"
    python -m qis.interf.summ --grad "$O" --out "$O/summ" $DS
    cp "$O/summ/summary.json" "$O/summ/small.npz" "$O/summ/groups_slim.json" "$AE_RESULTS/"
    for p in hard related unrelated control; do cp "$S/$p/summary.json" "$AE_RESULTS/sample_$p.json"; done
    ls -la "$AE_RESULTS"
'
