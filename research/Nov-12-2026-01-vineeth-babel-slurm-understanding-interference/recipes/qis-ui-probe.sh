#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=120G time=08:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS (domain study): candidate domains differ in (1) base success of Qwen3-1.7B (some / marginal / none) and (2) the
# sign of their gradient coupling with the hard-algebra domain at the base model. A domain with a negative mean cosine
# against hard algebra, or with zero success, gives a precondition of ray interference; a domain with a positive cosine
# does not. 16 base answers per problem; per-group GRPO gradients as count sketches, compared with the stored sketches of
# qis-ui-geom-base64 (same sketch seed). Sampling and backward passes only; no weight changes.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export UI="$PROJ_DATA/ui"
export PANEL="$UI/probe/panel_probe_v1.json"
export S="$UI/probe/samples"
export A="$UI/grad/base64"
export O="$UI/probe/grad"
export POOLS="mult2 mult3 mult4 mult5 mult6 add10 add9 mult10s mult9 gsm8k aime24 arc"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    test -f "$A/DONE"; test -s "$A/sk_global.npy"
    mkdir -p "$S" "$O"
'
phase panel "test -s $PANEL" '
    HF_HUB_OFFLINE=0 python -m qis.interf.build_probe_panel --out "$PANEL.tmp" --n 100
    mv "$PANEL.tmp" "$PANEL"
'
for pool in $POOLS; do
    export POOL=$pool
    phase "sample-$pool" "test -s $S/$pool/summary.json -o -f $S/$pool/ABSENT" '
        export HF_HUB_OFFLINE=1
        if python -c "import json,sys; sys.exit(0 if \"$POOL\" in json.load(open(\"$PANEL\"))[\"pools\"] else 1)"; then
            python -m qis.evaluate_ckpt --model "$MODEL" --ckpt base --panel "$PANEL" --panel-key "$POOL" --n 16 --out "$S/$POOL" --seed-label ui-probe-v1
        else
            mkdir -p "$S/$POOL"; touch "$S/$POOL/ABSENT"; echo "pool $POOL is not in the panel"
        fi
    '
done
phase export-samples "false" '
    python -c "import json; d=json.load(open(\"$PANEL\")); json.dump({\"notes\": d[\"notes\"], \"examples\": {k: v[0] for k, v in d[\"pools\"].items()}}, open(\"$AE_RESULTS/panel_info.json\", \"w\"), indent=1)"
    for p in $POOLS; do [ -s "$S/$p/summary.json" ] && cp "$S/$p/summary.json" "$AE_RESULTS/sample_$p.json"; done; true
'
phase grad "test -f $O/DONE" '
    export HF_HUB_OFFLINE=1
    SETS=""
    for p in $POOLS; do [ -s "$S/$p/summary.json" ] && SETS="$SETS --set $p=$S/$p/records.jsonl.gz:$PANEL:$p"; done
    python -m qis.interf.grad_geom --model "$MODEL" --ckpt base --out "$O" $SETS
'
phase summ "test -s $AE_RESULTS/summary.json" '
    python -m qis.interf.summ_reward --a "$A" --b "$O" --out "$O/summ"
    cp "$O/summ/summary.json" "$AE_RESULTS/"
'
