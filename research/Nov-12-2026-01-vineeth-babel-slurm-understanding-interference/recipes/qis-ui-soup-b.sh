#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=110G time=10:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: the six runs' weight updates have pairwise cosine 0.03-0.05 (qis-ui-wdelta). If the performance gain sits in
# their small common component, the mean of the six updates (norm about 0.45x) keeps the gain, and one run with the mean
# of the other five removed loses it. Evaluation only: dev panel (53 problems), 32 answers, same seeds for every model.
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
export CK="$PROJ_DATA/ckpt"
export SO="$PROJ_DATA/ui/soup"
export EV="$PROJ_DATA/ui/soup-eval"

phase precheck "false" '
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    mkdir -p "$SO" "$EV"; df -h /data/user_data/$USER | tail -1
'
for m in resid-R101 scaled-R101 loo5 mean6x; do
    export M=$m
    phase "eval-$m" "test -s $EV/$m/summary.json" '
        case "$M" in base) C=base;; R-s101-r128) C="$CK/qis-main-R-s101-sym/round_0128";; *) C="$SO/$M";; esac
        python -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$C" --panel "$BUNDLE" --panel-key dev --n 32 --out "$EV/$M" --seed-label ui-soup-v1
        cp "$EV/$M/summary.json" "$AE_RESULTS/summary_$M.json"
    '
done
phase export "false" '
    for f in "$EV"/*/summary.json; do m=$(basename "$(dirname "$f")"); cp "$f" "$AE_RESULTS/summary_$m.json"; done; ls "$AE_RESULTS"
'
