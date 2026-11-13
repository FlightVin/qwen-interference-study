#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=04:00:00 part=general constraint=VRAM_48GB
# HYPOTHESIS (precision): the change in log-probability is far from proportional to alpha (qis-ui-lp). If rounding of the
# FP32 weights to BF16 in the forward pass causes this, FP32 compute is close to linear and FP32 compute with BF16-rounded
# weights reproduces the BF16 result. Forward passes only. Earlier text:
# HYPOTHESIS: (1) linearity: is the change in log-probability of fixed base-policy answers proportional to alpha along
# theta_base + alpha (theta_run - theta_base)? (2) surrogate transfer: did each run move probability towards correct answers
# of domains it never trained on? Forward passes only, on the base samples of qis-ui-geom-base.
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
export O="$PROJ_DATA/ui/lp-prec"
export SO="$PROJ_DATA/ui/soup"

phase precheck "false" '
    for p in hard related unrelated control; do test -s "$S/$p/summary.json"; done
    test -s "$SO/soups.json"
    mkdir -p "$O"; nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
for P in fp32 fp32-wround train; do
    export P
    phase "lp-$P" "test -f $O/$P/DONE" '
        python -m qis.interf.logp_path --model "$MODEL" --out "$O/$P" --alphas 0.25,0.5,1,2 --samples 8 --precision "$P" \
            --set "hard=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related=$S/related/records.jsonl.gz:$BUNDLE:related" \
            --ckpt "R-s101-r128=$CK/qis-main-R-s101-sym/round_0128" --ckpt "mean6=$SO/mean6"
        mkdir -p "$AE_RESULTS/$P"; cp "$O/$P/lp.npz" "$O/$P/meta.json" "$AE_RESULTS/$P/"
    '
done
