#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=10:00:00 part=general constraint=VRAM_48GB
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
export O="$PROJ_DATA/ui/lp"
export SO="$PROJ_DATA/ui/soup"

phase precheck "false" '
    for p in hard related unrelated control; do test -s "$S/$p/summary.json"; done
    test -s "$SO/soups.json"
    mkdir -p "$O"; nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
'
phase lp "test -f $O/DONE" '
    python -m qis.interf.logp_path --model "$MODEL" --out "$O" --alphas 0.5,1,2 --samples 8 \
        --set "hard=$S/hard/records.jsonl.gz:$BUNDLE:hard" --set "related=$S/related/records.jsonl.gz:$BUNDLE:related" \
        --set "unrelated=$S/unrelated/records.jsonl.gz:$BUNDLE:unrelated" --set "control=$S/control/records.jsonl.gz:$CONTROL:control" \
        --ckpt "R-s101-r128=$CK/qis-main-R-s101-sym/round_0128" --ckpt "U-s101-r128=$CK/qis-main-U-s101-sym/round_0128" \
        --ckpt "H-s101-r128=$CK/qis-main-H-s101-sym/round_0128" --ckpt "R-s202-r128=$CK/qis-main-R-s202-sym/round_0128" \
        --ckpt "U-s202-r128=$CK/qis-main-U-s202-sym/round_0128" --ckpt "H-s202-r128=$CK/qis-main-H-s202-sym/round_0128" \
        --ckpt "R-s101-r64=$CK/qis-main-R-s101-sym/round_0064" --ckpt "H-s202-r64=$CK/qis-main-H-s202-sym/round_0064" \
        --ckpt "mean6=$SO/mean6" --ckpt "mean6x=$SO/mean6x" --ckpt "loo5=$SO/loo5" --ckpt "resid-R101=$SO/resid-R101" --ckpt "scaled-R101=$SO/scaled-R101"
'
phase export "test -s $AE_RESULTS/lp.npz" '
    cp "$O/lp.npz" "$O/meta.json" "$AE_RESULTS/"; ls -la "$AE_RESULTS"
'
