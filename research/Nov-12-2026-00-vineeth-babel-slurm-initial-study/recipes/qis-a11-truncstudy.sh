#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:20:00
# HYPOTHESIS: none -- A11 truncation study analysis on the dev 16,384 traces available (read-only).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
phase a "false" '
    D=""; for d in "$PROJ_DATA"/a11/dev16k/H-s101-r36 "$PROJ_DATA"/a11/dev16k/R-s101-r52-v2 "$PROJ_DATA"/a11/dev16k/U-s101-r48-v2; do [ -s "$d/summary.json" ] && D="$D $d"; done
    "$VENV/bin/python" -m qis.analysis.trunc_study "$PROJ_DATA/data/main-v1/bundle_main_v1.json" "$MODEL" "$AE_RESULTS" $D > /dev/null
    cat "$AE_RESULTS/trunc_study.json"
'
