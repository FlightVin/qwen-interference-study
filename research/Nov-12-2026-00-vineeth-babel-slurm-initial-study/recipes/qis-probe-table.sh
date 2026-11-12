#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- read-only table of probe panels (J_H derived from dev, J_R, J_U) pass@1/pass@8 per run and round.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
phase t "false" '
    "$VENV/bin/python" -m qis.analysis.probe_table "$PROJ_DATA/runs" "$PROJ_DATA/data/main-v1/bundle_main_v1.json" > "$AE_RESULTS/probe_table.tsv"
    cat "$AE_RESULTS/probe_table.tsv"
'
