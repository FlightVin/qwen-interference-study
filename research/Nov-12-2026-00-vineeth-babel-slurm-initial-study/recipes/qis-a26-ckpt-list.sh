#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:30:00
# HYPOTHESIS: none -- A26 pass A (CHECKLIST 2.6): list every directory under $PROJ_DATA holding model.pt or optim.pt, with file
# sizes, LATEST.json pointers and leftover *.tmp directories; plus a df and a du of ckpt/. Read-only.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"

phase list "test -s $AE_RESULTS/ckpt_list.json" '
    "$VENV/bin/python" research/audit-a26/tools/ckpt_audit.py "$PROJ_DATA" none "$AE_RESULTS/ckpt_list.json" --list-only
    df -h /data/user_data/$USER > "$AE_RESULTS/df.txt"
    du -sh "$PROJ_DATA"/ckpt/* > "$AE_RESULTS/du_ckpt.txt" 2>&1 || true
'
