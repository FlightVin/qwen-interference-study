#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=00:45:00
# HYPOTHESIS: none -- amendment A8 CPU tasks: §3.7 audit sample export, records reconciliation counts (§3.12/§3.16,
# calibration panels), truncation-by-role monitor of the main runs (D66), and storage check. Read-only on runs.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
phase test "false" '
    "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_analysis.py > "$AE_RESULTS/pytest.txt" 2>&1; rc=$?
    tail -3 "$AE_RESULTS/pytest.txt"; exit $rc
'
phase audit "test -s $AE_RESULTS/audit37_sample.jsonl.gz" '
    "$VENV/bin/python" -m qis.analysis.audit_export "$PROJ_DATA" data/annotations "$AE_RESULTS" "$MODEL"
'
phase reconcile "false" '
    "$VENV/bin/python" -m qis.analysis.reconcile_screening "$PROJ_DATA/data/calib-panels-k8/bundle.json" > "$AE_RESULTS/reconcile.json"
    cat "$AE_RESULTS/reconcile.json"
'
phase trunc "false" '
    "$VENV/bin/python" -m qis.analysis.trunc_monitor "$PROJ_DATA/runs" "$AE_RESULTS/trunc_monitor.json" | tail -60
'
phase storage "false" '
    { df -h /data/user_data/$USER | tail -1; du -sh "$PROJ_DATA"/ckpt "$PROJ_DATA"/runs "$PROJ_DATA"/screen "$PROJ_DATA"/a6 2>/dev/null; du -sh "$PROJ_DATA"/ckpt/qis-main-* 2>/dev/null; } > "$AE_RESULTS/storage.txt"
    cat "$AE_RESULTS/storage.txt"
'
