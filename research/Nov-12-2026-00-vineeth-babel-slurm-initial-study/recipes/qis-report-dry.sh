#!/usr/bin/env bash
#EXP gpus=0 cpus=2 mem=8G time=00:30:00
# HYPOTHESIS: none -- report pipeline dry run on non-sealed inputs (CHECKLIST 7.2-7.4; D114): report_data over the six
# main runs (attempt a2 asserted); separate plotting venv $PROJ_DATA/env/venv-report (matplotlib; training venv untouched) and figures (learning, POPE, local update if present). No test input is read.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PYTHONPATH="$PWD/src"
export PVENV="$PROJ_DATA/env/venv-report"
phase env "test -x $PVENV/bin/python && $PVENV/bin/python -c \"import matplotlib\"" '
    "$VENV/bin/python" -m venv "$PVENV" && "$PVENV/bin/pip" install -q "matplotlib==3.9.2" "numpy<2.1" && "$PVENV/bin/pip" freeze > "$AE_RESULTS/venv_report_freeze.txt"
'
phase a "test -s $AE_RESULTS/figures/figures.json" '
    "$PVENV/bin/python" -c "import matplotlib; print(\"matplotlib\", matplotlib.__version__)"
    "$VENV/bin/python" -m qis.analysis.report_data "$PROJ_DATA/runs" "$AE_RESULTS/report_data.json" a2
    "$PVENV/bin/python" -m qis.analysis.figures "$AE_RESULTS/report_data.json" "$AE_RESULTS/figures" --diag "$PROJ_DATA/diag-local/r0/diag_r0.jsonl" "$PROJ_DATA/diag-local/r64/diag_r64.jsonl"
    ls -la "$AE_RESULTS" "$AE_RESULTS/figures"
'
