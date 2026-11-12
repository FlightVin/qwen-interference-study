#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=16G time=01:00:00
# HYPOTHESIS: none -- report analysis after ALL training has ended (A13: test outcomes read only then). Precheck refuses
# unless R/U/H-s101 and R/U-s202 have RUN_COMPLETE (H-s202 not run, D122). Primary: report_tables (A9/A13 table) on the
# sealed-test summaries present; report_data (non-sealed inputs, attempt a2 asserted). EXPLORATORY (A15, D122):
# per_problem (reliability, bands), interference (shared hard trace). Figures in the plotting venv. Rerunnable: missing
# test summaries leave "missing" cells; run again when the last test ends.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export PVENV="$PROJ_DATA/env/venv-report"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
phase precheck "false" '
    for r in H-s101 R-s101 U-s101 R-s202 U-s202; do
        test -f "$PROJ_DATA/runs/qis-main-$r-sym/RUN_COMPLETE" || { echo "PRECONDITION FAILED: qis-main-$r-sym not complete; test outcomes stay sealed"; exit 1; }
    done
    ls "$PROJ_DATA"/test-eval/*/summary.json | tee "$AE_RESULTS/test_summaries_present.txt"
'
phase a "false" '
    O="$AE_RESULTS"; mkdir -p "$O/tables" "$O/figures"
    "$VENV/bin/python" -m qis.analysis.report_data "$PROJ_DATA/runs" "$O/tables/report_data.json" a2
    "$VENV/bin/python" -m qis.analysis.interference "$PROJ_DATA/runs" "$O/tables/interference.json"
    "$VENV/bin/python" -m qis.analysis.report_tables "$PROJ_DATA/test-eval" "$O/tables/a9_table"
    "$VENV/bin/python" -m qis.analysis.per_problem "$PROJ_DATA/test-eval" "$O/tables/per_problem"
    for f in "$PROJ_DATA"/test-eval/*/summary.json; do n=$(basename $(dirname "$f")); cp "$f" "$O/tables/summary_$n.json"; done
    for l in r0 r64; do test -s "$PROJ_DATA/diag-local/$l/diag_$l.jsonl" && cp "$PROJ_DATA/diag-local/$l/diag_$l.jsonl" "$O/tables/"; done
    "$PVENV/bin/python" -m qis.analysis.figures "$O/tables/report_data.json" "$O/figures" --table "$O/tables/a9_table.json" \
        --diag "$O/tables/diag_r0.jsonl" "$O/tables/diag_r64.jsonl" --per-problem "$O/tables/per_problem.json" --interference "$O/tables/interference.json"
    python3 - "$O/tables/summary_"*.json <<PY
import json, sys
for p in sys.argv[1:]:
    s = json.load(open(p)); print(p.rsplit("summary_", 1)[1], "per_problem", len(s["per_problem"]), "n", s["n"], "max_tokens", s["max_tokens"])
PY
    du -sh "$O"
'
