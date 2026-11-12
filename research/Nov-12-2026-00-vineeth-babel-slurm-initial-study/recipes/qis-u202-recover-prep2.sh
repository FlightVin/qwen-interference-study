#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:15:00
# HYPOTHESIS: none -- U-s202 second-retry prep (D117): job 11097811 FAILED (EINTR closing train records). Preserve records
# written after its last committed checkpoint: train_r{L}+ and eval_*_r{L+1}+ move to records/superseded-11097811/, so the
# resumed run regenerates them. If metrics.jsonl ends in a partial line, keep the original copy there and drop only that
# partial line. Nothing is deleted. Also lists checkpoint dirs (incl. .tmp) and LATEST.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/moved.txt" '
    C="$PROJ_DATA/ckpt/qis-main-U-s202-sym"; D="$PROJ_DATA/runs/qis-main-U-s202-sym"; R="$D/records"; S="$R/superseded-11097811"
    ls -la "$C" | tee "$AE_RESULTS/ckpt_ls.txt"; cat "$C/LATEST.json"; echo
    L=$(grep -o "\"round\": [0-9]*" "$C/LATEST.json" | grep -o "[0-9]*$"); echo "LATEST round=$L"
    case "$L" in 64|80) ;; *) echo "UNEXPECTED LATEST $L; stopping"; exit 1;; esac
    test -s "$C/round_$(printf %04d $L)/state.json" || { echo "no state.json for LATEST; stopping"; exit 1; }
    mkdir -p "$S"
    for f in "$R"/train_r*.jsonl.gz "$R"/eval_*_r*.jsonl.gz; do
        b=$(basename "$f"); n=$(echo "$b" | grep -o "r[0-9]\{4\}" | tr -d r); n=$((10#$n))
        case "$b" in train_*) [ "$n" -ge "$L" ] && mv "$f" "$S/";; eval_*) [ "$n" -gt "$L" ] && mv "$f" "$S/";; esac
    done
    M="$D/metrics.jsonl"
    if tail -n 1 "$M" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null && [ "$(tail -c 1 "$M" | od -An -c | tr -d " ")" = "\\n" ]; then
        echo "metrics.jsonl last line valid"
    else
        cp -p "$M" "$S/metrics.jsonl.orig"; python3 - "$M" <<PY
import sys; p=sys.argv[1]; b=open(p,"rb").read(); k=b.rfind(b"\n"); open(p,"wb").write(b[:k+1]); print("dropped partial tail bytes:", len(b)-k-1)
PY
    fi
    tail -n 1 "$M" | cut -c1-200
    ls -la "$S" | tee "$AE_RESULTS/moved.txt"; ls "$R" | tail -3
'
