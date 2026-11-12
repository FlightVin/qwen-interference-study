#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=2G time=00:20:00
# HYPOTHESIS: none -- storage floor (D117; project.md §11). Projected peak with two concurrent 22 GB saves (R-s202, U-s202)
# leaves < 50 GiB free. Retires optimizer state of the s101 endpoints whose sealed test evaluation completed and reloaded
# model.pt with a matching sha256 (R-s101 11095106, U-s101 11095118) and whose endpoint validation passed (checkpoint
# identity); no planned branch uses round_0128 optimizer state (the diagnostic uses R-s101 round_0064, kept). Keeps
# model.pt and state.json. Also removes U-s202's dead round_0080.tmp (job 11097811 ended; no writer). Each path is
# verified under $PROJ_DATA/ckpt and logged with its size.
export PROJ_DATA="/data/user_data/$USER/qis"
phase a "test -s $AE_RESULTS/deleted.txt" '
    C="$PROJ_DATA/ckpt"; df -h "$PROJ_DATA" | tee "$AE_RESULTS/df_before.txt"; : > "$AE_RESULTS/deleted.txt"
    for p in "$C"/qis-main-R-s101-sym/round_0128/optim.pt "$C"/qis-main-U-s101-sym/round_0128/optim.pt "$C"/qis-main-U-s202-sym/round_0080.tmp; do
        [ -e "$p" ] || { echo "absent $p" | tee -a "$AE_RESULTS/deleted.txt"; continue; }
        r=$(realpath "$p"); case "$r" in "$C"/*) ;; *) echo "REFUSE $r"; exit 1;; esac
        s=$(find "$r" -type f -printf "%s\n" | awk "{t+=\$1} END {printf \"%.1f\", t/1e9}")
        echo "$(date -u +%FT%TZ) rm -rf $r  ${s} GB" | tee -a "$AE_RESULTS/deleted.txt"; rm -rf "$r"
    done
    for r in R-s101 U-s101; do ls -la "$C/qis-main-$r-sym/round_0128"; done
    echo done >> "$AE_RESULTS/deleted.txt"; df -h "$PROJ_DATA" | tee "$AE_RESULTS/df_after.txt"
'
