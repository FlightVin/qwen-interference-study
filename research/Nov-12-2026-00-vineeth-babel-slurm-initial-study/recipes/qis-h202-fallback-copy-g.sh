#!/usr/bin/env bash
#EXP gpus=1 cpus=2 mem=16G time=01:00:00 part=general
# HYPOTHESIS: none -- D150 fallback, step 1 (general-pool variant, used only when preempt_cpu_qos cannot start; same body as qis-h202-fallback-copy): copy the latest saved H-s202 checkpoint (model.pt + state.json, not optim.pt) to a
# fixed directory that the trainer never prunes, and verify the model.pt hash against state.json. The six fallback shards
# (qis-test-H-s202-fb-k*) evaluate that copy. ROUND.txt records which round it is.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export SRC="$PROJ_DATA/ckpt/qis-main-H-s202-sym"
export DST="$PROJ_DATA/ckpt/qis-main-H-s202-sym-fallback"

phase copy "test -s $DST/ROUND.txt" '
    mkdir -p "$DST/ckpt"
    cat "$SRC/LATEST.json" | tee "$AE_RESULTS/latest.json"
    path=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"path\"])" "$SRC/LATEST.json")
    round=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"round\"])" "$SRC/LATEST.json")
    test -f "$path/model.pt" && test -f "$path/state.json" || { echo "PRECONDITION FAILED: $path incomplete"; exit 1; }
    cp "$path/state.json" "$DST/ckpt/state.json"
    cp "$path/model.pt" "$DST/ckpt/model.pt"
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$DST/ckpt/state.json")
    have=$(sha256sum "$DST/ckpt/model.pt" | cut -d" " -f1)
    echo "round $round from $path; model.pt sha256 recorded=$want copy=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH (source pruned during copy?)"; rm -f "$DST/ckpt/model.pt"; exit 1; }
    echo "$round" > "$DST/ROUND.txt"
    cp "$DST/ROUND.txt" "$AE_RESULTS/ROUND.txt"
'
