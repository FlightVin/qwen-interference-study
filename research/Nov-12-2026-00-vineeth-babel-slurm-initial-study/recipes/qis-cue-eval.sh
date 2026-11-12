#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=05:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: EXPLORATORY (A15.4, D123) strategy-cue panel: 24 frozen MATH test items (data/cue, IDs frozen before any
# trained-policy evaluation), uncued vs cued ("Consider whether a factorization could simplify the problem."), 8 samples
# per item per form, cap 12,288, frozen decoding; same pool key and seed label in both forms -> identical sampling seeds
# per (item, sample). Policies in order: base, R-s101, U-s101, R-s202, U-s202 (endpoints, model.pt sha256 checked).
# 24 x 2 x 5 x 8 = 1,920 completions. Phase analyse runs qis.analysis.cue (gain, DiD, lengths, fixed inspection sample). Each (policy, form) is skipped when its summary exists (requeue-safe).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export CUEDIR="$PWD/data/cue"
export OUTROOT="$PROJ_DATA/cue-eval/v1"

phase precheck "false" '
    printf "%s  %s\n%s  %s\n" c3399a42dbd346d2530ded4dacd230df0ca6d2f3825f9ae9f1066665a122aede "$CUEDIR/cue_v1_uncued.json" 1b3aa7bf6ff62f5c441bb468abec552ead0dec0f19f06360015b1d7361b23aaa "$CUEDIR/cue_v1_cued.json" | sha256sum -c - || { echo "CUE PANEL HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    mkdir -p "$OUTROOT"
'
phase eval "test \$(ls $OUTROOT/*/summary.json 2>/dev/null | wc -l) -eq 10" '
    for pol in base R-s101 U-s101 R-s202 U-s202; do
        if [ "$pol" = base ]; then CK=base; else
            run="qis-main-${pol%-*}-${pol#*-}-sym"; CK="$PROJ_DATA/ckpt/$run/round_0128"
            if ! test -f "$PROJ_DATA/runs/$run/RUN_COMPLETE"; then echo "SKIP $pol: run not complete"; continue; fi
            want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CK/state.json")
            have=$(sha256sum "$CK/model.pt" | cut -d" " -f1)
            echo "$pol model.pt sha256 recorded=$want on-disk=$have"; [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH $pol"; exit 1; }
        fi
        for form in uncued cued; do
            O="$OUTROOT/$pol-$form"; test -s "$O/summary.json" && continue; mkdir -p "$O"
            "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CK" --panel "$CUEDIR/cue_v1_$form.json" --panel-key cue --n 8 --out "$O" --seed-label cue-v1 --max-tokens 12288 || exit 1
        done
    done
    test $(ls $OUTROOT/*/summary.json | wc -l) -eq 10
'
phase analyse "test -s $OUTROOT/cue_analysis.json" '
    "$VENV/bin/python" -m qis.analysis.cue "$OUTROOT" "$CUEDIR/cue_v1_uncued.json" "$MODEL" "$OUTROOT/cue_analysis"
'
phase export "test -s $AE_RESULTS/summaries.tgz" '
    cp "$OUTROOT/cue_analysis.json" "$OUTROOT/cue_analysis.txt" "$AE_RESULTS/"
    tar -C "$OUTROOT" -czf "$AE_RESULTS/summaries.tgz" $(cd "$OUTROOT" && ls */summary.json)
'
