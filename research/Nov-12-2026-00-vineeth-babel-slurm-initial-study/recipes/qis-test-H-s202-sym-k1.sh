#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=06:00:00 part=general constraint=L40S
# HYPOTHESIS: none -- sealed target_test endpoint of qis-main-H-s202-sym, shard 1 of 6 (D149). Same recipe as qis-test-H-s202-sym
# (D141 common recipe: L40S, n 32, cap 12,288, seed label test-final); the panel is split by index mod 6, so per-problem seeds
# are unchanged. The last shard to finish merges all six with qis.eval_merge into the canonical output directory.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export TEST="$PROJ_DATA/data/main-v1/target_test_sealed.json"
export FINAL="$PROJ_DATA/test-eval/qis-test-H-s202-sym"
export ROOT="$PROJ_DATA/test-eval/qis-test-H-s202-sym-shards"
export K=1
export OUT="$ROOT/k$K"
export CKPT="$PROJ_DATA/ckpt/qis-main-H-s202-sym/round_0128"

phase precheck "false" '
    echo "2ffe70ddd5f81bee5c9feadee6f08335d8b189b33f38ccab474263182dec29ca  $TEST" | sha256sum -c - || { echo "SEALED TEST HASH MISMATCH"; exit 1; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || { echo "GPU already holds ${used} MiB on $(hostname)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -q "L40S" || { echo "NOT AN L40S GPU"; exit 1; }
    mkdir -p "$OUT"
    test -f "$PROJ_DATA/runs/qis-main-H-s202-sym/RUN_COMPLETE" || { echo "PRECONDITION FAILED: qis-main-H-s202-sym not complete"; exit 1; }
    test -f "$CKPT/model.pt" || { echo "PRECONDITION FAILED: no $CKPT/model.pt"; exit 1; }
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CKPT/state.json")
    have=$(sha256sum "$CKPT/model.pt" | cut -d" " -f1)
    echo "model.pt sha256 recorded=$want on-disk=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || { echo "CHECKPOINT IDENTITY MISMATCH"; exit 1; }
    "$VENV/bin/python" -c "import json,sys; st=json.load(open(sys.argv[1])); ok=(st[\"round\"],st[\"counters\"][\"opt_step\"],st[\"attempt_id\"])==(128,512,\"a2\"); print(\"provenance round\",st[\"round\"],\"opt_step\",st[\"counters\"][\"opt_step\"],\"attempt\",st[\"attempt_id\"],\"ok\" if ok else \"FAIL\")" "$CKPT/state.json" > "$AE_RESULTS/provenance.txt"
    cat "$AE_RESULTS/provenance.txt"; grep -q " ok$" "$AE_RESULTS/provenance.txt" || { echo "PROVENANCE CHECK FAILED (expected round 128, opt_step 512, a2)"; exit 1; }
    "$VENV/bin/python" -m qis.eval_merge --split 6 "$TEST" test "$OUT/panel_" > /dev/null
'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$OUT/panel_$K.json" --panel-key test --n 32 --out "$OUT" --seed-label test-final --max-tokens 12288
'
phase merge "test -s $AE_RESULTS/shard_summary.json" '
    "$VENV/bin/python" -c "import json,sys; s=json.load(open(sys.argv[1])); s.pop(\"per_problem\"); json.dump(s, open(sys.argv[2], \"w\"), indent=1)" "$OUT/summary.json" "$AE_RESULTS/shard_summary.json"
    (
        flock 9
        n=$(ls "$ROOT"/k*/summary.json 2>/dev/null | wc -l)
        echo "shards done: $n of 6" | tee -a "$AE_RESULTS/merge_status.txt"
        if [ "$n" = 6 ] && [ ! -s "$FINAL/summary.json" ]; then
            "$VENV/bin/python" -m qis.eval_merge "$TEST" test "$FINAL" "$ROOT"/k0 "$ROOT"/k1 "$ROOT"/k2 "$ROOT"/k3 "$ROOT"/k4 "$ROOT"/k5 > "$AE_RESULTS/merge.log" 2>&1 || { cat "$AE_RESULTS/merge.log"; exit 1; }
            cp "$FINAL/summary.json" "$AE_RESULTS/summary.json"
        fi
    ) 9> "$ROOT/.merge.lock"
'
