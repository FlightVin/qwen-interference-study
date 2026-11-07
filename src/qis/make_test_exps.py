"""Generate sealed target_test evaluation recipes (A9 scope; orchestration helper; writes exp/*.sh only).

One recipe for the common base model and one per in-scope run endpoint (round 128). 32 samples/problem,
128 problems, seed label `test-final` (EVAL_SEED stream, independent of training/selection). Endpoint recipes
first check that the run completed and that round_0128/model.pt matches the sha256 recorded in state.json
(checkpoint identity). Test outcomes never feed training decisions (all training is fixed before they exist).
Usage: python -m qis.make_test_exps --expdir DIR
"""
import argparse
import os

RUNS = [f"qis-main-{a}-s{s}-sym" for s in (101, 202) for a in "HRU"]
TEST_SHA = "2ffe70ddd5f81bee5c9feadee6f08335d8b189b33f38ccab474263182dec29ca"

HEAD = r'''#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=05:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: none -- sealed target_test evaluation (A9; PROTOCOL §7): {what}; 128 problems x 32 samples, cap 16,384 (A10; also success within 8,192 from the same traces).
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
export OUT="$PROJ_DATA/test-eval/{name}"
export CKPT="{ckpt}"

phase precheck "false" '
    echo "{sha}  $TEST" | sha256sum -c - || {{ echo "SEALED TEST HASH MISMATCH"; exit 1; }}
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || {{ echo "GPU already holds ${{used}} MiB on $(hostname)"; exit 1; }}
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$OUT"
{extra}'
phase eval "test -s $OUT/summary.json" '
    "$VENV/bin/python" -m qis.evaluate_ckpt --model "$MODEL" --ckpt "$CKPT" --panel "$TEST" --panel-key test --n 32 --out "$OUT" --seed-label test-final --max-tokens 16384
'
phase export "test -s $AE_RESULTS/summary.json" '
    cp "$OUT/summary.json" "$AE_RESULTS/"
'
'''

IDENT = r'''    test -f "$PROJ_DATA/runs/{run}/RUN_COMPLETE" || {{ echo "PRECONDITION FAILED: {run} not complete"; exit 1; }}
    test -f "$CKPT/model.pt" || {{ echo "PRECONDITION FAILED: no $CKPT/model.pt"; exit 1; }}
    want=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))[\"file_sha256\"][\"model.pt\"])" "$CKPT/state.json")
    have=$(sha256sum "$CKPT/model.pt" | cut -d" " -f1)
    echo "model.pt sha256 recorded=$want on-disk=$have" | tee "$AE_RESULTS/ckpt_identity.txt"
    [ "$want" = "$have" ] || {{ echo "CHECKPOINT IDENTITY MISMATCH"; exit 1; }}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True)
    a = ap.parse_args()
    jobs = [("qis-test-base", "base model (common baseline)", "base", "")]
    for r in RUNS:
        ck = f"$PROJ_DATA/ckpt/{r}/round_0128"
        jobs.append((f"qis-test-{r[len('qis-main-'):]}", f"endpoint of {r}", ck, IDENT.format(run=r)))
    for name, what, ck, extra in jobs:
        with open(os.path.join(a.expdir, f"{name}.sh"), "w") as f:
            f.write(HEAD.format(what=what, name=name, ckpt=ck, sha=TEST_SHA, extra=extra))
        print(name)


if __name__ == "__main__":
    main()
