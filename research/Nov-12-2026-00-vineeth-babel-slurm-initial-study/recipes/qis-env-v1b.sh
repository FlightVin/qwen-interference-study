#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=8G time=00:30:00
# HYPOTHESIS: none -- env patch. omegaconf resolved to 1.4.1 (verl cannot import); add ninja for
# any JIT kernels. Then rerun all CPU unit tests (incl. verl agg_loss parity).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export UV_CACHE_DIR="$PROJ_DATA/uvcache"
export UV="$HOME/.local/bin/uv"
export PYTHONPATH="$PWD/src"
export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
phase patch "test -f $VENV/.patch-b" '
    "$UV" pip freeze --python "$VENV/bin/python" > "$AE_RESULTS/freeze_before.txt"
    "$UV" pip install --python "$VENV/bin/python" "omegaconf>=2.3,<2.5" ninja 2>&1 | tail -20 > "$AE_RESULTS/patch.log"
    "$UV" pip freeze --python "$VENV/bin/python" > "$AE_RESULTS/freeze_after.txt"
    diff "$AE_RESULTS/freeze_before.txt" "$AE_RESULTS/freeze_after.txt" > "$AE_RESULTS/freeze_diff.txt" || true
    "$VENV/bin/python" -c "import verl.trainer.ppo.core_algos as c; print(\"verl import ok\")"
    touch "$VENV/.patch-b"
'
phase tests "test -s $AE_RESULTS/pytest.txt && grep -q \" passed\" $AE_RESULTS/pytest.txt && ! grep -qE \"failed|error\" $AE_RESULTS/pytest.txt" '
    rc=0; "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/ > "$AE_RESULTS/pytest.txt" 2>&1 || rc=$?
    tail -15 "$AE_RESULTS/pytest.txt"; exit $rc
'
