#!/usr/bin/env bash
#EXP gpus=0 cpus=4 mem=24G time=01:30:00
# HYPOTHESIS: none -- engineering. Builds the project environment venv-v1 (verl 0.9.0 + vLLM +
# math-verify) under PROJ_DATA, and downloads the pinned model and dataset. Emits pip freeze,
# resolved revisions and file hashes for the record.
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export UV_CACHE_DIR="$PROJ_DATA/uvcache"
export UV="$HOME/.local/bin/uv"

phase dirs "test -d $PROJ_DATA/data" '
    mkdir -p "$PROJ_DATA"/{env,hf,data,runs,ckpt,uvcache,records}
    df -h /data/user_data/$USER | tail -1 > "$AE_RESULTS/df.txt"
'
phase venv "test -f $VENV/.complete" '
    rm -rf "$VENV"
    "$UV" venv --python 3.12 "$VENV"
    rc=0
    "$UV" pip install --python "$VENV/bin/python" "verl[vllm]==0.9.0" "math-verify[antlr4_13_2]==0.9.0" \
        datasets pytest wandb 2>&1 | tail -40 > "$AE_RESULTS/install.log" || rc=$?
    "$UV" pip freeze --python "$VENV/bin/python" > "$AE_RESULTS/freeze.txt"
    [ $rc -eq 0 ] || exit $rc
    "$VENV/bin/python" -c "import verl, vllm, torch, transformers, math_verify; print(verl.__version__, vllm.__version__, torch.__version__, transformers.__version__, torch.version.cuda)" > "$AE_RESULTS/versions.txt"
    touch "$VENV/.complete"
'
phase model "test -f $PROJ_DATA/hf/.model-complete" '
    "$VENV/bin/python" - <<PY > "$AE_RESULTS/model.txt"
from huggingface_hub import snapshot_download, HfApi
info = HfApi().model_info("Qwen/Qwen3-1.7B")
p = snapshot_download("Qwen/Qwen3-1.7B", revision=info.sha)
print("revision", info.sha); print("path", p)
PY
    touch "$PROJ_DATA/hf/.model-complete"
'
phase quota "test -s $AE_RESULTS/du.txt" '
    du -sh "$PROJ_DATA"/* > "$AE_RESULTS/du.txt" 2>&1 || true
    df -h /data/user_data/$USER | tail -1 >> "$AE_RESULTS/du.txt"
'
