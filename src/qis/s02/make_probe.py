"""Write an E2 local-update probe recipe (study s02). Orchestration helper only.

From one saved state (model + AdamW) of a run, take ONE round under several loss weightings of the same mixture and
evaluate before and after. Branches share the parent's schedule, sampling seeds and data; only role weights differ:
  Tonly (T=1, C=0), Conly (T=0, C=1), null (T=0, C=0: zero gradient, Adam momentum only), null2 (replicate of null:
  measures evaluation noise, since vLLM sampling is not bit-reproducible).
The start-round evaluation is repeated in every branch (four estimates of the same state). No checkpoints are written.
Usage: python -m qis.s02.make_probe --parent EXPID --round R --expdir DIR [--part general]
"""
import argparse
import json
import os

HEAD = r'''#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=96G time=03:00:00 part={part}
# HYPOTHESIS: E2 local update effect at {parent} round {rnd}: one round of C-only updates lowers caesar (T) success
# relative to a null round, by more in low-success strata? T-only round shows the size of T's own one-round gain.
# Branches Tonly / Conly / null / null2 from the same model + Adam state; eval conf panels before and after.
export PROJ_DATA="/data/user_data/$USER/qis"
export S02="$PROJ_DATA/s02"
export VENV="$PROJ_DATA/env/venv-s02"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export PARENT="{parent}"
export RND={rnd}
export PW="$S02/runs/$PARENT"
export INIT="$S02/ckpt/$PARENT/round_$(printf %04d $RND)"
export P="$S02/runs/{expid}"
phase precheck "false" '
    test -f "$INIT/state.json" || {{ echo "PRECONDITION FAILED: $INIT missing"; exit 1; }}
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || {{ echo "GPU already holds ${{used}} MiB on $(hostname)"; exit 1; }}
    nvidia-smi --query-gpu=name,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$P"
'
'''

BRANCH = r'''
export B_{name}='{weights}'
phase {name} "test -f $P/{name}/RUN_COMPLETE" '
    mkdir -p "$P/{name}"
    python - <<PY
import json, os
cfg = json.load(open(os.environ["PW"] + "/config.json"))
w = json.loads(os.environ["B_{name}"])
cfg["mix"] = [dict(m, weight=w[m["role"]]) for m in cfg["mix"]]
cfg.update(run_id="{expid}-{name}", init_ckpt=os.environ["INIT"], init_round=int(os.environ["RND"]),
           rounds=int(os.environ["RND"]) + 1, eval_rounds=[int(os.environ["RND"]), int(os.environ["RND"]) + 1],
           eval_panels={panels}, out_dir=os.environ["P"] + "/{name}", ckpt_dir=os.environ["P"] + "/{name}/ckpt",
           no_ckpt=True, retain_rounds=[])
json.dump(cfg, open(os.environ["P"] + "/{name}/config.json", "w"), indent=1)
PY
    rc=0; python -m qis.train.run --config "$P/{name}/config.json" >> "$P/{name}/stdout.log" 2>&1 || rc=$?
    grep -v "^INFO" "$P/{name}/stdout.log" | tail -5
    cp "$P/{name}/metrics.jsonl" "$AE_RESULTS/{name}.metrics.jsonl" 2>/dev/null || true
    cp "$P/{name}/config.json" "$AE_RESULTS/{name}.config.json"
    if [ $rc -eq 143 ]; then scontrol requeue $SLURM_JOB_ID; sleep 900; fi
    exit $rc
'
'''

PANELS = {"caesar_s4_conf": 16, "caesar_s8_conf": 16, "caesar_s12_conf": 16, "terse10_k2_conf": 8}
BRANCHES = [("Tonly", {"T": 1.0, "C": 0.0}), ("Conly", {"T": 0.0, "C": 1.0}),
            ("null", {"T": 0.0, "C": 0.0}), ("null2", {"T": 0.0, "C": 0.0})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--expdir", required=True)
    ap.add_argument("--part", default="general")
    a = ap.parse_args()
    expid = f"qis-s02-e2-{a.parent.replace('qis-s02-', '')}-r{a.round}"
    text = HEAD.format(part=a.part, parent=a.parent, rnd=a.round, expid=expid)
    for name, w in BRANCHES:
        text += BRANCH.format(name=name, weights=json.dumps(w), expid=expid, panels=json.dumps(PANELS))
    path = os.path.join(a.expdir, expid + ".sh")
    open(path, "w").write(text)
    print(path)


if __name__ == "__main__":
    main()
