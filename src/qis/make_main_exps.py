"""Generate the main-study experiment recipes (orchestration helper; writes exp/*.sh only).

15 runs: primary H/R/U x seeds 101/202/303 at the selected clipping; sensitivity R/U x 3 seeds at the
other clipping. Endpoint T (128 or 64) and dev evaluation schedule (0, T/4, T/2, 3T/4, T) per
project.md §9; panels dev (8), probe_J_R (8), probe_J_U (8) (J_H is a subset of dev). The R seed-101
selected-clipping run retains its halfway full checkpoint for the local-update diagnostic (§7).
Usage: python -m qis.make_main_exps --lr 1e-6 --clip-selected sym --rounds 128 --expdir DIR
"""
import argparse
import os

from qis.limits import MAX_TOKENS

CLIP = {"sym": 0.2, "asym": 0.28}

TEMPLATE = r'''#!/usr/bin/env bash
#EXP gpus=1 cpus=8 mem=64G time=40:00:00 part=preempt constraint=VRAM_48GB
# HYPOTHESIS: main study ({phase}) — arm {arm}, seed {seed}, lr {lr}, clip (0.2,{eh}), {rounds} rounds,
# response cap {maxtok} tokens, bundle main-v1 (sha 0e255243…). Primary contrast R−U on target_test (separate job).
export PROJ_DATA="/data/user_data/$USER/qis"
export VENV="$PROJ_DATA/env/venv-v1"
export HF_HOME="$PROJ_DATA/hf"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VENV/bin:$PATH"
export WANDB_MODE=online
export MODEL="$PROJ_DATA/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export BUNDLE="$PROJ_DATA/data/main-v1/bundle_main_v1.json"
export RUN="{run}"
export W="$PROJ_DATA/runs/$RUN"
export CK="$PROJ_DATA/ckpt/$RUN"

phase precheck "false" '
    test -s "$BUNDLE" || {{ echo "PRECONDITION FAILED: bundle missing"; exit 1; }}
    echo "0e255243d8cca981a653c4fd42b38beeeb163cee924886fa166b972d585d8c51  $BUNDLE" | sha256sum -c - || {{ echo "BUNDLE HASH MISMATCH"; exit 1; }}
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2048 ] || {{ echo "GPU already holds ${{used}} MiB on $(hostname)"; exit 1; }}
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$AE_RESULTS/gpu.txt"
    mkdir -p "$W" "$CK"
    if [ -f "$W/metrics.jsonl" ] && ! head -1 "$W/metrics.jsonl" | grep -q "\"attempt_id\": \"{attempt}\""; then echo "STALE RUN DIR (other attempt) in $W"; exit 1; fi
    if [ -f "$CK/LATEST.json" ] && [ ! -f "$W/metrics.jsonl" ]; then echo "STALE CHECKPOINTS in $CK"; exit 1; fi
'
phase config "false" '
    cat > "$W/config.json" <<JSON
{{"run_id": "$RUN", "attempt_id": "{attempt}", "phase": "{phase}", "arm": "{arm}", "seed": {seed},
 "lr": {lr}, "eps_low": 0.2, "eps_high": {eh}, "rounds": {rounds}, "ckpt_every": {ckpt_every},
 "eval_rounds": {evals}, "eval_panels": {{"dev": 8, "probe_J_R": 8, "probe_J_U": 8}},
 "retain_rounds": {retain},
 "bundle": "$BUNDLE", "model": "$MODEL", "max_tokens": {maxtok}, "gpu_mem_util": 0.30,
 "out_dir": "$W", "ckpt_dir": "$CK", "ckpt_lock_root": "$PROJ_DATA/ckpt",
 "wandb": {{"enabled": true, "project": "qis-qwen-interference"}}}}
JSON
'
phase train "test -f $W/RUN_COMPLETE" '
    rc=0; "$VENV/bin/python" -m qis.train.run --config "$W/config.json" >> "$W/stdout.log" 2>&1 || rc=$?
    tail -20 "$W/stdout.log"
    if [ $rc -eq 143 ]; then scontrol requeue $SLURM_JOB_ID; sleep 900; fi
    exit $rc
'
phase export "test -s $AE_RESULTS/metrics.jsonl" '
    cp "$W/metrics.jsonl" "$W/config.json" "$AE_RESULTS/"
'
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True)
    ap.add_argument("--clip-selected", choices=list(CLIP), required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--expdir", required=True)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="response cap for every round")
    ap.add_argument("--attempt", default="a1")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=4, help="checkpoint cadence in rounds (operational; D104)")
    a = ap.parse_args()
    seed_list = [int(x) for x in a.seeds.split(",")]
    T = a.rounds
    evals = [0, T // 4, T // 2, 3 * T // 4, T]
    other = "asym" if a.clip_selected == "sym" else "sym"
    runs = [("primary", arm, s, a.clip_selected) for arm in "HRU" for s in seed_list] + \
           ([] if a.no_sensitivity else [("sensitivity", arm, s, other) for arm in "RU" for s in seed_list])
    for phase, arm, seed, clip in runs:
        run = f"qis-main-{arm}-s{seed}-{clip}" if phase == "primary" else f"qis-sens-{arm}-s{seed}-{clip}"
        retain = [T // 2] if (phase == "primary" and arm == "R" and seed == 101) else []
        with open(os.path.join(a.expdir, f"{run}.sh"), "w") as f:
            f.write(TEMPLATE.format(phase=phase, arm=arm, seed=seed, lr=a.lr, eh=CLIP[clip], rounds=T, run=run,
                                    evals=evals, retain=retain, attempt=a.attempt, maxtok=a.max_tokens,
                                    ckpt_every=a.ckpt_every))
        print(run)


if __name__ == "__main__":
    main()
