#!/usr/bin/env bash
#EXP gpus=0 cpus=1 mem=4G time=00:10:00
# HYPOTHESIS: none -- export candidate ID order (seeded) for pass-2 annotation planning.
export PROJ_DATA="/data/user_data/$USER/qis"
export C="$PROJ_DATA/data/candidates-v1-study"
export C2="$PROJ_DATA/data/candidates-v1-calib"
phase export "test -s $AE_RESULTS/ids.json" '
    python3 - <<PY
import json, os
out = {}
for d in ("$C", "$C2"):
    if not os.path.isdir(d): continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            out[f[:-6]] = [json.loads(l)["id"] for l in open(os.path.join(d, f))]
json.dump(out, open("$AE_RESULTS/ids.json", "w"))
print({k: len(v) for k, v in out.items()})
PY
'
