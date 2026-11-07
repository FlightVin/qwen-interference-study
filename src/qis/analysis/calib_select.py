"""Apply the frozen §7 calibration selection rule to calibration metrics files (stdlib only; run in a CPU job).

Inputs: `<dir>/cal8k-lr{lr}-{clip}-s{seed}.metrics.jsonl` (seed 11 for all six configs; seed 22 for the top two).
Rule (project.md §7): reject non-finite updates, broken verification (unknown verifier results) or > 5% capped
validation responses at any evaluation; rank the rest by seed-11 endpoint gain; top two repeat with seed 22 (a
repeat that breaks a rejection criterion invalidates that finalist); select by mean gain; differences ≤ 0.01 are
ties, broken lr 1e-6 > 5e-7 > 2e-6 then symmetric > asymmetric; require positive mean gain and neither seed below
its own baseline by more than 0.02.
Usage: python3 -m qis.analysis.calib_select research/evidence/calibration
"""
import glob
import json
import math
import os
import sys

LR_PREF = {"1e-6": 0, "5e-7": 1, "2e-6": 2}
CLIP_PREF = {"sym": 0, "asym": 1}
TIE = 0.01
TRUNC_MAX = 0.05


def run_summary(path):
    rs = [json.loads(l) for l in open(path)]
    ev = {r["round"]: r for r in rs if r.get("kind") == "eval"}
    steps = [r for r in rs if r.get("kind") == "step"]
    finite = all(math.isfinite(v) for r in steps for v in r.values() if isinstance(v, float))
    complete = any(r.get("kind") == "event" and r.get("what") == "complete" for r in rs)
    reasons = []
    if not complete or 0 not in ev or 32 not in ev:
        reasons.append("incomplete")
    if not finite:
        reasons.append("non-finite")
    if any(e["unknown"] > 0 for e in ev.values()):
        reasons.append("verifier unknown")
    tmax = max((e["truncated"] for e in ev.values()), default=1.0)
    if tmax > TRUNC_MAX:
        reasons.append(f"truncation {tmax:.4f} > {TRUNC_MAX}")
    b, e = ev.get(0, {}).get("pass1"), ev.get(32, {}).get("pass1")
    return {"base": b, "end": e, "gain": (e - b) if b is not None and e is not None else None,
            "trunc_max": tmax, "valid": not reasons, "reasons": reasons}


def select(d):
    runs = {}
    for p in glob.glob(os.path.join(d, "cal8k-lr*-s*.metrics.jsonl")):
        name = os.path.basename(p)[len("cal8k-"):-len(".metrics.jsonl")]
        cfg, seed = name.rsplit("-s", 1)
        runs.setdefault(cfg, {})[int(seed)] = run_summary(p)
    first = {c: v[11] for c, v in runs.items() if 11 in v}
    if len(first) != 6:
        return {"status": "waiting", "detail": f"{len(first)}/6 seed-11 runs"}
    valid = sorted((c for c in first if first[c]["valid"]), key=lambda c: -first[c]["gain"])
    out = {"seed11": first, "ranking": valid, "finalists": valid[:2]}
    if len(valid) < 2:
        out["status"] = "fewer than two valid configurations: diagnose and rerun once (§7)"
        return out
    fin = []
    for c in valid[:2]:
        s22 = runs[c].get(22)
        if s22 is None:
            out["status"] = "waiting"; out["detail"] = f"seed 22 missing for {c}"; return out
        ok = s22["valid"]
        mean = (first[c]["gain"] + s22["gain"]) / 2 if s22["gain"] is not None else None
        gate = ok and mean is not None and mean > 0 and min(first[c]["gain"], s22["gain"]) >= -0.02
        fin.append({"config": c, "gain11": first[c]["gain"], "gain22": s22["gain"], "mean": mean,
                    "seed22_valid": ok, "seed22_reasons": s22["reasons"], "gate": gate})
    out["finalist_results"] = fin
    ok = [f for f in fin if f["gate"]]
    if not ok:
        out["status"] = "validation-gain gate failed: diagnose, then extend finalists to 64 rounds (§7)"
        return out
    best = max(f["mean"] for f in ok)
    tied = [f for f in ok if best - f["mean"] <= TIE]
    lr = lambda c: c[len("lr"):].rsplit("-", 1)[0]      # "lr2e-6-sym" -> "2e-6"
    clip = lambda c: c.rsplit("-", 1)[1]                # "lr2e-6-sym" -> "sym"
    pick = min(tied, key=lambda f: (LR_PREF[lr(f["config"])], CLIP_PREF[clip(f["config"])]))
    out.update(status="selected", selected=pick["config"], tie=len(tied) > 1)
    return out


if __name__ == "__main__":
    print(json.dumps(select(sys.argv[1]), indent=1))
