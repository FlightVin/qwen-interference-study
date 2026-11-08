"""Endpoint validation for one finished main run (A9.3; run in a CPU job; read-only).

Checks, each reported as pass/fail with its numbers:
- rounds: metrics has one `round` record for every round 1..128 (no gaps, no duplicates after resume);
- attempts: records/train_rNNNN.jsonl.gz exist for rounds 0..127 with 16 prompts x 8 responses each;
- verifier: no reward=None / status=unknown in training or scheduled evaluation records; no held rounds;
- optimizer: 4 steps per round (512 total), no non-finite step;
- evaluations: dev, probe_J_R, probe_J_U at rounds 0, 32, 64, 96, 128 (8 responses/problem);
- checkpoint identity: round_0128 model.pt/optim.pt sha256 equal the values recorded in state.json;
- attempt: every metrics row and the round_0128 state.json carry the retained attempt id (A11/D99; default a2);
- truncation: capped/total for hard and auxiliary training prompts; cost inputs: start/complete times per attempt.
Usage: python -m qis.analysis.validate_endpoint RUN_DIR CKPT_DIR OUT.json [ATTEMPT_ID]
"""
import glob
import gzip
import hashlib
import json
import os
import sys

ROUNDS, PROMPTS, RESP, STEPS = 128, 16, 8, 4
EVAL_ROUNDS, PANELS = (0, 32, 64, 96, 128), ("dev", "probe_J_R", "probe_J_U")


def sha(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def main(run_dir, ckpt_dir, out, attempt="a2"):
    M = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    res, ok = {}, True
    att = sorted({str(r.get("attempt_id")) for r in M})
    rounds = [r["round"] for r in M if r.get("kind") == "round"]
    res["rounds"] = {"n": len(rounds), "distinct": len(set(rounds)), "missing": sorted(set(range(1, ROUNDS + 1)) - set(rounds)),
                     "pass": sorted(set(rounds)) == list(range(1, ROUNDS + 1))}
    steps = [r for r in M if r.get("kind") == "step"]
    res["optimizer"] = {"steps": len({r["opt_step"] for r in steps}), "nonfinite": sum(bool(r.get("nonfinite")) for r in steps),
                        "pass": len({r["opt_step"] for r in steps}) == ROUNDS * STEPS and not any(r.get("nonfinite") for r in steps)}
    ev = [r for r in M if r.get("kind") == "event"]
    held = [e for e in ev if "held" in e.get("what", "")]
    res["events"] = {"starts": sum(e["what"] == "start" for e in ev), "complete": any(e["what"] == "complete" for e in ev),
                     "held": len(held), "attempt_windows": [(e.get("job"), e.get("restart"), e["utc"]) for e in ev if e["what"] in ("start", "complete")]}
    tr = {"hard": [0, 0], "aux": [0, 0]}
    bad_counts, unknown_train, files = [], 0, 0
    for rnd in range(ROUNDS):
        p = os.path.join(run_dir, "records", f"train_r{rnd:04d}.jsonl.gz")
        if not os.path.exists(p):
            bad_counts.append((rnd, "missing")); continue
        files += 1
        recs = [json.loads(l) for l in gzip.open(p, "rt")]
        if len(recs) != PROMPTS * RESP or len({r["slot"] for r in recs}) != PROMPTS:
            bad_counts.append((rnd, len(recs)))
        for r in recs:
            unknown_train += r["reward"] is None or r["status"] == "unknown"
            k = "hard" if r["role"].startswith("hard") else "aux"
            tr[k][0] += r["finish"] == "length"; tr[k][1] += 1
    res["attempts"] = {"files": files, "bad": bad_counts[:10], "pass": files == ROUNDS and not bad_counts}
    unknown_eval, ev_found = 0, {}
    for p in sorted(glob.glob(os.path.join(run_dir, "records", "eval_*_r*.jsonl.gz"))):
        name = os.path.basename(p)[5:-9]
        panel, r = name.rsplit("_r", 1)
        recs = [json.loads(l) for l in gzip.open(p, "rt")]
        unknown_eval += sum(x["reward"] is None or x["status"] == "unknown" for x in recs)
        ev_found[f"{panel}@{int(r)}"] = len(recs)
    need = [f"{p}@{r}" for p in PANELS for r in EVAL_ROUNDS]
    res["evaluations"] = {"found": ev_found, "missing": [k for k in need if k not in ev_found], "pass": all(k in ev_found for k in need)}
    res["verifier"] = {"unknown_train": unknown_train, "unknown_eval": unknown_eval, "held_rounds": len(held),
                       "pass": unknown_train == 0 and unknown_eval == 0 and not held}
    fin = os.path.join(ckpt_dir, f"round_{ROUNDS:04d}")
    ident = {"path": fin}
    if os.path.exists(os.path.join(fin, "state.json")):
        meta = json.load(open(os.path.join(fin, "state.json")))
        ident.update({n: {"recorded": h, "on_disk": sha(os.path.join(fin, n))} for n, h in meta.get("file_sha256", {}).items()})
        ident["pass"] = all(v["recorded"] == v["on_disk"] for k, v in ident.items() if isinstance(v, dict))
    else:
        ident["pass"] = False
    res["checkpoint_identity"] = ident
    st = os.path.join(fin, "state.json")
    ck_att = json.load(open(st)).get("attempt_id") if os.path.exists(st) else None
    res["attempt"] = {"expected": attempt, "metrics": att, "checkpoint": ck_att, "pass": att == [attempt] and ck_att == attempt}
    res["truncation_train"] = {k: {"capped": a, "total": b, "frac": a / b if b else None} for k, (a, b) in tr.items()}
    for k in ("rounds", "optimizer", "attempts", "evaluations", "verifier", "checkpoint_identity", "attempt"):
        ok &= bool(res[k]["pass"])
    res["all_pass"] = ok
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
