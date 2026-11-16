"""Re-grade stored evaluation records with the current grader (study s02). No sampling, no GPU.

Reads eval record files (gz JSONL written by qis.train.run.evaluate / qis.s02.calib), decodes each response, rebuilds
the grader item from the bundle, scores it again, and writes a summary per pool: old and new pass@1, per-problem
counts, and a table of status transitions (old -> new) with up to 5 example texts per changed transition.
Usage: python -m qis.s02.regrade --model M --bundle B --records DIR --label calib --out FILE.json
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from qis.verify import Verifier
    tok = AutoTokenizer.from_pretrained(a.model)
    items = {it["id"]: it for v in json.load(open(a.bundle))["pools"].values() for it in v}
    ver = Verifier(2)
    res = {}
    for path in sorted(glob.glob(os.path.join(a.records, "eval_*.jsonl.gz"))):
        recs = [json.loads(x) for x in gzip.open(path, "rt")]
        texts = [tok.decode(r["ids"], skip_special_tokens=True) for r in recs]
        q = []
        for r, t in zip(recs, texts):
            it = items[r["prompt_id"]]
            q.append({"gold": it["gold"], "response": t, "truncated": r["finish"] == "length", "n_tokens": r["n_tokens"],
                      "match": it.get("match"), "len_limit": it.get("len_limit"), "base": it.get("base"),
                      "numbers": it.get("numbers"), "target": it.get("target"), "tests": it.get("tests"),
                      "test_setup": it.get("test_setup", "")})
        new = ver.score(q)
        trans, ex = {}, {}
        per_old, per_new = {}, {}
        for r, n, t in zip(recs, new, texts):
            key = f"{r['status']}->{n['status']}"
            trans[key] = trans.get(key, 0) + 1
            if r["status"] != n["status"] and len(ex.setdefault(key, [])) < 5:
                ex[key].append({"old_extracted": r["extracted"], "new_extracted": n.get("extracted"), "tail": t[-300:]})
            per_old[r["prompt_id"]] = per_old.get(r["prompt_id"], 0) + (r["reward"] == 1)
            per_new[r["prompt_id"]] = per_new.get(r["prompt_id"], 0) + (n["reward"] == 1)
        npp = len(recs) // max(1, len(per_old))
        name = os.path.basename(path)[len("eval_"):-len(".jsonl.gz")]
        res[name] = {"responses": len(recs), "n": npp,
                     "pass1_old": float(np.mean([c / npp for c in per_old.values()])),
                     "pass1_new": float(np.mean([c / npp for c in per_new.values()])),
                     "per_problem_c_new": [per_new[k] for k in sorted(per_new)], "transitions": trans,
                     "examples": ex, "fallback_used": sum(bool(n.get("fallback")) for n in new)}
        print(name, json.dumps({k: res[name][k] for k in ("pass1_old", "pass1_new", "transitions")}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    ver.close()


if __name__ == "__main__":
    main()
