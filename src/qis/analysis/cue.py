"""EXPLORATORY (A15.4, panel frozen in D123): strategy-cue panel analysis. Kept separate from the interference evidence.

Input: `<OUTROOT>/<policy>-<form>/{summary.json,records.jsonl.gz}` from `qis-cue-eval` (24 items x 8 samples; forms
uncued / cued share sampling seeds per (item, sample), so items and samples are paired). No new generations. Output JSON:
- per policy and form: pass@1 (equal weight per item), truncated share, mean and median response tokens, and the share
  of responses whose text contains "factor" (a crude method marker; the cued prompt itself names factorization);
- cue gain per policy: mean over items of (cued c/8 - uncued c/8), 95% item bootstrap (10,000 draws);
- per seed: difference in differences (R cued - R uncued) - (U cued - U uncued), paired by item, item bootstrap;
- a fixed inspection sample: items at panel positions 0, 6, 12, 18, sample 0, for every policy and form; the first
  1,200 and last 300 characters of each response, with its tokens, finish reason and reward (written to OUT.txt).
Usage: python -m qis.analysis.cue OUTROOT PANEL_UNCUED.json MODEL_DIR OUT_PREFIX
"""
import gzip
import json
import os
import sys

import numpy as np

from qis import seeds

POLICIES, FORMS, DRAWS, INSPECT = ("base", "R-s101", "U-s101", "R-s202", "U-s202"), ("uncued", "cued"), 10_000, (0, 6, 12, 18)


def boot(diff, name):
    D = np.asarray(diff, dtype=np.float64)
    r = seeds.rng(seeds.BOOTSTRAP_SEED, f"a15cue/{name}")
    bs = D[r.integers(0, len(D), (DRAWS, len(D)))].mean(axis=1)
    return {"est": float(D.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975))}


def load(d, tok):
    recs = [json.loads(l) for l in gzip.open(os.path.join(d, "records.jsonl.gz"), "rt")]
    for r in recs:
        r["text"] = tok.decode(r.pop("ids"), skip_special_tokens=True)
    return recs


def main(outroot, panel, model, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ids = [it["id"] for it in json.load(open(panel))["pools"]["cue"]]
    res = {"items": len(ids), "policies": {}, "cue_gain": {}, "did": {}, "missing": []}
    c, lines = {}, []
    for pol in POLICIES:
        for form in FORMS:
            d = os.path.join(outroot, f"{pol}-{form}")
            if not os.path.exists(os.path.join(d, "summary.json")):
                res["missing"].append(f"{pol}-{form}")
                continue
            recs = load(d, tok)
            cc = {i: 0 for i in ids}
            for r in recs:
                cc[r["prompt_id"]] += r["reward"] == 1
            n = len(recs) // len(ids)
            assert n * len(ids) == len(recs) and set(cc) == set(ids), f"{pol}-{form}: incomplete records"
            c[(pol, form)] = {i: cc[i] / n for i in ids}
            T = [r["n_tokens"] for r in recs]
            res["policies"].setdefault(pol, {})[form] = {
                "pass1": float(np.mean(list(c[(pol, form)].values()))), "completions": len(recs),
                "truncated": float(np.mean([r["finish"] == "length" for r in recs])),
                "mean_tokens": float(np.mean(T)), "median_tokens": float(np.median(T)),
                "mentions_factor": float(np.mean(["factor" in r["text"].lower() for r in recs]))}
            by = {(r["prompt_id"], r["sample"]): r for r in recs}
            for k in INSPECT:
                r = by[(ids[k], 0)]
                lines += [f"=== {pol} {form} item#{k} {ids[k]} sample 0: tokens {r['n_tokens']} finish {r['finish']} "
                          f"reward {r['reward']}", r["text"][:1200], "[...]", r["text"][-300:], ""]
    for pol in POLICIES:
        if all((pol, f) in c for f in FORMS):
            res["cue_gain"][pol] = boot([c[(pol, "cued")][i] - c[(pol, "uncued")][i] for i in ids], f"gain/{pol}")
    for s in ("s101", "s202"):
        R, U = f"R-{s}", f"U-{s}"
        if all((p, f) in c for p in (R, U) for f in FORMS):
            res["did"][s] = boot([(c[(R, "cued")][i] - c[(R, "uncued")][i]) - (c[(U, "cued")][i] - c[(U, "uncued")][i])
                                  for i in ids], f"did/{s}")
        else:
            res["did"][s] = "missing"
    json.dump(res, open(out + ".json", "w"), indent=1)
    open(out + ".txt", "w").write("\n".join(lines))
    print(json.dumps({k: res[k] for k in ("policies", "cue_gain", "did", "missing")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
