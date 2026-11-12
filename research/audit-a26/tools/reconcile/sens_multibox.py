"""A26 reconcile: sensitivity (b) "any box" rule. For every finished reward-0 response, test whether an EARLIER
\\boxed{}/\\fbox{} content (not the last one, which the frozen rule grades) is verifier-equivalent to gold, and whether
the gold is split across several boxes (joined box contents verify). Then recompute pass@1 and per-seed contrasts with
"correct = last box correct OR any earlier box correct" (capped attempts stay 0; unknown stays 0).

Uses the project verifier qis.verify.Verifier (unchanged 779c54d..9b2fd05) on compute, as the reconcile brief allows,
and the Qwen3 tokenizer to decode stored token ids. Run:
  env PYTHONPATH=<worktree>/src python sens_multibox.py OUT_JSON
"""
import gzip
import json
import re
import sys

import numpy as np
from transformers import AutoTokenizer

from qis.verify import Verifier

P = "/data/user_data/vineethb/qis"


def main():
    global tok, V, cache
    OUT = sys.argv[1]
    SEED, B = 20261112, 10000
    tok = AutoTokenizer.from_pretrained(P + "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


    def boxes(text):
        out = []
        for m in BOX.finditer(text):
            i, d, j = m.end(), 1, m.end()
            while j < len(text) and d:
                d += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            out.append(text[i:j - 1].strip() if d == 0 else None)
        return out


    def norm(s):
        return re.sub(r"\s+", "", s or "")


    TEST = {it["id"]: it for it in json.load(open(P + "/data/main-v1/target_test_sealed.json"))["pools"]["test"]}
    STRESS = {it["id"]: it for it in json.load(open(P + "/data/stress/stress_v1.json"))["pools"]["stress"]}
    CELLS = {("test", "base"): "test-eval/qis-test-base", ("test", "H-s101"): "test-eval/qis-test-H-s101-sym",
             ("test", "R-s101"): "test-eval/qis-test-R-s101-sym", ("test", "U-s101"): "test-eval/qis-test-U-s101-sym",
             ("test", "H-s202"): "test-eval/qis-test-H-s202-sym", ("test", "R-s202"): "test-eval/qis-test-R-s202-sym",
             ("test", "U-s202"): "test-eval/qis-test-U-s202-sym", ("test", "H-s202-fb116"): "test-eval/qis-test-H-s202-sym-fallback",
             ("test", "cgpu-R-s101"): "test-eval-cgpu/qis-cgpu-R-s101", ("test", "cgpu-R-s202"): "test-eval-cgpu/qis-cgpu-R-s202",
             ("test", "cgpu-U-s101"): "test-eval-cgpu/qis-cgpu-U-s101", ("test", "cgpu-U-s202"): "test-eval-cgpu/qis-cgpu-U-s202"}
    for c in ("base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "H-s202-r128", "R-s202-r128", "U-s202-r128", "R-s101-r64", "H-s202-r64"):
        CELLS[("stress", c)] = f"stress-eval/{c}"

    V = Verifier(workers=3)
    cache = {}


    def equiv(gold, pred):
        """Verifier status for pred (a box CONTENT) against gold; cached."""
        k = (gold, pred)
        if k not in cache:
            r = V.score([{"gold": gold, "response": "\\boxed{" + pred + "}", "truncated": False}])[0]
            cache[k] = r["status"]
        return cache[k]


    res = {"rule": __doc__, "cells": {}, "contrasts": {}, "examples": []}
    PER = {}
    for (panel, cell), d in CELLS.items():
        items = TEST if panel == "test" else STRESS
        cnt = {"attempts": 0, "correct_last_box": 0, "finished_reward0": 0, "finished_reward0_ge2_boxes": 0,
               "earlier_box_equiv_gold": 0, "earlier_box_exact_gold": 0, "earlier_box_verifier_unknown": 0,
               "split_across_boxes": 0, "correct_any_box": 0, "unknown_status_kept_0": 0}
        per = {}
        with gzip.open(f"{P}/{d}/records.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                pid = r["prompt_id"]
                x = per.setdefault(pid, [0, 0, 0])
                x[2] += 1
                cnt["attempts"] += 1
                if r["reward"] == 1:
                    cnt["correct_last_box"] += 1; cnt["correct_any_box"] += 1; x[0] += 1; x[1] += 1
                    continue
                if r["status"] == "unknown":
                    cnt["unknown_status_kept_0"] += 1
                    continue
                if r["finish"] == "length":
                    continue
                cnt["finished_reward0"] += 1
                b = boxes(tok.decode(r["ids"], skip_special_tokens=True))
                if len(b) < 2:
                    continue
                cnt["finished_reward0_ge2_boxes"] += 1
                gold = items[pid]["gold"]
                last = b[-1]
                earlier = []
                for s in b[:-1]:
                    if s is not None and s != last and s not in earlier:
                        earlier.append(s)
                st = [equiv(gold, s) for s in earlier]
                hit = "correct" in st
                if any(norm(s) == norm(gold) for s in earlier):
                    cnt["earlier_box_exact_gold"] += 1
                if "unknown" in st and not hit:
                    cnt["earlier_box_verifier_unknown"] += 1
                split = False
                if not hit:
                    dist = []
                    for s in b:
                        if s is not None and s not in dist:
                            dist.append(s)
                    if len(dist) >= 2:
                        cands = {", ".join(dist)} | {", ".join(dist[-k:]) for k in range(2, min(4, len(dist)) + 1)}
                        split = any(equiv(gold, c) == "correct" for c in sorted(cands))
                if hit:
                    cnt["earlier_box_equiv_gold"] += 1
                if split:
                    cnt["split_across_boxes"] += 1
                if hit or split:
                    cnt["correct_any_box"] += 1; x[1] += 1
                    if len(res["examples"]) < 40:
                        res["examples"].append({"panel": panel, "cell": cell, "pid": pid, "sample": r["sample"], "gold": gold,
                                                "boxes": [s[:120] if s else s for s in b][-6:], "extracted": r["extracted"],
                                                "kind": "earlier_box" if hit else "split"})
        PER[(panel, cell)] = per
        n = cnt["attempts"]
        cnt["pass1_last_box"] = float(np.mean([v[0] / v[2] for v in per.values()]))
        cnt["pass1_any_box"] = float(np.mean([v[1] / v[2] for v in per.values()]))
        cnt["change"] = cnt["pass1_any_box"] - cnt["pass1_last_box"]
        res["cells"][f"{panel}|{cell}"] = cnt
        print(panel, cell, cnt, flush=True)
    V.close()


    def boot(a, b, j):
        ids = sorted(a)
        d = np.array([a[p][j] / a[p][2] - b[p][j] / b[p][2] for p in ids])
        rng = np.random.default_rng(SEED)
        m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
        return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


    PAIRS = {"test": [("R-s101", "U-s101"), ("R-s101", "H-s101"), ("U-s101", "H-s101"), ("R-s202", "U-s202"), ("R-s202", "H-s202"),
                      ("U-s202", "H-s202"), ("cgpu-R-s101", "cgpu-U-s101"), ("cgpu-R-s202", "cgpu-U-s202")],
             "stress": [("R-s101-r128", "U-s101-r128"), ("R-s101-r128", "H-s101-r128"), ("U-s101-r128", "H-s101-r128"),
                        ("R-s202-r128", "U-s202-r128"), ("R-s202-r128", "H-s202-r128"), ("U-s202-r128", "H-s202-r128")]}
    for panel, pairs in PAIRS.items():
        for a, b in pairs:
            res["contrasts"][f"{panel}|{a} - {b}"] = {"last_box": boot(PER[(panel, a)], PER[(panel, b)], 0),
                                                     "any_box": boot(PER[(panel, a)], PER[(panel, b)], 1)}
    res["verifier_calls"] = len(cache)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
