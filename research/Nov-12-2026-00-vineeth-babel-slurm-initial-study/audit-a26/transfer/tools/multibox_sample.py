"""A26 x-transfer: describe multi-box and single-box wrong attempts (box counts, distinct contents, examples). Read-only.
Usage: python multibox_sample.py P MODEL OUT_JSON"""
import gzip, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import all_boxes
P, model, out = sys.argv[1:4]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model)
pools = json.load(open(os.path.join(P, "data/main-v1/bundle_main_v1.json")))["pools"]
gold = {it["id"]: it["gold"] for v in pools.values() for it in v}
jh = {it["id"] for it in pools["probe_J_H"]}
res = {}
for run in ["qis-main-H-s101-sym", "qis-main-R-s202-sym"]:
    for rnd in (0, 128):
        for pn, fn in (("J_U", "probe_J_U"), ("J_R", "probe_J_R"), ("J_H", "dev")):
            nb, nd, ex, tail_after = Counter(), Counter(), [], Counter()
            for line in gzip.open(os.path.join(P, "runs", run, "records", f"eval_{fn}_r{rnd:04d}.jsonl.gz"), "rt"):
                r = json.loads(line)
                if pn == "J_H" and r["prompt_id"] not in jh:
                    continue
                if r["reward"] != 0 or r["finish"] == "length":
                    continue
                t = tok.decode(r["ids"], skip_special_tokens=True)
                b = all_boxes(t)
                if len(b) < 2:
                    continue
                nb[min(len(b), 6)] += 1
                nd[min(len({x for x in b if x is not None}), 6)] += 1
                last = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", t))[-1].start()
                tail_after[min(len(t) - last, 2000) // 200 * 200] += 1
                if len(ex) < 4:
                    ex.append({"id": r["prompt_id"], "gold": gold[r["prompt_id"]], "boxes": b[:8], "n_tokens": r["n_tokens"],
                               "end": t[-600:]})
            res[f"{run}|{rnd}|{pn}"] = {"n_box": dict(nb), "n_distinct": dict(nd), "chars_after_last_box_open": dict(tail_after),
                                        "examples": ex}
json.dump(res, open(out, "w"), indent=1)
