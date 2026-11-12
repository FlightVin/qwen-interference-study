"""A26 x-transfer: per-problem counts and failure classes from the control-panel records (read-only; compute).

Reads   $P/control-eval/<policy>/{summary.json,records.jsonl.gz} for every policy directory present.
Writes  OUT_JSON: {policy: {"summary": {...without per_problem}, "per_problem": {id: counts}}}
Classes are those of taxonomy.py (correct, unknown, truncated, no_box, box_unbalanced, multi_box_wrong, single_box_wrong);
per problem also c_within_8192 (correct with n_tokens <= 8192) and any_box rescues are not computed here.
Usage: python control_extract.py P MODEL OUT_JSON
"""
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import CLASSES, all_boxes  # noqa: E402


def main():
    P, model, out = sys.argv[1:4]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    root = os.path.join(P, "control-eval")
    res = {}
    for pol in sorted(os.listdir(root)):
        d = os.path.join(root, pol)
        if not os.path.exists(os.path.join(d, "summary.json")):
            res[pol] = {"status": "no summary.json"}
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        per = defaultdict(lambda: dict({c: 0 for c in CLASSES}, n=0, c_within_8192=0, tokens=0))
        for line in gzip.open(os.path.join(d, "records.jsonl.gz"), "rt"):
            r = json.loads(line)
            fin = r["finish"] != "length"
            if r["reward"] == 1:
                cls = "correct"
            elif r["reward"] is None:
                cls = "unknown"
            elif not fin:
                cls = "truncated"
            else:
                b = all_boxes(tok.decode(r["ids"], skip_special_tokens=True))
                cls = ("no_box" if not b else "box_unbalanced" if b[-1] is None else
                       "multi_box_wrong" if len(b) >= 2 else "single_box_wrong")
            p = per[r["prompt_id"]]
            p[cls] += 1
            p["n"] += 1
            p["tokens"] += r["n_tokens"]
            p["c_within_8192"] += cls == "correct" and r["n_tokens"] <= 8192
        for x in s["per_problem"]:                     # cross-check against the job's own counts
            assert per[x["id"]]["correct"] == x["c"] and per[x["id"]]["n"] == x["n"], (pol, x)
        res[pol] = {"summary": {k: v for k, v in s.items() if k != "per_problem"}, "per_problem": dict(per)}
        print(pol, s["pass1"], s["truncated"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
