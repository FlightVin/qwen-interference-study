"""A26 x-transfer: lenient re-score of control-panel failures (read-only; compute).

For every reward-0 attempt in $P/control-eval/<policy>/records.jsonl.gz: does any box (not only the last) or the last number
verify against the gold (project verifier)? Also the number of distinct box contents in multi-box wrong attempts.
Usage: python control_lenient.py P MODEL PANEL_JSON OUT_JSON
"""
import gzip
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import NUM, all_boxes  # noqa: E402


def main():
    from qis.verify import Verifier
    from transformers import AutoTokenizer
    P, model, panel, out = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(model)
    gold = {x["id"]: x["gold"] for x in json.load(open(panel))["pools"]["control"]}
    ver, res = Verifier(4), {}
    for pol in sorted(os.listdir(os.path.join(P, "control-eval"))):
        rows, checks = [], []
        for line in gzip.open(os.path.join(P, "control-eval", pol, "records.jsonl.gz"), "rt"):
            r = json.loads(line)
            if r["reward"] == 1:
                continue
            t = tok.decode(r["ids"], skip_special_tokens=True)
            b = all_boxes(t)
            k = len(rows)
            g = gold[r["prompt_id"]]
            rows.append({"fin": r["finish"] != "length", "nbox": len(b), "ndist": len({x for x in b if x is not None}),
                         "any": False, "num": False})
            for x in {x for x in b if x is not None}:
                checks.append((k, "any", g, x))
            n = NUM.findall(t)
            if n:
                checks.append((k, "num", g, n[-1].replace(",", "")))
        sc = ver.score([{"gold": g, "response": "\\boxed{" + v + "}", "truncated": False} for _, _, g, v in checks])
        for (k, w, _, _), s in zip(checks, sc):
            if s["reward"] == 1:
                rows[k][w] = True
        fin = [x for x in rows if x["fin"]]
        multi = [x for x in fin if x["nbox"] >= 2]
        res[pol] = {"failures": len(rows), "finished_failures": len(fin), "finished_any_box": sum(x["any"] for x in fin),
                    "finished_last_num": sum(x["num"] for x in fin),
                    "truncated_any_box": sum(x["any"] for x in rows if not x["fin"]),
                    "multi_box_wrong": len(multi), "multi_box_wrong_distinct": dict(Counter(min(x["ndist"], 4) for x in multi)),
                    "multi_box_wrong_any_box": sum(x["any"] for x in multi)}
        print(pol, res[pol], flush=True)
    ver.close()
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
