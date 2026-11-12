"""A26 x-transfer task 1: failure taxonomy of in-run probe attempts (read-only on $P).

Reads   $P/runs/qis-main-{H,R,U}-s{101,202}-sym/records/eval_{probe_J_U,probe_J_R,dev}_rNNNN.jsonl.gz
        $P/data/main-v1/bundle_main_v1.json (golds; J_H = pool probe_J_H, its attempts come from the dev file)
Writes  OUT/taxonomy.json (per run x round x panel counts, lenient re-scores, per-problem c/8)

Classes (exclusive, in this order):
  correct           reward == 1
  unknown           reward is None (verifier failure)
  truncated         finish == "length"
  no_box            finished, no \\boxed{ or \\fbox{ in the text
  box_unbalanced    finished, a box opener exists but the last box has unbalanced braces (last_boxed -> None)
  multi_box_wrong   finished, >= 2 boxes, reward 0
  single_box_wrong  finished, exactly 1 box, reward 0
Lenient re-scores (project verifier, same worker pool as training, on compute):
  any_box   some box (not only the last) verifies against the gold
  last_num  the last number in the response verifies against the gold
  within_8192  correct with n_tokens <= 8192 (the round-0/32 cap; counterfactual view of the same traces)
Usage: python taxonomy.py P MODEL_DIR OUT_DIR   (PYTHONPATH must contain the project src for qis.verify)
"""
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

from qis.verify import Verifier

RUNS = [f"qis-main-{a}-s{s}-sym" for a in "HRU" for s in (101, 202)]
ROUNDS = [0, 32, 64, 96, 128]
PANELS = {"J_U": "probe_J_U", "J_R": "probe_J_R", "J_H": "dev"}
OPEN = re.compile(r"\\(?:boxed|fbox)\s*\{")
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
CLASSES = ["correct", "unknown", "truncated", "no_box", "box_unbalanced", "multi_box_wrong", "single_box_wrong"]


def all_boxes(text):
    out = []
    for m in OPEN.finditer(text):
        i, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append(text[i:j - 1].strip() if depth == 0 else None)
    return out


def main():
    P, model, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    pools = json.load(open(os.path.join(P, "data/main-v1/bundle_main_v1.json")))["pools"]
    gold = {it["id"]: it["gold"] for v in pools.values() for it in v}
    ids_of = {"J_U": [it["id"] for it in pools["probe_J_U"]], "J_R": [it["id"] for it in pools["probe_J_R"]],
              "J_H": [it["id"] for it in pools["probe_J_H"]]}
    ver = Verifier(4)
    res = {"meta": {"P": P, "rounds": ROUNDS, "runs": RUNS, "classes": CLASSES,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "cells": {}, "missing": []}
    for run in RUNS:
        for rnd in ROUNDS:
            for pn, fname in PANELS.items():
                path = os.path.join(P, "runs", run, "records", f"eval_{fname}_r{rnd:04d}.jsonl.gz")
                if not os.path.exists(path):
                    res["missing"].append(path)
                    continue
                keep = set(ids_of[pn])
                rows, checks = [], []
                with gzip.open(path, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        if r["prompt_id"] not in keep:
                            continue
                        text = tok.decode(r["ids"], skip_special_tokens=True)
                        boxes = all_boxes(text)
                        fin = r["finish"] != "length"
                        if r["reward"] == 1:
                            cls = "correct"
                        elif r["reward"] is None:
                            cls = "unknown"
                        elif not fin:
                            cls = "truncated"
                        elif not boxes:
                            cls = "no_box"
                        elif boxes[-1] is None:
                            cls = "box_unbalanced"
                        elif len(boxes) >= 2:
                            cls = "multi_box_wrong"
                        else:
                            cls = "single_box_wrong"
                        nums = NUM.findall(text)
                        row = {"id": r["prompt_id"], "cls": cls, "fin": fin, "n_tokens": r["n_tokens"],
                               "n_box": len(boxes), "n_distinct_box": len({b for b in boxes if b is not None}),
                               "reward": r["reward"], "any_box": False, "last_num": False,
                               "has_think": "<think>" in text}
                        k = len(rows)
                        rows.append(row)
                        if cls != "correct":
                            g = gold[r["prompt_id"]]
                            for b in {b for b in boxes if b is not None}:
                                checks.append((k, "any_box", g, b))
                            if nums:
                                checks.append((k, "last_num", g, nums[-1].replace(",", "")))
                if checks:
                    sc = ver.score([{"gold": g, "response": "\\boxed{" + v + "}", "truncated": False}
                                    for _, _, g, v in checks])
                    for (k, what, _, _), s in zip(checks, sc):
                        if s["reward"] == 1:
                            rows[k][what] = True
                cell = {"attempts": len(rows), "classes": dict(Counter(x["cls"] for x in rows))}
                for c in CLASSES:
                    cell["classes"].setdefault(c, 0)
                nonc = [x for x in rows if x["cls"] != "correct"]
                cell["lenient"] = {
                    "finished_any_box": sum(x["any_box"] for x in nonc if x["fin"]),
                    "finished_last_num": sum(x["last_num"] for x in nonc if x["fin"]),
                    "finished_any_box_or_last_num": sum(x["any_box"] or x["last_num"] for x in nonc if x["fin"]),
                    "truncated_any_box": sum(x["any_box"] for x in nonc if not x["fin"]),
                    "truncated_last_num": sum(x["last_num"] for x in nonc if not x["fin"]),
                    "multi_box_wrong_any_box": sum(x["any_box"] for x in nonc if x["cls"] == "multi_box_wrong"),
                    "single_box_wrong_last_num": sum(x["last_num"] for x in nonc if x["cls"] == "single_box_wrong"),
                    "no_box_last_num": sum(x["last_num"] for x in nonc if x["cls"] == "no_box"),
                }
                cell["correct_within_8192"] = sum(x["cls"] == "correct" and x["n_tokens"] <= 8192 for x in rows)
                cell["over_8192"] = sum(x["n_tokens"] > 8192 for x in rows)
                cell["has_think"] = sum(x["has_think"] for x in rows)
                cell["mean_tokens"] = {c: (sum(x["n_tokens"] for x in rows if x["cls"] == c) /
                                           max(1, sum(x["cls"] == c for x in rows))) for c in CLASSES}
                per = defaultdict(lambda: {"c": 0, "n": 0, "trunc": 0, "c_lenient": 0, "c_within_8192": 0, "fmt_fail": 0})
                for x in rows:
                    p = per[x["id"]]
                    p["n"] += 1
                    p["c"] += x["cls"] == "correct"
                    p["trunc"] += x["cls"] == "truncated"
                    p["c_lenient"] += x["cls"] == "correct" or (x["fin"] and x["any_box"])
                    p["c_within_8192"] += x["cls"] == "correct" and x["n_tokens"] <= 8192
                    p["fmt_fail"] += x["cls"] in ("no_box", "box_unbalanced", "multi_box_wrong")
                cell["per_problem"] = {i: per[i] for i in ids_of[pn] if i in per}
                res["cells"][f"{run}|{rnd}|{pn}"] = cell
                print(run, rnd, pn, cell["attempts"], cell["classes"], flush=True)
    ver.close()
    res["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(os.path.join(out_dir, "taxonomy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
