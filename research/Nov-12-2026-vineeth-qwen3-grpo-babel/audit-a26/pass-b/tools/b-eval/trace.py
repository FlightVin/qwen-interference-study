"""A26 pass B task 7a-c: traced cases + multi-box census + independent last-box extraction check (sealed test).

Uses only the Qwen3 tokenizer (transformers) to decode token ids; no project code.
"""
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import P, dump

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(P + "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
TEST = {it["id"]: it for it in json.load(open(P + "/data/main-v1/target_test_sealed.json"))["pools"]["test"]}
ORDER = list(TEST)
BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


def boxes(text):
    out = []
    for m in BOX.finditer(text):
        i, d = m.end(), 1
        j = i
        while j < len(text) and d:
            d += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append(text[i:j - 1].strip() if d == 0 else None)
    return out


def recs(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


out = {}
CELLS = {"base": "qis-test-base", "H-s101": "qis-test-H-s101-sym", "R-s101": "qis-test-R-s101-sym",
         "U-s101": "qis-test-U-s101-sym", "H-s202": "qis-test-H-s202-sym", "R-s202": "qis-test-R-s202-sym",
         "U-s202": "qis-test-U-s202-sym"}
census, example = {}, None
for cell, d in CELLS.items():
    c = {"responses": 0, "extract_mismatch": 0, "ge2_boxes": 0, "ge2_distinct": 0, "ge2_reward1": 0,
         "ge2_reward0_finished": 0, "ge2_reward0_capped": 0, "ge2_reward0_finished_nonlast_equals_gold": 0}
    for r in recs(f"{P}/test-eval/{d}/records.jsonl.gz"):
        c["responses"] += 1
        text = tok.decode(r["ids"], skip_special_tokens=True)
        b = boxes(text)
        mine = b[-1] if b else None
        if mine != r["extracted"]:
            c["extract_mismatch"] += 1
        if len(b) >= 2:
            c["ge2_boxes"] += 1
            distinct = len({x for x in b}) > 1
            c["ge2_distinct"] += distinct
            if r["reward"] == 1:
                c["ge2_reward1"] += 1
            elif r["finish"] == "length":
                c["ge2_reward0_capped"] += 1
            else:
                c["ge2_reward0_finished"] += 1
                g = TEST[r["prompt_id"]]["gold"].replace(" ", "")
                if any((x or "").replace(" ", "") == g for x in b[:-1]):
                    c["ge2_reward0_finished_nonlast_equals_gold"] += 1
                    if example is None:
                        example = {"cell": cell, "problem": r["prompt_id"], "sample": r["sample"], "gold": TEST[r["prompt_id"]]["gold"],
                                   "boxes": b, "record_extracted": r["extracted"], "status": r["status"], "reward": r["reward"],
                                   "finish": r["finish"], "n_tokens": r["n_tokens"]}
    census[cell] = c
out["multibox_census"] = census
out["trace_c_example"] = example
if example is None:  # fall back to any finished multi-box response with distinct boxes
    for r in recs(f"{P}/test-eval/qis-test-base/records.jsonl.gz"):
        b = boxes(tok.decode(r["ids"], skip_special_tokens=True))
        if len(b) >= 2 and len(set(b)) > 1 and r["finish"] != "length":
            out["trace_c_example"] = {"cell": "base", "problem": r["prompt_id"], "sample": r["sample"],
                                      "gold": TEST[r["prompt_id"]]["gold"], "boxes": b, "record_extracted": r["extracted"],
                                      "status": r["status"], "reward": r["reward"]}
            break

# 7a: base, intermediate_algebra/test/00874 sample 4
for r in recs(f"{P}/test-eval/qis-test-base/records.jsonl.gz"):
    if r["prompt_id"] == "intermediate_algebra/test/00874" and r["sample"] == 4:
        text = tok.decode(r["ids"], skip_special_tokens=True)
        out["trace_a"] = {"problem": r["prompt_id"], "sample": 4, "gold": TEST[r["prompt_id"]]["gold"], "seed": r["seed"],
                          "n_tokens": r["n_tokens"], "finish": r["finish"], "status": r["status"], "reward": r["reward"],
                          "extracted": r["extracted"], "my_last_box": boxes(text)[-1], "tail": text[-400:]}
        break

# 7b: first extended + recovered attempt of R-s101 on the sealed test (panel order)
orig = {}
for r in recs(f"{P}/test-eval/qis-test-R-s101-sym/records.jsonl.gz"):
    orig[(r["prompt_id"], r["sample"])] = r
rows = {}
for r in recs(f"{P}/length-test/R-s101-r128/all/rows.jsonl.gz"):
    if r["extended"]:
        rows[(r["prompt_id"], r["sample"])] = r
pick = None
for p in ORDER:
    for j in range(32):
        r = rows.get((p, j))
        if r and r["reward_2"] == 1:
            pick = (p, j); break
    if pick:
        break
o, w = orig[pick], rows[pick]
t_orig = tok.decode(o["ids"], skip_special_tokens=True)
t_full = tok.decode(o["ids"] + w["new_ids"], skip_special_tokens=True)
out["trace_b"] = {"problem": pick[0], "sample": pick[1], "gold": TEST[pick[0]]["gold"],
                  "original": {k: o[k] for k in ("seed", "n_tokens", "finish", "status", "reward", "extracted")},
                  "row": {k: w[k] for k in w if k != "new_ids"}, "len_new_ids": len(w["new_ids"]),
                  "original_tail": t_orig[-300:], "full_tail": t_full[-400:], "my_last_box_full": (boxes(t_full) or [None])[-1],
                  "problem_c_12288": sum(orig[(pick[0], j)]["reward"] == 1 for j in range(32)),
                  "problem_c_16384": sum((rows[(pick[0], j)]["reward_2"] if (pick[0], j) in rows else orig[(pick[0], j)]["reward"]) == 1
                                         for j in range(32))}

# ID overlap between sealed test, stress and cue (prefix-normalised)
norm = lambda i: i.split(":", 1)[1] if i.startswith("math:") else i
st = {norm(it["id"]) for it in json.load(open(P + "/data/stress/stress_v1.json"))["pools"]["stress"]}
cue = {norm(it["id"]) for it in json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/cue_v1_uncued.json"))["pools"]["cue"]}
te = {norm(i) for i in TEST}
out["id_overlap"] = {"test_stress": len(te & st), "test_cue": len(te & cue), "stress_cue": len(st & cue),
                     "stress_ids_from_math_test_split": sum("/test/" in i for i in st)}
dump("eval_trace.json", out)
print(json.dumps(out, indent=1)[:5000])
