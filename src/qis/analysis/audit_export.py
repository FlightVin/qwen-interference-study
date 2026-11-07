"""Export the §3.7 baseline error-audit sample (project.md §3 audit rule; amendment A8.3; run in a CPU job).

Rule (fixed before any record is read): candidates = every H candidate re-screened at the 8,192 cap
(`h_k8.jsonl`, native + Omni-MATH). An attempt is "incorrect" when reward == 0 and status != unknown
(wrong, no_answer or truncated). Items with at least one incorrect attempt are eligible. Seeded
selection (CONSTRUCTION_SEED, stream "audit37"): up to 64 items in permuted order, then one incorrect
attempt per item (at most one per problem). Each record carries the problem statement, gold answer,
primary technique tag, the full decoded non-thinking response, and every \\boxed{} content (split-box check).
Usage: python -m qis.analysis.audit_export PROJ_DATA ANN_DIR OUT_DIR MODEL
"""
import glob
import gzip
import json
import os
import sys

from qis import seeds

N_MAX = 64


def all_boxed(text):
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + len("\\boxed{"):k - 1])
        i = k


def primary_tags(ann):
    tags = {}
    for name in ("tags_v13.jsonl", "omni_pass2.jsonl", "omni_third_read.jsonl"):
        p = os.path.join(ann, name)
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if r.get("primary") not in (None, "none"):
                    tags[r["id"]] = r["primary"]
    return tags


def main(pd, ann, out, model):
    os.makedirs(out, exist_ok=True)
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{pd}/data/rescreen-k8-h/h_k8.jsonl")}
    wrong = {}
    for p in sorted(glob.glob(f"{pd}/screen/s2k8/h/s2k8_records_*.jsonl.gz")):
        for line in gzip.open(p, "rt"):
            r = json.loads(line)
            if r["id"] in items and r["reward"] == 0 and r["status"] != "unknown":
                wrong.setdefault(r["id"], []).append(r)
    elig = sorted(wrong)
    rng = seeds.rng(seeds.CONSTRUCTION_SEED, "audit37")
    chosen = [elig[i] for i in rng.permutation(len(elig))[:N_MAX]]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tags = primary_tags(ann)
    with gzip.open(os.path.join(out, "audit37_sample.jsonl.gz"), "wt") as f:
        for i in chosen:
            att = sorted(wrong[i], key=lambda r: r["sample"])
            r = att[int(rng.integers(0, len(att)))]
            text = tok.decode(r["ids"], skip_special_tokens=True)
            f.write(json.dumps({"id": i, "source": items[i].get("source"), "primary_tag": tags.get(i),
                                "problem": items[i]["problem"], "gold": items[i]["gold"], "sample": r["sample"],
                                "status": r["status"], "finish": r["finish"], "n_tokens": r["n_tokens"],
                                "extracted": r["extracted"], "all_boxed": all_boxed(text), "response": text}) + "\n")
    man = {"candidates": len(items), "eligible_items": len(elig), "sampled": len(chosen),
           "rule": "seeded (1701, 'audit37'): permute eligible ids, take <=64, one incorrect attempt each",
           "ids": chosen}
    json.dump(man, open(os.path.join(out, "audit37_manifest.json"), "w"), indent=1)
    print({k: v for k, v in man.items() if k != "ids"})


if __name__ == "__main__":
    main(*sys.argv[1:5])
