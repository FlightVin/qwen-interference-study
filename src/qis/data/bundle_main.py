"""Build the frozen main-study bundle (pools + evaluation panels) and its manifest.

Pools: hard (H_train, 46; D47), related / unrelated (63 pairs in matched order; D48: related[k] is
paired with unrelated[k]). Evaluation panels: dev (target_dev, 53), probe_J_H (32, subset of dev),
probe_J_R (28), probe_J_U (32) — rubric-selected (D34). target_test is written to a separate sealed
file, never to the training bundle.
Usage: python -m qis.data.bundle_main PROJ_DATA OUT_DIR
"""
import hashlib
import json
import os
import sys


def text_index(pd):
    idx = {}
    for line in open(f"{pd}/data/splits-v1/manifest.jsonl"):
        r = json.loads(line)
        idx[r["id"]] = {"id": r["id"], "problem": r["problem"], "gold": r["gold"], "source": "MATH", "subject": r["subject"], "level": r["level"]}
    for line in open("data/external/omni_tranche_v1.jsonl"):
        r = json.loads(line)
        idx[r["id"]] = {"id": r["id"], "problem": r["problem"], "gold": r["gold"], "source": "Omni-MATH", "subject": "omni", "level": str(r["difficulty"])}
    for b in ("R1", "R2", "R3", "R4", "R5", "U1", "U2", "U3"):
        for line in open(f"data/authored/{b}.jsonl"):
            r = json.loads(line)
            idx[r["id"]] = {"id": r["id"], "problem": r["problem"], "gold": r["answer"], "source": f"authored-{b}",
                            "subject": "authored", "level": "authored", "template_family": r["template_family"]}
    return idx


def main(pd, out):
    os.makedirs(out, exist_ok=True)
    idx = text_index(pd)
    hsel = json.load(open("data/screening/hsel_k8.json"))["selected"]
    match = json.load(open("data/screening/match_k8.json"))["pairs"]
    panels = json.load(open("data/panels/panels.json"))
    pools = {"hard": [{**idx[h["id"]], "band": h["band"], "c32_k8": h["c32"], "family": h["family"]} for h in sorted(hsel, key=lambda x: x["id"])],
             "related": [idx[p["r"]] for p in match], "unrelated": [idx[p["u"]] for p in match]}
    for k, name in (("target_dev", "dev"), ("probe_J_H", "probe_J_H"), ("probe_J_R", "probe_J_R"), ("probe_J_U", "probe_J_U")):
        pools[name] = [idx[i] for i in panels[k]["ids"]]
    bundle = {"pools": pools, "note": "main-study bundle v1 (cap 8,192; D45-D48); related[k] paired with unrelated[k]"}
    with open(f"{out}/bundle_main_v1.json", "w") as f:
        json.dump(bundle, f, sort_keys=True)
    with open(f"{out}/target_test_sealed.json", "w") as f:
        json.dump({"pools": {"test": [idx[i] for i in panels["target_test"]["ids"]]}}, f, sort_keys=True)
    man = {k: {"n": len(v), "ids": [x["id"] for x in v]} for k, v in pools.items()}
    man["sha256"] = {n: hashlib.sha256(open(f"{out}/{n}", "rb").read()).hexdigest()
                     for n in ("bundle_main_v1.json", "target_test_sealed.json")}
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print({k: v["n"] for k, v in man.items() if k != "sha256"}, man["sha256"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
