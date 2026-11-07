"""Screening candidates from pass-1 annotations under rubric v1.1 (research/RUBRIC.md).

Pools (construction split): hard (H_train candidates), related, unrelated. Calibration:
seeded subsets of calib_train / calib_val from the same three subjects (any focal label).
Each pool is ordered by a fixed seeded permutation (construction seed 1701) BEFORE screening.
Writes <out>/<pool>.jsonl (id, problem, gold) and <out>/candidates_summary.json.
"""
import argparse
import json
import os
import re

from qis import seeds

SUBJ_ALG = ("algebra", "intermediate_algebra")


def level(r):
    m = re.search(r"\d", r["level"])
    return int(m.group()) if m else 0


def role(a):
    """Rubric v1.1 role from a pass-1 annotation record, or None."""
    if a["figure"] == "yes" or a["confidence"] == "low":
        return None
    if a["pool"] == "factor" and a["label"] == "central":
        if a["direct_practice"] == "no" and level(a) >= 4:
            return "hard"
        if a["direct_practice"] == "yes":
            return "related"
    if a["pool"] == "modular" and a["label"] == "absent" and a["modular"] == "yes":
        return "unrelated"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {}
    for line in open(a.manifest):
        r = json.loads(line)
        man[r["id"]] = r
    pools = {"hard": [], "related": [], "unrelated": [], "target_dev": [], "target_test": [],
             "probe_related_dev": [], "probe_unrelated_dev": []}
    for line in open(a.annotations):
        ann = json.loads(line)
        r = man[ann["id"]]
        if not r.get("gold_eligible") or r["split"] == "excluded":
            continue
        ro = role(ann)
        if ro is None:
            continue
        if r["split"] == "construction":
            pools[ro].append(r)
        elif r["split"] == "study_dev":
            {"hard": pools["target_dev"], "related": pools["probe_related_dev"],
             "unrelated": pools["probe_unrelated_dev"]}[ro].append(r)
        elif r["split"] == "test" and ro == "hard":
            pools["target_test"].append(r)
    for sp in ("calib_train", "calib_val"):
        pools[sp] = [r for r in man.values() if r["split"] == sp and r.get("gold_eligible")
                     and r["subject"] in SUBJ_ALG + ("number_theory",) and "[asy]" not in r["problem"]]
    summary = {}
    for name, rows in pools.items():
        rows.sort(key=lambda r: r["id"])
        perm = seeds.rng(seeds.CONSTRUCTION_SEED, f"candidates/{name}").permutation(len(rows))
        rows = [rows[i] for i in perm]
        if name.startswith("calib"):
            rows = rows[:a.n_calib]
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "problem": r["problem"], "gold": r["gold"],
                                    "subject": r["subject"], "level": r["level"]}) + "\n")
        summary[name] = len(rows)
    json.dump(summary, open(os.path.join(a.out, "candidates_summary.json"), "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
