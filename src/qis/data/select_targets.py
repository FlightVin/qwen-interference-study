"""Held-out target and probe panels, selected by rubric only (never by model success).

Final label per item = pass-1/pass-2 agreement, else the third-read resolution (unresolved -> excluded).
target_dev  : study_dev, central, direct_practice=no, level 4-5, conf not low, no figure; all, seeded order.
target_test : official test, same rule; first 128 in seeded candidate order.
J_H probe   : first 32 of target_dev (seeded order).
J_R probe   : study_dev, central, direct_practice=yes; up to 32 in seeded order.
J_U probe   : study_dev number theory, absent, modular=yes; up to 32 in seeded order.
Candidate files carry the seeded order (qis.data.candidates).
"""
import argparse
import json
import os


def final_labels(p1, p2, third):
    out = {}
    for i, b in p2.items():
        a = p1[i]
        if (a["label"], a["direct_practice"], a["modular"]) == (b["label"], b["direct_practice"], b["modular"]):
            out[i] = {**b, "source": "agree"}
        elif i in third and third[i]["resolved"]:
            out[i] = {**b, **{k: third[i][k] for k in ("label", "direct_practice", "modular", "confidence")}, "source": "third"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ann", required=True, help="data/annotations directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(f)}  # noqa: E731
    p1 = rd(os.path.join(a.ann, "pass1.jsonl"))
    p2 = rd(os.path.join(a.ann, "pass2_targets_probes.jsonl"))
    third = rd(os.path.join(a.ann, "third_read.jsonl"))
    fin = final_labels(p1, p2, third)
    ok = lambda i: i in fin and fin[i]["confidence"] != "low" and fin[i]["figure"] == "no"  # noqa: E731
    order = lambda name: [json.loads(l) for l in open(os.path.join(a.cand, f"{name}.jsonl"))]  # noqa: E731
    hard_rule = lambda i: ok(i) and fin[i]["label"] == "central" and fin[i]["direct_practice"] == "no"  # noqa: E731
    panels = {
        "target_dev": [r for r in order("target_dev") if hard_rule(r["id"])],
        "target_test": [r for r in order("target_test") if r["id"] in p2 and hard_rule(r["id"])][:128],
        "probe_J_R": [r for r in order("probe_related_dev") if ok(r["id"]) and fin[r["id"]]["label"] == "central"
                      and fin[r["id"]]["direct_practice"] == "yes"][:32],
        "probe_J_U": [r for r in order("probe_unrelated_dev") if ok(r["id"]) and fin[r["id"]]["label"] == "absent"
                      and fin[r["id"]]["modular"] == "yes"][:32],
    }
    panels["probe_J_H"] = panels["target_dev"][:32]
    os.makedirs(a.out, exist_ok=True)
    for k, rows in panels.items():
        with open(os.path.join(a.out, f"{k}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    summary = {k: {"n": len(v), "ids": [r["id"] for r in v]} for k, v in panels.items()}
    json.dump(summary, open(os.path.join(a.out, "panels.json"), "w"), indent=1)
    print({k: len(v) for k, v in panels.items()})


if __name__ == "__main__":
    main()
