"""H_train selection over the combined native + external pool (A4.2-3), committed before any
external screening count was viewed. Same bands and composition as select_hard.py (D28/D31):
eligible items from (a) the native selection's eligible list and (b) Omni-MATH items whose pass-1 and
pass-2 annotations agree (or third read resolves) on central, direct_practice=no, proof_like=no,
confidence not low, no figure, with a v1.3 primary operation (pass 2), 32 fresh attempts, no
unknowns, band core/moderate, truncated attempts < 50% of failures. Composition rule identical
(all core up to 64, then moderate up to min(64-C, C)); pooled family round-robin in seeded order.
"""
import argparse
import json

from qis.data.select_hard import band, select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="native selection.json (uses its 'eligible')")
    ap.add_argument("--omni-s2", required=True)
    ap.add_argument("--omni-p1", required=True)
    ap.add_argument("--omni-p2", required=True)
    ap.add_argument("--omni-third", required=True)
    ap.add_argument("--omni-intake", required=True, help="intake_report.jsonl (excluded field)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(f)}  # noqa: E731
    p1, p2, th, s2, intake = rd(a.omni_p1), rd(a.omni_p2), rd(a.omni_third), rd(a.omni_s2), rd(a.omni_intake)
    key = lambda r: (r["label"], r["direct_practice"], r["proof_like"])  # noqa: E731
    elig, excl = list(json.load(open(a.native))["eligible"]), []
    for i, x2 in p2.items():
        x1, why = p1[i], None
        fin = x2 if key(x1) == key(x2) else (th[i] if i in th and th[i]["resolved"] else None)
        c = s2.get(i)
        if intake.get(i, {}).get("excluded"):
            why = "intake: " + intake[i]["excluded"]
        elif fin is None:
            why = "unresolved disagreement"
        elif not (fin["label"] == "central" and fin["direct_practice"] == "no" and fin["proof_like"] == "no"):
            why = "not central multi-step short-answer"
        elif x2["confidence"] == "low" or x1["confidence"] == "low" or x2["figure"] == "yes":
            why = "low confidence or figure"
        elif x2["primary"] == "none":
            why = "no primary operation"
        elif c is None or c["n"] != 32:
            why = "no 32-attempt count"
        elif c["unknown"]:
            why = "verifier unknown"
        elif band(c["c"]) is None:
            why = f"outside bands (c={c['c']}/32)"
        elif c["truncated"] >= 0.5 * max(1, c["n"] - c["c"]):
            why = "difficulty driven by truncation"
        if why:
            excl.append({"id": i, "reason": why}); continue
        elig.append({"id": i, "source": "omni", "level": "omni", "band": band(c["c"]), "c32": c["c"], "family": x2["primary"]})
    sel = select(elig)
    json.dump({"selected": sel, "eligible": elig, "omni_excluded": excl}, open(a.out, "w"), indent=1)
    from collections import Counter
    print(json.dumps({"eligible": len(elig), "eligible_by_source_band": Counter(f"{e['source']}/{e['band']}" for e in elig),
                      "selected": len(sel), "selected_by_band": Counter(e["band"] for e in sel),
                      "selected_by_source": Counter(e["source"] for e in sel),
                      "selected_by_family": Counter(e["family"] for e in sel),
                      "omni_excluded": Counter(e["reason"].split(" (")[0] for e in excl)}, indent=1))


if __name__ == "__main__":
    main()
