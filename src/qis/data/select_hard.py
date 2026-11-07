"""H_train selection rule (A4.2), fixed and committed before full stage-2 counts were viewed.

Eligible: native hard (L4-5) or hard_reserve (L3) construction candidate; pass-1 AND pass-2 agree
(or the third read resolves a disagreement) on central + direct_practice=no; a v1.3 primary operation (pass-2 confidence not low, figure no); stage-2 counts on 32 fresh
attempts with no verifier unknown; c in core 2-8 or moderate 9-12; difficulty not driven by
truncation (truncated attempts < 50% of failed attempts).
Composition: target 64 distinct problems; core share >= 1/2.
  C = #core eligible. If C >= 64: take 64 core. Else take all core, then moderate up to
  min(64 - C, C) so that core stays >= half.
Order inside a band: round-robin over structure families (family with fewest selected first,
ties by family name), items within a family in the recorded seeded order
(seeds.rng(1701, "select_hard/order")). Output: manifest with id, source, level, band, c32, family.
"""
import argparse
import json

from qis import seeds

TARGET = 64


def round_robin(items, k, already):
    fam_count = dict(already)
    by_fam = {}
    for it in items:
        by_fam.setdefault(it["family"], []).append(it)
    out = []
    while len(out) < k and any(by_fam.values()):
        fam = min((f for f in by_fam if by_fam[f]), key=lambda f: (fam_count.get(f, 0), f))
        out.append(by_fam[fam].pop(0))
        fam_count[fam] = fam_count.get(fam, 0) + 1
    return out, fam_count


def select(cands):
    """cands: dicts with id, band ('core'|'moderate'), family. Returns selected list."""
    order = seeds.rng(seeds.CONSTRUCTION_SEED, "select_hard/order").permutation(len(cands))
    ranked = [sorted(cands, key=lambda x: x["id"])[i] for i in order]
    core = [c for c in ranked if c["band"] == "core"]
    mod = [c for c in ranked if c["band"] == "moderate"]
    take_core, fams = round_robin(core, min(len(core), TARGET), {})
    n_mod = min(TARGET - len(take_core), len(take_core))
    take_mod, fams = round_robin(mod, n_mod, fams)
    return take_core + take_mod


def band(c):
    return "core" if 2 <= c <= 8 else ("moderate" if 9 <= c <= 12 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="jsonl: id, source(hard|hard_reserve), level")
    ap.add_argument("--s2", required=True, help="jsonl stage-2 counts (id, c, n, unknown, truncated)")
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--pass2", required=True, help="jsonl with id, label, direct_practice, confidence, figure")
    ap.add_argument("--third", required=True, help="third-read resolutions (id, label, direct_practice, resolved)")
    ap.add_argument("--tags", required=True, help="v1.3 technique tags (id, primary); primary is the diversity family (D30)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.pass1)}
    p2 = {json.loads(l)["id"]: json.loads(l) for l in open(a.pass2)}
    s2 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s2)}
    third = {json.loads(l)["id"]: json.loads(l) for l in open(a.third)}
    tags = {json.loads(l)["id"]: json.loads(l) for l in open(a.tags)}
    cands, excl = [], []
    for line in open(a.candidates):
        it = json.loads(line)
        i = it["id"]
        why = None
        x1, x2, c = p1.get(i), p2.get(i), s2.get(i)
        if c is None or c["n"] != 32:
            why = "no 32-attempt stage-2 count"
        elif c["unknown"]:
            why = "verifier unknown"
        elif x2 is None:
            why = "no pass-2 label"
        elif (x1["label"], x1["direct_practice"]) != (x2["label"], x2["direct_practice"]) and \
                not (i in third and third[i]["resolved"] and third[i]["label"] == "central" and third[i]["direct_practice"] == "no"):
            why = "pass-1/pass-2 disagreement not resolved to central multi-step by third read"
        elif (x1["label"], x1["direct_practice"]) == (x2["label"], x2["direct_practice"]) and \
                not (x2["label"] == "central" and x2["direct_practice"] == "no"):
            why = f"not central multi-step ({x2['label']}/{x2['direct_practice']})"
        elif tags.get(i, {}).get("primary") in (None, "none"):
            why = "no v1.3 primary factoring operation"
        elif x2["confidence"] == "low" or x2["figure"] == "yes":
            why = "pass-2 low confidence or figure"
        elif band(c["c"]) is None:
            why = f"outside bands (c={c['c']}/32)"
        elif c["truncated"] >= 0.5 * max(1, c["n"] - c["c"]):
            why = "difficulty driven by truncation"
        if why:
            excl.append({"id": i, "reason": why})
            continue
        cands.append({"id": i, "source": it["source"], "level": it.get("level"), "band": band(c["c"]),
                      "c32": c["c"], "family": tags[i]["primary"]})
    sel = select(cands)
    json.dump({"selected": sel, "eligible": cands, "excluded": excl}, open(a.out, "w"), indent=1)
    from collections import Counter
    print(json.dumps({"eligible": len(cands), "eligible_by_band": Counter(c["band"] for c in cands),
                      "selected": len(sel), "selected_by_band": Counter(c["band"] for c in sel),
                      "selected_by_family": Counter(c["family"] for c in sel),
                      "selected_by_source": Counter(c["source"] for c in sel),
                      "excluded_reasons": Counter(e["reason"].split(" (")[0] for e in excl)}, indent=1))


if __name__ == "__main__":
    main()
