"""A17 §5: select, verify and freeze the Harder-128 and SuperHard-128 stress panels (rubric v1).

Rubric: research/evidence/a17/panels/RUBRIC-stress-v1.md (frozen before candidate export). Selection uses solution
structure only (agent annotations); no model output is read.

Inputs
- CANDIDATES: data/external/a17/candidates_v1.jsonl (problem, solution, gold, provenance)
- ANNOT_DIR: research/evidence/a17/panels/annotations/batch_*.jsonl (agent annotations, one line per candidate)
- USED_ROOTS: directories scanned recursively for study items (any JSON/JSONL object with a "problem" string, any
  depth); every training, calibration, practice, dev, probe, cue and sealed-test file lives under these roots.
Steps
1. Eligible = annotation eligible, tier harder/superhard, cue_free yes, confidence not low, gold_check agrees, and the
   gold passes `verify.gold_eligible` (parses and verifies against itself).
   Golds with units or text (\\circ, \\text, \\mbox, \\%, $) are excluded (rubric 1.1).
2. Exclusion against study items: exact text, normalized text, or word-3-gram Jaccard >= 0.5 with numbers masked.
3. Families: union of candidates whose masked 3-gram Jaccard >= 0.5 or whose normalized family_hint is identical.
4. Order within tier by sha256("stress-v1:" + tier + ":" + id). SuperHard is filled first (scarcer tier); a family
   used in SuperHard is closed to Harder. Accept in order with <= 2 items per family and no source above 64 items
   while another source still has unused eligible items; two items of one family with Jaccard >= 0.8 or the same gold are one problem (one
   enters); stop at 128.
Outputs (OUT_DIR): superhard_v1.json, harder_v1.json ({"pools": {"superhard"|"harder": [...]}}), stress_v1.json (both
tiers in pool "stress", the evaluation file), selection_report.json (counts, exclusions by reason, composition, family
and gold-form checks, SHA-256 of every output).
Usage: python -m qis.data.stress_panels CANDIDATES ANNOT_DIR OUT_DIR USED_ROOT [USED_ROOT ...]
"""
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

from qis.verify import Verifier, gold_eligible

TARGET, JACC, SRC_CAP, FAM_CAP, DUP = 128, 0.5, 64, 2, 0.8
WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]")


def norm(t):
    return re.sub(r"\s+|\\[,;:! ]|\\left|\\right|\$|[{}]|\\dfrac|\\tfrac", "", t.lower().replace("\\dfrac", "\\frac"))


def grams(t):
    # braces, dollar signs and LaTeX spacing are dropped so that x^{4} and x^4 give the same tokens
    t = re.sub(r"[{}$]|\\[,;:! ]|\\left|\\right", " ", t.lower())
    w = ["#" if x[0].isdigit() else x for x in WORD.findall(t)]
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def used_items(roots):
    out = []

    def walk(o, src):
        if isinstance(o, dict):
            if isinstance(o.get("problem"), str):
                out.append((src, o.get("id"), o["problem"]))
            for v in o.values():
                walk(v, src)
        elif isinstance(o, list):
            for v in o:
                walk(v, src)
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True)):
            if "/stress" in p or "/external/a17" in p or os.path.getsize(p) > 200_000_000:
                continue
            try:
                if p.endswith(".jsonl"):
                    for l in open(p):
                        if l.strip():
                            walk(json.loads(l), p)
                elif p.endswith(".json"):
                    walk(json.load(open(p)), p)
            except (ValueError, UnicodeDecodeError):
                continue
    return out


def same_problem(a, b):
    """Same family and either near-identical text or the same normalized gold answer."""
    return a["family_id"] == b["family_id"] and (jacc(a["grams"], b["grams"]) >= DUP or norm(a["gold"]) == norm(b["gold"]))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    cand_path, annot_dir, out_dir, roots = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    cands = {json.loads(l)["id"]: json.loads(l) for l in open(cand_path)}
    ann = {}
    for p in sorted(glob.glob(os.path.join(annot_dir, "batch_*.jsonl"))):
        for l in open(p):
            if l.strip():
                a = json.loads(l)
                ann[a["id"]] = a
    rep = {"rubric": "research/evidence/a17/panels/RUBRIC-stress-v1.md", "annotated": len(ann),
           "candidates": len(cands), "excluded": Counter(), "used_files": 0}
    used = used_items(roots)
    rep["used_items"], rep["used_files"] = len(used), len({u[0] for u in used})
    u_exact = {u[2] for u in used}
    u_norm = {norm(u[2]) for u in used}
    u_grams = [(u[0], u[1], grams(u[2])) for u in used]
    ver = Verifier(4)
    elig = []
    for cid, a in ann.items():
        c = cands.get(cid)
        why = None
        if c is None:
            why = "annotation id not in candidates"
        elif not a.get("eligible") or a.get("tier") not in ("harder", "superhard"):
            why = "annotation: not eligible"
        elif a.get("cue_free") != "yes":
            why = "annotation: cue present"
        elif a.get("confidence") == "low":
            why = "annotation: low confidence"
        elif a.get("gold_check") != "agrees":
            why = "annotation: gold not confirmed"
        if why:
            rep["excluded"][why] += 1
            continue
        if re.search(r"\\circ|\\text|\\mbox|\\%|\$", c["gold"]):
            rep["excluded"]["gold carries units or text (rubric 1.1)"] += 1
            continue
        if re.search(r"(?<![\\a-zA-Z])[a-zA-Z]{4,}", c["gold"]):     # prose words (not LaTeX commands)
            rep["excluded"]["gold carries units or text (rubric 1.1)"] += 1
            continue
        g = gold_eligible(ver, c["gold"])
        if not g["eligible"]:
            rep["excluded"]["gold not verifier-eligible"] += 1
            continue
        if c["problem"] in u_exact:
            rep["excluded"]["exact text of a study item"] += 1
            continue
        if norm(c["problem"]) in u_norm:
            rep["excluded"]["normalized text of a study item"] += 1
            continue
        cg = grams(c["problem"])
        hit = max(((jacc(cg, ug), src, uid) for src, uid, ug in u_grams), default=(0.0, None, None))
        if hit[0] >= JACC:
            rep["excluded"]["close variant of a study item (3-gram Jaccard >= 0.5)"] += 1
            rep.setdefault("variant_hits", []).append({"id": cid, "jaccard": round(hit[0], 3), "file": hit[1], "item": hit[2]})
            continue
        forms = []
        fs = [f for f in (a.get("gold_equivalent_forms") or []) if isinstance(f, str) and f.strip()]
        if fs:   # the frozen grader must accept each listed equivalent form (a refusal is a grader limit, recorded)
            rs = ver.score([{"gold": c["gold"], "response": "\\boxed{" + f + "}", "truncated": False} for f in fs])
            forms = [{"form": f, "status": r["status"]} for f, r in zip(fs, rs)]
        elig.append({**c, "tier": a["tier"], "annotation": a, "grams": cg, "gold_forms_checked": forms})
    # families
    parent = {e["id"]: e["id"] for e in elig}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hint = defaultdict(list)
    for e in elig:
        h = norm(e["annotation"].get("family_hint", ""))
        if h:
            hint[h].append(e["id"])
    for ids in hint.values():
        for x in ids[1:]:
            parent[find(x)] = find(ids[0])
    for i, e in enumerate(elig):
        for f in elig[i + 1:]:
            if jacc(e["grams"], f["grams"]) >= JACC:
                parent[find(f["id"])] = find(e["id"])
    for e in elig:
        e["family_id"] = "fam-" + hashlib.sha256(find(e["id"]).encode()).hexdigest()[:10]
    rep["eligible_by_tier"] = Counter(e["tier"] for e in elig)
    rep["eligible_by_tier_source"] = Counter(f"{e['tier']}|{e['source_collection']}" for e in elig)
    closed, panels = set(), {}
    for tier in ("superhard", "harder"):
        pool = sorted((e for e in elig if e["tier"] == tier and e["family_id"] not in closed),
                      key=lambda e: hashlib.sha256(f"stress-v1:{tier}:{e['id']}".encode()).hexdigest())
        acc, fam, src = [], Counter(), Counter()
        remaining = Counter(e["source_collection"] for e in pool)
        for e in pool:
            remaining[e["source_collection"]] -= 1
            if len(acc) >= TARGET:
                break
            if fam[e["family_id"]] >= FAM_CAP:
                continue
            if any(same_problem(a, e) for a in acc):
                rep["excluded"]["same problem as an accepted item (same family; Jaccard >= 0.8 or same gold)"] += 1
                continue
            others_left = any(v > 0 for s, v in remaining.items() if s != e["source_collection"])
            if src[e["source_collection"]] >= SRC_CAP and others_left:
                continue
            acc.append(e)
            fam[e["family_id"]] += 1
            src[e["source_collection"]] += 1
        # second pass: fill from sources skipped by the cap if still short (balance target relaxed, recorded)
        if len(acc) < TARGET:
            taken = {e["id"] for e in acc}
            for e in pool:
                if len(acc) >= TARGET:
                    break
                if e["id"] not in taken and fam[e["family_id"]] < FAM_CAP and not any(
                        same_problem(a, e) for a in acc):
                    acc.append(e)
                    fam[e["family_id"]] += 1
                    rep.setdefault("source_cap_relaxed", Counter())[tier] += 1
        closed |= {e["family_id"] for e in acc}
        panels[tier] = acc
        rep[f"{tier}_selected"] = len(acc)
        rep[f"{tier}_composition"] = {
            "source": Counter(e["source_collection"] for e in acc), "technique": Counter(e["annotation"].get("technique") for e in acc),
            "answer_type": Counter(e["annotation"].get("answer_type") for e in acc),
            "obstacle": Counter(e["annotation"].get("obstacle") for e in acc if tier == "superhard"),
            "families": len({e["family_id"] for e in acc}), "eligible_pool": len(pool)}
    keep = ("id", "source_collection", "source_ref", "level", "subject", "problem", "gold", "tier", "family_id", "gold_forms_checked")
    akeep = ("technique", "technique_secondary", "factoring_quote", "required_reasoning", "obstacle", "obstacle_note",
             "alt_route", "answer_type", "confidence", "annotator")
    files = {}
    allitems = []
    for tier, acc in panels.items():
        items = [{**{k: e[k] for k in keep}, "annotation": {k: e["annotation"].get(k) for k in akeep},
                  "problem_sha256": hashlib.sha256(e["problem"].encode()).hexdigest()} for e in acc]
        allitems += items
        p = os.path.join(out_dir, f"{tier}_v1.json")
        json.dump({"version": "stress-v1", "tier": tier, "pools": {tier: items}}, open(p, "w"), indent=1, ensure_ascii=False)
        files[os.path.basename(p)] = sha_file(p)
    p = os.path.join(out_dir, "stress_v1.json")
    json.dump({"version": "stress-v1", "pools": {"stress": allitems}}, open(p, "w"), indent=1, ensure_ascii=False)
    files["stress_v1.json"] = sha_file(p)
    ids = [e["id"] for e in allitems]
    rep["cross_panel_checks"] = {"duplicate_ids": len(ids) - len(set(ids)),
                                 "shared_families": len({e["family_id"] for e in panels["harder"]} & {e["family_id"] for e in panels["superhard"]}),
                                 "max_cross_jaccard": round(max((jacc(a["grams"], b["grams"]) for a in panels["harder"] for b in panels["superhard"]), default=0.0), 3)}
    rep["sha256"] = files
    ver.close()
    json.dump(rep, open(os.path.join(out_dir, "selection_report.json"), "w"), indent=1, default=dict)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("variant_hits",)}, indent=1, default=dict))


if __name__ == "__main__":
    main()
