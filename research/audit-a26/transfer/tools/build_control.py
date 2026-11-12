"""A26 x-transfer: build the control panel control_v1 by rule (PREREG.md §2). No model output is read.

Usage (compute, PYTHONPATH = project src):
  python build_control.py SPLITS_MANIFEST OUT_DIR ROOT [ROOT ...]
ROOTs are scanned for study items (any JSON/JSONL object with a "problem" string, any depth; any string that looks like a
MATH ID). Writes OUT_DIR/control_v1.json ({"version", "pools": {"control": [...]}}) and OUT_DIR/control_report.json.
"""
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

from qis.data.stress_panels import grams, jacc, norm
from qis.verify import Verifier, gold_eligible

SUBJECTS = ["geometry", "counting_and_probability", "precalculus", "prealgebra"]
LEVELS = {"Level 4", "Level 5"}
PER_SUBJECT, JACC = 16, 0.5
MID = re.compile(r"^(?:math:)?((?:algebra|counting_and_probability|geometry|intermediate_algebra|number_theory|prealgebra|"
                 r"precalculus)/(?:train|test)/\d{5})$")


def scan(roots, skip):
    texts, ids, files = [], set(), set()

    def walk(o, src):
        if isinstance(o, dict):
            if isinstance(o.get("problem"), str):
                texts.append((src, o.get("id"), o["problem"]))
            for v in o.values():
                walk(v, src)
        elif isinstance(o, list):
            for v in o:
                walk(v, src)
        elif isinstance(o, str):
            m = MID.match(o)
            if m:
                ids.add(m.group(1))
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True)):
            if any(s in p for s in skip) or os.path.getsize(p) > 200_000_000:
                continue
            try:
                if p.endswith(".jsonl"):
                    for line in open(p):
                        if line.strip():
                            walk(json.loads(line), p)
                elif p.endswith(".json"):
                    walk(json.load(open(p)), p)
                else:
                    continue
                files.add(p)
            except (ValueError, UnicodeDecodeError):
                continue
    return texts, ids, files


def main():
    manifest, out_dir, roots = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    skip = ["/splits-v1/", "/raw-21a5633/", "/audit-a26/", "/data/control/"]
    texts, used_ids, files = scan(roots, skip)
    for _, i, _ in texts:
        m = MID.match(i or "")
        if m:
            used_ids.add(m.group(1))
    u_exact = {t for _, _, t in texts}
    u_norm = {norm(t) for _, _, t in texts}
    u_grams = [(s, i, grams(t)) for s, i, t in texts]
    rep = {"rule": "research/audit-a26/transfer/PREREG.md §2", "roots": roots, "skip": skip, "used_files": len(files),
           "used_texts": len(texts), "used_ids": len(used_ids), "candidates": Counter(), "excluded": Counter(),
           "excluded_items": [], "eligible": Counter()}
    cands = []
    for line in open(manifest):
        r = json.loads(line)
        if r["orig_split"] == "test" and r["subject"] in SUBJECTS and r["level"] in LEVELS:
            cands.append(r)
    ver = Verifier(4)
    elig = {s: [] for s in SUBJECTS}
    for r in cands:
        rep["candidates"][r["subject"]] += 1
        why, detail = None, None
        if r["id"] in used_ids:
            why = "id in a study file"
        elif r["problem"] in u_exact:
            why = "exact text of a study item"
        elif norm(r["problem"]) in u_norm:
            why = "normalized text of a study item"
        else:
            cg = grams(r["problem"])
            hit = max(((jacc(cg, ug), s, i) for s, i, ug in u_grams), default=(0.0, None, None))
            if hit[0] >= JACC:
                why, detail = "close variant (3-gram Jaccard >= 0.5)", {"jaccard": round(hit[0], 3), "file": hit[1],
                                                                        "item": hit[2]}
        if why is None:
            if r.get("gold") is None:
                why = "no boxed gold"
            elif not gold_eligible(ver, r["gold"])["eligible"]:
                why = "gold not verifier-eligible"
        if why:
            rep["excluded"][why] += 1
            rep["excluded_items"].append({"id": r["id"], "subject": r["subject"], "why": why, **({"hit": detail} if detail else {})})
            continue
        rep["eligible"][r["subject"]] += 1
        elig[r["subject"]].append(r)
    ver.close()
    panel = []
    for s in SUBJECTS:
        order = sorted(elig[s], key=lambda r: hashlib.sha256(("a26-control/" + r["id"]).encode()).hexdigest())
        for r in order[:PER_SUBJECT]:
            panel.append({"id": r["id"], "source": "MATH", "source_ref": "EleutherAI/hendrycks_math@21a5633 " + r["id"],
                          "subject": s, "level": r["level"], "problem": r["problem"], "gold": r["gold"],
                          "problem_sha": r["problem_sha"], "gold_sha": r["gold_sha"],
                          "order_key": hashlib.sha256(("a26-control/" + r["id"]).encode()).hexdigest()})
    rep["panel"] = Counter(x["subject"] for x in panel)
    rep["panel_levels"] = Counter(x["level"] for x in panel)
    out = {"version": "control_v1 (A26 x-transfer; rule research/audit-a26/transfer/PREREG.md §2)", "pools": {"control": panel}}
    p = os.path.join(out_dir, "control_v1.json")
    json.dump(out, open(p, "w"), indent=1)
    rep["sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(rep, open(os.path.join(out_dir, "control_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "excluded_items"}, indent=1))


if __name__ == "__main__":
    main()
