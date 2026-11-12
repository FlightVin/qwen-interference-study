"""a-data (A26 pass A): skill labels, pass agreement, screening counts, pairing, exclusions, recipe hash references.

Stdlib only; reads repo files on the laptop (no project code). Set membership comes from a-data-inventory_overlap.json
(written by inventory_overlap.py on compute). Usage (repo root): python3 -I THIS OUT.json
"""
import collections
import glob
import hashlib
import json
import os
import re
import sys

R = os.getcwd()
A = f"{R}/data/annotations"
rd = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(f) if l.strip()}  # noqa: E731
inv = json.load(open(f"{R}/research/audit-a26/pass-a/data/a-data-inventory_overlap.json"))
sets = {k: v["ids"] for k, v in inv["sets"].items()}
out = {"labels": {}, "screening": {}, "pairing": {}, "exclusions": {}, "recipe_hashes": {}, "local_hashes": {}}

# ---------- annotations
p1, th = rd(f"{A}/pass1.jsonl"), rd(f"{A}/third_read.jsonl")
p2 = {}
for f in ("pass2_hard", "pass2_targets_probes", "pass2_aux_natural"):
    for i, r in rd(f"{A}/{f}.jsonl").items():
        p2.setdefault(i, (f, r))
o1, o2, ot = rd(f"{A}/omni_pass1.jsonl"), rd(f"{A}/omni_pass2.jsonl"), rd(f"{A}/omni_third_read.jsonl")
a1 = {**rd(f"{A}/authored_pass1.jsonl"), **rd(f"{A}/authored2_pass1.jsonl")}
a2 = {**rd(f"{A}/authored_pass2.jsonl"), **rd(f"{A}/authored2_pass2.jsonl")}
at = {**rd(f"{A}/authored_third_read.jsonl"), **rd(f"{A}/authored2_third_read.jsonl")}
tags = rd(f"{A}/tags_v13.jsonl")
K = ("label", "direct_practice", "modular")


def label_of(i):
    if i.startswith("omni/"):
        x1, x2 = o1.get(i), o2.get(i)
        if not x1 or not x2:
            return "omni: missing pass", None
        key = lambda r: (r["label"], r["direct_practice"], r["proof_like"])  # noqa: E731
        if key(x1) == key(x2):
            return "omni: pass1=pass2", (x2["label"], x2["direct_practice"], x2["proof_like"], x2.get("primary"))
        if i in ot and ot[i]["resolved"]:
            return "omni: third read", (ot[i]["label"], ot[i]["direct_practice"], ot[i]["proof_like"], x2.get("primary"))
        return "omni: unresolved", None
    if i.startswith("syn-"):
        x1, x2 = a1.get(i), a2.get(i)
        if not x1 or not x2:
            return "authored: missing pass", None
        if tuple(x1[k] for k in K) == tuple(x2[k] for k in K):
            return "authored: pass1=pass2", tuple(x2[k] for k in K) + (x2.get("primary"),)
        if i in at and at[i]["resolved"]:
            return "authored: third read", tuple(at[i][k] for k in K) + (x2.get("primary"),)
        return "authored: pass1!=pass2, no third", tuple(x2[k] for k in K)
    x1 = p1.get(i)
    x2 = p2.get(i, (None, None))[1]
    if x1 is None:
        return "native: no pass1", None
    if x2 is None:
        return "native: pass1 only", tuple(x1[k] for k in K)
    if tuple(x1[k] for k in K) == tuple(x2[k] for k in K):
        return f"native: pass1=pass2 ({p2[i][0]})", tuple(x2[k] for k in K) + ((tags.get(i) or {}).get("primary"),)
    if i in th and th[i]["resolved"]:
        return f"native: third read ({p2[i][0]})", tuple(th[i][k] for k in K) + ((tags.get(i) or {}).get("primary"),)
    return "native: disagreement unresolved", tuple(x2[k] for k in K)


for name, ids in sets.items():
    if name.startswith("stress"):
        continue
    src, lab = collections.Counter(), collections.Counter()
    for i in ids:
        s, l = label_of(i)
        src[s] += 1
        lab[str(l[:3]) if l else "none"] += 1
    out["labels"][name] = {"source": dict(src), "final_label": dict(lab)}
# H_train: families also from tags_v13 (native) / omni pass-2 primary
hsel = json.load(open(f"{R}/data/screening/hsel_k8.json"))
fam_chk = collections.Counter()
for h in hsel["selected"]:
    want = o2[h["id"]]["primary"] if h["id"].startswith("omni/") else tags[h["id"]]["primary"]
    fam_chk["family == rule" if want == h["family"] else "family != rule"] += 1
out["labels"]["H_train_family_rule_check"] = dict(fam_chk)
# R pool primary technique (v1.3) and authored template families
rp = collections.Counter()
for i in sets["R_pool"]:
    s, l = label_of(i)
    rp[l[3] if l and len(l) > 3 else "?"] += 1
out["labels"]["R_pool_primary"] = dict(rp)
# stress: first read (batch annotations) vs blind second read
first = {}
for p in glob.glob(f"{R}/research/evidence/a17/panels/annotations/batch_*.jsonl"):
    first.update(rd(p))
second = {}
for p in glob.glob(f"{R}/research/evidence/a17/panels/second_read/read_*.jsonl"):
    second.update(rd(p))
for tier in ("stress_SuperHard", "stress_Harder"):
    ids = ["math:" + i if re.match(r"^[a-z_]+/(train|test)/", i) else ("omni:" + i if i.startswith("omni/") else i) for i in sets[tier]]
    c = collections.Counter()
    for i in ids:
        f, s = first.get(i), second.get(i)
        c["first read present"] += f is not None
        if s is None:
            c["no second read"] += 1
            continue
        c["second read"] += 1
        c["tier agrees"] += f["tier"] == s["tier"]
        c["eligible agrees (second eligible)"] += bool(s.get("eligible"))
        c["gold agrees"] += s.get("gold_check") == "agrees"
    out["labels"][tier] = dict(c)

# ---------- screening counts
def counts(path):
    return rd(path)


cnt = {"H_train": counts(f"{R}/data/screening/h_s2k8_counts.jsonl"),
       "R_pool": counts(f"{R}/data/screening/related_k8_s2k8_counts.jsonl"),
       "U_pool": counts(f"{R}/data/screening/unrelated_k8_s2k8_counts.jsonl"),
       "calib_train": counts(f"{R}/data/screening/calib_train_calib16k8_counts.jsonl"),
       "calib_val": counts(f"{R}/data/screening/calib_val_calib16k8_counts.jsonl")}
for name, c in cnt.items():
    ids = sets[name]
    have = [c[i] for i in ids if i in c]
    cs = [x["c"] for x in have]
    out["screening"][name] = {"items_with_count": len(have), "of": len(ids), "n": sorted({x["n"] for x in have}),
                              "c_min": min(cs), "c_max": max(cs), "c_mean": sum(cs) / len(cs),
                              "c_hist": dict(sorted(collections.Counter(cs).items())), "unknown": sum(x["unknown"] for x in have),
                              "truncated_attempts": sum(x["truncated"] for x in have), "stage": sorted({x["stage"] for x in have})}
hc = {h["id"]: h["c32"] for h in hsel["selected"]}
out["screening"]["H_train"]["hsel_c32_equals_count_file"] = sum(hc[i] == cnt["H_train"][i]["c"] for i in hc)
out["screening"]["H_train"]["band"] = dict(collections.Counter(h["band"] for h in hsel["selected"]))
out["screening"]["H_train"]["source"] = dict(collections.Counter(h["source"] for h in hsel["selected"]))
for s in ("dev", "J_H", "J_R", "J_U", "test", "cue", "stress_SuperHard", "stress_Harder"):
    hit = 0
    for f in glob.glob(f"{R}/data/screening/*_counts.jsonl"):
        ids = set(rd(f))
        hit += len(ids & set(sets[s]))
    out["screening"][s] = {"items_in_any_screening_count_file": hit}

# ---------- pairing
match = json.load(open(f"{R}/data/screening/match_k8.json"))
man = json.load(open(f"{R}/data/main_v1_manifest.json"))
pr = match["pairs"]
out["pairing"] = {"n_pairs": len(pr), "bundle_related_order_equals_match_r": [p["r"] for p in pr] == man["related"]["ids"],
                  "bundle_unrelated_order_equals_match_u": [p["u"] for p in pr] == man["unrelated"]["ids"],
                  "manifest_equals_inventory_bundle": man["related"]["ids"] == sets["R_pool"] and man["unrelated"]["ids"] == sets["U_pool"],
                  "exact_count_matches": sum(p["exact"] for p in pr), "same_answer_type": sum(p["same_answer_type"] for p in pr),
                  "provenance": dict(collections.Counter(("natural" if not p["r"].startswith("syn-") else "authored",
                                                          "natural" if not p["u"].startswith("syn-") else "authored") for p in pr)),
                  "abs_c_diff_hist": dict(sorted(collections.Counter(abs(cnt["R_pool"][p["r"]]["c"] - cnt["U_pool"][p["u"]]["c"]) for p in pr).items())),
                  "dropped_in_balance": match["final_balance"].get("dropped"), "smd": {k: match["final_balance"][k] for k in ("smd_success", "smd_resp_len")},
                  "eligible": match["eligible"]}
pr_list = [str(x) for x in out["pairing"]["provenance"].items()]
out["pairing"]["provenance"] = pr_list

# ---------- exclusions
out["exclusions"]["H_train"] = {"eligible": len(hsel["eligible"]), "selected": len(hsel["selected"]),
                                "excluded": dict(collections.Counter(e["reason"].split(" (")[0] for e in hsel["excluded"])),
                                "excluded_n": len(hsel["excluded"])}
for name, cand in (("dev", "target_dev"), ("test", "target_test"), ("J_R", "probe_related_dev"), ("J_U", "probe_unrelated_dev")):
    out["exclusions"][name] = {"note": "candidate list lives on compute ($P/data/candidates-v1-study); counts there"}

# ---------- recipe hash references
known = {}
for k, v in inv["hashes"].items():
    known.setdefault(v, []).append(k)
for p in sorted(glob.glob(f"{R}/slurm-agentic-engineering/exp/*.sh") + glob.glob(f"{R}/slurm-agentic-engineering/exp-deferred/*.sh")):
    txt = open(p).read()
    for h in set(re.findall(r"\b[0-9a-f]{64}\b", txt)):
        out["recipe_hashes"].setdefault(h, {"files": known.get(h, ["(not a data file hashed here)"]), "recipes": []})["recipes"].append(os.path.basename(p))

# ---------- local hashes of repo copies
for p in ["data/main_v1_manifest.json", "data/panels/panels.json", "data/stress/stress_v1.json", "data/stress/harder_v1.json",
          "data/stress/superhard_v1.json", "data/cue/cue_v1_uncued.json", "data/cue/cue_v1_cued.json", "data/screening/hsel_k8.json",
          "data/screening/match_k8.json", "data/external/omni_tranche_v1.jsonl", "data/external/a17/candidates_v1.jsonl",
          "research/evidence/a17/panels/audit_sample_v1.json"]:
    out["local_hashes"][p] = hashlib.sha256(open(f"{R}/{p}", "rb").read()).hexdigest()
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "recipe_hashes"}, indent=1, default=str)[:9000])
print(json.dumps({h[:12]: (v["files"], sorted(set(v["recipes"]))[:6], len(v["recipes"])) for h, v in out["recipe_hashes"].items()}, indent=0))
