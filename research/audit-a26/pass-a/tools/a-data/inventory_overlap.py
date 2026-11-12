"""a-data (A26 pass A): set inventory, fresh sha256 and overlap matrices across every QIS data set.

Stdlib only; read-only. Runs ON COMPUTE (needs $P/data). W = a pinned worktree of the repo (data/ unchanged since 577aa9f).
Similarity rules reproduced from the project (copied, not imported):
  J5  = splits.py: normalize() then 5-word shingles, Jaccard >= 0.7 = near-duplicate (src/qis/data/splits.py:17-34,80-82)
  J3m = stress_panels.py: grams() masked word 3-grams, Jaccard >= 0.5 = close variant (src/qis/data/stress_panels.py:43-53,121-124)
  norm = stress_panels.norm (lowercase, drop whitespace, LaTeX spacing, \\left/\\right, $, braces, dfrac/tfrac)
  simple = lowercase, drop LaTeX spacing commands and $, collapse whitespace
Usage: python inventory_overlap.py W OUT.json
"""
import collections
import glob
import gzip
import hashlib
import json
import os
import re
import sys

P = "/data/user_data/vineethb/qis"
J5_T, J3_T = 0.7, 0.5
WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]")


def s_normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\]+", " ", text)
    return " ".join(text.split())


def sh5(text):
    toks = s_normalize(text).split()
    if len(toks) < 5:
        return frozenset({" ".join(toks)})
    return frozenset(" ".join(toks[i:i + 5]) for i in range(len(toks) - 4))


def norm(t):
    return re.sub(r"\s+|\\[,;:! ]|\\left|\\right|\$|[{}]|\\dfrac|\\tfrac", "", t.lower().replace("\\dfrac", "\\frac"))


def simple(t):
    t = re.sub(r"\\[,;:! ]|\$", " ", t.lower())
    return " ".join(t.split())


def g3(t):
    t = re.sub(r"[{}$]|\\[,;:! ]|\\left|\\right", " ", t.lower())
    w = ["#" if x[0].isdigit() else x for x in WORD.findall(t)]
    return frozenset(tuple(w[i:i + 3]) for i in range(len(w) - 2))


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cid(i):
    return i[5:] if i.startswith("math:") or i.startswith("omni:") else i


class Item:
    __slots__ = ("id", "text", "s5", "s3", "nm", "sp", "src")

    def __init__(self, i, text, src):
        self.id, self.text, self.src = cid(i), text, src
        self.s5, self.s3, self.nm, self.sp = sh5(text), g3(text), norm(text), simple(text)


def index(items, attr):
    inv = collections.defaultdict(list)
    for k, it in enumerate(items):
        for g in getattr(it, attr):
            inv[g].append(k)
    return inv


def best_matches(A, B, attr, thr, same=False, skip_same_id=True):
    """Max Jaccard over pairs (a in A, b in B); list pairs >= thr. same=True: within one set (a<b)."""
    inv = index(B, attr)
    best, hits = (0.0, None, None), []
    for ka, a in enumerate(A):
        cands = set()
        for g in getattr(a, attr):
            cands.update(inv.get(g, ()))
        for kb in cands:
            if same and kb <= ka:
                continue
            b = B[kb]
            if skip_same_id and a.id == b.id:
                continue
            j = jac(getattr(a, attr), getattr(b, attr))
            if j > best[0]:
                best = (j, a.id, b.id)
            if j >= thr:
                hits.append((round(j, 4), a.id, b.id))
    hits.sort(reverse=True)
    return {"max": best[0], "max_pair": best[1:], "n_ge_thr": len(hits), "hits": hits[:60]}


def main(W, out_path):
    out = {"hashes": {}, "sets": {}, "pairs": {}, "extended": {}, "within": {}, "checks": {}}
    # ---------- MATH universe
    M = {}
    for line in open(f"{P}/data/splits-v1/manifest.jsonl"):
        r = json.loads(line)
        M[r["id"]] = r
    omni_all = [json.loads(l) for l in open(f"{W}/data/external/a17/omni_math_test.jsonl")]
    tranche = [json.loads(l) for l in open(f"{W}/data/external/omni_tranche_v1.jsonl")]
    out["checks"]["omni_tranche_text_equals_omni_test_line"] = sum(omni_all[t["omni_index"]]["problem"] == t["problem"] for t in tranche)
    out["checks"]["omni_tranche_id_equals_index"] = sum(t["id"] == f"omni/{t['omni_index']:05d}" for t in tranche)
    # ---------- study sets
    B = json.load(open(f"{P}/data/main-v1/bundle_main_v1.json"))["pools"]
    T = json.load(open(f"{P}/data/main-v1/target_test_sealed.json"))["pools"]["test"]
    C = json.load(open(f"{P}/data/calib-panels-k8/bundle.json"))["pools"]
    S = json.load(open(f"{P}/data/stress/stress_v1.json"))["pools"]["stress"]
    CU = json.load(open(f"{W}/data/cue/cue_v1_uncued.json"))["pools"]["cue"]
    CC = json.load(open(f"{W}/data/cue/cue_v1_cued.json"))["pools"]["cue"]
    AUD = json.load(open(f"{W}/research/evidence/a17/panels/audit_sample_v1.json"))["items"]
    sets = {
        "H_train": B["hard"], "R_pool": B["related"], "U_pool": B["unrelated"], "dev": B["dev"],
        "J_H": B["probe_J_H"], "J_R": B["probe_J_R"], "J_U": B["probe_J_U"], "test": T,
        "calib_train": C["calib_train"], "calib_val": C["calib_val"],
        "stress_SuperHard": [x for x in S if x["tier"] == "superhard"], "stress_Harder": [x for x in S if x["tier"] == "harder"],
        "cue": CU,
    }
    items = {k: [Item(x["id"], x["problem"], x.get("source") or x.get("source_collection") or "") for x in v] for k, v in sets.items()}
    # per-set descriptors
    for k, v in sets.items():
        ids = [cid(x["id"]) for x in v]
        d = {"n_rows": len(v), "distinct_ids": len(set(ids)), "distinct_norm_text": len({norm(x["problem"]) for x in v}),
             "source": dict(collections.Counter(x.get("source") or x.get("source_collection") or "?" for x in v))}
        mi = [i for i in ids if i in M]
        d["math_items"] = len(mi)
        d["math_internal_split"] = dict(collections.Counter(M[i]["split"] for i in mi))
        d["math_orig_split"] = dict(collections.Counter(M[i]["orig_split"] for i in mi))
        d["math_subject"] = dict(collections.Counter(M[i]["subject"] for i in mi))
        d["math_level"] = dict(collections.Counter(M[i]["level"] for i in mi))
        d["math_dup_groups"] = len({M[i]["group"] for i in mi})
        d["math_group_size_gt1"] = sum(int(M[i]["group_size"]) > 1 for i in mi)
        d["math_text_equals_manifest"] = sum(M[i]["problem"] == x["problem"] for i, x in zip(ids, v) if i in M)
        d["math_gold_equals_manifest"] = sum(M[i]["gold"] == x["gold"] for i, x in zip(ids, v) if i in M)
        d["omni_text_equals_omni_test"] = sum(omni_all[int(i[5:])]["problem"] == x["problem"] for i, x in zip(ids, v) if i.startswith("omni/"))
        if k == "stress_SuperHard" or k == "stress_Harder":
            d["families"] = len({x["family_id"] for x in v})
        if "family" in v[0]:
            d["families"] = dict(collections.Counter(x["family"] for x in v))
        if "template_family" in v[0]:
            d["template_families"] = len({x.get("template_family") for x in v})
        d["ids"] = ids
        out["sets"][k] = d
    out["checks"]["cue_cued_vs_uncued_same_ids_order"] = [cid(a["id"]) for a in CU] == [cid(a["id"]) for a in CC]
    out["checks"]["cue_cued_is_uncued_plus_cue"] = sum(b["problem"].startswith(a["problem"]) and b["gold"] == a["gold"] for a, b in zip(CU, CC))
    aud = [cid(i) for t in AUD.values() for i in t]
    out["checks"]["audit_items"] = {"n": len(aud), "in_stress": sum(i in {cid(x["id"]) for x in S} for i in aud),
                                    "tier_match": sum(1 for t, l in AUD.items() for i in l
                                                      if any(cid(x["id"]) == cid(i) and x["tier"] == t for x in S))}
    # intended relations
    dev_ids = [cid(x["id"]) for x in B["dev"]]
    jh_ids = [cid(x["id"]) for x in B["probe_J_H"]]
    out["checks"]["J_H_equals_first32_dev_in_order"] = jh_ids == dev_ids[:32]
    out["checks"]["J_H_text_gold_equal_dev"] = sum(a == b for a, b in zip(B["probe_J_H"], B["dev"][:32]))
    # ---------- pairwise matrices over study sets
    names = list(items)
    for a_i, a in enumerate(names):
        out["within"][a] = {"J5": best_matches(items[a], items[a], "s5", J5_T, same=True, skip_same_id=False),
                            "J3m": best_matches(items[a], items[a], "s3", J3_T, same=True, skip_same_id=False),
                            "dup_norm_text": len(items[a]) - len({x.nm for x in items[a]})}
        for b in names[a_i + 1:]:
            A, Bs = items[a], items[b]
            ida, idb = {x.id for x in A}, {x.id for x in Bs}
            r = {"id_overlap": len(ida & idb), "exact_text": len({x.text for x in A} & {x.text for x in Bs}),
                 "simple_norm_text": len({x.sp for x in A} & {x.sp for x in Bs}),
                 "norm_text": len({x.nm for x in A} & {x.nm for x in Bs})}
            r["J5"] = best_matches(A, Bs, "s5", J5_T)          # same-ID pairs skipped (they are the ID overlap)
            r["J3m"] = best_matches(A, Bs, "s3", J3_T)
            out["pairs"][f"{a}|{b}"] = r
        print("pairs done for", a, flush=True)
    # ---------- extended universes
    train = items["H_train"] + items["R_pool"] + items["U_pool"]
    evals = [x for k in ("dev", "J_R", "J_U", "test", "calib_train", "calib_val", "stress_SuperHard", "stress_Harder", "cue") for x in items[k]]
    study_all = [x for k in names for x in items[k]]
    study_ids = {x.id for x in study_all}
    math_test = [Item(i, r["problem"], "MATH") for i, r in M.items() if r["orig_split"] == "test"]
    math_all = [Item(i, r["problem"], "MATH") for i, r in M.items()]
    omni_items = [Item(f"omni/{k:05d}", r["problem"], "Omni-MATH") for k, r in enumerate(omni_all)]
    out["extended"]["train_vs_all_MATH_test"] = {"J5": best_matches(train, math_test, "s5", J5_T),
                                                 "J3m": best_matches(train, math_test, "s3", J3_T)}
    print("train vs MATH test done", flush=True)
    omni_used = [x for x in study_all if x.id.startswith("omni/")]
    out["extended"]["omni_used_vs_all_MATH"] = {"n_omni_used": len(omni_used), "J5": best_matches(omni_used, math_all, "s5", J5_T),
                                                "J3m": best_matches(omni_used, math_all, "s3", J3_T)}
    print("omni used vs MATH done", flush=True)
    non_study_omni = [x for x in omni_items if x.id not in study_ids]
    out["extended"]["all_omni_test_vs_study_sets"] = {"n": len(omni_items), "J5": best_matches(study_all, omni_items, "s5", J5_T),
                                                      "J3m": best_matches(study_all, omni_items, "s3", J3_T)}
    print("omni all done", flush=True)
    # screening candidate pools (exposed to the base model only) vs evaluation sets
    scr = {}
    for p in sorted(glob.glob(f"{P}/data/candidates-v1-study/*.jsonl") + glob.glob(f"{P}/data/omni-v1/omni_hard.jsonl")
                    + glob.glob(f"{P}/data/candidates-authored-v*/*.jsonl") + glob.glob(f"{P}/data/rescreen-k8-h/*.jsonl")):
        rows = [json.loads(l) for l in open(p)]
        scr[os.path.relpath(p, f"{P}/data")] = rows
    out["sets_screening"] = {k: {"n_rows": len(v), "distinct_ids": len({r["id"] for r in v})} for k, v in scr.items()}
    screened = {}
    for k, v in scr.items():
        if k.startswith("candidates-v1-study/") and k.split("/")[1] in ("target_dev.jsonl", "target_test.jsonl", "probe_related_dev.jsonl",
                                                                        "probe_unrelated_dev.jsonl", "calib_train.jsonl", "calib_val.jsonl"):
            continue          # these are the evaluation/calibration candidate lists, not training-screening pools
        for r in v:
            screened[cid(r["id"])] = Item(r["id"], r["problem"], k)
    scr_items = list(screened.values())
    out["extended"]["training_screen_pool_union"] = {"n": len(scr_items)}
    ev_sets = ("dev", "J_R", "J_U", "test", "stress_SuperHard", "stress_Harder", "cue")
    for k in ev_sets:
        ids = {x.id for x in items[k]}
        out["extended"][f"screen_union_vs_{k}"] = {"id_overlap": len(ids & set(screened)),
                                                   "norm_text": len({x.nm for x in items[k]} & {x.nm for x in scr_items}),
                                                   "J5": best_matches(items[k], scr_items, "s5", J5_T),
                                                   "J3m": best_matches(items[k], scr_items, "s3", J3_T)}
    print("screening done", flush=True)
    # ---------- hashes (fresh)
    files = [f"{P}/data/main-v1/bundle_main_v1.json", f"{P}/data/main-v1/target_test_sealed.json", f"{P}/data/main-v1/manifest.json",
             f"{P}/data/calib-panels-k8/bundle.json", f"{P}/data/calib-panels-v1/bundle.json", f"{P}/data/splits-v1/manifest.jsonl",
             f"{P}/data/splits-v1/dup_pairs.json", f"{P}/data/targets-v1/panels.json", f"{P}/data/hsel-combined-v1/selection.json",
             f"{P}/data/match-k8/match.json", f"{P}/data/stress/stress_v1.json", f"{P}/data/stress/superhard_v1.json",
             f"{P}/data/stress/harder_v1.json", f"{P}/data/stress/selection_report.json",
             f"{P}/data/stress.superseded-d143/stress_v1.json", f"{W}/data/cue/cue_v1_uncued.json", f"{W}/data/cue/cue_v1_cued.json",
             f"{W}/data/main_v1_manifest.json", f"{W}/data/panels/panels.json", f"{W}/data/stress/stress_v1.json",
             f"{W}/data/screening/hsel_k8.json", f"{W}/data/screening/match_k8.json", f"{W}/data/external/omni_tranche_v1.jsonl",
             f"{W}/data/external/a17/candidates_v1.jsonl", f"{W}/research/evidence/a17/panels/audit_sample_v1.json"]
    files += sorted(glob.glob(f"{P}/data/candidates-v1-study/*.jsonl")) + sorted(glob.glob(f"{P}/data/rescreen-k8-h/*.jsonl"))
    files += sorted(glob.glob(f"{P}/data/raw-21a5633/*/*.parquet"))
    for p in files:
        out["hashes"][p.replace(W, "W").replace(P, "$P")] = sha(p) if os.path.exists(p) else "unavailable"
    # sha recorded in run configs of calibration runs and other bundle consumers
    cfgsha = {}
    for mp in sorted(glob.glob(f"{P}/runs/*/metrics.jsonl")) + sorted(glob.glob(f"{P}/runs/superseded/*/metrics.jsonl")):
        s = collections.Counter()
        for line in open(mp):
            if '"kind": "config"' in line:
                x = json.loads(line)
                s[(x.get("bundle_sha"), x["config"].get("bundle"))] += 1
        cfgsha[os.path.relpath(mp, P)] = [[a, b, n] for (a, b), n in s.items()]
    out["config_bundle_sha"] = cfgsha
    # summary.json keys of evaluation outputs (does a panel hash appear?)
    summ = {}
    for p in sorted(glob.glob(f"{P}/test-eval/*/summary.json") + glob.glob(f"{P}/stress-eval/*/summary.json")
                    + glob.glob(f"{P}/cue-eval/*/*/summary.json") + glob.glob(f"{P}/test-eval-cgpu/*/summary.json")):
        x = json.load(open(p))
        summ[os.path.relpath(p, P)] = {k: x[k] for k in x if "sha" in k.lower() or k in ("panel", "n", "problems", "seed_label", "max_tokens")}
    out["eval_summaries"] = summ
    json.dump(out, open(out_path, "w"), indent=1, sort_keys=True, default=list)
    print("written", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
