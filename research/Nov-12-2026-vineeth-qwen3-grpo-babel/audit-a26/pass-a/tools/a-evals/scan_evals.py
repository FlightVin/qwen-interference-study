"""A26 pass A (agent a-evals): scan every published evaluation's raw records on compute.

Read-only on $P. Stdlib + numpy only; no project imports. The seed formula is re-implemented from
src/qis/seeds.py (SeedSequence(entropy=master, spawn_key=sha256(name) as four little-endian uint32)),
so the per-problem request seeds can be checked against what an unsharded run would have used.

For each evaluation directory: attempts per problem, sample indices, request seeds, verifier status,
truncation, finish/n_tokens consistency, per-problem counts vs summary.json, summary scalars recomputed.
For sharded evaluations: shard panels partition the full panel (items[s::k]); shard records cover their
shard panel; merged records equal the union of the shard records (line hashes).
For the length extension: rows joined to the original records on (prompt_id, sample).
Writes $P/audit-a26/a-evals/scan.json (unrounded) and per-attempt outcome strings for analysis.
Usage: python scan_evals.py OUT.json
"""
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

P = "/data/user_data/vineethb/qis"
REPO = os.path.expanduser("~/ae-qis/code/qis-cue-eval-6d4e33a")   # pinned worktree of the completed cue job 11104738
EVAL_SEED = 4242
PANELS = {"test": (f"{P}/data/main-v1/target_test_sealed.json", "test"),
          "stress": (f"{P}/data/stress/stress_v1.json", "stress"),
          "cue_cued": (f"{REPO}/data/cue/cue_v1_cued.json", "cue"),
          "cue_uncued": (f"{REPO}/data/cue/cue_v1_uncued.json", "cue")}


def int_seed(master, name):
    d = hashlib.sha256(name.encode()).digest()
    key = tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))
    return int(np.random.SeedSequence(entropy=master, spawn_key=key).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def code(r):
    if r["status"] == "unknown":
        return "U"
    if r["finish"] == "length":
        return "T" if r["reward"] == 0 else "X"     # X would be a capped attempt with reward 1 (must not occur)
    if r["reward"] == 1:
        return "C"
    return "W"


def scan_records(path, key, n, label, panel_ids, cap, keep_hash=False, keep_tokhash=False):
    per = defaultdict(list)
    status = Counter()
    finish = Counter()
    bad = []
    hashes = {}
    tokh = {}
    lines = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            lines += 1
            pid = r["prompt_id"]
            status[r["status"]] += 1
            finish[r["finish"]] += 1
            nt = r["n_tokens"]
            if len(r["ids"]) != nt:
                bad.append(("ids_len", pid, r["sample"]))
            if r["finish"] == "length" and nt != cap:
                bad.append(("length_not_cap", pid, r["sample"], nt))
            if nt > cap:
                bad.append(("over_cap", pid, r["sample"], nt))
            if r["finish"] == "length" and r["reward"] != 0:
                bad.append(("capped_reward1", pid, r["sample"]))
            per[pid].append((r["sample"], r["seed"], code(r), nt))
            if keep_hash:
                hashes[(pid, r["sample"])] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
            if keep_tokhash:
                tokh[(pid, r["sample"])] = hashlib.sha1(json.dumps(r["ids"]).encode()).hexdigest()
    ids_seen = set(per)
    out = {"path": path, "lines": lines, "status": dict(status), "finish": dict(finish), "bad": bad[:20], "n_bad": len(bad),
           "problems_seen": len(ids_seen), "panel_problems": len(panel_ids),
           "missing_problems": sorted(set(panel_ids) - ids_seen), "extra_problems": sorted(ids_seen - set(panel_ids))}
    wrong_n, dup_samples, seed_multi, seed_mismatch = [], [], [], []
    pp = {}
    for pid in panel_ids:
        g = sorted(per.get(pid, []))
        samples = [s for s, *_ in g]
        if len(g) != n:
            wrong_n.append((pid, len(g)))
        if len(set(samples)) != len(samples) or sorted(set(samples)) != list(range(len(set(samples)))):
            dup_samples.append(pid)
        sds = {sd for _, sd, _, _ in g}
        if len(sds) != 1:
            seed_multi.append(pid)
        exp = int_seed(EVAL_SEED, f"eval/{key}/{pid}/{label}/n{n}")
        if sds != {exp}:
            seed_mismatch.append((pid, sorted(sds)[:2], exp))
        s = "".join(c for _, _, c, _ in g)
        pp[pid] = {"s": s, "c": s.count("C"), "t": s.count("T"), "u": s.count("U"),
                   "c8k": sum(1 for _, _, c, nt in g if c == "C" and nt <= 8192),
                   "tok": sum(nt for *_, nt in g), "seed": sorted(sds)[0] if sds else None}
    out.update({"wrong_n": wrong_n, "dup_or_gap_samples": dup_samples, "seed_not_single": seed_multi,
                "seed_mismatch": seed_mismatch[:5], "n_seed_mismatch": len(seed_mismatch)})
    return out, pp, hashes, tokh


def summarize(pp, ids, n):
    c = np.array([pp[i]["c"] for i in ids], float)
    t = np.array([pp[i]["t"] for i in ids], float)
    u = np.array([pp[i]["u"] for i in ids], float)
    c8 = np.array([pp[i]["c8k"] for i in ids], float)
    tot = n * len(ids)
    return {"problems": len(ids), "attempts": int(tot), "correct": int(c.sum()), "truncated_n": int(t.sum()),
            "unknown_n": int(u.sum()), "pass1": float((c / n).mean()), "pass8": float(np.mean([pass_at_k(n, int(x), 8) for x in c])),
            "trunc": float(t.sum() / tot), "acc_finished": float(c.sum() / (tot - t.sum())) if tot > t.sum() else None,
            "pass1_within_8192": float((c8 / n).mean()), "solved_any": int((c > 0).sum()), "zero": int((c == 0).sum()),
            "tokens": int(sum(pp[i]["tok"] for i in ids))}


def compare_summary(summ, rec, pp, ids, n):
    diffs = []
    sp = {p["id"]: p for p in summ.get("per_problem", [])}
    if sorted(sp) != sorted(ids):
        diffs.append("summary per_problem ids differ from panel")
    for i in ids:
        if i in sp and (sp[i]["c"] != pp[i]["c"] or sp[i].get("c_within_8192", pp[i]["c8k"]) != pp[i]["c8k"] or sp[i]["n"] != n):
            diffs.append(f"per_problem {i}: summary c={sp[i]['c']} c8k={sp[i].get('c_within_8192')} n={sp[i]['n']} records c={pp[i]['c']} c8k={pp[i]['c8k']}")
    for k_s, k_r, tol in (("pass1", "pass1", 1e-12), ("pass8", "pass8", 1e-9), ("truncated", "trunc", 1e-12),
                          ("unknown", "unknown_n", 0), ("completions", "attempts", 0), ("pass1_within_8192", "pass1_within_8192", 1e-12),
                          ("tokens", "tokens", 0)):
        if k_s in summ and abs(float(summ[k_s]) - float(rec[k_r])) > tol:
            diffs.append(f"{k_s}: summary {summ[k_s]} records {rec[k_r]}")
    return diffs


def load_panel(kind):
    path, key = PANELS[kind]
    items = json.load(open(path))["pools"][key]
    return path, key, items


def main(out_path):
    res = {"panels": {}, "evals": {}, "shards": {}, "length": {}, "cgpu_identity": {}, "diag": {}}
    pan = {}
    for kind in PANELS:
        path, key, items = load_panel(kind)
        pan[kind] = (key, items)
        res["panels"][kind] = {"path": path, "sha256": sha_file(path), "items": len(items), "ids": [it["id"] for it in items]}
        if kind == "stress":
            res["panels"][kind]["tier"] = {it["id"]: it["tier"] for it in items}
            res["panels"][kind]["source"] = {it["id"]: it["source_collection"] for it in items}
    EV = []
    for lab, d in [("base", "qis-test-base"), ("H-s101", "qis-test-H-s101-sym"), ("H-s202", "qis-test-H-s202-sym"),
                   ("H-s202-fb", "qis-test-H-s202-sym-fallback"), ("R-s101", "qis-test-R-s101-sym"), ("R-s202", "qis-test-R-s202-sym"),
                   ("U-s101", "qis-test-U-s101-sym"), ("U-s202", "qis-test-U-s202-sym")]:
        EV.append((f"test/{lab}", f"{P}/test-eval/{d}", "test", 32, "test-final", 12288))
    for lab in ("R-s101", "R-s202", "U-s101", "U-s202"):
        EV.append((f"cgpu/{lab}", f"{P}/test-eval-cgpu/qis-cgpu-{lab}", "test", 32, "test-final", 12288))
    for lab in ("base", "H-s101-r128", "H-s202-r128", "H-s202-r64", "R-s101-r128", "R-s101-r64", "R-s202-r128", "U-s101-r128", "U-s202-r128"):
        EV.append((f"stress/{lab}", f"{P}/stress-eval/{lab}", "stress", 8, "stress-v1", 12288))
    for pol in ("base", "R-s101", "R-s202", "U-s101", "U-s202"):
        for form in ("uncued", "cued"):
            EV.append((f"cue/{pol}-{form}", f"{P}/cue-eval/v1/{pol}-{form}", f"cue_{form}", 8, "cue-v1", 12288))
    SH = [("test/H-s202", f"{P}/test-eval/qis-test-H-s202-sym-shards", 6, "test", 32, "test-final"),
          ("test/H-s202-fb", f"{P}/test-eval/qis-test-H-s202-sym-fallback-shards", 6, "test", 32, "test-final"),
          ("stress/H-s202-r128", f"{P}/stress-eval/H-s202-r128-shards", 3, "stress", 8, "stress-v1")]
    sharded = {s[0] for s in SH}
    tokhash = {}
    pp_all = {}
    merged_hash = {}
    for lab, d, kind, n, label, cap in EV:
        key, items = pan[kind]
        ids = [it["id"] for it in items]
        rec_path = os.path.join(d, "records.jsonl.gz")
        if not os.path.exists(rec_path):
            res["evals"][lab] = {"dir": d, "status": "unavailable (no records.jsonl.gz)"}
            continue
        keep_tok = lab.startswith("cgpu/") or lab in ("test/R-s101", "test/R-s202", "test/U-s101", "test/U-s202")
        info, pp, hashes, th = scan_records(rec_path, key, n, label, ids, cap, keep_hash=lab in sharded, keep_tokhash=keep_tok)
        summ = json.load(open(os.path.join(d, "summary.json")))
        rec = summarize(pp, ids, n)
        info["summary_fields"] = {k: summ.get(k) for k in ("ckpt", "ckpt_model_sha256", "panel", "n", "max_tokens", "seed_label",
                                                           "pass1", "pass8", "truncated", "unknown", "completions", "problems", "seconds",
                                                           "shards")}
        info["summary_vs_records"] = compare_summary(summ, rec, pp, ids, n)
        info["records"] = rec
        info["records_sha256"] = sha_file(rec_path)
        info["summary_sha256"] = sha_file(os.path.join(d, "summary.json"))
        if kind == "stress":
            tiers = res["panels"]["stress"]["tier"]
            info["tiers"] = {t: summarize(pp, [i for i in ids if tiers[i] == t], n) for t in ("harder", "superhard")}
        res["evals"][lab] = info
        pp_all[lab] = {i: {k: pp[i][k] for k in ("s", "c", "t", "u", "c8k", "tok")} for i in ids}
        if th:
            tokhash[lab] = th
        if hashes:
            merged_hash[lab] = hashes
        print(lab, rec["pass1"], info["n_bad"], len(info["wrong_n"]), info["n_seed_mismatch"], info["summary_vs_records"][:2], flush=True)
    # common-GPU vs original identity
    for lab in ("R-s101", "R-s202", "U-s101", "U-s202"):
        o, g = tokhash.get(f"test/{lab}"), tokhash.get(f"cgpu/{lab}")
        if o and g:
            same = sum(o[k] == g.get(k) for k in o)
            po, pg = pp_all[f"test/{lab}"], pp_all[f"cgpu/{lab}"]
            same_c = sum(po[i]["c"] == pg[i]["c"] for i in po)
            same_reward = sum((a == "C") == (b == "C") for i in po for a, b in zip(po[i]["s"], pg[i]["s"]))
            res["cgpu_identity"][lab] = {"identical_token_ids": same, "attempts": len(o), "identical_c": same_c,
                                         "identical_reward": same_reward}
    # shards
    for lab, root, k, kind, n, label in SH:
        key, items = pan[kind]
        ids = [it["id"] for it in items]
        sh = {"root": root, "k": k, "shards": {}}
        union = {}
        seen = Counter()
        for s in range(k):
            sd = os.path.join(root, f"k{s}")
            want = ids[s::k]
            pj = json.load(open(os.path.join(sd, f"panel_{s}.json")))["pools"][key]
            pj_ids = [it["id"] for it in pj]
            # each shard dir holds all k shard panels; check they agree with the split rule
            all_panels_ok = all([it["id"] for it in json.load(open(os.path.join(sd, f"panel_{j}.json")))["pools"][key]] == ids[j::k]
                                for j in range(k))
            same_items = pj == items[s::k]
            info, pp, hashes, _ = scan_records(os.path.join(sd, "records.jsonl.gz"), key, n, label, want, 12288, keep_hash=True)
            ssum = json.load(open(os.path.join(sd, "summary.json")))
            info["summary_vs_records"] = compare_summary(ssum, summarize(pp, want, n), pp, want, n)
            info["panel_ids_equal_split"] = pj_ids == want
            info["panel_items_identical_to_full_panel_slice"] = same_items
            info["all_k_panel_files_follow_split"] = all_panels_ok
            info["ckpt_model_sha256"] = ssum.get("ckpt_model_sha256")
            info["seed_label"] = ssum.get("seed_label")
            info["max_tokens"] = ssum.get("max_tokens")
            info["pass1"] = ssum.get("pass1")
            info["seconds"] = ssum.get("seconds")
            info["n_records"] = len(hashes)
            for kk, h in hashes.items():
                seen[kk] += 1
                union[kk] = h
            sh["shards"][f"k{s}"] = info
        mh = merged_hash.get(lab, {})
        sh["union_records"] = len(union)
        sh["duplicate_keys_across_shards"] = sum(1 for v in seen.values() if v > 1)
        sh["merged_records"] = len(mh)
        sh["merged_equals_union"] = (set(mh) == set(union)) and all(mh[x] == union[x] for x in union)
        sh["merged_minus_union"] = len(set(mh) - set(union))
        sh["union_minus_merged"] = len(set(union) - set(mh))
        sh["differing_lines"] = sum(1 for x in union if x in mh and mh[x] != union[x])
        shard_ids = [set(ids[s::k]) for s in range(k)]
        sh["shard_id_sets_disjoint"] = sum(len(x) for x in shard_ids) == len(set().union(*shard_ids))
        sh["shard_id_sets_cover_panel"] = set().union(*shard_ids) == set(ids)
        res["shards"][lab] = sh
        print("shards", lab, sh["merged_equals_union"], sh["union_records"], flush=True)
    # length extension
    LT = [("test", ck, f"{P}/length-test/{ck}-r128/all", f"test/{ck}") for ck in ("H-s101", "R-s101", "R-s202", "U-s101", "U-s202")]
    LS = [("stress", ck, f"{P}/length-diag/{ck}-r128", f"stress/{ck}-r128") for ck in ("H-s101", "R-s101", "R-s202", "U-s101", "U-s202")]
    orig_cache = {}

    def originals(lab, kind):
        if lab not in orig_cache:
            d = {"test": {"test/H-s101": "qis-test-H-s101-sym", "test/R-s101": "qis-test-R-s101-sym", "test/R-s202": "qis-test-R-s202-sym",
                          "test/U-s101": "qis-test-U-s101-sym", "test/U-s202": "qis-test-U-s202-sym"}}
            path = (f"{P}/test-eval/{d['test'][lab]}/records.jsonl.gz" if kind == "test" else f"{P}/stress-eval/{lab.split('/')[1]}/records.jsonl.gz")
            o = {}
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    o[(r["prompt_id"], r["sample"])] = (r["n_tokens"], r["finish"], r["reward"], r["status"])
            orig_cache.clear()
            orig_cache[lab] = o
        return orig_cache[lab]

    for kind, ck, d, olab in LT + LS:
        o = originals(olab, kind)
        parts = ["all"] if kind == "test" else ["t1", "rest"]
        rows = {}
        part_of = {}
        summ = {}
        dup = 0
        for part in parts:
            pth = os.path.join(d, part, "rows.jsonl.gz") if kind == "stress" else os.path.join(d, "rows.jsonl.gz")
            summ[part] = json.load(open(os.path.join(os.path.dirname(pth), "summary.json")))
            with gzip.open(pth, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    kk = (r["prompt_id"], r["sample"])
                    if kk in rows:
                        dup += 1
                    r.pop("new_ids", None)
                    rows[kk] = r
                    part_of[kk] = part
        chk = Counter()
        bad = []
        for kk, r in rows.items():
            on = o.get(kk)
            if on is None:
                bad.append(("not_in_original", kk)); continue
            if (r["n_tokens_1"], r["finish_1"], r["reward_1"], r["status_1"]) != on:
                bad.append(("orig_mismatch", kk))
            if r["extended"]:
                chk["extended"] += 1
                if r["n_tokens_1"] != 12288 or r["finish_1"] != "length":
                    bad.append(("extended_not_capped", kk))
                if r["n_tokens_2"] != r["n_tokens_1"] + r["new_tokens"]:
                    bad.append(("token_sum", kk))
                if r["finish_2"] == "length" and r["n_tokens_2"] != 16384:
                    bad.append(("still_capped_not_16384", kk))
                if r["finish_2"] != "length" and r["n_tokens_2"] > 16384:
                    bad.append(("over_16384", kk))
                if r["finish_2"] == "length" and r["reward_2"] != 0:
                    bad.append(("capped2_reward1", kk))
            else:
                if r["finish_1"] == "length":
                    bad.append(("capped_not_extended", kk))
                if (r["n_tokens_2"], r["finish_2"], r["reward_2"]) != (r["n_tokens_1"], r["finish_1"], r["reward_1"]):
                    bad.append(("kept_row_changed", kk))
        capped_orig = {kk for kk, v in o.items() if v[1] == "length"}
        not_extended = sorted(capped_orig - {kk for kk, r in rows.items() if r["extended"]})
        L = {"dir": d, "rows": len(rows), "duplicates": dup, "orig_attempts": len(o), "rows_cover_all_original": set(rows) == set(o),
             "capped_original": len(capped_orig), "capped_original_with_ntokens_ne_12288": sum(1 for kk in capped_orig if o[kk][0] != 12288),
             "capped_not_extended": [list(x) for x in not_extended[:20]], "n_capped_not_extended": len(not_extended),
             "bad": [list(map(str, b)) for b in bad[:20]], "n_bad": len(bad), "summaries": summ}
        # per problem and per set
        sets = {"all": list(rows)}
        if kind == "stress":
            sets["t1"] = [kk for kk in rows if part_of[kk] == "t1"]
            tiers = res["panels"]["stress"]["tier"]
            for t in ("harder", "superhard"):
                sets[f"all_{t}"] = [kk for kk in rows if tiers[kk[0]] == t]
                sets[f"t1_{t}"] = [kk for kk in rows if part_of[kk] == "t1" and tiers[kk[0]] == t]
        L["sets"] = {}
        L["per_problem"] = {}
        for sname, keys in sets.items():
            byp = defaultdict(lambda: [0, 0, 0])
            for kk in keys:
                r = rows[kk]
                b = byp[kk[0]]
                b[0] += 1; b[1] += r["reward_1"] == 1; b[2] += r["reward_2"] == 1
            ext = [rows[kk] for kk in keys if rows[kk]["extended"]]
            L["sets"][sname] = {"attempts": len(keys), "problems": len(byp),
                                "pass1_1": float(np.mean([b[1] / b[0] for b in byp.values()])),
                                "pass1_2": float(np.mean([b[2] / b[0] for b in byp.values()])),
                                "correct_1": sum(rows[kk]["reward_1"] == 1 for kk in keys),
                                "correct_2": sum(rows[kk]["reward_2"] == 1 for kk in keys),
                                "capped_1": len(ext), "recovered": sum(r["reward_2"] == 1 for r in ext),
                                "finished_wrong": sum(r["finish_2"] != "length" and r["reward_2"] != 1 for r in ext),
                                "still_capped": sum(r["finish_2"] == "length" for r in ext),
                                "new_tokens": sum(r["new_tokens"] for r in ext),
                                "new_tokens_recovered_only": sum(r["new_tokens"] for r in ext if r["reward_2"] == 1),
                                "attempts_per_problem": sorted(Counter(b[0] for b in byp.values()).items())}
            if sname in ("all", "t1"):
                L["per_problem"][sname] = {p: b for p, b in byp.items()}
        res["length"][f"{kind}/{ck}"] = L
        print("length", kind, ck, L["n_bad"], L["n_capped_not_extended"], L["sets"]["all"]["recovered"], flush=True)
    # diagnostic records (n per problem, cap)
    for rnd in ("r0", "r64"):
        root = f"{P}/diag-local/{rnd}/records"
        for br in sorted(os.listdir(root)):
            bd = os.path.join(root, br)
            files = sorted(os.listdir(bd)) if os.path.isdir(bd) else []
            info = {}
            for fn in files:
                fp = os.path.join(bd, fn)
                if not fn.endswith(".jsonl.gz"):
                    info[fn] = "non-records"; continue
                per = Counter(); fin = Counter(); mx = 0; st = Counter()
                with gzip.open(fp, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        per[r.get("prompt_id")] += 1; fin[r.get("finish")] += 1; mx = max(mx, r.get("n_tokens", 0)); st[r.get("status")] += 1
                info[fn] = {"lines": sum(per.values()), "problems": len(per), "n_per_problem": sorted(Counter(per.values()).items()),
                            "finish": dict(fin), "max_n_tokens": mx, "status": dict(st)}
            res["diag"][f"{rnd}/{br}"] = info
    json.dump(res, open(out_path, "w"))
    json.dump(pp_all, open(out_path.replace(".json", "_pp.json"), "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
