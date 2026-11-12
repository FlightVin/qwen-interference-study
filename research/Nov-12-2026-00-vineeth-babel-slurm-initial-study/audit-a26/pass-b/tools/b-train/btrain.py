"""A26 audit pass B, agent b-train: independent reconstruction of training-side results from raw records.

Reads only raw run outputs under $P (metrics.jsonl rows written by the trainer, records/train_r*.jsonl.gz,
in-run eval records, diagnostic per-branch eval records and rows). Uses stdlib + numpy only; no project code.
Usage (compute node): python btrain.py OUT_DIR
Writes OUT_DIR/train_*.json (unrounded).
"""
import collections
import glob
import gzip
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

P = "/data/user_data/vineethb/qis"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
RUNS = [f"{a}-s{s}" for s in (101, 202) for a in "HRU"]
EVAL_ROUNDS = (0, 32, 64, 96, 128)
BOOT_SEED = 26111111          # b-train's own bootstrap seed (not the project's 2027)
B = 10000
PANELS = ("dev", "probe_J_R", "probe_J_U")

bundle = json.load(open(f"{P}/data/main-v1/bundle_main_v1.json"))
pools = bundle["pools"]
ORDER = {"dev": [x["id"] for x in pools["dev"]], "probe_J_H": [x["id"] for x in pools["probe_J_H"]],
         "probe_J_R": [x["id"] for x in pools["probe_J_R"]], "probe_J_U": [x["id"] for x in pools["probe_J_U"]]}
HARD_IDS = {x["id"] for x in pools["hard"]}


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else sorted(o) if isinstance(o, set) else str(o))


def rundir(run):
    return f"{P}/runs/qis-main-{run}-sym"


# ---------------------------------------------------------------- record readers
def strip_ids(line):
    k = line.find(', "ids": [')
    return json.loads(line[:k] + "}") if k > 0 else json.loads(line)


def read_train(fn):
    """Return list of 16 groups (by slot) with per-attempt fields; checks 8 samples per slot."""
    g = {}
    n = 0
    with gzip.open(fn, "rt") as f:
        for line in f:
            r = strip_ids(line)
            n += 1
            s = g.setdefault(r["slot"], {"round": r["round"], "slot": r["slot"], "role": r["role"],
                                         "pid": r["prompt_id"], "seed": r["seed"], "att": {}})
            assert s["pid"] == r["prompt_id"] and s["role"] == r["role"]
            s["att"][r["sample"]] = (r["reward"], r["adv"], r["n_tokens"], r["finish"], r["status"], r["extracted"])
    out = []
    for slot in sorted(g):
        s = g[slot]
        assert sorted(s["att"]) == list(range(8)), (fn, slot)
        a = [s["att"][j] for j in range(8)]
        out.append({"round": s["round"], "slot": slot, "role": s["role"], "pid": s["pid"], "seed": s["seed"],
                    "rew": [x[0] for x in a], "adv": [x[1] for x in a], "ntok": [x[2] for x in a],
                    "fin": [x[3] for x in a], "status": [x[4] for x in a], "extracted": [x[5] for x in a]})
    return out, n


def read_eval(fn):
    """prompt_id -> dict(c, n, trunc, ntok_max, trunc_ntoks set, status counter, seeds set)."""
    d = {}
    with gzip.open(fn, "rt") as f:
        for line in f:
            r = strip_ids(line)
            e = d.setdefault(r["prompt_id"], {"c": 0, "n": 0, "trunc": 0, "ntok_max": 0, "trunc_ntok": set(),
                                              "status": collections.Counter(), "samples": set(), "seed": set()})
            e["n"] += 1
            e["c"] += r["reward"] == 1
            e["status"][r["status"]] += 1
            e["samples"].add(r["sample"])
            e["seed"].add(r["seed"])
            e["ntok_max"] = max(e["ntok_max"], r["n_tokens"])
            if r["finish"] == "length":
                e["trunc"] += 1
                e["trunc_ntok"].add(r["n_tokens"])
    return d


def summarize_eval(d, ids):
    miss = [i for i in ids if i not in d]
    assert not miss, miss
    c = np.array([d[i]["c"] for i in ids], float)
    n = np.array([d[i]["n"] for i in ids], float)
    st = collections.Counter()
    for i in ids:
        st.update(d[i]["status"])
    tn = sorted(set().union(*[d[i]["trunc_ntok"] for i in ids]))
    return {"problems": len(ids), "n_per_problem": sorted(set(n.astype(int).tolist())), "attempts": int(n.sum()),
            "correct": int(c.sum()), "pass1": float(np.mean(c / n)), "truncated": int(sum(d[i]["trunc"] for i in ids)),
            "trunc_share": float(sum(d[i]["trunc"] for i in ids) / n.sum()),
            "ntok_max": int(max(d[i]["ntok_max"] for i in ids)), "trunc_ntok_values": tn,
            "status": dict(st), "unknown": int(st.get("unknown", 0)),
            "per_problem_c": [int(x) for x in c], "extra_ids_in_file": len(set(d) - set(ids))}


# ---------------------------------------------------------------- metrics reader with segments
def read_metrics(fn):
    rows, seg = [], -1
    segs = []
    for i, line in enumerate(open(fn)):
        r = json.loads(line)
        if r["kind"] == "event" and r["what"] == "start":
            seg += 1
            segs.append({"seg": seg, "line": i, "job": r.get("job"), "resume_round": r.get("resume_round"),
                         "resume_from": r.get("resume_from"), "gpu": r.get("gpu"), "code_sha": r.get("code_sha"),
                         "utc": r["utc"], "attempt_id": r["attempt_id"]})
        r["_line"], r["_seg"] = i, seg
        rows.append(r)
    return rows, segs


# ---------------------------------------------------------------- seeds / minibatch plan (re-implemented)
def name_key(name):
    d = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))


def nrng(master, name):
    return np.random.default_rng(np.random.SeedSequence(entropy=master, spawn_key=name_key(name)))


def mb_plan(roles, rng, nmb=4):
    by = {}
    for g, r in enumerate(roles):
        by.setdefault(r, []).append(g)
    plan = [[] for _ in range(nmb)]
    for r in sorted(by):
        gs = by[r]
        perm = rng.permutation(len(gs))
        k = len(gs) // nmb
        for m in range(nmb):
            plan[m] += [gs[i] for i in perm[m * k:(m + 1) * k]]
    order = rng.permutation(nmb)
    return [sorted(plan[m]) for m in order]


def adv_formula(rew):
    r = np.array(rew, np.float64)
    if (r == r[0]).all():
        return [0.0] * len(r)
    return list((r - r.mean()) / (r.std() + 1e-6))


def gclass(rew):
    c = sum(1 for x in rew if x == 1)
    return "all_wrong" if c == 0 else ("all_correct" if c == len(rew) else "mixed")


def boot_paired(a, b, seed=BOOT_SEED):
    """95% percentile interval of mean(a-b) resampling problems (same draw both)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "n_problems": int(len(d)), "draws": B, "seed": seed}


# ================================================================ 1. segments, canonical rows, records
segments, canon, train, superseded_train = {}, {}, {}, {}
row_match = {}
SUPERSEDED_DIRS = {"R-s101": {"records/superseded-11081724": "11081724"},
                   "R-s202": {"records/superseded-11081729": "11081729"},
                   "U-s202": {"records/superseded-11094037": "11094037", "records/superseded-11097811": "11097811"},
                   "H-s202": {"records_a2orig_r60-64": "11082192"}}
for run in RUNS:
    rows, segs = read_metrics(f"{rundir(run)}/metrics.jsonl")
    # canonical = last write per key
    last = {}
    for r in rows:
        if r["kind"] == "round":
            last[("round", r["round"])] = r
        elif r["kind"] == "step":
            last[("step", r["round"], r["minibatch"])] = r
        elif r["kind"] == "eval":
            last[("eval", r["panel"], r["round"])] = r
    canon[run] = last
    for s in segs:
        mine = [r for r in rows if r["_seg"] == s["seg"]]
        rr = [r["round"] for r in mine if r["kind"] == "round"]
        s["round_rows"] = len(rr)
        s["rounds_1based"] = [min(rr), max(rr)] if rr else None
        s["step_rows"] = sum(r["kind"] == "step" for r in mine)
        s["eval_rows"] = sorted({(r["panel"], r["round"]) for r in mine if r["kind"] == "eval"})
        s["canonical_round_rows"] = sum(1 for r in mine if r["kind"] == "round" and last[("round", r["round"])] is r)
        s["canonical_step_rows"] = sum(1 for r in mine if r["kind"] == "step" and last[("step", r["round"], r["minibatch"])] is r)
        s["canonical_eval_rows"] = sum(1 for r in mine if r["kind"] == "eval" and last[("eval", r["panel"], r["round"])] is r)
        s["last_event"] = [r.get("what") for r in mine if r["kind"] == "event"][-1]
        s["attempt_ids"] = sorted({r["attempt_id"] for r in mine})
    segments[run] = segs
    # canonical records
    tr = []
    for k in range(128):
        groups, n = read_train(f"{rundir(run)}/records/train_r{k:04d}.jsonl.gz")
        assert n == 128 and len(groups) == 16 and all(g["round"] == k for g in groups)
        tr.append(groups)
    train[run] = tr
    # superseded records
    sup = {}
    for sub, job in SUPERSEDED_DIRS.get(run, {}).items():
        for fn in sorted(glob.glob(f"{rundir(run)}/{sub}/train_r*.jsonl.gz")):
            k = int(fn[-13:-9])
            sup[(job, k)] = read_train(fn)[0]
    superseded_train[run] = sup
    # match every round row to records (canonical and superseded)
    def gcounts(groups):
        cls = collections.Counter(gclass(g["rew"]) for g in groups)
        return (cls["all_wrong"], cls["mixed"], cls["all_correct"],
                sum(f == "length" for g in groups for f in g["fin"]),
                round(sum(sum(g["ntok"]) for g in groups) / 128, 2))
    m = []
    for r in rows:
        if r["kind"] != "round":
            continue
        k = r["round"] - 1
        rc = (r["groups_all0"], r["groups_mixed"], r["groups_all1"], r["truncated"], round(r["len_mean"], 2))
        job = segs[r["_seg"]]["job"]
        is_c = last[("round", r["round"])] is r
        mc = gcounts(tr[k]) == rc
        ms = gcounts(sup[(job, k)]) == rc if (job, k) in sup else None
        if not is_c or (job, k) in sup:
            m.append({"round_1based": r["round"], "line": r["_line"], "job": job, "canonical": is_c,
                      "matches_canonical_records": mc, "matches_superseded_records_of_job": ms,
                      "counts_all0_mixed_all1_trunc_lenmean": rc})
    row_match[run] = {"noncanonical_or_superseded_rows": m,
                      "canonical_rows_matching_records": sum(
                          1 for k in range(128) if gcounts(tr[k]) == (last[("round", k + 1)]["groups_all0"],
                                                                     last[("round", k + 1)]["groups_mixed"],
                                                                     last[("round", k + 1)]["groups_all1"],
                                                                     last[("round", k + 1)]["truncated"],
                                                                     round(last[("round", k + 1)]["len_mean"], 2)))}
    print(run, "loaded", flush=True)

dump("train_segments.json", {"segments": segments, "row_match": row_match,
                             "rule": "canonical row = last write of each key ((round), (round, minibatch), (panel, round)) "
                                     "in metrics.jsonl; canonical records = files in records/ (superseded attempts moved to "
                                     "records/superseded-<job> or records_a2orig_r60-64)"})

# ================================================================ 1b. schedule sanity (shared slots, pairs)
sched = {}
for s in (101, 202):
    same_shared = sum(1 for k in range(128) if [g["pid"] for g in train[f"H-s{s}"][k][:8]] ==
                      [g["pid"] for g in train[f"R-s{s}"][k][:8]] == [g["pid"] for g in train[f"U-s{s}"][k][:8]])
    same_seed = sum(1 for k in range(128) if [g["seed"] for g in train[f"H-s{s}"][k]] ==
                    [g["seed"] for g in train[f"R-s{s}"][k]] == [g["seed"] for g in train[f"U-s{s}"][k]])
    rel = {x["id"]: i for i, x in enumerate(pools["related"])}
    unr = {x["id"]: i for i, x in enumerate(pools["unrelated"])}
    pair_ok = sum(1 for k in range(128) for j in range(8, 16)
                  if rel[train[f"R-s{s}"][k][j]["pid"]] == unr[train[f"U-s{s}"][k][j]["pid"]])
    roles_ok = {run: all([g["role"] for g in train[run][k]] == (["hard_common"] * 8 + (["hard_extra"] * 8 if run[0] == "H" else ["aux"] * 8))
                         for k in range(128)) for run in RUNS if run.endswith(str(s))}
    pres = {}
    for run in (f"H-s{s}", f"R-s{s}", f"U-s{s}"):
        cnt = collections.Counter(g["pid"] for k in range(128) for g in train[run][k] if g["role"].startswith("hard"))
        pres[run] = {"distinct_hard": len(cnt), "min": min(cnt.values()), "max": max(cnt.values()),
                     "in_H_train": len(set(cnt) & HARD_IDS)}
    sched[s] = {"rounds_shared_slot_ids_identical_HRU": same_shared, "rounds_request_seeds_identical_HRU": same_seed,
                "R_U_practice_slots_same_pair_index": pair_ok, "of_practice_slots": 128 * 8, "roles_as_expected": roles_ok,
                "hard_presentations_per_problem": pres}
dump("train_schedule_check.json", sched)

# ================================================================ 4. group statistics, advantages, steps
groups_out, steps_out, adv_check = {}, {}, {}
for run in RUNS:
    seed = int(run[-3:])
    per_round = []
    tot = collections.Counter()
    adv_maxdiff, adv_nonzero_in_nonmixed, adv_zero_in_mixed = 0.0, 0, 0
    for k in range(128):
        rc = collections.Counter()
        for g in train[run][k]:
            cl = gclass(g["rew"])
            rc[(g["role"], cl)] += 1
            tot[(g["role"], cl)] += 1
            f = adv_formula(g["rew"])
            adv_maxdiff = max(adv_maxdiff, max(abs(a - b) for a, b in zip(f, g["adv"])))
            if cl != "mixed":
                adv_nonzero_in_nonmixed += sum(a != 0.0 for a in g["adv"])
            else:
                adv_zero_in_mixed += sum(a == 0.0 for a in g["adv"])
        per_round.append({"round_1based": k + 1, **{f"{r}|{c}": v for (r, c), v in sorted(rc.items())}})
    groups_out[run] = {"per_round": per_round, "totals": {f"{r}|{c}": v for (r, c), v in sorted(tot.items())},
                       "totals_all_roles": {c: sum(v for (r, cc), v in tot.items() if cc == c)
                                            for c in ("all_wrong", "mixed", "all_correct")}}
    adv_check[run] = {"max_abs_diff_recorded_vs_formula": adv_maxdiff,
                      "nonzero_adv_in_nonmixed_groups": adv_nonzero_in_nonmixed,
                      "zero_adv_in_mixed_groups": adv_zero_in_mixed}
    # steps
    st, vt_ok, vt_bad, zero_pred, zero_obs, mom = [], 0, [], 0, 0, []
    comp_ok = True
    for k in range(128):
        roles = [g["role"] for g in train[run][k]]
        plan = mb_plan(roles, nrng(seed, f"minibatch/r{k}"))
        for m, mb in enumerate(plan):
            comp_ok &= collections.Counter(roles[i] for i in mb) == collections.Counter({r: 2 for r in set(roles)})
            row = canon[run][("step", k, m)]
            vt = sum(sum(train[run][k][gi]["ntok"]) for gi in mb)
            if vt == row["valid_tokens"]:
                vt_ok += 1
            else:
                vt_bad.append((k, m, vt, row["valid_tokens"]))
            cls = [gclass(train[run][k][gi]["rew"]) for gi in mb]
            pred0 = all(c != "mixed" for c in cls)
            obs0 = row["grad_norm_preclip"] == 0.0
            zero_pred += pred0
            zero_obs += obs0
            rec = {"round_1based": k + 1, "step_round_field": k, "minibatch": m, "opt_step": row["opt_step"],
                   "grad_norm_preclip": row["grad_norm_preclip"], "grad_clipped": row["grad_clipped"], "lr": row["lr"],
                   "loss": row["loss"], "valid_tokens": row["valid_tokens"], "groups": mb,
                   "classes": cls, "roles": [roles[i] for i in mb], "pred_zero": pred0, "obs_zero": obs0}
            if pred0:
                rec["all_32_adv_zero"] = all(a == 0.0 for gi in mb for a in train[run][k][gi]["adv"])
                rec["prompt_ids"] = [train[run][k][gi]["pid"] for gi in mb]
                mom.append(rec)
            st.append(rec)
    mism = [(r["round_1based"], r["minibatch"]) for r in st if r["pred_zero"] != r["obs_zero"]]
    gn = np.array([r["grad_norm_preclip"] for r in st])
    steps_out[run] = {"n_steps": len(st), "valid_tokens_match": vt_ok, "valid_tokens_mismatch": vt_bad,
                      "role_composition_2_2_all": bool(comp_ok),
                      "zero_grad_predicted": zero_pred, "zero_grad_observed": zero_obs, "pred_obs_mismatch": mism,
                      "momentum_only_steps": mom,
                      "first_momentum_only_round_1based": min([r["round_1based"] for r in mom], default=None),
                      "momentum_only_all_adv_zero": all(r["all_32_adv_zero"] for r in mom),
                      "momentum_only_by_quarter": [sum(1 for r in mom if q * 32 < r["round_1based"] <= (q + 1) * 32) for q in range(4)],
                      "momentum_only_composition": dict(collections.Counter(
                          "+".join(sorted(f"{ro}:{c}" for ro, c in zip(r["roles"], r["classes"]))) for r in mom)),
                      "grad_clipped_steps": int(sum(r["grad_clipped"] for r in st)),
                      "grad_norm_nonzero_median": float(np.median(gn[gn > 0])),
                      "lr_values": sorted({r["lr"] for r in st}),
                      "opt_step_first_last": [st[0]["opt_step"], st[-1]["opt_step"]]}
    # superseded step rows vs superseded records (valid tokens)
    sup_chk = []
    rows, segs = read_metrics(f"{rundir(run)}/metrics.jsonl")
    for r in rows:
        if r["kind"] != "step":
            continue
        job = segs[r["_seg"]]["job"]
        key = (job, r["round"])
        if key in superseded_train[run] and canon[run][("step", r["round"], r["minibatch"])]["_line"] != r["_line"]:
            sg = superseded_train[run][key]
            plan = mb_plan([g["role"] for g in sg], nrng(seed, f"minibatch/r{r['round']}"))
            vt = sum(sum(sg[gi]["ntok"]) for gi in plan[r["minibatch"]])
            sup_chk.append(vt == r["valid_tokens"])
    steps_out[run]["superseded_step_rows_valid_tokens_match"] = [sum(sup_chk), len(sup_chk)]
    print(run, "steps done", flush=True)
dump("train_groups.json", {"groups": groups_out, "advantage_check": adv_check})
dump("train_steps.json", steps_out)

# ================================================================ 2b. training reward per role per round
rewards = {}
for run in RUNS:
    rr = []
    for k in range(128):
        row = canon[run][("round", k + 1)]
        d = {"round_1based": k + 1, "cap_round_row": row["max_tokens"],
             "ntok_max_records": max(max(g["ntok"]) for g in train[run][k]),
             "trunc_ntok_values": sorted({n for g in train[run][k] for n, f in zip(g["ntok"], g["fin"]) if f == "length"}),
             "cum_hard_attempts": 8 * sum(1 for kk in range(k + 1) for g in train[run][kk] if g["role"].startswith("hard"))}
        for role in sorted({g["role"] for g in train[run][k]}):
            gs = [g for g in train[run][k] if g["role"] == role]
            att = [x for g in gs for x in g["rew"]]
            d[f"success|{role}"] = sum(att) / len(att)
            d[f"trunc|{role}"] = sum(f == "length" for g in gs for f in g["fin"]) / len(att)
            d[f"row_success|{role}"] = row["success"].get(role)
        hard = [x for g in train[run][k] if g["role"].startswith("hard") for x in g["rew"]]
        d["success|all_hard"] = sum(hard) / len(hard)
        d["status_unknown"] = sum(s == "unknown" for g in train[run][k] for s in g["status"])
        rr.append(d)
    # window summaries
    win = {}
    for lo, hi in ((1, 32), (33, 64), (65, 96), (97, 128)):
        sub = rr[lo - 1:hi]
        win[f"{lo}-{hi}"] = {key: float(np.mean([x[key] for x in sub])) for key in sub[0] if key.startswith(("success|", "trunc|"))}
    # at equal hard exposure: hard success over first X cumulative hard attempts
    eq = {}
    for X in (2048, 4096, 8192):
        kmax = next(i for i, x in enumerate(rr) if x["cum_hard_attempts"] >= X)
        eq[str(X)] = {"rounds_1based": [1, kmax + 1],
                      "success_all_hard_mean": float(np.mean([x["success|all_hard"] for x in rr[:kmax + 1]])),
                      "success_hard_common_mean": float(np.mean([x["success|hard_common"] for x in rr[:kmax + 1]]))}
    rewards[run] = {"per_round": rr, "windows": win, "equal_hard_exposure": eq,
                    "row_vs_records_max_abs_diff": max(abs(x[f"success|{r}"] - x[f"row_success|{r}"])
                                                       for x in rr for r in ("hard_common", "aux", "hard_extra")
                                                       if f"success|{r}" in x)}
dump("train_rewards.json", rewards)

# ================================================================ 2. learning curves from eval records
curves, evalfiles = {}, {}
for run in RUNS:
    cr = {}
    for rnd in EVAL_ROUNDS:
        ent = {}
        for panel in PANELS:
            d = read_eval(f"{rundir(run)}/records/eval_{panel}_r{rnd:04d}.jsonl.gz")
            evalfiles[(run, panel, rnd)] = d
            ent[panel] = summarize_eval(d, ORDER[panel])
            if panel == "dev":
                ent["probe_J_H"] = summarize_eval(d, ORDER["probe_J_H"])
        # compare with metric rows (all rows for this key)
        rows, segs = read_metrics(f"{rundir(run)}/metrics.jsonl")
        for panel in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
            mrows = [r for r in rows if r["kind"] == "eval" and r["panel"] == panel and r["round"] == rnd]
            comp = []
            for r in mrows:
                pp = {x["id"]: x["c"] for x in r["per_problem"]}
                comp.append({"line": r["_line"], "job": segs[r["_seg"]]["job"], "pass1_row": r["pass1"],
                             "trunc_row": r.get("truncated"), "cap_row": r.get("max_tokens"),
                             "per_problem_equal_records": [pp.get(i) for i in ORDER[panel]] == ent[panel]["per_problem_c"],
                             "canonical": canon[run][("eval", panel, rnd)]["_line"] == r["_line"]})
            ent[panel]["metric_rows"] = comp
            ent[panel]["cap_canonical_row"] = canon[run][("eval", "dev" if panel == "probe_J_H" else panel, rnd)].get("max_tokens")
        cr[str(rnd)] = ent
    curves[run] = cr
# extra (non-canonical) evaluations
extra = {}
for run, sub, rnd in (("H-s202", "records_a2orig_r60-64", 64), ("R-s101", "records/superseded-11081724", 128)):
    ent = {}
    for panel in PANELS:
        d = read_eval(f"{rundir(run)}/{sub}/eval_{panel}_r{rnd:04d}.jsonl.gz")
        evalfiles[(run + "|" + sub, panel, rnd)] = d
        ent[panel] = summarize_eval(d, ORDER[panel])
        if panel == "dev":
            ent["probe_J_H"] = summarize_eval(d, ORDER["probe_J_H"])
    rows, segs = read_metrics(f"{rundir(run)}/metrics.jsonl")
    for panel in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
        mrows = [r for r in rows if r["kind"] == "eval" and r["panel"] == panel and r["round"] == rnd]
        ent[panel]["metric_rows"] = [{"line": r["_line"], "job": segs[r["_seg"]]["job"], "pass1_row": r["pass1"],
                                      "per_problem_equal_records": [{x["id"]: x["c"] for x in r["per_problem"]}.get(i) for i in ORDER[panel]] == ent[panel]["per_problem_c"]}
                                     for r in mrows]
    extra[f"{run} {sub} r{rnd}"] = ent
# H-s202 r64: paired comparison a2-orig vs a2-r60
h64 = {}
for panel in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
    a = extra["H-s202 records_a2orig_r60-64 r64"][panel]["per_problem_c"]
    b = curves["H-s202"]["64"][panel]["per_problem_c"]
    h64[panel] = boot_paired(np.array(a) / 8, np.array(b) / 8)
dump("train_curves.json", {"curves": curves, "noncanonical_evals": extra, "H_s202_r64_orig_minus_r60": h64,
                           "J_H_definition": "first 32 dev IDs (bundle probe_J_H); computed from dev records"})

# ================================================================ 3. earliest observed ordering R vs U
order_out = {}
for s in (101, 202):
    for panel in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
        diffs = {}
        for rnd in EVAL_ROUNDS:
            cR = np.array(curves[f"R-s{s}"][str(rnd)][panel]["per_problem_c"]) / 8
            cU = np.array(curves[f"U-s{s}"][str(rnd)][panel]["per_problem_c"]) / 8
            diffs[rnd] = boot_paired(cR, cU)
        final = np.sign(diffs[128]["diff"])
        signs = {r: float(np.sign(diffs[r]["diff"])) for r in EVAL_ROUNDS}
        trained = [r for r in EVAL_ROUNDS if r > 0]
        first = next((r for r in trained if signs[r] == final), None)
        stays_from = next((r for r in trained if all(signs[q] == final for q in trained if q >= r)), None)
        order_out[f"s{s}|{panel}"] = {"R_minus_U": {str(r): diffs[r] for r in EVAL_ROUNDS}, "sign_at_128": float(final),
                                      "signs": {str(r): signs[r] for r in EVAL_ROUNDS},
                                      "first_trained_round_same_order_as_128": first,
                                      "earliest_round_from_which_order_stays": stays_from,
                                      "round0_sign_same_as_128": signs[0] == final}
dump("train_ordering.json", {"ordering": order_out,
                             "note": "rounds 0 and 32 evaluated at cap 8192, rounds 64-128 at 12288 (see cap fields in train_curves.json)",
                             "stress": "no R/U pair of intermediate checkpoints in either seed (only R-s101 r64 and H-s202 r64 retained); not computed"})

# ================================================================ 5. local update diagnostic
diag = {}
for lab in ("r0", "r64"):
    base = f"{P}/diag-local/{lab}"
    rowsd = {}
    for fn in (f"{base}/diag_{lab}.jsonl", f"{base}/diag_{lab}_ref.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            rowsd[f"{r['branch']}-{r['seed']}" if r["seed"] else r["branch"]] = r
    tags = sorted(os.listdir(f"{base}/records"))
    res = {}
    for t in tags:
        key = t[len(lab) + 1:]
        ent = {}
        for panel in ("probe_J_H", "probe_J_R", "probe_J_U"):
            d = read_eval(f"{base}/records/{t}/eval_{panel}_r7064.jsonl.gz")
            ent[panel] = summarize_eval(d, ORDER[panel])
            row = rowsd.get(key)
            if row is not None:
                ent[panel]["row_pass1"] = row["eval"][panel]["pass1"]
                ent[panel]["row_matches_records"] = [x["c"] for x in row["eval"][panel]["per_problem"]] == ent[panel]["per_problem_c"]
            ent[panel]["seeds_set"] = None
        row = rowsd.get(key, {})
        ent["update_row"] = {k: row.get(k) for k in ("batch_reward", "mixed_groups", "grad_norm_preclip", "capped", "prompt_ids", "prompt_idx")}
        res[key] = ent
    # seeds paired across branches?
    seeds_equal = all(read_eval(f"{base}/records/{t}/eval_probe_J_H_r7064.jsonl.gz")[ORDER["probe_J_H"][0]]["seed"] ==
                      read_eval(f"{base}/records/{tags[0]}/eval_probe_J_H_r7064.jsonl.gz")[ORDER["probe_J_H"][0]]["seed"]
                      for t in tags)
    delta = {}
    for key in res:
        if key == "none":
            continue
        delta[key] = {}
        for panel in ("probe_J_H", "probe_J_R", "probe_J_U"):
            a = np.array(res[key][panel]["per_problem_c"]) / 16
            b = np.array(res["none"][panel]["per_problem_c"]) / 16
            delta[key][panel] = boot_paired(a, b)
    # branch-type comparison
    types = {}
    for panel in ("probe_J_H", "probe_J_R", "probe_J_U"):
        vals = {bt: [delta[f"{bt}-{s}"][panel]["diff"] for s in (4101, 4102, 4103, 4104)] for bt in ("hard", "related", "unrelated")}
        allv = sum(vals.values(), [])
        obs = max(np.mean(v) for v in vals.values()) - min(np.mean(v) for v in vals.values())
        # exact permutation over assignments of 12 draws to 3 labelled groups of 4
        cnt, ge = 0, 0
        idx = list(range(12))
        for g1 in itertools.combinations(idx, 4):
            rest = [i for i in idx if i not in g1]
            for g2 in itertools.combinations(rest, 4):
                g3 = [i for i in rest if i not in g2]
                ms = [np.mean([allv[i] for i in g]) for g in (g1, g2, g3)]
                cnt += 1
                ge += (max(ms) - min(ms)) >= obs - 1e-12
        rel_minus_unr = [a - b for a, b in zip(vals["related"], vals["unrelated"])]
        types[panel] = {"per_draw": vals, "type_means": {k: float(np.mean(v)) for k, v in vals.items()},
                        "type_sd": {k: float(np.std(v, ddof=1)) for k, v in vals.items()},
                        "range_of_type_means": float(obs), "perm_p_range": ge / cnt, "perm_n": cnt,
                        "related_minus_unrelated_by_seed": rel_minus_unr,
                        "rerun_minus_none": res["none_rerun"][panel]["pass1"] - res["none"][panel]["pass1"],
                        "cycled_minus_rerun": res["none_cycled"][panel]["pass1"] - res["none_rerun"][panel]["pass1"]}
    # alternative reference: no-update source evaluated after the same sleep/restore/reload cycle (none_cycled)
    for panel in ("probe_J_H", "probe_J_R", "probe_J_U"):
        ref = res["none_cycled"][panel]["pass1"]
        types[panel]["type_means_vs_none_cycled"] = {bt: float(np.mean([res[f"{bt}-{s}"][panel]["pass1"] - ref for s in (4101, 4102, 4103, 4104)]))
                                                     for bt in ("hard", "related", "unrelated")}
    # branches whose three panels are identical attempt-for-attempt in per-problem counts
    sig = collections.defaultdict(list)
    for key in res:
        sig[json.dumps([res[key][p]["per_problem_c"] for p in ("probe_J_H", "probe_J_R", "probe_J_U")])].append(key)
    identical = [v for v in sig.values() if len(v) > 1]
    diag[lab] = {"evals": res, "delta_vs_none": delta, "branch_types": types, "eval_seeds_identical_across_branches": seeds_equal,
                 "identical_eval_groups": identical}
# compare diagnostic none with in-run evaluation of the same policy
diag["compare_inrun"] = {
    "r64_diag_none_vs_Rs101_inrun_r64": {p: {"diag_none": diag["r64"]["evals"]["none"][p]["pass1"],
                                            "diag_none_trunc": diag["r64"]["evals"]["none"][p]["trunc_share"],
                                            "inrun": curves["R-s101"]["64"][p]["pass1"],
                                            "inrun_trunc": curves["R-s101"]["64"][p]["trunc_share"]}
                                         for p in ("probe_J_H", "probe_J_R", "probe_J_U")},
    "r0_diag_none_vs_inrun_r0_all_runs": {p: {"diag_none": diag["r0"]["evals"]["none"][p]["pass1"],
                                             "inrun": {run: curves[run]["0"][p]["pass1"] for run in RUNS}}
                                          for p in ("probe_J_H", "probe_J_R", "probe_J_U")}}
dump("train_diag.json", diag)

# ================================================================ 6. loss and recovery on H_train prompts
lr_out = {}
for run in RUNS:
    for scope in ("all_hard", "shared_only"):
        seqs = collections.defaultdict(list)
        for k in range(128):
            for g in train[run][k]:
                if g["role"] == "hard_common" or (scope == "all_hard" and g["role"] == "hard_extra"):
                    seqs[g["pid"]].append((k + 1, sum(g["rew"])))
        opp, eps, rec, exp_pool, exp_local, lags, never = 0, [], 0, 0.0, 0.0, [], 0
        for pid, sq in seqs.items():
            cs = [c for _, c in sq]
            pooled_tot = sum(cs)
            for i in range(len(sq) - 1):
                if cs[i] < 1:
                    continue
                opp += 1
                # sampling-only expectation for presentation i+1
                p_pool = (pooled_tot - cs[i + 1]) / (8 * (len(cs) - 1))
                blk = (sq[i + 1][0] - 1) // 32
                others = [c for j, (r, c) in enumerate(sq) if j != i + 1 and (r - 1) // 32 == blk]
                p_loc = sum(others) / (8 * len(others)) if others else p_pool
                exp_pool += (1 - p_pool) ** 8
                exp_local += (1 - p_loc) ** 8
                if cs[i + 1] == 0:
                    later = [j for j in range(i + 2, len(sq)) if cs[j] >= 1]
                    e = {"pid": pid, "solved_round": sq[i][0], "c_solved": cs[i], "lost_round": sq[i + 1][0],
                         "recovered_round": sq[later[0]][0] if later else None,
                         "presentations_to_recover": (later[0] - (i + 1)) if later else None,
                         "p_local": p_loc, "cap_change_between": sq[i][0] <= 32 < sq[i + 1][0]}
                    eps.append(e)
                    if later:
                        rec += 1
                        lags.append(sq[later[0]][0] - sq[i + 1][0])
                    else:
                        never += 1
        lr_out[f"{run}|{scope}"] = {"problems": len(seqs), "presentations": sum(len(v) for v in seqs.values()),
                                    "opportunities": opp, "loss_episodes": len(eps), "loss_rate": len(eps) / opp if opp else None,
                                    "recovered": rec, "not_recovered_by_128": never,
                                    "median_rounds_to_recover": float(np.median(lags)) if lags else None,
                                    "expected_losses_sampling_pooled": exp_pool, "expected_losses_sampling_local": exp_local,
                                    "episodes": eps,
                                    "problems_never_solved": sum(1 for v in seqs.values() if all(c == 0 for _, c in v)),
                                    "problems_all_correct_last_presentation": sum(1 for v in seqs.values() if v[-1][1] == 8)}
dump("train_loss_recovery.json", lr_out)

# ================================================================ 8. traced cases
tr_out = {}
# (a) resumed run: R-s101 round 125 (records index 124) and H-s202 round 64 evaluation
rows, segs = read_metrics(f"{rundir('R-s101')}/metrics.jsonl")
rr = [r for r in rows if r["kind"] == "round" and r["round"] == 125]
tr_out["a_resume_R_s101_round125"] = {
    "round_rows": [{"line": r["_line"], "job": segs[r["_seg"]]["job"], "groups_all0": r["groups_all0"], "groups_mixed": r["groups_mixed"],
                    "groups_all1": r["groups_all1"], "truncated": r["truncated"], "success": r["success"], "max_tokens": r["max_tokens"],
                    "canonical": canon["R-s101"][("round", 125)]["_line"] == r["_line"]} for r in rr],
    "records_main_dir": {"success_by_role": {ro: float(np.mean([x for g in train["R-s101"][124] if g["role"] == ro for x in g["rew"]])) for ro in ("hard_common", "aux")},
                         "classes": dict(collections.Counter(gclass(g["rew"]) for g in train["R-s101"][124])),
                         "trunc": sum(f == "length" for g in train["R-s101"][124] for f in g["fin"])},
    "records_superseded_11081724": {"success_by_role": {ro: float(np.mean([x for g in superseded_train["R-s101"][("11081724", 124)] if g["role"] == ro for x in g["rew"]])) for ro in ("hard_common", "aux")},
                                    "classes": dict(collections.Counter(gclass(g["rew"]) for g in superseded_train["R-s101"][("11081724", 124)])),
                                    "trunc": sum(f == "length" for g in superseded_train["R-s101"][("11081724", 124)] for f in g["fin"])},
    "same_prompt_ids_and_seeds": [(g["pid"], g["seed"]) for g in train["R-s101"][124]] == [(g["pid"], g["seed"]) for g in superseded_train["R-s101"][("11081724", 124)]],
    "step_rows_round_index_124": [{"line": r["_line"], "job": segs[r["_seg"]]["job"], "minibatch": r["minibatch"], "opt_step": r["opt_step"],
                                   "grad_norm_preclip": r["grad_norm_preclip"], "valid_tokens": r["valid_tokens"],
                                   "canonical": canon["R-s101"][("step", 124, r["minibatch"])]["_line"] == r["_line"]}
                                  for r in rows if r["kind"] == "step" and r["round"] == 124]}


def find_group(run, cls, role=None):
    for k in range(128):
        for g in train[run][k]:
            if gclass(g["rew"]) == cls and (role is None or g["role"] == role):
                return k, g


def group_step(run, k, slot):
    seed = int(run[-3:])
    plan = mb_plan([g["role"] for g in train[run][k]], nrng(seed, f"minibatch/r{k}"))
    m = next(i for i, mb in enumerate(plan) if slot in mb)
    row = canon[run][("step", k, m)]
    return m, plan[m], {x: row[x] for x in ("opt_step", "round", "minibatch", "grad_norm_preclip", "loss", "valid_tokens", "lr", "grad_clipped")}


k, g = find_group("R-s101", "all_correct", "aux")
m, mb, srow = group_step("R-s101", k, g["slot"])
tr_out["b_all_correct_group"] = {"run": "R-s101", "round_1based": k + 1, "slot": g["slot"], "role": g["role"], "pid": g["pid"],
                                 "rewards": g["rew"], "adv": g["adv"], "ntok": g["ntok"], "minibatch": m, "minibatch_groups": mb,
                                 "minibatch_classes": [gclass(train["R-s101"][k][gi]["rew"]) for gi in mb], "step_row": srow,
                                 "loss_contribution": "A=0 for all 8 attempts -> per-token objective min(q*0, clip(q)*0)=0 -> zero gradient; still counted in the 1/32 divisor"}
# (c) earliest momentum-only step over all runs
cand = [(r["round_1based"], run, r) for run in RUNS for r in steps_out[run]["momentum_only_steps"]]
cand.sort(key=lambda x: x[0])
r0, run0, mrec = cand[0]
tr_out["c_momentum_only_step"] = {"run": run0, "step": mrec,
                                  "groups": [{"slot": gi, "role": train[run0][r0 - 1][gi]["role"], "pid": train[run0][r0 - 1][gi]["pid"],
                                              "rewards": train[run0][r0 - 1][gi]["rew"], "adv": train[run0][r0 - 1][gi]["adv"]}
                                             for gi in mrec["groups"]]}
k, g = find_group("R-s101", "mixed", "hard_common")
m, mb, srow = group_step("R-s101", k, g["slot"])
c = sum(g["rew"])
tr_out["d_mixed_group"] = {"run": "R-s101", "round_1based": k + 1, "slot": g["slot"], "role": g["role"], "pid": g["pid"],
                           "rewards": g["rew"], "adv": g["adv"], "formula_pos": math.sqrt((8 - c) / c), "formula_neg": -math.sqrt(c / (8 - c)),
                           "adv_formula_exact": adv_formula(g["rew"]), "ntok": g["ntok"], "finish": g["fin"], "minibatch": m, "step_row": srow}
dump("train_traces.json", tr_out)

# ---------------------------------------------------------------- superseded training time (for cost section)
sup_time = {}
for run in RUNS:
    rows, segs = read_metrics(f"{rundir(run)}/metrics.jsonl")
    t_sup = sum(r["t_round"] for r in rows if r["kind"] == "round" and canon[run][("round", r["round"])]["_line"] != r["_line"])
    e_sup = sum(r["seconds"] for r in rows if r["kind"] == "eval" and "seconds" in r and canon[run][("eval", r["panel"], r["round"])]["_line"] != r["_line"])
    t_all = sum(r["t_round"] for r in rows if r["kind"] == "round" and canon[run][("round", r["round"])]["_line"] == r["_line"])
    per_job = {}
    for r in rows:
        if r["kind"] == "round" and canon[run][("round", r["round"])]["_line"] != r["_line"]:
            pj = per_job.setdefault(segs[r["_seg"]]["job"], {"round_s": 0.0, "eval_s": 0.0, "rounds": 0})
            pj["round_s"] += r["t_round"]; pj["rounds"] += 1
        if r["kind"] == "eval" and "seconds" in r and canon[run][("eval", r["panel"], r["round"])]["_line"] != r["_line"]:
            pj = per_job.setdefault(segs[r["_seg"]]["job"], {"round_s": 0.0, "eval_s": 0.0, "rounds": 0})
            pj["eval_s"] += r["seconds"]
    sup_time[run] = {"superseded_round_seconds": t_sup, "superseded_eval_seconds": e_sup, "canonical_round_seconds": t_all,
                     "superseded_round_rows": sum(1 for r in rows if r["kind"] == "round" and canon[run][("round", r["round"])]["_line"] != r["_line"]),
                     "per_job": per_job}
dump("train_superseded_time.json", sup_time)
print("done", flush=True)
