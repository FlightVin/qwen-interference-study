#!/usr/bin/env python3
"""A26 pass A (a-code): check one training run's executed behaviour against the code, from its own records.

Read-only on $P. Needs numpy only (venv-v1); does NOT import project code: the seed streams, schedule and
minibatch plan are re-implemented here from src/qis/seeds.py, train/schedule.py, train/grpo.py (@9d6349a = HEAD).
Checks per canonical round (last `round` row per round in metrics.jsonl):
  schedule  prompt_id/role per slot == round_plan(arm, seed, rnd)            (schedule.py:45-56, run.py:248-249)
  seeds     record seed == int_seed(seed, "gen/r{rnd}/p{slot}")             (run.py:252)
  grading   finish=length -> status truncated, reward 0, n_tokens == cap(rnd); stop -> last id is EOS;
            correct -> reward 1; no unknown; n_tokens == len(ids) <= cap     (verify.py:122-149, run.py:254)
  adv       stored adv == population-std group advantage recomputed in float32 (grpo.py:16-23)
  groups    all0/mixed/all1 from records == round row                       (run.py:338-340)
  minibatch step-row valid_tokens == sum of n_tokens of the predicted plan's 4 groups, and
            grad_norm_preclip == 0 <=> all 4 predicted groups have zero advantage (grpo.py:44-64, run.py:289-316);
            also scored for the alternative plan without sorted(role) order, to show the test discriminates.
  files     train_rNNNN mtime within 900 s of its canonical round row; eval file mtime within 900 s before its
            canonical eval row; eval per_problem counts in the row == counts recomputed from the file.
  segments  start events: job, code_sha, gpu, resume_round; config rows per segment.
Usage: python runs_check.py --run-dir $P/runs/qis-main-R-s101-sym --bundle $P/data/main-v1/bundle_main_v1.json --out X.json
"""
import argparse, gzip, hashlib, json, os, collections
import numpy as np

EOS = {151645, 151643}


# ---- seeds.py re-implementation
def name_key(name):
    d = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))


def rng(master, name):
    return np.random.default_rng(np.random.SeedSequence(entropy=master, spawn_key=name_key(name)))


def int_seed(master, name):
    return int(np.random.SeedSequence(entropy=master, spawn_key=name_key(name)).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


# ---- schedule.py re-implementation (sequential walk; same stream semantics as _draws)
_P = {}


def perm(n, seed, name, c):
    k = (n, seed, name, c)
    if k not in _P:
        _P[k] = [int(i) for i in rng(seed, f"{name}/cycle{c}").permutation(n)]
    return _P[k]


def draws(n, seed, name, k, R, excl=None):
    pos, out = 0, []
    for rnd in range(R):
        ex = set(excl[rnd]) if excl else set()
        got = []
        while len(got) < k:
            i = perm(n, seed, name, pos // n)[pos % n]
            pos += 1
            if i in ex or i in got:
                continue
            got.append(i)
        out.append(got)
    return out


def plans(arm, seed, R, n_hard, n_pairs):
    hc = draws(n_hard, seed, "sched/hard_common", 8, R)
    if arm == "H":
        ex = draws(n_hard, seed, "sched/hard_extra", 8, R, hc)
        return [[("hard_common", "hard", i) for i in hc[r]] + [("hard_extra", "hard", i) for i in ex[r]] for r in range(R)]
    pr = draws(n_pairs, seed, "sched/aux_pairs", 8, R)
    pool = {"R": "related", "U": "unrelated"}[arm]
    return [[("hard_common", "hard", i) for i in hc[r]] + [("aux", pool, p) for p in pr[r]] for r in range(R)]


def minibatch_plan(roles, nmb, g, sort_roles=True):
    by = {}
    for gi, r in enumerate(roles):
        by.setdefault(r, []).append(gi)
    plan = [[] for _ in range(nmb)]
    for r in (sorted(by) if sort_roles else list(by)):
        gs = by[r]
        p = g.permutation(len(gs))
        k = len(gs) // nmb
        for m in range(nmb):
            plan[m] += [gs[i] for i in p[m * k:(m + 1) * k]]
    order = g.permutation(nmb)
    return [sorted(plan[m]) for m in order]


def cap_for(sched, rnd, default):
    if not sched:
        return default
    cap = sched[0][1]
    for s, c in sched:
        if rnd >= s:
            cap = c
    return cap


def adv_f32(rw):
    r = np.asarray(rw, dtype=np.float32)
    if (r == r[0]).all():
        return np.zeros(8, dtype=np.float32)
    m = r.mean(dtype=np.float32)
    s = np.sqrt(((r - m) ** 2).mean(dtype=np.float32), dtype=np.float32)
    return ((r - m) / (s + np.float32(1e-6))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.run_dir
    res = {"run_dir": d, "problems": []}
    pools = json.load(open(a.bundle))["pools"]

    rows, bad = [], 0
    for line in open(os.path.join(d, "metrics.jsonl")):
        try:
            rows.append(json.loads(line))
        except ValueError:
            bad += 1
    res["metrics_rows"], res["metrics_unparsable"] = len(rows), bad
    starts = [r for r in rows if r.get("kind") == "event" and r.get("what") == "start"]
    res["segments"] = [{k: s.get(k) for k in ("utc", "job", "restart", "resume_round", "resume_from", "code_sha", "gpu", "host", "bundle_sha")}
                       for s in starts]
    cfgs = [r["config"] for r in rows if r.get("kind") == "config"]
    keys = ("run_id", "attempt_id", "arm", "seed", "lr", "eps_low", "eps_high", "rounds", "ckpt_every", "retain_rounds",
            "eval_rounds", "eval_panels", "max_tokens", "cap_schedule", "gpu_mem_util", "init_ckpt", "grad_clip", "max_num_seqs")
    res["configs_distinct"] = [json.loads(x) for x in sorted({json.dumps({k: c.get(k) for k in keys}, sort_keys=True) for c in cfgs})]
    res["events"] = collections.Counter(r.get("what") for r in rows if r.get("kind") == "event")
    cfg = cfgs[-1]
    arm, seed = cfg["arm"], cfg["seed"]

    def seg_of(utc):
        j = None
        for i, s in enumerate(starts):
            if s["utc"] <= utc:
                j = i
        return j

    rr, st, ev = {}, {}, {}
    cnt_r, cnt_s, cnt_e = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        if r.get("kind") == "round":
            rr[r["round"]] = r; cnt_r[r["round"]] += 1
        elif r.get("kind") == "step":
            st[(r["round"], r["minibatch"])] = r; cnt_s[(r["round"], r["minibatch"])] += 1
        elif r.get("kind") == "eval" and not r.get("derived_from"):
            ev[(r["panel"], r["round"])] = r; cnt_e[(r["panel"], r["round"])] += 1
    R = cfg["rounds"]
    res["canonical_round_rows"] = len(rr)
    res["canonical_step_rows"] = len(st)
    res["rounds_with_duplicate_rows"] = sorted(k for k, v in cnt_r.items() if v > 1)
    res["round_rows_by_segment"] = collections.Counter(seg_of(rr[k]["utc"]) for k in rr)
    res["step_rows_by_segment"] = collections.Counter(seg_of(st[k]["utc"]) for k in st)
    res["opt_step_labels_distinct"] = len({r.get("opt_step") for r in rows if r.get("kind") == "step"})
    res["opt_step_label_canonical_minmax"] = [min(r["opt_step"] for r in st.values()), max(r["opt_step"] for r in st.values())]

    P = plans(arm, seed, R, len(pools["hard"]), len(pools.get("related", [])))
    tot = collections.Counter()
    zg = {"sorted": [0, 0], "unsorted": [0, 0]}           # [matches, total] for zero-grad prediction
    vt = {"sorted": [0, 0], "unsorted": [0, 0]}           # valid_tokens equality
    mtime_bad, adv_maxdiff = [], 0.0
    for rnd in range(R):
        p = os.path.join(d, "records", f"train_r{rnd:04d}.jsonl.gz")
        if not os.path.exists(p):
            res["problems"].append(f"missing {p}"); continue
        recs = [json.loads(l) for l in gzip.open(p, "rt")]
        tot["files"] += 1
        tot["attempts"] += len(recs)
        if len(recs) != 128:
            res["problems"].append(f"r{rnd}: {len(recs)} records")
            continue
        cap = cap_for(cfg.get("cap_schedule"), rnd, cfg["max_tokens"])
        row = rr.get(rnd + 1)
        if row is None:
            res["problems"].append(f"r{rnd}: no round row"); continue
        if row.get("max_tokens") != cap:
            res["problems"].append(f"r{rnd}: round row cap {row.get('max_tokens')} != {cap}")
        if abs(os.path.getmtime(p) - row["utc"]) > 900:
            mtime_bad.append(rnd)
        plan = P[rnd]
        groups = [[] for _ in range(16)]
        for k, x in enumerate(recs):
            gi, j = k // 8, k % 8
            role, pool, idx = plan[gi]
            exp_id = pools[pool][idx]["id"]
            ok = (x["round"] == rnd and x["slot"] == gi and x["sample"] == j and x["role"] == role and x["prompt_id"] == exp_id)
            tot["schedule_ok"] += ok
            tot["seed_ok"] += x["seed"] == int_seed(seed, f"gen/r{rnd}/p{gi}")
            ids = x["ids"]
            tot["ntok_eq_len"] += x["n_tokens"] == len(ids)
            tot["le_cap"] += x["n_tokens"] <= cap
            if x["finish"] == "length":
                tot["capped"] += 1
                tot["capped_ok"] += (x["n_tokens"] == cap and x["reward"] == 0 and x["status"] == "truncated")
            elif x["finish"] == "stop":
                tot["stop"] += 1
                tot["stop_eos"] += bool(ids) and ids[-1] in EOS
            else:
                tot["finish_other"] += 1
            tot["unknown"] += x["status"] == "unknown"
            tot["status_reward_ok"] += (x["status"] == "correct") == (x["reward"] == 1) and x["reward"] in (0, 1)
            groups[gi].append(x)
        ng = []
        for gi, g in enumerate(groups):
            rw = [x["reward"] for x in g]
            A = adv_f32(rw)
            for x, aa in zip(g, A):
                dlt = abs(float(aa) - x["adv"])
                adv_maxdiff = max(adv_maxdiff, dlt)
                tot["adv_ok"] += dlt <= 1e-5
            ng.append(sum(rw))
        gc = (sum(c == 0 for c in ng), sum(0 < c < 8 for c in ng), sum(c == 8 for c in ng))
        tot["group_counts_ok"] += gc == (row["groups_all0"], row["groups_mixed"], row["groups_all1"])
        roles = [r for r, _, _ in plan]
        ntok = [sum(x["n_tokens"] for x in g) for g in groups]
        mixed = [0 < c < 8 for c in ng]
        for key, srt in (("sorted", True), ("unsorted", False)):
            mb = minibatch_plan(roles, 4, rng(seed, f"minibatch/r{rnd}"), sort_roles=srt)
            for m, gis in enumerate(mb):
                s = st.get((rnd, m))
                if s is None:
                    continue
                pred_zero = not any(mixed[gi] for gi in gis)
                zg[key][0] += pred_zero == (s["grad_norm_preclip"] == 0.0); zg[key][1] += 1
                vt[key][0] += sum(ntok[gi] for gi in gis) == s["valid_tokens"]; vt[key][1] += 1
        tot["zero_grad_steps_observed"] += sum(st[(rnd, m)]["grad_norm_preclip"] == 0.0 for m in range(4) if (rnd, m) in st)
    res["train_records"] = dict(tot)
    res["adv_max_abs_diff"] = adv_maxdiff
    res["train_file_mtime_off_row"] = mtime_bad
    res["minibatch_zero_grad_match"] = zg
    res["minibatch_valid_tokens_match"] = vt

    # ---- in-run evaluation files vs canonical eval rows
    evres = []
    for (panel, rnd), r in sorted(ev.items()):
        p = os.path.join(d, "records", f"eval_{panel}_r{rnd:04d}.jsonl.gz")
        x = {"panel": panel, "round": rnd, "rows": cnt_e[(panel, rnd)], "segment": seg_of(r["utc"])}
        if not os.path.exists(p):
            x["file"] = "missing"; evres.append(x); continue
        c = collections.Counter(); n = collections.Counter(); seeds_ok = 0; m = 0
        items = {it["id"] for it in pools[panel]}
        for line in gzip.open(p, "rt"):
            q = json.loads(line); m += 1
            n[q["prompt_id"]] += 1; c[q["prompt_id"]] += q["reward"] == 1
            seeds_ok += q["seed"] == int_seed(4242, f"eval/{panel}/{q['prompt_id']}/r{rnd}/n{r['n']}")
        per = {pp["id"]: (pp["c"], pp["n"]) for pp in r["per_problem"]}
        x.update(records=m, ids_eq_panel=set(n) == items, n_each=sorted(set(n.values())), seeds_ok=seeds_ok,
                 counts_eq_row=all(per.get(k) == (c[k], n[k]) for k in n) and len(per) == len(n),
                 mtime_minus_row_s=round(os.path.getmtime(p) - r["utc"], 1), max_tokens=r.get("max_tokens"))
        evres.append(x)
    res["evals"] = evres
    # ---- sibling and nested record dirs (superseded segments)
    side = {}
    for base in [os.path.join(d, "records")] + [os.path.join(d, x) for x in os.listdir(d) if x.startswith("records") and x != "records"]:
        for x in sorted(os.listdir(base)):
            q = os.path.join(base, x)
            if os.path.isdir(q):
                side[os.path.relpath(q, d)] = sorted(os.listdir(q))
    res["record_side_dirs"] = side
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("canonical_round_rows", "canonical_step_rows", "train_records", "adv_max_abs_diff",
                                          "minibatch_zero_grad_match", "minibatch_valid_tokens_match", "train_file_mtime_off_row")}, default=str))
    print("problems:", res["problems"][:10])


if __name__ == "__main__":
    main()
