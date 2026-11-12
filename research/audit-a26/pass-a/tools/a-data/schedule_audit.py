"""a-data (A26 pass A): verify the training schedules of the six canonical runs from raw training records.

Runs ON COMPUTE with the project code on PYTHONPATH (schedule.round_plan, grpo.minibatch_plan, seeds) - read-only.
Inputs : $P/runs/qis-main-{H,R,U}-s{101,202}-sym/{config.json,metrics.jsonl,records/train_r0000..0127.jsonl.gz},
         superseded record dirs (records/superseded-*, H-s202 records_a2orig_r60-64), the bundle in config.json.
Output : one JSON (argv[1]) with every check, unrounded counts and per-problem presentation counts.
Usage  : PYTHONPATH=<repo>/src python schedule_audit.py OUT.json
"""
import collections
import gzip
import hashlib
import json
import os
import sys

from qis import seeds
from qis.train import grpo
from qis.train.schedule import round_plan

P = "/data/user_data/vineethb/qis"
RUNS = [f"qis-main-{a}-s{s}-sym" for s in (101, 202) for a in "HRU"]
N_ROUNDS, N_SLOTS, N_SAMPLES, N_MB = 128, 16, 8, 4


def read_records(path):
    """Per-slot summary of one training record file (streamed)."""
    slots = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            s = slots.setdefault(r["slot"], {"prompt_id": r["prompt_id"], "role": r["role"], "seed": r["seed"],
                                             "round": r["round"], "rewards": {}, "adv": {}, "status": collections.Counter(),
                                             "ids_bad": 0, "tokens": 0})
            if (r["prompt_id"], r["role"], r["seed"], r["round"]) != (s["prompt_id"], s["role"], s["seed"], s["round"]):
                s["ids_bad"] += 1
            s["rewards"][r["sample"]] = r["reward"]
            s["adv"][r["sample"]] = r["adv"]
            s["status"][r["status"]] += 1
            s["tokens"] += r["n_tokens"]
    return slots


def adv_expected(rew):
    n = len(rew)
    m = sum(rew) / n
    sd = (sum((x - m) ** 2 for x in rew) / n) ** 0.5
    if all(x == rew[0] for x in rew):
        return [0.0] * n
    return [(x - m) / (sd + 1e-6) for x in rew]


def group_counts(slots):
    c = collections.Counter()
    for s in slots.values():
        k = sum(s["rewards"].values())
        c["all0" if k == 0 else ("all1" if k == N_SAMPLES else "mixed")] += 1
    return c


def main(out_path):
    out = {"runs": {}, "cross": {}}
    per_round = {}           # run -> rnd -> dict
    for run in RUNS:
        rd = f"{P}/runs/{run}"
        cfg = json.load(open(f"{rd}/config.json"))
        arm, seed = cfg["arm"], cfg["seed"]
        bpath = cfg["bundle"]
        bsha = hashlib.sha256(open(bpath, "rb").read()).hexdigest()
        pools = json.load(open(bpath))["pools"]
        pid = {k: [x["id"] for x in v] for k, v in pools.items()}
        pair_of = {}
        for k, i in enumerate(pid["related"]):
            pair_of[i] = k
        for k, i in enumerate(pid["unrelated"]):
            pair_of[i] = k
        # metrics rows in file order
        rounds_rows, step_rows, configs, events = collections.defaultdict(list), collections.defaultdict(list), [], []
        for line in open(f"{rd}/metrics.jsonl"):
            x = json.loads(line)
            if x["kind"] == "round":
                rounds_rows[x["round"]].append(x)
            elif x["kind"] == "step":
                step_rows[(x["round"], x["minibatch"])].append(x)
            elif x["kind"] == "config":
                configs.append({"bundle_sha": x["bundle_sha"], "attempt_id": x.get("attempt_id"),
                                "cap_schedule": x["config"].get("cap_schedule"), "seed": x["config"]["seed"],
                                "arm": x["config"]["arm"], "bundle": x["config"]["bundle"]})
            elif x["kind"] == "event" and x["what"] in ("start", "resume", "complete"):
                events.append({k: x.get(k) for k in ("what", "job", "resume_round", "round", "code_sha", "bundle_sha", "attempt_id", "utc")})
        R = {"arm": arm, "seed": seed, "bundle": bpath, "bundle_sha256_now": bsha, "config_rows": configs, "events": events,
             "checks": collections.Counter(), "fail_examples": [], "round_rows_multi": {}, "zero_grad_steps": [],
             "plan_zero_grad_mismatch": [], "group_totals": collections.Counter(), "presentations": {}}
        pres = {"hard_common": collections.Counter(), "hard_extra": collections.Counter(), "aux": collections.Counter()}

        def fail(kind, **kv):
            R["checks"]["FAIL " + kind] += 1
            if len(R["fail_examples"]) < 40:
                R["fail_examples"].append({"kind": kind, **kv})

        per_round[run] = {}
        for rnd in range(N_ROUNDS):
            path = f"{rd}/records/train_r{rnd:04d}.jsonl.gz"
            slots = read_records(path)
            plan = round_plan(arm, seed, rnd, len(pid["hard"]), len(pid["related"]), 0)
            if sorted(slots) != list(range(N_SLOTS)):
                fail("slot set", rnd=rnd, slots=sorted(slots))
            for j, (role, pool, idx) in enumerate(plan):
                s = slots.get(j)
                if s is None:
                    continue
                R["checks"]["slots"] += 1
                if s["ids_bad"]:
                    fail("inconsistent rows within a slot", rnd=rnd, slot=j)
                if s["round"] != rnd:
                    fail("round field", rnd=rnd, slot=j, got=s["round"])
                if s["role"] != role:
                    fail("role", rnd=rnd, slot=j, want=role, got=s["role"])
                if s["prompt_id"] != pid[pool][idx]:
                    fail("prompt_id vs round_plan", rnd=rnd, slot=j, want=pid[pool][idx], got=s["prompt_id"])
                else:
                    R["checks"]["prompt_id matches round_plan"] += 1
                want_seed = seeds.int_seed(seed, f"gen/r{rnd}/p{j}")
                if s["seed"] != want_seed:
                    fail("request seed", rnd=rnd, slot=j)
                else:
                    R["checks"]["request seed matches"] += 1
                if sorted(s["rewards"]) != list(range(N_SAMPLES)):
                    fail("samples", rnd=rnd, slot=j)
                rew = [s["rewards"][k] for k in range(N_SAMPLES)]
                if any(v not in (0, 1) for v in rew):
                    fail("non-binary reward", rnd=rnd, slot=j, rew=rew)
                ea = adv_expected(rew)
                if max(abs(ea[k] - s["adv"][k]) for k in range(N_SAMPLES)) > 1e-4:
                    fail("adv recompute", rnd=rnd, slot=j)
                else:
                    R["checks"]["adv recompute ok"] += 1
                pres[role][s["prompt_id"]] += 1
            roles = [slots[j]["role"] for j in range(N_SLOTS)]
            ids = [slots[j]["prompt_id"] for j in range(N_SLOTS)]
            shared = ids[:8]
            if len(set(shared)) != 8:
                fail("shared not distinct", rnd=rnd)
            extra = ids[8:] if arm == "H" else []
            if arm == "H":
                if len(set(extra)) != 8:
                    fail("extra not distinct", rnd=rnd)
                if set(extra) & set(shared):
                    fail("extra repeats shared", rnd=rnd, overlap=sorted(set(extra) & set(shared)))
                else:
                    R["checks"]["extra disjoint from shared"] += 1
            pairs = [pair_of[i] for i in ids[8:]] if arm != "H" else []
            if arm != "H" and len(set(pairs)) != 8:
                fail("pairs not distinct", rnd=rnd)
            # group composition vs round rows
            gc = group_counts(slots)
            R["group_totals"].update(gc)
            rows = rounds_rows.get(rnd + 1, [])
            match = [int((r["groups_all0"], r["groups_mixed"], r["groups_all1"]) == (gc["all0"], gc["mixed"], gc["all1"])) for r in rows]
            if not rows:
                fail("no round row", rnd=rnd)
            elif not match[-1]:
                fail("records do not match LAST round row", rnd=rnd, rec=dict(gc), rows=[(r["groups_all0"], r["groups_mixed"], r["groups_all1"]) for r in rows])
            else:
                R["checks"]["records match last round row"] += 1
            if len(rows) > 1:
                R["round_rows_multi"][rnd + 1] = {"n_rows": len(rows), "match_each": match,
                                                   "rows": [(r["groups_all0"], r["groups_mixed"], r["groups_all1"]) for r in rows]}
            # minibatch plan vs step rows
            mixed = [0 < sum(slots[j]["rewards"].values()) < N_SAMPLES for j in range(N_SLOTS)]
            mb = grpo.minibatch_plan(roles, N_MB, seeds.rng(seed, f"minibatch/r{rnd}"))
            for m, groups in enumerate(mb):
                if sorted(roles[g] for g in groups) != sorted([r for r in sorted(set(roles)) for _ in range(2)]):
                    fail("minibatch role balance", rnd=rnd, m=m)
                srows = step_rows.get((rnd, m), [])
                if not srows:
                    fail("no step row", rnd=rnd, m=m)
                    continue
                gn = srows[-1]["grad_norm_preclip"]
                plan_zero = not any(mixed[g] for g in groups)
                if gn == 0.0:
                    R["zero_grad_steps"].append([rnd, m])
                    alladv0 = all(v == 0.0 for g in groups for v in slots[g]["adv"].values())
                    R["checks"]["zero-grad step: all 32 adv == 0" if alladv0 else "FAIL zero-grad step has nonzero adv"] += 1
                if plan_zero != (gn == 0.0):
                    R["plan_zero_grad_mismatch"].append({"rnd": rnd, "m": m, "plan_zero": plan_zero, "grad_norm": gn})
                else:
                    R["checks"]["step zero-grad <-> plan minibatch all non-mixed"] += 1
                tok = sum(slots[g]["tokens"] for g in groups)
                if tok == srows[-1].get("valid_tokens"):
                    R["checks"]["step valid_tokens == planned minibatch token sum"] += 1
                else:
                    fail("valid_tokens mismatch", rnd=rnd, m=m, plan=tok, row=srows[-1].get("valid_tokens"))
                if srows[-1].get("responses") != 32:
                    fail("responses != 32", rnd=rnd, m=m)
            per_round[run][rnd] = {"shared": shared, "extra": extra, "pairs": pairs, "mb": mb, "roles": roles,
                                   "seeds": [slots[j]["seed"] for j in range(N_SLOTS)], "ids": ids}
        R["n_step_rows_total"] = sum(len(v) for v in step_rows.values())
        R["n_round_rows_total"] = sum(len(v) for v in rounds_rows.values())
        R["n_zero_grad_steps"] = len(R["zero_grad_steps"])
        R["first_zero_grad_round_1based"] = (min(r for r, _ in R["zero_grad_steps"]) + 1) if R["zero_grad_steps"] else None
        # superseded segments: which round rows do they match?
        sup = {}
        dirs = [d for d in os.listdir(f"{rd}/records") if d.startswith("superseded")]
        dirs = [f"{rd}/records/{d}" for d in dirs]
        if os.path.isdir(f"{rd}/records_a2orig_r60-64"):
            dirs.append(f"{rd}/records_a2orig_r60-64")
        for d in sorted(dirs):
            files = sorted(f for f in os.listdir(d) if f.startswith("train_r"))
            info = []
            for fn in files:
                rnd = int(fn[7:11])
                gc = group_counts(read_records(f"{d}/{fn}"))
                rows = rounds_rows.get(rnd + 1, [])
                info.append({"round_1based": rnd + 1, "rec": [gc["all0"], gc["mixed"], gc["all1"]],
                             "match_row_index": [k for k, r in enumerate(rows) if (r["groups_all0"], r["groups_mixed"], r["groups_all1"]) == (gc["all0"], gc["mixed"], gc["all1"])],
                             "n_rows": len(rows)})
            sup[os.path.relpath(d, rd)] = {"files": len(files), "rounds": info,
                                           "other_files": sorted(f for f in os.listdir(d) if not f.startswith("train_r"))[:20]}
        R["superseded"] = sup
        # presentations
        exp = {"hard_common": 8 * 128 / 46, "hard_extra": 8 * 128 / 46, "aux": 8 * 128 / 63}
        for role, cnt in pres.items():
            if not cnt:
                continue
            pool = "hard" if role.startswith("hard") else {"R": "related", "U": "unrelated"}[arm]
            vals = [cnt.get(i, 0) for i in pid[pool]]
            R["presentations"][role] = {"expected": exp[role], "n_problems": len(vals), "never": sum(v == 0 for v in vals),
                                        "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "total": sum(vals),
                                        "hist": dict(sorted(collections.Counter(vals).items())),
                                        "per_problem": {i: cnt.get(i, 0) for i in pid[pool]}}
        if arm == "H":
            tot = {i: pres["hard_common"].get(i, 0) + pres["hard_extra"].get(i, 0) for i in pid["hard"]}
            v = list(tot.values())
            R["presentations"]["hard_total"] = {"expected": 16 * 128 / 46, "min": min(v), "max": max(v), "mean": sum(v) / len(v),
                                                "total": sum(v), "hist": dict(sorted(collections.Counter(v).items())), "per_problem": tot}
        R["checks"] = dict(R["checks"]); R["group_totals"] = dict(R["group_totals"])
        out["runs"][run] = R
        print(run, "done", {k: v for k, v in R["checks"].items()}, flush=True)
    # cross-arm checks per seed
    for seed in (101, 202):
        H, Rr, U = (per_round[f"qis-main-{a}-s{seed}-sym"] for a in "HRU")
        c = collections.Counter(); ex = []
        part_same = collections.Counter()
        for rnd in range(N_ROUNDS):
            if H[rnd]["shared"] == Rr[rnd]["shared"] == U[rnd]["shared"]:
                c["shared ids+order identical H=R=U"] += 1
            else:
                c["FAIL shared differ"] += 1; ex.append(rnd)
            if H[rnd]["seeds"][:8] == Rr[rnd]["seeds"][:8] == U[rnd]["seeds"][:8]:
                c["shared request seeds identical"] += 1
            if Rr[rnd]["pairs"] == U[rnd]["pairs"]:
                c["R/U pair indices identical (same slot order)"] += 1
            else:
                c["FAIL R/U pairs differ"] += 1
            if Rr[rnd]["seeds"] == U[rnd]["seeds"] == H[rnd]["seeds"]:
                c["all 16 request seeds identical H=R=U"] += 1

            def shared_partition(x):
                return sorted(tuple(sorted(g for g in grp if g < 8)) for grp in x["mb"])

            def mb_order(x):
                return [tuple(g for g in grp if g < 8) for grp in x["mb"]]
            part_same["R vs U shared-slot partition equal"] += shared_partition(Rr[rnd]) == shared_partition(U[rnd])
            part_same["R vs U full minibatch plan equal"] += Rr[rnd]["mb"] == U[rnd]["mb"]
            part_same["H vs R shared-slot partition equal"] += shared_partition(H[rnd]) == shared_partition(Rr[rnd])
            part_same["H vs R shared groups in same minibatch order"] += mb_order(H[rnd]) == mb_order(Rr[rnd])
        out["cross"][str(seed)] = {"checks": dict(c), "fail_rounds": ex[:20], "minibatch_grouping": dict(part_same),
                                   "example_round0": {a: per_round[f"qis-main-{a}-s{seed}-sym"][0]["mb"] for a in "HRU"},
                                   "example_round0_roles": {a: per_round[f"qis-main-{a}-s{seed}-sym"][0]["roles"] for a in "HRU"}}
    json.dump(out, open(out_path, "w"), indent=1, sort_keys=True)
    print(json.dumps(out["cross"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
