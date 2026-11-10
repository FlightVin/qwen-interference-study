"""EXPLORATORY (A15.2, rules frozen in D122): interference trace on the hard prompts shared by R and U.

Per seed, R and U see the same `hard_common` prompt at the same round. From training records (run in a CPU job):
- per 16-round window (1-16, 17-32 | cap 8,192 -> 12,288 | 33-48, ..., 113-128) and arm (H uses its hard_common role too):
  hard success; share of 8-answer groups all-wrong / mixed / all-correct; all-wrong groups split into all-finished vs
  >= 1 capped; capped share of responses;
- mastery vs signal loss: for each problem, consecutive appearances; a transition out of "mixed" is classed by its
  destination (all-correct = mastery, all-wrong = signal loss), split by window of the later appearance;
- previously solvable: a shared hard problem with >= 1 correct answer in an earlier appearance in the same run; share of
  its later appearances that are all-wrong (finished / capped), per window;
- paired R vs U per seed on identical (problem, round) cells: counts where one arm is all-wrong and the other is not;
- dev panel (scheduled evaluations, 8 samples per problem): per pair of successive evaluations, the number of problems
  with c > 0 then c = 0 ("lost") and c = 0 then c > 0 ("gained"), per run;
- complete flag (RUN_COMPLETE) per run; H vs R schedule check per seed (A16): share of common rounds where the
  hard_common prompt sets are identical, so H can be compared with R/U on the same shared prompts.
Usage: python -m qis.analysis.interference RUNS_DIR OUT.json
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

N = 8
WINDOWS = [(a, a + 15) for a in range(1, 129, 16)]


def window(rnd):
    return next(f"{a}-{b}" for a, b in WINDOWS if a <= rnd <= b)


def groups(run_dir):
    """{(prompt_id, round): {"c": correct, "capped": n_capped, "n": 8}} for role hard_common (1-based rounds)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "records", "train_r*.jsonl.gz"))):
        rnd = int(re.search(r"train_r(\d+)", p).group(1)) + 1
        try:
            recs = [json.loads(l) for l in gzip.open(p, "rt")]
        except (EOFError, OSError):
            continue
        by = collections.defaultdict(list)
        for r in recs:
            if r["role"] == "hard_common":
                by[(r["prompt_id"], r["slot"])].append(r)
        for (pid, _), g in by.items():
            out[(pid, rnd)] = {"c": sum(x["reward"] == 1 for x in g), "capped": sum(x["finish"] == "length" for x in g),
                               "n": len(g)}
    return out


def kind(g):
    return "all_wrong" if g["c"] == 0 else ("all_correct" if g["c"] == g["n"] else "mixed")


def per_window(G):
    W = collections.defaultdict(lambda: collections.Counter())
    for (pid, rnd), g in G.items():
        w = W[window(rnd)]
        w["groups"] += 1; w["responses"] += g["n"]; w["correct"] += g["c"]; w["capped"] += g["capped"]
        k = kind(g); w[k] += 1
        if k == "all_wrong":
            w["all_wrong_finished" if g["capped"] == 0 else "all_wrong_capped"] += 1
    res = {}
    for wk, c in W.items():
        n = c["groups"]
        res[wk] = {"groups": n, "success": c["correct"] / c["responses"], "capped_share": c["capped"] / c["responses"],
                   **{k: c[k] / n for k in ("all_wrong", "mixed", "all_correct", "all_wrong_finished", "all_wrong_capped")},
                   "counts": dict(c)}
    return dict(sorted(res.items(), key=lambda kv: int(kv[0].split("-")[0])))


def transitions(G):
    """Consecutive appearances of each problem: where do groups that were mixed go, and do solved problems fail later?"""
    seq = collections.defaultdict(list)
    for (pid, rnd), g in G.items():
        seq[pid].append((rnd, g))
    T = collections.defaultdict(lambda: collections.Counter())
    P = collections.defaultdict(lambda: collections.Counter())
    for pid, s in seq.items():
        s.sort(key=lambda x: x[0])
        solved = False
        for i, (rnd, g) in enumerate(s):
            w = window(rnd)
            if i:
                prev = kind(s[i - 1][1])
                if prev == "mixed":
                    T[w]["from_mixed"] += 1; T[w]["to_" + kind(g)] += 1
            if solved:
                P[w]["later_appearances"] += 1
                if g["c"] == 0:
                    P[w]["all_wrong"] += 1
                    P[w]["all_wrong_finished" if g["capped"] == 0 else "all_wrong_capped"] += 1
            solved = solved or g["c"] > 0
    order = lambda d: dict(sorted(((k, dict(v)) for k, v in d.items()), key=lambda kv: int(kv[0].split("-")[0])))  # noqa: E731
    return order(T), order(P)


def paired(GR, GU):
    C = collections.defaultdict(lambda: collections.Counter())
    for key in set(GR) & set(GU):
        w = C[window(key[1])]
        a, b = kind(GR[key]) == "all_wrong", kind(GU[key]) == "all_wrong"
        w["cells"] += 1; w["R_only_all_wrong"] += a and not b; w["U_only_all_wrong"] += b and not a; w["both"] += a and b
    return {k: dict(v) for k, v in sorted(C.items(), key=lambda kv: int(kv[0].split("-")[0]))}


def dev_changes(run_dir):
    ev = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "records", "eval_dev_r*.jsonl.gz"))):
        rnd = int(re.search(r"eval_dev_r(\d+)", p).group(1))
        c = collections.Counter()
        n = collections.Counter()
        for l in gzip.open(p, "rt"):
            r = json.loads(l); c[r["prompt_id"]] += r["reward"] == 1; n[r["prompt_id"]] += 1
        ev[rnd] = (c, n)
    rs = sorted(ev)
    out = []
    for a, b in zip(rs, rs[1:]):
        (ca, na), (cb, nb) = ev[a], ev[b]
        ids = set(na) & set(nb)
        out.append({"from": a, "to": b, "problems": len(ids), "lost": sum(ca[i] > 0 and cb[i] == 0 for i in ids),
                    "gained": sum(ca[i] == 0 and cb[i] > 0 for i in ids),
                    "zero_at_from": sum(ca[i] == 0 for i in ids), "zero_at_to": sum(cb[i] == 0 for i in ids)})
    return out


def per_round(G):
    """{round: [correct, responses]} on hard_common, for exact round matching with a partial run (A16)."""
    out = collections.defaultdict(lambda: [0, 0])
    for (pid, rnd), g in G.items():
        out[rnd][0] += g["c"]; out[rnd][1] += g["n"]
    return {r: out[r] for r in sorted(out)}


def schedule_match(GA, GB):
    """Rounds present in both runs, and how many of them have identical hard_common prompt sets."""
    ra, rb = collections.defaultdict(set), collections.defaultdict(set)
    for pid, rnd in GA:
        ra[rnd].add(pid)
    for pid, rnd in GB:
        rb[rnd].add(pid)
    common = sorted(set(ra) & set(rb))
    return {"common_rounds": len(common), "identical": sum(ra[r] == rb[r] for r in common),
            "max_common_round": common[-1] if common else 0}


def main(runs, out):
    res = {"windows": [f"{a}-{b}" for a, b in WINDOWS], "cap_change_after_round": 32, "runs": {}, "paired_R_vs_U": {}}
    G = {}
    for d in sorted(glob.glob(os.path.join(runs, "qis-main-*"))):
        run = os.path.basename(d)
        G[run] = groups(d)
        T, P = transitions(G[run])
        res["runs"][run] = {"windows": per_window(G[run]), "from_mixed": T, "previously_solvable": P,
                            "dev_changes": dev_changes(d), "problems": len({k[0] for k in G[run]}),
                            "complete": os.path.exists(os.path.join(d, "RUN_COMPLETE")),
                            "per_round": per_round(G[run])}
    for run in G:
        m = re.match(r"qis-main-R-(s\d+)-(\w+)", run)
        u = run.replace("-R-", "-U-")
        if m and u in G:
            res["paired_R_vs_U"][m.group(1)] = paired(G[run], G[u])
    res["schedule_H_vs_R"] = {}
    for run in G:
        m = re.match(r"qis-main-H-(s\d+)-(\w+)", run)
        r = run.replace("-H-", "-R-")
        if m and r in G:
            res["schedule_H_vs_R"][m.group(1)] = schedule_match(G[run], G[r])
    json.dump(res, open(out, "w"), indent=1)
    for run, v in res["runs"].items():
        print(run, {w: (round(x["success"], 3), round(x["all_wrong"], 3), round(x["mixed"], 3), round(x["all_correct"], 3))
                    for w, x in v["windows"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
