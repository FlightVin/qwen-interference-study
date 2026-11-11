"""EXPLORATORY (A17): GRPO signal, apparent forgetting, dev transitions and local-update tables (read-only; CPU job).

Inputs
- RUNS_ROOT/qis-main-{H,R,U}-s{101,202}-sym/: metrics.jsonl and records/{train_rNNNN,eval_<panel>_rNNNN}.jsonl.gz,
  as written by qis.train.run. Round numbers: metrics `round` rows and eval rows are 1-based / "rounds done";
  `step` rows and train records carry the 0-based round (run.py:312, 338, 358; file name train_r{rnd:04d}).
- H-s202 (A17) was stopped after round 64 and resumed from the round-60 checkpoint in the same directory. Two
  trajectories are built from it:
    H-s202-orig = metrics rows before the A17 `resume` event (rounds to 64);
    H-s202-r60  = rows for rounds <= 60 from before, plus every row after the resume event.
  Inside a trajectory, a `resume` event at round s drops earlier rows of rounds > s, then the last occurrence per
  (kind, round[, panel | minibatch]) wins (the existing convention). The A17 resume event is the first `resume` event
  after the rows copied to metrics.pre_resume_a17.jsonl (when that file is a prefix of metrics.jsonl), else the first
  `resume` event with round 60 logged after a round > 60. The resumed run overwrites records/train_r0060..0063 in place, so a records file is
  assigned to a trajectory by its mtime against the resume event's `utc`. A sibling directory records* (other than
  records/) is read as a backup of the original records. Original records that were overwritten and not backed up are
  listed as missing; nothing is filled in for them. The resumed run may still be running: partial runs are kept and
  flagged, unreadable (partly written) gz files are skipped and listed.
- --diag D1 D2: diag_<label>.jsonl from qis.diag_local (r0 = base model, r64 = R-s101 round 64);
  --diag-ref R1 R2: diag_<label>_ref.jsonl (no-update rows none_rerun, none_cycled).
- --gpu-hours COST_JSON (optional): qis.analysis.cost_table output; categories "main <arm>-s<seed>".

Outputs in OUT_DIR (JSON + a short Markdown table each)
1. signal.json / signal.md: per trajectory, 32-round window (r1-32, r33-64, r65-96, r97-128) and logged role
   (hard_common = shared hard prompts; hard_extra = H's extra hard prompts; aux = R's related/factoring pool or U's
   unrelated/modular pool): all-wrong / mixed / all-correct groups (counts, fractions); flag: second pool's all-correct
   fraction > 0.5; per-round series of mixed groups and cumulative counts, generated tokens, cumulative hard responses;
   compute (cost JSON GPU-h if given; active hours = sum of round-row t_round / 3600); optimiser statistics at the
   logged granularity (grad norm and loss per optimizer step; clip fractions per round).
2. persistence.json / persistence.md: on hard_common, per trajectory: all-wrong groups by capped class (0 / 1-7 / all);
   previously solvable items (>= 1 correct at an earlier appearance in the same trajectory) that become all-wrong, with
   (prompt, round) and the outcome at the prompt's next appearance; denominators; split r1-32 / r33-128 and by window.
   Matched cells (same prompt, same round): R vs U per seed; H-s202-orig and H-s202-r60 vs R-s202 and U-s202.
3. dev_transitions.json / dev_transitions.md: lost (>= 1 correct -> 0) and gained (0 -> >= 1) dev problems between
   consecutive scheduled evaluations (0, 32, 64, 96, 128) from the eval rows' per_problem counts, cross-checked against
   the per-sample eval records; verified attempts n per problem; repeated evaluations of one (panel, round); no-update
   reference pairs from the diag files (same request seeds, so they measure non-reproducibility, not resampling).
4. branches.json / branches.md / delta_j_table.md: per diag checkpoint, branch and minibatch seed: absolute pre and
   post J_H, J_R, J_U, changes, mixed groups; no-update rows; unrelated - related and unrelated - hard; spreads; whether
   a minibatch without reward variation still takes an AdamW step (source check of diag_local.py).
5. curves.json (data for the learning-curve figures). With --figures FIG_DIR: learning_curves.png,
   pope_JH_vs_JR.png, pope_JH_vs_JU.png, delta_j.png, transitions.png, figures.json.

Usage
  python -m qis.analysis.a17_train RUNS_ROOT OUT_DIR --diag D1 D2 --diag-ref R1 R2 [--figures FIG_DIR]
         [--gpu-hours COST_JSON]
  python -m qis.analysis.a17_train plot OUT_DIR FIG_DIR      (figures only, from the JSON outputs; matplotlib venv)
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
import zlib

import numpy as np

N_RESP = 8
WINDOWS = ((1, 32), (33, 64), (65, 96), (97, 128))
PHASES = (("r1-32", 1, 32), ("r33-128", 33, 128))
CAP_LAST_OLD = 32                       # cap 8,192 for rounds 1-32, 12,288 from round 33 (cap_schedule [[0,8192],[32,12288]])
EVAL_ROUNDS = (0, 32, 64, 96, 128)
PANELS = ("dev", "probe_J_H", "probe_J_R", "probe_J_U")
J_PANELS = ("J_H", "J_R", "J_U")
BRANCHES = ("hard", "related", "unrelated")
H202 = "qis-main-H-s202-sym"
A17_ROUND = 60
PRE_RESUME = "metrics.pre_resume_a17.jsonl"
SECOND_ROLE = {"H": "hard_extra", "R": "aux", "U": "aux"}
ROLE_DESC = {"hard_common": "shared hard prompts (same schedule in H, R and U of one seed)",
             "hard_extra": "H: extra hard prompts", "aux_R": "R: related pool (factoring exercises)",
             "aux_U": "U: unrelated pool (modular-arithmetic exercises)"}
OUTCOMES = ("all_wrong", "mixed", "all_correct", "no_later_observation")
U101_R64_NOTE = ("The U-s101 round-64 checkpoint (model and optimizer state) was not retained, so no local-update "
                 "diagnostic from U-s101 exists (REPORT.md section 3).")


# ---------------------------------------------------------------- small helpers
def wname(r):
    return next(f"r{a}-{b}" for a, b in WINDOWS if a <= r <= b) if 1 <= r <= WINDOWS[-1][1] else f"r{r}"


def phase(r):
    return "r1-32" if r <= CAP_LAST_OLD else "r33-128"


def kind(g):
    return "all_wrong" if g["c"] == 0 else ("all_correct" if g["c"] == g["n"] else "mixed")


def cap_class(g):
    return "zero" if g["capped"] == 0 else ("all" if g["capped"] == g["n"] else "some")


def summ(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": len(xs), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "p90": float(np.quantile(a, 0.9))}


def spread(xs):
    a = np.asarray([float(x) for x in xs])
    if not len(a):
        return None
    return {"n": int(len(a)), "values": a.tolist(), "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


def frac(k, n):
    return k / n if n else None


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def dstr(a, b):
    """'a − b = ±d' with both absolute values (direction: first minus second)."""
    if a is None or b is None:
        return "n/a"
    return f"{a:.4f} − {b:.4f} = {a - b:+.4f}"


def read_jsonl(path):
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def parse_rec(line):
    """Record line without the (large, last) token-id list; run.py:358-361 and 135-137 write "ids" last."""
    i = line.rfind(', "ids": [')
    if i > 0:
        try:
            return json.loads(line[:i] + "}")
        except json.JSONDecodeError:
            pass
    return json.loads(line)


def short(run):
    p = run.split("-")                                          # qis-main-R-s101-sym
    return f"{p[2]}-{p[3]}", p[2], int(p[3][1:])


# ---------------------------------------------------------------- metrics trajectories
def pos(r):
    """Rounds done that a row belongs to (step rows carry the 0-based round, run.py:312)."""
    k = r.get("kind")
    if k in ("round", "eval"):
        return r.get("round")
    if k == "step":
        return r.get("round", -1) + 1
    return None


def is_resume(r):
    return r.get("kind") == "event" and r.get("what") == "resume"


def canonical(rows):
    kept = []
    for r in rows:
        if is_resume(r):
            s = r["round"]
            kept = [x for x in kept if pos(x) is None or pos(x) <= s]
        kept.append(r)
    out = {"round": {}, "step": {}, "eval": {}, "events": [], "resumes": []}
    for r in kept:
        k = r.get("kind")
        if k == "round":
            out["round"][r["round"]] = r
        elif k == "step":
            out["step"][(r["round"] + 1, r["minibatch"])] = r
        elif k == "eval":
            out["eval"][(r["panel"], r["round"])] = r
        elif k == "event":
            out["events"].append(r)
            if is_resume(r):
                out["resumes"].append({"round": r["round"], "utc": r.get("utc")})
    return out


def a17_split(d, rows):
    """Index of the A17 resume event in metrics.jsonl, or None; plus how it was found."""
    info = {"pre_resume_file": None}
    pre = os.path.join(d, PRE_RESUME)
    if os.path.exists(pre):
        P, bad = read_jsonl(pre)
        prefix = len(P) <= len(rows) and all(a == b for a, b in zip(P, rows))
        info["pre_resume_file"] = {"rows": len(P), "unparsable_lines": bad, "is_prefix_of_metrics": prefix}
        if prefix:
            idx = next((i for i in range(len(P), len(rows)) if is_resume(rows[i])), None)
            if idx is not None:
                info["method"] = "first resume event after the rows of metrics.pre_resume_a17.jsonl"
                return idx, info
    reached, idx = 0, None
    for i, r in enumerate(rows):
        if r.get("kind") == "round":
            reached = max(reached, r["round"])
        if is_resume(r) and r.get("round") == A17_ROUND and reached > A17_ROUND:
            idx = i
            break
    info["method"] = (f"first resume event with round {A17_ROUND} after a round > {A17_ROUND} was logged"
                      if idx is not None else "no A17 resume event found")
    return idx, info


def rec_index(dirs):
    """{("train", round1) | (panel, round): path} over record directories, later dirs not overriding earlier."""
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.jsonl.gz")):
            b = os.path.basename(p)
            m = re.match(r"train_r(\d+)\.jsonl\.gz$", b)
            if m:
                out.setdefault(("train", int(m.group(1)) + 1), p)
                continue
            m = re.match(r"eval_(.+)_r(\d+)\.jsonl\.gz$", b)
            if m:
                out.setdefault((m.group(1), int(m.group(2))), p)
    return out


def trajectories(root):
    T = []
    for d in sorted(glob.glob(os.path.join(root, "qis-main-*"))):
        run = os.path.basename(d)
        mp = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(mp):
            continue
        name, arm, seed = short(run)
        rows, bad = read_jsonl(mp)
        complete = os.path.exists(os.path.join(d, "RUN_COMPLETE"))
        recs = rec_index([os.path.join(d, "records")])
        base = {"run": run, "arm": arm, "seed": seed, "dir": d, "unparsable_metric_lines": bad}
        if run != H202:
            T.append({**base, "name": name, "M": canonical(rows), "files": recs, "complete": complete, "notes": []})
            continue
        idx, info = a17_split(d, rows)
        backups = sorted(p for p in glob.glob(os.path.join(d, "records*")) if os.path.isdir(p) and os.path.basename(p) != "records")
        info["backup_record_dirs"] = backups
        if idx is None:
            T.append({**base, "name": name + "-orig", "M": canonical(rows), "files": recs, "complete": complete,
                      "notes": ["no A17 resume event in metrics.jsonl: one trajectory only"], "a17": info})
            continue
        s, utc = rows[idx]["round"], rows[idx].get("utc")
        info.update({"resume_row_index": idx, "resume_round": s, "resume_utc": utc})
        orig_f, r60_f, rewritten_shared = {}, {}, []
        for k, p in recs.items():
            r = k[1]
            new = os.path.getmtime(p) >= utc
            if new:
                r60_f[k] = p
                if r <= s:
                    rewritten_shared.append(list(k))
            else:
                orig_f[k] = p
                if r <= s:
                    r60_f[k] = p
        for k, p in rec_index(backups).items():
            orig_f.setdefault(k, p)
        Mo = canonical(rows[:idx])
        want = [("train", r) for r in sorted(Mo["round"])] + [k for k in Mo["eval"] if k[0] in PANELS and k[0] != "probe_J_H"]
        missing = sorted([list(k) for k in want if k not in orig_f], key=lambda x: (x[0], x[1]))
        notes_o = [f"rows before the A17 resume event (row {idx}); stopped after round {max(Mo['round'], default=0)}"]
        if missing:
            notes_o.append("original records overwritten by the resumed run and not backed up: " + json.dumps(missing))
        Mr = canonical(rows)
        notes_r = [f"rows to round {s} from before the resume plus rows after it"]
        if ("dev", 64) not in Mr["eval"] and max(Mr["round"], default=0) >= 64:
            notes_r.append("no round-64 evaluation after the resume: run.py:218-220 skips an evaluation whose "
                           "records/eval_<panel>_r0064.jsonl.gz already exists (written by the original attempt)")
        if rewritten_shared:
            notes_r.append(f"records of rounds <= {s} rewritten after the resume: {rewritten_shared}")
        T.append({**base, "name": name + "-orig", "M": Mo, "files": orig_f, "complete": False, "notes": notes_o,
                  "a17": info, "missing_records": missing})
        T.append({**base, "name": name + "-r60", "M": Mr, "files": r60_f, "complete": complete, "notes": notes_r,
                  "a17": info})
    return T


# ---------------------------------------------------------------- records
_TRAIN = {}


def train_groups(path, r1):
    """[{round, slot, prompt_id, role, c, n, capped, tokens}] for one train records file (None if unreadable)."""
    if path in _TRAIN:
        return _TRAIN[path]
    by, bad_round = {}, 0
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                r = parse_rec(line)
                bad_round += r["round"] != r1 - 1
                g = by.setdefault(r["slot"], {"round": r1, "slot": r["slot"], "prompt_id": r["prompt_id"], "role": r["role"],
                                              "c": 0, "n": 0, "capped": 0, "tokens": 0})
                g["c"] += r["reward"] == 1
                g["n"] += 1
                g["capped"] += r["finish"] == "length"
                g["tokens"] += r["n_tokens"]
    except (EOFError, OSError, ValueError, zlib.error):
        _TRAIN[path] = None
        return None
    out = [by[k] for k in sorted(by)]
    if bad_round:
        for g in out:
            g["record_round_mismatch"] = True
    _TRAIN[path] = out
    return out


def eval_counts(path):
    """{prompt_id: (correct, samples)} from one eval records file (run.py:135-141), None if unreadable."""
    c, n = collections.Counter(), collections.Counter()
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                if line.strip():
                    r = parse_rec(line)
                    c[r["prompt_id"]] += r["reward"] == 1
                    n[r["prompt_id"]] += 1
    except (EOFError, OSError, ValueError, zlib.error):
        return None
    return {k: (c[k], n[k]) for k in n}


def load_groups(t):
    G, unreadable = [], []
    for k in sorted(k for k in t["files"] if k[0] == "train"):
        gs = train_groups(t["files"][k], k[1])
        if gs is None:
            unreadable.append(k[1])
        else:
            G += gs
    t["groups"], t["unreadable_train_rounds"] = G, unreadable
    t["record_rounds"] = sorted({g["round"] for g in G})


# ---------------------------------------------------------------- 1. signal
def counts(gs):
    c = collections.Counter(kind(g) for g in gs)
    n = len(gs)
    return {"groups": n, **{k: c[k] for k in ("all_wrong", "mixed", "all_correct")},
            **{f"frac_{k}": frac(c[k], n) for k in ("all_wrong", "mixed", "all_correct")}}


def signal(t, cost):
    G, M = t["groups"], t["M"]
    roles = sorted({g["role"] for g in G})
    second = SECOND_ROLE.get(t["arm"], "aux")
    by_round = collections.defaultdict(list)
    for g in G:
        by_round[g["round"]].append(g)
    rr = M["round"]
    # per-round series
    series, cum_mixed, cum_tok, cum_hard, hard_src = [], collections.Counter(), 0, 0, {}
    cum_known = True                    # once a round has neither records nor counter rows, the cumulative is unknown
    for r in sorted(set(by_round) | set(rr)):
        gs = by_round.get(r, [])
        row = {"round": r, "records": bool(gs)}
        if gs:
            mixed = collections.Counter(g["role"] for g in gs if kind(g) == "mixed")
            for role in roles:
                cum_mixed[role] += mixed[role]
            tok = sum(g["tokens"] for g in gs)
            hard = sum(g["n"] for g in gs if g["role"].startswith("hard"))
            cum_tok += tok
            hard_src[r] = "records"
            row.update({"mixed": {k: mixed[k] for k in roles}, "mixed_total": sum(mixed.values()),
                        "gen_tokens": tok})
        else:
            hard = None
            if r in rr and r - 1 in rr and "hard_responses" in rr[r]:
                hard = rr[r]["hard_responses"] - rr[r - 1]["hard_responses"]
                hard_src[r] = "metrics hard_responses counter difference (records missing)"
        if hard is None:
            cum_known = False
        else:
            cum_hard += hard
        row.update({"cum_mixed": dict(cum_mixed), "cum_mixed_total": sum(cum_mixed.values()),
                    "cum_gen_tokens_from_records": cum_tok, "cum_hard_responses": cum_hard if cum_known else None})
        if r in rr:
            row["metrics"] = {k: rr[r].get(k) for k in ("groups_all0", "groups_mixed", "groups_all1", "max_tokens", "gen_tokens")}
        series.append(row)
    # consistency of records with round rows (run.py:339-340)
    mismatch = []
    for r, gs in by_round.items():
        if r in rr:
            c = counts(gs)
            m = (rr[r].get("groups_all0"), rr[r].get("groups_mixed"), rr[r].get("groups_all1"))
            if (c["all_wrong"], c["mixed"], c["all_correct"]) != m:
                mismatch.append({"round": r, "records": [c["all_wrong"], c["mixed"], c["all_correct"]], "metrics": list(m)})
    # optimiser statistics at the logged granularity
    steps = M["step"]
    windows = {}
    cum_at_end = 0
    for a, b in WINDOWS:
        w = f"r{a}-{b}"
        gs = [g for g in G if a <= g["round"] <= b]
        rounds_rec = sorted({g["round"] for g in gs})
        tok = {role: sum(g["tokens"] for g in gs if g["role"] == role) for role in roles}
        cum_at_end += sum(tok.values())
        sec = counts([g for g in gs if g["role"] == second])
        st = [s for (r, _), s in steps.items() if a <= r <= b]
        rws = [rr[r] for r in rr if a <= r <= b]
        windows[w] = {
            "rounds_with_records": len(rounds_rec), "complete_window": len(rounds_rec) == b - a + 1,
            "rounds_logged": len(rws),
            "by_role": {role: counts([g for g in gs if g["role"] == role]) for role in roles},
            "all_roles": counts(gs),
            "second_pool_role": second,
            "second_pool_all_correct_frac_gt_0.5": (sec["frac_all_correct"] > 0.5) if sec["groups"] else None,
            "gen_tokens": {**tok, "all": sum(tok.values())},
            "cum_gen_tokens_at_window_end": cum_at_end,
            "optimiser": {
                "grad_norm_preclip_per_step": summ([s.get("grad_norm_preclip") for s in st]),
                "grad_clipped_steps": sum(bool(s.get("grad_clipped")) for s in st), "steps": len(st),
                "loss_per_step": summ([s.get("loss") for s in st]),
                "clipfrac_pos_per_round": summ([r.get("clipfrac_pos") for r in rws]),
                "clipfrac_neg_per_round": summ([r.get("clipfrac_neg") for r in rws]),
            }}
    per_round_opt = []
    for r in sorted(rr):
        st = [steps[k] for k in sorted(steps) if k[0] == r]
        per_round_opt.append({"round": r, "grad_norm_preclip": [s.get("grad_norm_preclip") for s in st],
                              "grad_clipped": [bool(s.get("grad_clipped")) for s in st],
                              "loss": [s.get("loss") for s in st],
                              "clipfrac_pos": rr[r].get("clipfrac_pos"), "clipfrac_neg": rr[r].get("clipfrac_neg")})
    compute = {"active_seconds_sum_t_round": sum(r.get("t_round", 0) for r in rr.values()),
               "active_hours": sum(r.get("t_round", 0) for r in rr.values()) / 3600,
               "active_hours_note": "sum of round-row t_round (run.py:348): generation, verification, old log probs, "
                                    "training and weight sync of each round on 1 GPU; scheduled evaluations and "
                                    "checkpoint writes are not included"}
    if cost is not None:
        cat = cost.get("categories", {}).get(f"main {t['arm']}-s{t['seed']}")
        compute["cost_json_category"] = f"main {t['arm']}-s{t['seed']}"
        compute["gpu_hours_cost_json"] = None if cat is None else {k: cat.get(k) for k in ("gpu_h", "gpu_h_canonical", "allocations", "states")}
        if t["run"] == H202:
            compute["cost_json_note"] = "one category for the whole run directory (both H-s202 trajectories)"
    compute["gpu_hours_source"] = "cost JSON" if cost is not None else "active_hours (no cost JSON given)"
    return {"run": t["run"], "complete": t["complete"], "notes": t["notes"],
            "roles": {r: (ROLE_DESC.get(f"aux_{t['arm']}", r) if r == "aux" else ROLE_DESC.get(r, r)) for r in roles},
            "rounds_logged": len(rr), "max_round_logged": max(rr, default=0),
            "rounds_with_records": len(t["record_rounds"]), "unreadable_train_rounds": t["unreadable_train_rounds"],
            "missing_records": t.get("missing_records", []),
            "records_vs_round_rows_mismatch": mismatch,
            "cum_hard_responses_source": {"records": sum(v == "records" for v in hard_src.values()),
                                          "metrics_counter": sum(v != "records" for v in hard_src.values())},
            "gen_tokens_from_records": cum_tok,
            "gen_tokens_metrics_counter_last": rr[max(rr)].get("gen_tokens") if rr else None,
            "windows": windows, "per_round": series, "per_round_optimiser": per_round_opt, "compute": compute}


SIGNAL_GRANULARITY = {
    "grad_norm_preclip": "per optimizer step: 4 steps per round, one per minibatch of 4 groups (step rows, run.py:312-316); "
                         "global L2 norm before clipping at 1.0",
    "grad_clipped": "per optimizer step: grad_norm_preclip > grad_clip (run.py:314)",
    "loss": "per optimizer step: sum over the minibatch of -objective/32 (run.py:302-304, 313); loss == policy_loss",
    "clipfrac_pos / clipfrac_neg": "per round (round rows, run.py:344-345): share of response tokens where the clipped "
                                   "objective branch was active (grpo.py:37-38), among tokens with positive / negative advantage",
}


# ---------------------------------------------------------------- 2. persistence
def hc(t):
    return {(g["prompt_id"], g["round"]): g for g in t["groups"] if g["role"] == "hard_common"}


def persistence_run(t):
    G = hc(t)
    seq = collections.defaultdict(list)
    for (pid, r), g in G.items():
        seq[pid].append((r, g))
    events = []
    denom = collections.Counter()
    for pid, s in seq.items():
        s.sort(key=lambda x: x[0])
        solved = False
        for i, (r, g) in enumerate(s):
            if solved:
                denom[phase(r)] += 1
                denom[wname(r)] += 1
                if g["c"] == 0:
                    nxt = s[i + 1] if i + 1 < len(s) else None
                    events.append({"prompt_id": pid, "round": r, "phase": phase(r), "window": wname(r),
                                   "capped": g["capped"], "n": g["n"], "cap_class": cap_class(g),
                                   "next_round": nxt[0] if nxt else None,
                                   "next_outcome": kind(nxt[1]) if nxt else "no_later_observation",
                                   "next_correct": nxt[1]["c"] if nxt else None})
            solved = solved or g["c"] > 0
    events.sort(key=lambda e: (e["round"], e["prompt_id"]))

    def block(keyname, key):
        ev = [e for e in events if e[keyname] == key]
        aw = [g for (pid, r), g in G.items() if (phase(r) if keyname == "phase" else wname(r)) == key and g["c"] == 0]
        allg = [g for (pid, r), g in G.items() if (phase(r) if keyname == "phase" else wname(r)) == key]
        oc = collections.Counter(e["next_outcome"] for e in ev)
        cc = collections.Counter(e["cap_class"] for e in ev)
        later = sum(oc[k] for k in OUTCOMES[:3])
        return {"hard_common_groups": len(allg),
                "all_wrong_groups": len(aw),
                "all_wrong_by_capped": {k: sum(cap_class(g) == k for g in aw) for k in ("zero", "some", "all")},
                "later_appearances_of_previously_solvable": denom[key],
                "previously_solvable_became_all_wrong": len(ev),
                "frac_of_later_appearances": frac(len(ev), denom[key]),
                "ps_all_wrong_by_capped": {k: cc[k] for k in ("zero", "some", "all")},
                "next_outcome": {k: oc[k] for k in OUTCOMES},
                "with_later_observation": later}
    rounds = sorted({r for _, r in G})
    return {"run": t["run"], "complete": t["complete"], "notes": t["notes"],
            "rounds_with_records": len(rounds), "max_round_with_records": rounds[-1] if rounds else 0,
            "problems": len(seq),
            "by_phase": {p: block("phase", p) for p, _, _ in PHASES},
            "by_window": {f"r{a}-{b}": block("window", f"r{a}-{b}") for a, b in WINDOWS},
            "events": events}


def matched(ta, tb, ev):
    GA, GB = hc(ta), hc(tb)
    ra, rb = collections.defaultdict(set), collections.defaultdict(set)
    for pid, r in GA:
        ra[r].add(pid)
    for pid, r in GB:
        rb[r].add(pid)
    common = sorted(set(ra) & set(rb))
    keys = set(GA) & set(GB)
    ea = {(e["prompt_id"], e["round"]) for e in ev[ta["name"]]}
    eb = {(e["prompt_id"], e["round"]) for e in ev[tb["name"]]}
    out = {}
    for p, lo, hi in PHASES:
        ks = [k for k in keys if lo <= k[1] <= hi]
        a = {k for k in ks if GA[k]["c"] == 0}
        b = {k for k in ks if GB[k]["c"] == 0}
        out[p] = {"cells": len(ks), "a_only_all_wrong": len(a - b), "b_only_all_wrong": len(b - a),
                  "both_all_wrong": len(a & b), "neither": len(ks) - len(a | b),
                  "ps_all_wrong_a": sum(k in ea for k in ks), "ps_all_wrong_b": sum(k in eb for k in ks),
                  "ps_all_wrong_both": sum(k in ea and k in eb for k in ks)}
    return {"a": ta["name"], "b": tb["name"], "common_rounds": len(common),
            "rounds_identical_prompt_sets": sum(ra[r] == rb[r] for r in common),
            "max_common_round": common[-1] if common else 0, "by_phase": out,
            "note": "a/b_only_all_wrong: all 8 answers wrong in that arm only; ps_*: cells where that arm's group is a "
                    "previously-solvable item turned all-wrong (solvability judged within each arm's own run)"}


# ---------------------------------------------------------------- 3. dev transitions
def pp_map(per_problem):
    return {x["id"]: (x["c"], x["n"]) for x in per_problem}


def lost_gained(a, b):
    ids = sorted(set(a) & set(b))
    lost = [i for i in ids if a[i][0] > 0 and b[i][0] == 0]
    gained = [i for i in ids if a[i][0] == 0 and b[i][0] > 0]
    return {"problems": len(ids), "lost": len(lost), "gained": len(gained),
            "zero_at_from": sum(a[i][0] == 0 for i in ids), "zero_at_to": sum(b[i][0] == 0 for i in ids),
            "n_per_problem_from": sorted({a[i][1] for i in ids}), "n_per_problem_to": sorted({b[i][1] for i in ids}),
            "lost_ids": lost, "gained_ids": gained}


def dev_run(t):
    ev = sorted((r for (p, r) in t["M"]["eval"] if p == "dev"))
    evals = {}
    for r in ev:
        row = t["M"]["eval"][("dev", r)]
        pm = pp_map(row.get("per_problem", []))
        e = {"round": r, "row_n": row.get("n"), "problems": row.get("problems"), "pass1": row.get("pass1"),
             "n_per_problem": sorted({v[1] for v in pm.values()}), "max_tokens": row.get("max_tokens")}
        f = t["files"].get(("dev", r))
        if f:
            rc = eval_counts(f)
            if rc is None:
                e["records_check"] = "unreadable"
            else:
                e["records_samples_per_problem"] = sorted({v[1] for v in rc.values()})
                e["records_agree_with_row_per_problem"] = rc == pm
        else:
            e["records_check"] = "no records file for this trajectory"
        evals[r] = (e, pm)
    sched = [r for r in ev if r in EVAL_ROUNDS]
    trans = [{"from": a, "to": b, **lost_gained(evals[a][1], evals[b][1])} for a, b in zip(sched, sched[1:])]
    tot = {"lost": sum(x["lost"] for x in trans), "gained": sum(x["gained"] for x in trans)}
    if sched:
        tot["zero_first"] = sum(v[0] == 0 for v in evals[sched[0]][1].values())
        tot["zero_last"] = sum(v[0] == 0 for v in evals[sched[-1]][1].values())
        tot["first_round"], tot["last_round"] = sched[0], sched[-1]
    return {"run": t["run"], "notes": t["notes"], "evaluations": [evals[r][0] for r in ev], "transitions": trans,
            "totals": tot}


def repeated_evals(run, rows):
    out = []
    occ = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("kind") == "eval" and "per_problem" in r:
            occ[(r["panel"], r["round"])].append(i)
    for (panel, rnd), idx in sorted(occ.items()):
        for i, j in zip(idx, idx[1:]):
            res = [rows[k]["round"] for k in range(i, j) if is_resume(rows[k])]
            retrained = any(s < rnd for s in res)
            out.append({"run": run, "panel": panel, "round": rnd, "row_index": [i, j],
                        "utc": [rows[i].get("utc"), rows[j].get("utc")],
                        "pass1": [rows[i].get("pass1"), rows[j].get("pass1")],
                        "resume_events_between": res,
                        "checkpoint": ("retrained from an earlier checkpoint between the two evaluations: not a "
                                       "no-update reference") if retrained else
                                      "no resume to an earlier round between them: same weights presumed",
                        "same_request_seeds": True,
                        **lost_gained(pp_map(rows[i]["per_problem"]), pp_map(rows[j]["per_problem"]))})
    return out


# ---------------------------------------------------------------- 4. branches
def read_diag(paths):
    rows, missing = [], []
    for p in paths or []:
        if not os.path.exists(p):
            missing.append(p)
            continue
        for line in open(p):
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            r["_file"] = p
            rows.append(r)
    return rows, missing


def jv(row, j):
    e = row.get("eval", {}).get(f"probe_{j}")
    return None if e is None else e["pass1"]


def jpp(row, j):
    e = row.get("eval", {}).get(f"probe_{j}")
    return None if e is None else pp_map(e["per_problem"])


def zero_variation_source_check():
    """Text check of diag_local.one_update: is learner.step() called with no guard on mixed groups?"""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    try:
        src = open(os.path.join(here, os.pardir, "diag_local.py")).read()
        m = re.search(r"def one_update\(.*?(?=\ndef )", src, re.S)
        body = m.group(0) if m else ""
        out["one_update_calls_learner_step"] = "learner.step()" in body
        out["guard_on_reward_variation_before_step"] = bool(re.search(r"\bif\b[^\n]*(mixed|rewards|adv)", body))
    except OSError as e:
        out["error"] = str(e)
    try:
        lsrc = open(os.path.join(here, os.pardir, "train", "learner.py")).read()
        out["learner_uses_torch_AdamW"] = "torch.optim.AdamW(" in lsrc
        out["weight_decay_0"] = "weight_decay=0.0" in lsrc
    except OSError as e:
        out["learner_error"] = str(e)
    return out


def branches(diag, refs):
    labels = sorted({r["label"] for r in diag + refs})
    res = {}
    for lab in labels:
        D = [r for r in diag if r["label"] == lab]
        Rf = {r["branch"]: r for r in refs if r["label"] == lab}
        nones = [r for r in D if r["branch"] == "none"]
        pre = nones[-1] if nones else None
        rer = Rf.get("none_rerun")
        src = next(({k: r.get(k) for k in ("source", "round", "fresh_optimizer")} for r in D + list(Rf.values())), {})
        noupd = {}
        for tag, r in (("none_original", pre), ("none_rerun", rer), ("none_cycled", Rf.get("none_cycled"))):
            if r is not None:
                noupd[tag] = {"file": r["_file"], **{j: jv(r, j) for j in J_PANELS},
                              "truncated": {j: r["eval"].get(f"probe_{j}", {}).get("truncated") for j in J_PANELS},
                              "problems": {j: len(r["eval"].get(f"probe_{j}", {}).get("per_problem", [])) for j in J_PANELS},
                              "n_per_problem": {j: sorted({x["n"] for x in r["eval"].get(f"probe_{j}", {}).get("per_problem", [])}) for j in J_PANELS}}
        draws = []
        for r in D:
            if r["branch"] not in BRANCHES:
                continue
            d = {"branch": r["branch"], "seed": r.get("seed"), "mixed_groups": r.get("mixed_groups"),
                 "batch_reward": r.get("batch_reward"), "grad_norm_preclip": r.get("grad_norm_preclip"),
                 "capped": r.get("capped"), "prompt_ids": r.get("prompt_ids"),
                 "post": {j: jv(r, j) for j in J_PANELS},
                 "pre_original": {j: jv(pre, j) if pre else None for j in J_PANELS},
                 "pre_rerun": {j: jv(rer, j) if rer else None for j in J_PANELS}}
            d["change_post_minus_original"] = {j: (d["post"][j] - d["pre_original"][j]) if None not in (d["post"][j], d["pre_original"][j]) else None for j in J_PANELS}
            d["change_post_minus_rerun"] = {j: (d["post"][j] - d["pre_rerun"][j]) if None not in (d["post"][j], d["pre_rerun"][j]) else None for j in J_PANELS}
            draws.append(d)
        draws.sort(key=lambda d: (BRANCHES.index(d["branch"]), d["seed"] or 0))
        by = {b: {d["seed"]: d for d in draws if d["branch"] == b} for b in BRANCHES}
        comps = {}
        for a, b in (("unrelated", "related"), ("unrelated", "hard")):
            seeds = sorted(set(by[a]) & set(by[b]))
            per = [{"seed": s, **{j: {a: by[a][s]["post"][j], b: by[b][s]["post"][j],
                                      f"{a}_minus_{b}": by[a][s]["post"][j] - by[b][s]["post"][j]} for j in J_PANELS}}
                   for s in seeds]
            mean = {j: {f"mean_{a}": float(np.mean([by[a][s]["post"][j] for s in seeds])) if seeds else None,
                        f"mean_{b}": float(np.mean([by[b][s]["post"][j] for s in seeds])) if seeds else None} for j in J_PANELS}
            for j in J_PANELS:
                ma, mb = mean[j][f"mean_{a}"], mean[j][f"mean_{b}"]
                mean[j][f"mean_{a}_minus_mean_{b}"] = None if ma is None else ma - mb
            comps[f"{a}_vs_{b}"] = {"seeds": seeds, "per_seed": per, "over_seeds": mean,
                                    "note": "post values only; both share the same no-update reference, so it cancels"}
        spreads = {b: {j: spread([d["post"][j] for d in by[b].values()]) for j in J_PANELS} for b in BRANCHES if by[b]}
        ref_vals = {j: spread([v[j] for v in noupd.values() if v.get(j) is not None]) for j in J_PANELS}
        zero = [{"branch": d["branch"], "seed": d["seed"], "grad_norm_preclip": d["grad_norm_preclip"], "post": d["post"]}
                for d in draws if d["mixed_groups"] == 0]
        zrows = [r for r in D if r["branch"] in BRANCHES and r.get("mixed_groups") == 0]
        ident = None
        if len(zrows) > 1:
            ident = all(jpp(r, j) == jpp(zrows[0], j) for r in zrows[1:] for j in J_PANELS)
        res[lab] = {"source": src, "no_update": noupd, "draws": draws, "comparisons": comps,
                    "spread_post_within_branch": spreads, "spread_no_update_rows": ref_vals,
                    "mixed_groups_per_draw": {b: {str(s): d["mixed_groups"] for s, d in by[b].items()} for b in BRANCHES},
                    "zero_mixed_draws": zero,
                    "zero_mixed_draws_identical_per_problem_counts": ident}
    return res


ZERO_VARIATION = {
    "statement": "A minibatch whose 4 groups all have equal rewards still takes an AdamW step. Every advantage is 0 "
                 "(grpo.py:22-23), so every gradient is exactly 0 (recorded grad_norm_preclip 0.0), but "
                 "diag_local.one_update calls learner.step() unconditionally (diag_local.py:90-93) and torch AdamW "
                 "(learner.py:50, betas 0.9/0.999, weight_decay 0) moves the weights by lr * m_hat / (sqrt(v_hat) + eps) "
                 "from its stored moments. From a saved optimizer state (r64) this is a non-zero 'momentum-only' step. "
                 "From a fresh optimizer (r0, zero moments) the same step would leave the weights unchanged. The "
                 "training loop does the same (run.py:310-319: learner.step() for every minibatch).",
}


# ---------------------------------------------------------------- markdown
def md_signal(S, cost_given):
    L = ["# GRPO signal (A17, exploratory)", "",
         "Groups of 8 answers per prompt: AW = all wrong, M = mixed, AC = all correct. Only mixed groups carry a GRPO "
         "gradient. Second pool: H = extra hard prompts, R = related (factoring), U = unrelated (modular). "
         "Cap 8,192 in r1-32, 12,288 from round 33.", "",
         "| Run | Window | Rounds (records) | Shared hard AW / M / AC | Second pool AW / M / AC | Second pool AC > 0.5 | Gen. tokens (M) | Cumulative (M) |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in S["runs"].items():
        for w, x in s["windows"].items():
            if not x["rounds_with_records"]:
                continue
            h = x["by_role"].get("hard_common")
            a = x["by_role"].get(x["second_pool_role"])
            fmt = lambda c: f"{c['all_wrong']} / {c['mixed']} / {c['all_correct']} of {c['groups']}" if c and c["groups"] else "n/a"  # noqa: E731
            flag = {True: "yes", False: "no", None: "n/a"}[x["second_pool_all_correct_frac_gt_0.5"]]
            fr = f" ({a['frac_all_correct']:.2f})" if a and a["groups"] else ""
            L.append(f"| {name} | {w} | {x['rounds_with_records']}{'' if x['complete_window'] else ' (partial)'} | {fmt(h)} | "
                     f"{fmt(a)} | {flag}{fr} | {x['gen_tokens']['all'] / 1e6:.1f} | {x['cum_gen_tokens_at_window_end'] / 1e6:.1f} |")
    L += ["", "**Compute.** " + ("GPU-h from the cost JSON (per run directory). " if cost_given else "No cost JSON: ") +
          "Active hours = sum of round times (t_round), 1 GPU.", "",
          "| Run | GPU-h (cost JSON) | of which canonical | Active hours (t_round) | Rounds logged |", "|---|---|---|---|---|"]
    for name, s in S["runs"].items():
        c = s["compute"]
        g = c.get("gpu_hours_cost_json") or {}
        L.append(f"| {name} | {f4(g.get('gpu_h')) if g else 'n/a'} | {f4(g.get('gpu_h_canonical')) if g else 'n/a'} | "
                 f"{c['active_hours']:.2f} | {s['rounds_logged']} |")
    L += ["", "**Optimiser.** Grad norm and loss are logged per optimizer step (4 per round); clip fractions per round.", "",
          "| Run | Window | Steps | Grad norm median (min–max) | Steps clipped | clipfrac_pos mean | clipfrac_neg mean |",
          "|---|---|---|---|---|---|---|"]
    for name, s in S["runs"].items():
        for w, x in s["windows"].items():
            o = x["optimiser"]
            g = o["grad_norm_preclip_per_step"]
            if not o["steps"] or g is None:
                continue
            cp, cn = o["clipfrac_pos_per_round"], o["clipfrac_neg_per_round"]
            L.append(f"| {name} | {w} | {o['steps']} | {g['median']:.3f} ({g['min']:.3f}–{g['max']:.3f}) | "
                     f"{o['grad_clipped_steps']} | {f4(cp and cp['mean'])} | {f4(cn and cn['mean'])} |")
    notes = [(n, s["notes"]) for n, s in S["runs"].items() if s["notes"]]
    if notes:
        L += [""] + [f"- {n}: " + "; ".join(v) for n, v in notes]
    return "\n".join(L) + "\n"


def md_persistence(P):
    L = ["# Apparent forgetting on the shared hard prompts (A17, exploratory)", "",
         "Previously solvable (PS): >= 1 correct answer at an earlier appearance of the prompt in the same trajectory. "
         "Denominator: later appearances of PS prompts. Capped: answers cut at the length cap (0 / 1-7 / all 8). "
         "Next outcome: the group at the prompt's next appearance in the same trajectory.", "",
         "| Run | Rounds | Later PS appearances | PS → all wrong | capped 0 / some / all | next: AW / M / AC / none | All-wrong groups (capped 0 / some / all) |",
         "|---|---|---|---|---|---|---|"]
    for name, x in P["runs"].items():
        for p, b in x["by_phase"].items():
            if not b["hard_common_groups"]:
                continue
            fr = f" ({b['frac_of_later_appearances']:.3f})" if b["frac_of_later_appearances"] is not None else ""
            cc, oc, ac = b["ps_all_wrong_by_capped"], b["next_outcome"], b["all_wrong_by_capped"]
            L.append(f"| {name} | {p} | {b['later_appearances_of_previously_solvable']} | {b['previously_solvable_became_all_wrong']}{fr} | "
                     f"{cc['zero']} / {cc['some']} / {cc['all']} | {oc['all_wrong']} / {oc['mixed']} / {oc['all_correct']} / "
                     f"{oc['no_later_observation']} | {b['all_wrong_groups']} of {b['hard_common_groups']} ({ac['zero']} / {ac['some']} / {ac['all']}) |")
    L += ["", "**Matched cells** (same prompt, same round; a = first run, b = second run):", "",
          "| a vs b | Rounds | Cells | AW in a only | AW in b only | AW in both | PS→AW a / b / both |", "|---|---|---|---|---|---|---|"]
    for m in P["matched"]:
        for p, x in m["by_phase"].items():
            if x["cells"]:
                L.append(f"| {m['a']} vs {m['b']} | {p} | {x['cells']} | {x['a_only_all_wrong']} | {x['b_only_all_wrong']} | "
                         f"{x['both_all_wrong']} | {x['ps_all_wrong_a']} / {x['ps_all_wrong_b']} / {x['ps_all_wrong_both']} |")
    return "\n".join(L) + "\n"


def md_dev(Dv):
    L = ["# Dev-panel transitions (A17, exploratory)", "",
         f"Verified attempts per problem: {Dv['n_verification']['summary']}", "",
         "| Run | From → to | Problems | n per problem (from / to) | Lost | Gained | Zero-correct problems (from → to) |",
         "|---|---|---|---|---|---|---|"]
    for name, x in Dv["runs"].items():
        for t in x["transitions"]:
            L.append(f"| {name} | {t['from']} → {t['to']} | {t['problems']} | {t['n_per_problem_from']} / {t['n_per_problem_to']} | "
                     f"{t['lost']} | {t['gained']} | {t['zero_at_from']} → {t['zero_at_to']} |")
    L += ["", "**Repeated evaluations and no-update references** (same request seeds: differences come from "
          "non-reproducible generation, not from independent resampling):", "",
          "| Source | Panel | Rows compared | n per problem | Lost | Gained | pass@1 (first − second) | Checkpoint |", "|---|---|---|---|---|---|---|---|"]
    for r in Dv["repeated_evaluations"]:
        L.append(f"| {r['run']} r{r['round']} | {r['panel']} | rows {r['row_index'][0]} / {r['row_index'][1]} | {r['n_per_problem_from']} | "
                 f"{r['lost']} | {r['gained']} | {dstr(r['pass1'][0], r['pass1'][1])} | {r['checkpoint']} |")
    for r in Dv["diag_reference_pairs"]:
        L.append(f"| diag {r['label']} | {r['panel']} | {r['first']} / {r['second']} | {r['n_per_problem_from']} | {r['lost']} | "
                 f"{r['gained']} | {dstr(r['pass1_first'], r['pass1_second'])} | no update |")
    return "\n".join(L) + "\n"


def md_branches(B):
    L = ["# One-step local update (A17, exploratory)", "",
         "pre = no-update evaluation of the source checkpoint (original row; rerun row in brackets). post = after ONE "
         "GRPO/AdamW step on 4 prompts × 8 answers. Changes are post − pre. J values are pass@1 at 16 samples per problem.", ""]
    for lab, x in B["checkpoints"].items():
        src = x["source"]
        L += [f"## {lab}: source {src.get('source')}" + (f" (round {src.get('round')})" if src.get("round") is not None else ""), "",
              "| No-update row | J_H | J_R | J_U |", "|---|---|---|---|"]
        for tag, v in x["no_update"].items():
            L.append(f"| {tag} | {f4(v['J_H'])} | {f4(v['J_R'])} | {f4(v['J_U'])} |")
        L += ["", "| Branch | Seed | Mixed groups (of 4) | ΔJ_H = post − pre (original) | ΔJ_H vs rerun | grad norm |", "|---|---|---|---|---|---|"]
        for d in x["draws"]:
            L.append(f"| {d['branch']} | {d['seed']} | {d['mixed_groups']} | {dstr(d['post']['J_H'], d['pre_original']['J_H'])} | "
                     f"{dstr(d['post']['J_H'], d['pre_rerun']['J_H'])} | {f4(d['grad_norm_preclip'])} |")
        L += ["", "| Comparison (post values) | J_H | J_R | J_U |", "|---|---|---|---|"]
        for cname, c in x["comparisons"].items():
            a, b = cname.split("_vs_")
            for p in c["per_seed"]:
                L.append(f"| {a} − {b}, seed {p['seed']} | " + " | ".join(dstr(p[j][a], p[j][b]) for j in J_PANELS) + " |")
            o = c["over_seeds"]
            L.append(f"| mean {a} − mean {b} ({len(c['seeds'])} seeds) | " +
                     " | ".join(dstr(o[j][f'mean_{a}'], o[j][f'mean_{b}']) for j in J_PANELS) + " |")
        L += ["", "| Spread of single rows | J_H min–max (SD) | J_R min–max (SD) | J_U min–max (SD) |", "|---|---|---|---|"]
        rows = list(x["spread_post_within_branch"].items()) + [("no-update rows", x["spread_no_update_rows"])]
        for b, sp in rows:
            cells = []
            for j in J_PANELS:
                s = sp.get(j)
                cells.append("n/a" if not s else f"{s['min']:.4f}–{s['max']:.4f} ({f4(s['sd'])}; n={s['n']})")
            L.append(f"| {b} | " + " | ".join(cells) + " |")
        z = x["zero_mixed_draws"]
        L += ["", f"Draws with zero mixed groups: {len(z)}" +
              (": " + ", ".join(f"{d['branch']} {d['seed']}" for d in z) if z else "") +
              ("" if x["zero_mixed_draws_identical_per_problem_counts"] is None else
               f"; identical per-problem counts on all three probes: {x['zero_mixed_draws_identical_per_problem_counts']}"), ""]
    L += [B["zero_variation"]["statement"], "", B["u101_r64"]]
    return "\n".join(L) + "\n"


def md_delta_table(B):
    L = ["# One-step changes ΔJ = post − pre (pre = original no-update row), absolute values shown", "",
         "| Checkpoint | Branch | Seed | Mixed | J_H post − pre = Δ | J_R post − pre = Δ | J_U post − pre = Δ |", "|---|---|---|---|---|---|---|"]
    for lab, x in B["checkpoints"].items():
        for d in x["draws"]:
            L.append(f"| {lab} | {d['branch']} | {d['seed']} | {d['mixed_groups']} | " +
                     " | ".join(dstr(d["post"][j], d["pre_original"][j]) for j in J_PANELS) + " |")
        nu = x["no_update"]
        if "none_original" in nu:
            for tag in ("none_rerun", "none_cycled"):
                if tag in nu:
                    L.append(f"| {lab} | no update ({tag} − original) | – | – | " +
                             " | ".join(dstr(nu[tag][j], nu["none_original"][j]) for j in J_PANELS) + " |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- curves (figure data)
def curves(t, sig):
    ev = {}
    for p in PANELS:
        ev[p] = [{"round": r, "pass1": row.get("pass1"), "problems": row.get("problems"), "n": row.get("n")}
                 for (pp, r), row in sorted(t["M"]["eval"].items(), key=lambda kv: kv[0][1]) if pp == p]
    cum = {0: 0}
    for row in sig["per_round"]:
        if row["cum_hard_responses"] is not None:
            cum[row["round"]] = row["cum_hard_responses"]
    G = [g for g in t["groups"] if g["role"] == "hard_common"]
    bins = []
    for b in range(1, 129, 8):
        gs = [g for g in G if b <= g["round"] < b + 8]
        if not gs:
            continue
        k = len({g["round"] for g in gs})
        bins.append({"bin": f"{b}-{b + 7}", "start": b, "rounds": k, "complete": k == 8,
                     "correct": sum(g["c"] for g in gs), "responses": sum(g["n"] for g in gs),
                     "success": sum(g["c"] for g in gs) / sum(g["n"] for g in gs)})
    caps = {r: row.get("max_tokens") for r, row in t["M"]["round"].items()}
    old = [r for r, c in caps.items() if c == 8192]
    new = [r for r, c in caps.items() if c is not None and c > 8192]
    return {"run": t["run"], "arm": t["arm"], "seed": t["seed"], "complete": t["complete"], "evals": ev,
            "cum_hard_responses_at_round": {str(r): cum[r] for r in sorted(cum) if r in EVAL_ROUNDS or r == CAP_LAST_OLD},
            "practised_shared_hard_bins": bins,
            "cap_check": {"last_round_cap_8192": max(old, default=None), "first_round_cap_above_8192": min(new, default=None)}}


# ---------------------------------------------------------------- figures
ARM_COLOR = {"H": "#4c72b0", "R": "#dd8452", "U": "#55a868"}
SEED_STYLE = {101: "-", 202: "--"}
SEED_MARK = {101: "o", 202: "s"}
OUTCOME_COLOR = {"all_wrong": "#8c2d04", "mixed": "#fd8d3c", "all_correct": "#41ab5d", "no_later_observation": "#bdbdbd"}


def _style(name, c):
    if name.endswith("-orig"):
        return ":", "v", f"{c['arm']} seed {c['seed']} a2-orig (to r64)"
    if name.endswith("-r60"):
        return "-.", "D", f"{c['arm']} seed {c['seed']} a2-r60 (resumed at r60)" + ("" if c["complete"] else ", partial")
    return SEED_STYLE.get(c["seed"], ":"), SEED_MARK.get(c["seed"], "^"), f"{c['arm']} seed {c['seed']}"


def plot(fig_dir, cur, pers, br):
    """Draw the five figures from curves.json, persistence.json and branches.json contents. Returns names drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    drawn = []
    runs = cur.get("runs", {})

    def problems(panel):
        for c in runs.values():
            for e in c["evals"].get(panel, []):
                return e.get("problems"), e.get("n")
        return None, None

    if runs:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4)
        held = [("dev", "dev"), ("probe_J_H", "J_H"), ("probe_J_R", "J_R"), ("probe_J_U", "J_U")]
        ax_r = [fig.add_subplot(gs[0, k]) for k in range(4)]            # held-out vs round
        ax_x = [fig.add_subplot(gs[1, k]) for k in range(4)]            # held-out vs cumulative hard responses
        ax_p = fig.add_subplot(gs[2, 0:2])                              # practised prompts (training groups)
        ax_l = fig.add_subplot(gs[2, 2:4]); ax_l.axis("off")
        cap32, handles = {}, []
        for name, c in sorted(runs.items()):
            ls, mk, lab = _style(name, c)
            col = ARM_COLOR.get(c["arm"], "black")
            cum = {int(k): v for k, v in c["cum_hard_responses_at_round"].items()}
            if CAP_LAST_OLD in cum:
                cap32.setdefault(cum[CAP_LAST_OLD], set()).add(c["arm"])
            for k, (p, _) in enumerate(held):
                ev = [e for e in c["evals"].get(p, []) if e["pass1"] is not None]
                h, = ax_r[k].plot([e["round"] for e in ev], [e["pass1"] for e in ev], ls, marker=mk, color=col, ms=4, label=lab)
                xs = [(cum[e["round"]], e["pass1"]) for e in ev if e["round"] in cum]
                ax_x[k].plot([x for x, _ in xs], [y for _, y in xs], ls, marker=mk, color=col, ms=4)
                if k == 0:
                    handles.append(h)
            bins = [b for b in c["practised_shared_hard_bins"] if b["complete"]]
            ax_p.plot([b["start"] + 3.5 for b in bins], [b["success"] for b in bins], ls, marker=mk, color=col, ms=4)
        for k, (p, short_name) in enumerate(held):
            npb, kp = problems(p)
            ax_r[k].set_title(f"HELD-OUT {short_name} ({npb} problems × {kp} samples) vs round", fontsize=8)
            ax_x[k].set_title(f"HELD-OUT {short_name} vs cumulative hard-prompt training responses", fontsize=8)
            for a in (ax_r[k], ax_x[k]):
                a.set_ylabel(f"{short_name} pass@1")
            ax_r[k].set_xlabel("training round")
            ax_x[k].set_xlabel("cumulative hard-prompt training responses")
        ax_p.set_title("PRACTISED prompts (training groups, not a held-out evaluation): success on the 8 shared hard "
                       "prompts per round, complete 8-round bins", fontsize=8)
        ax_p.set_facecolor("#f4f4f4")
        ax_p.set_ylabel("training success, shared hard prompts"); ax_p.set_xlabel("training round (bin centre)")
        for a in ax_r + [ax_p]:
            a.axvline(CAP_LAST_OLD + 0.5, color="grey", lw=0.8, ls=":")
            a.text(CAP_LAST_OLD + 1.5, 0.02, "cap 8,192 → 12,288", fontsize=6, color="grey", transform=a.get_xaxis_transform())
        for a in ax_x:
            for x, arms in sorted(cap32.items()):
                a.axvline(x, color="grey", lw=0.8, ls=":")
                a.text(x, 0.02, f" cap change ({', '.join(sorted(arms))})", fontsize=6, color="grey",
                       transform=a.get_xaxis_transform())
        for a in ax_r + ax_x + [ax_p]:
            a.grid(alpha=0.3)
        ax_l.legend(handles=handles, loc="center", fontsize=9, title="trajectory (colour = arm)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=150); plt.close(fig)
        drawn.append("learning_curves")

        vals = lambda p: [e["pass1"] for c in runs.values() for e in c["evals"].get(p, []) if e["pass1"] is not None]  # noqa: E731
        xv, yv = vals("probe_J_R") + vals("probe_J_U"), vals("probe_J_H")
        if xv and yv:
            xl, yl = (min(xv) - 0.02, max(xv) + 0.02), (min(yv) - 0.02, max(yv) + 0.02)
            (nh, kh), (nr, _), (nu, _) = problems("probe_J_H"), problems("probe_J_R"), problems("probe_J_U")
            for aux, fname in (("R", "pope_JH_vs_JR"), ("U", "pope_JH_vs_JU")):
                fig, a = plt.subplots(figsize=(6.4, 6.0))
                for name, c in sorted(runs.items()):
                    ls, mk, lab = _style(name, c)
                    col = ARM_COLOR.get(c["arm"], "black")
                    jx = {e["round"]: e["pass1"] for e in c["evals"].get(f"probe_J_{aux}", [])}
                    jh = {e["round"]: e["pass1"] for e in c["evals"].get("probe_J_H", [])}
                    rs = sorted(set(jx) & set(jh))
                    if not rs:
                        continue
                    a.plot([jx[r] for r in rs], [jh[r] for r in rs], ls, marker=mk, color=col, lw=1, ms=4, label=lab)
                    for r in rs:
                        a.annotate(str(r), (jx[r], jh[r]), textcoords="offset points", xytext=(3, 3), fontsize=6, color=col)
                a.set_xlim(*xl); a.set_ylim(*yl)
                a.set_xlabel(f"J_{aux} pass@1")
                a.set_ylabel("J_H pass@1")
                probe = (f"J_R = {nr} related (factoring) probe problems" if aux == "R"
                         else f"J_U = {nu} unrelated (modular-arithmetic) probe problems")
                a.set_title(f"J_H vs J_{aux} at each scheduled evaluation (labels = round; lines in time order)\n"
                            f"J_H = fixed {nh}-problem hard subset of the dev panel; {probe}; {kh} samples per problem;\n"
                            f"axis ranges identical in both J_H-vs-J_R and J_H-vs-J_U plots", fontsize=8)
                a.legend(fontsize=6); a.grid(alpha=0.3)
                fig.tight_layout(); fig.savefig(os.path.join(fig_dir, fname + ".png"), dpi=150); plt.close(fig)
                drawn.append(fname)

    cps = (br or {}).get("checkpoints", {})
    if cps:
        fig, axes = plt.subplots(1, len(cps), figsize=(5.6 * len(cps), 4.4), squeeze=False)
        for a, (lab, x) in zip(axes[0], cps.items()):
            cats = list(BRANCHES) + ["no update"]
            for j, dx, mk in (("J_H", -0.2, "o"), ("J_R", 0.0, "^"), ("J_U", 0.2, "s")):
                for k, b in enumerate(BRANCHES):
                    ds = [d["change_post_minus_original"][j] for d in x["draws"] if d["branch"] == b
                          and d["change_post_minus_original"][j] is not None]
                    a.plot([k + dx] * len(ds), ds, mk, color="black" if j == "J_H" else "grey", ms=4, mfc="none" if j != "J_H" else None,
                           label=j if k == 0 else None)
                nu = x["no_update"]
                if "none_original" in nu:
                    ds = [nu[t][j] - nu["none_original"][j] for t in ("none_rerun", "none_cycled") if t in nu]
                    a.plot([3 + dx] * len(ds), ds, mk, color="#c44e52", ms=4,
                           label="no-update row − original" if j == "J_H" else None)
            a.axhline(0, color="grey", lw=0.8)
            a.set_xticks(range(4)); a.set_xticklabels(["hard", "related", "unrelated", "no update\n(rerun, cycled)"], fontsize=8)
            src = x["source"]
            a.set_title(f"{lab}: {'base model' if src.get('source') == 'base' else 'R-s101 round ' + str(src.get('round'))}"
                        "\neach point = one measured change; see delta_j_table.md", fontsize=8)
            a.set_ylabel("post − pre pass@1 (pre = original no-update row)")
            a.legend(fontsize=7); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "delta_j.png"), dpi=150); plt.close(fig)
        drawn.append("delta_j")

    pr = (pers or {}).get("runs", {})
    if pr:
        names = list(pr)
        fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3.8), sharey=True, squeeze=False)
        ws = [f"r{a}-{b}" for a, b in WINDOWS]
        for a, name in zip(axes[0], names):
            bw = pr[name]["by_window"]
            bottom = np.zeros(len(ws))
            for o in OUTCOMES:
                v = np.array([bw[w]["next_outcome"][o] for w in ws], dtype=float)
                a.bar(range(len(ws)), v, bottom=bottom, color=OUTCOME_COLOR[o], label=o.replace("_", " ") if name == names[0] else None)
                bottom += v
            for i, w in enumerate(ws):
                if bw[w]["hard_common_groups"]:
                    a.text(i, bottom[i], f"{bw[w]['previously_solvable_became_all_wrong']}/{bw[w]['later_appearances_of_previously_solvable']}",
                           ha="center", va="bottom", fontsize=6)
            a.axvline(0.5, color="grey", lw=0.8, ls=":")
            a.set_xticks(range(len(ws))); a.set_xticklabels(ws, fontsize=6, rotation=30)
            a.set_title(name, fontsize=8)
        axes[0][0].set_ylabel("previously solvable → all-wrong groups\n(stacked by outcome at next appearance)", fontsize=8)
        axes[0][0].legend(fontsize=6, title="next appearance", title_fontsize=6)
        fig.suptitle("Shared hard prompts: label = events / later appearances of previously solvable prompts; "
                     "dotted line = cap change after round 32", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "transitions.png"), dpi=150); plt.close(fig)
        drawn.append("transitions")
    json.dump({"drawn": drawn}, open(os.path.join(fig_dir, "figures.json"), "w"), indent=1)
    return drawn


# ---------------------------------------------------------------- main
def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        out, fig = sys.argv[2], sys.argv[3]
        ld = lambda n: json.load(open(os.path.join(out, n))) if os.path.exists(os.path.join(out, n)) else {}  # noqa: E731
        print("drawn:", plot(fig, ld("curves.json"), ld("persistence.json"), ld("branches.json")))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root"); ap.add_argument("out_dir")
    ap.add_argument("--diag", nargs="*", default=[]); ap.add_argument("--diag-ref", nargs="*", default=[])
    ap.add_argument("--figures"); ap.add_argument("--gpu-hours")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cost = json.load(open(a.gpu_hours)) if a.gpu_hours else None
    T = trajectories(a.runs_root)
    for t in T:
        load_groups(t)
    traj_info = {t["name"]: {"run": t["run"], "complete": t["complete"], "notes": t["notes"],
                             "a17": t.get("a17"), "missing_records": t.get("missing_records", [])} for t in T}
    # 1
    S = {t["name"]: signal(t, cost) for t in T}
    sig = {"windows": [f"r{a}-{b}" for a, b in WINDOWS], "cap": "8,192 for rounds 1-32; 12,288 from round 33",
           "roles_logged": "role field of train records (run.py:358; schedule.py:49-56)",
           "granularity": SIGNAL_GRANULARITY, "cost_json": a.gpu_hours, "trajectories": traj_info, "runs": S}
    dump(sig, os.path.join(a.out_dir, "signal.json"))
    open(os.path.join(a.out_dir, "signal.md"), "w").write(md_signal(sig, cost is not None))
    # 2
    P = {t["name"]: persistence_run(t) for t in T}
    ev = {n: p["events"] for n, p in P.items()}
    by = {t["name"]: t for t in T}
    pairs = []
    for s in sorted({t["seed"] for t in T}):
        pairs.append((f"R-s{s}", f"U-s{s}"))
    for h in ("H-s202-r60", "H-s202-orig"):
        pairs += [(h, "R-s202"), (h, "U-s202")]
    pers = {"definition": {"previously_solvable": ">= 1 correct answer at an earlier hard_common appearance of the same "
                                                  "prompt in the same trajectory",
                           "phases": "r1-32 (cap 8,192) and r33-128 (cap 12,288)",
                           "next_outcome": "kind of the group at the prompt's next hard_common appearance in the same "
                                           "trajectory; no_later_observation if none was recorded (includes run end)",
                           "capped": "answers with finish == 'length' (run.py:360)"},
            "runs": P, "matched": [matched(by[x], by[y], ev) for x, y in pairs if x in by and y in by]}
    dump(pers, os.path.join(a.out_dir, "persistence.json"))
    open(os.path.join(a.out_dir, "persistence.md"), "w").write(md_persistence(pers))
    # 3
    Dv = {t["name"]: dev_run(t) for t in T}
    raw = {}
    for t in T:
        if t["run"] not in raw:
            raw[t["run"]] = read_jsonl(os.path.join(t["dir"], "metrics.jsonl"))[0]
    rep = [x for run, rows in raw.items() for x in repeated_evals(run, rows)]
    diag, miss_d = read_diag(a.diag)
    refs, miss_r = read_diag(a.diag_ref)
    ref_pairs = []
    for lab in sorted({r["label"] for r in diag + refs}):
        rows = {r["branch"]: r for r in diag + refs if r["label"] == lab and r["branch"] in ("none", "none_rerun", "none_cycled")}
        for x, y in (("none", "none_rerun"), ("none_rerun", "none_cycled")):
            if x in rows and y in rows:
                for j in J_PANELS:
                    if jpp(rows[x], j) is None or jpp(rows[y], j) is None:
                        continue
                    ref_pairs.append({"label": lab, "panel": f"probe_{j}", "first": x, "second": y,
                                      "pass1_first": jv(rows[x], j), "pass1_second": jv(rows[y], j),
                                      **lost_gained(jpp(rows[x], j), jpp(rows[y], j))})
    ns = sorted({n for x in Dv.values() for e in x["evaluations"] for n in e["n_per_problem"]})
    rs = sorted({n for x in Dv.values() for e in x["evaluations"] for n in e.get("records_samples_per_problem", [])})
    agree = [e.get("records_agree_with_row_per_problem") for x in Dv.values() for e in x["evaluations"]]
    nver = {"n_per_problem_values": ns, "records_samples_per_problem_values": rs,
            "fields": "metrics eval row per_problem[].n (run.py:138, 224-225) and row n (run.py:143); samples per "
                      "prompt_id in records/eval_dev_rNNNN.jsonl.gz (run.py:135-141); n comes from config eval_panels.dev",
            "records_agree": {"true": sum(v is True for v in agree), "false": sum(v is False for v in agree),
                              "not_checked": sum(v is None for v in agree)},
            "report_md_claim": "REPORT.md section 4 says 'of 16'",
            "summary": (f"n = {ns[0]} attempts per dev problem in every scheduled evaluation" if len(ns) == 1 else
                        f"n values found: {ns}") + (f"; the per-sample records show {rs}" if rs else "") +
                       ("; the REPORT.md 'of 16' is wrong" if ns and 16 not in ns else "")}
    dev = {"definition": {"lost": ">= 1 correct at the earlier evaluation and 0 at the later one",
                          "gained": "0 correct at the earlier evaluation and >= 1 at the later one",
                          "source": "per_problem counts of canonical dev eval rows; cross-checked with eval records"},
           "n_verification": nver, "runs": Dv, "repeated_evaluations": rep, "diag_reference_pairs": ref_pairs,
           "reference_note": "Evaluation request seeds depend only on panel, problem, round and n (run.py:123); the "
                             "diagnostic uses one constant round label for every row (diag_local.py:51). Two evaluations "
                             "of the same weights therefore use the same seeds: their differences measure non-bitwise-"
                             "reproducible generation (and GPU type where it differs), not independent resampling.",
           "missing_inputs": miss_d + miss_r}
    dump(dev, os.path.join(a.out_dir, "dev_transitions.json"))
    open(os.path.join(a.out_dir, "dev_transitions.md"), "w").write(md_dev(dev))
    # 4
    B = {"checkpoints": branches(diag, refs), "inputs": {"diag": a.diag, "diag_ref": a.diag_ref, "missing": miss_d + miss_r},
         "zero_variation": {**ZERO_VARIATION, "source_check": zero_variation_source_check(),
                            "recorded_zero_mixed_draws": sum(1 for r in diag if r.get("branch") in BRANCHES and r.get("mixed_groups") == 0),
                            "recorded_draws": sum(1 for r in diag if r.get("branch") in BRANCHES)},
         "u101_r64": U101_R64_NOTE,
         "definitions": {"pre": "no-update evaluation of the source (diag 'none' row = original; ref 'none_rerun' = fresh "
                                "process; 'none_cycled' = after the save-restore-reload cycle)",
                         "post": "evaluation after one GRPO/AdamW step on 4 prompts x 8 answers",
                         "J": "pass@1 at 16 samples per problem (diag_local.py:50) on probe_J_H / probe_J_R / probe_J_U",
                         "mixed_groups": "groups of the minibatch with 0 < correct < 8 (diag_local.py:94)"}}
    dump(B, os.path.join(a.out_dir, "branches.json"))
    open(os.path.join(a.out_dir, "branches.md"), "w").write(md_branches(B))
    open(os.path.join(a.out_dir, "delta_j_table.md"), "w").write(md_delta_table(B))
    # 5
    C = {"runs": {t["name"]: curves(t, S[t["name"]]) for t in T}}
    dump(C, os.path.join(a.out_dir, "curves.json"))
    for name, s in S.items():
        print(name, "rounds", s["rounds_logged"], "records", s["rounds_with_records"], "complete", s["complete"],
              "mismatch", len(s["records_vs_round_rows_mismatch"]), "notes", s["notes"])
    print("dev n:", nver["summary"])
    if a.figures:
        print("drawn:", plot(a.figures, C, pers, B))


if __name__ == "__main__":
    main()
