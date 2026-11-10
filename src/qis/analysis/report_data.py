"""Non-sealed report inputs from the six main runs (read-only; run in a CPU job; CHECKLIST 7.2-7.4, A1.2, A13).

Per run (runs/qis-main-*; superseded attempts live elsewhere and are not read):
- attempt check: every metrics row that carries attempt_id has the expected one (D99; default a2);
- scheduled evaluations (dev, probe_J_H/J_R/J_U), one row per (panel, round); after a resume the LAST occurrence is
  canonical; scalar fields only, plus the number of logged occurrences;
- exposure: cumulative hard-prompt responses and generated training tokens at every round (from training records and
  round rows), so curves can be drawn against rounds and against cumulative hard rollouts;
- active compute on the canonical trajectory: sum of round times (seconds);
- truncation by role (hard_common, hard_extra, hard, aux) for rounds 1-32 (cap 8,192) and 33-128 (cap 12,288).
Usage: python -m qis.analysis.report_data RUNS_DIR OUT.json [ATTEMPT_ID]
"""
import glob
import json
import os
import sys

from qis.analysis.curves import round_rows

PANELS = ("dev", "probe_J_H", "probe_J_R", "probe_J_U")
ROLES = ("hard_common", "hard_extra", "hard", "aux")


def run_data(d, attempt):
    M = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    bad = sorted({r.get("attempt_id") for r in M if "attempt_id" in r and r["attempt_id"] != attempt})
    evals, seen = {}, {}
    for r in M:
        if r.get("kind") == "eval" and r.get("panel") in PANELS:
            k = (r["panel"], r["round"])
            seen[k] = seen.get(k, 0) + 1
            evals[k] = {x: v for x, v in r.items() if not isinstance(v, (list, dict))}
    for k in evals:
        evals[k]["occurrences"] = seen[k]
    rounds = {}
    for r in M:
        if r.get("kind") == "round":
            rounds[r["round"]] = r                                   # last occurrence canonical
    rows = round_rows(d)
    cum_hard, cum_tokens, expo = 0, 0, {}
    for rnd in sorted(rows):
        cum_hard += rows[rnd].get("hard", {}).get("total", 0)
        # generated tokens from the records (metrics counters are offset after a resume, D127)
        cum_tokens += sum(rows[rnd][k]["total"] * rows[rnd][k]["mean_tokens"] for k in ("hard_common", "hard_extra", "aux") if k in rows[rnd])
        expo[rnd] = {"cum_hard_responses": cum_hard, "gen_tokens": cum_tokens, "cap": rows[rnd]["cap"]}
    trunc = {}
    for name, lo, hi in (("r1-32", 1, 32), ("r33-128", 33, 128)):
        t = {}
        for role in ROLES:
            c = sum(rows[x][role]["capped"] for x in rows if lo <= x <= hi and role in rows[x])
            n = sum(rows[x][role]["total"] for x in rows if lo <= x <= hi and role in rows[x])
            if n:
                t[role] = {"capped": c, "total": n, "share": c / n}
        trunc[name] = t
    return {"attempt_ok": not bad, "other_attempt_ids": bad, "rounds_logged": len(rounds),
            "max_round": max(rounds) if rounds else 0, "records_rounds": len(rows),
            "active_seconds": sum(r.get("t_round", 0) for r in rounds.values()),
            "gen_tokens": cum_tokens, "gen_tokens_metrics_counter": max((r.get("gen_tokens", 0) for r in rounds.values()), default=0),
            "evals": sorted(evals.values(), key=lambda e: (e["panel"], e["round"])),
            "exposure": expo, "truncation": trunc}


def main(runs, out, attempt="a2"):
    res = {os.path.basename(d): run_data(d, attempt) for d in sorted(glob.glob(os.path.join(runs, "qis-main-*")))}
    json.dump({"attempt": attempt, "runs": res}, open(out, "w"))
    for k, v in res.items():
        print(k, "attempt_ok", v["attempt_ok"], "rounds", v["rounds_logged"], "max", v["max_round"],
              "evals", len(v["evals"]), "active_h", round(v["active_seconds"] / 3600, 1),
              "trunc>32", {r: round(x["share"], 3) for r, x in v["truncation"]["r33-128"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
