"""Companion W&B runs with complete curves (read-only on runs; logging only; run in a CPU job).

The live training runs mirrored only flat numeric fields before the A10 code (evaluation panels shared one key;
per-role success and quantile dicts were dropped; J_H was not logged). For each main run this writes a companion
W&B run `<run>-curves` (same project, group = run id, job_type "derived") from the local metrics/records:
- round/success/<role>, round/len_mean, round/truncated, round/groups_*, clip fractions, ratio and mismatch tails,
  entropy (every 8 rounds), round/max_tokens (cap; marks the 8k→16k transition);
- curves/<hard|aux|hard_common|hard_extra>/{all_wrong,mixed,all_correct,success,capped_frac,mean_tokens};
- eval/<panel>/{pass1,pass8,truncated} for dev, probe_J_R, probe_J_U, and probe_J_H derived from dev per-problem.
Run ids carry the trajectory's attempt id (`<run>-<attempt>-curves`) so superseded and resumed trajectories never
mix (A11). One extra run `qis-RvU-s<seed>-sym-<attempt>-curves` logs R and U on the shared hard schedule at matched rounds.
Incremental: a state file under PROJ_DATA records the last logged round per companion run; reruns resume the same
W&B run id and log only newer rounds. No account-wide sync; only these project runs.
Usage: python -m qis.analysis.wandb_companion RUNS_DIR BUNDLE STATE.json PROJECT
"""
import json
import os
import sys

import numpy as np

from qis.analysis.curves import round_rows
from qis.train import grpo

PANELS = ("dev", "probe_J_R", "probe_J_U")


def run_rows(d, jh):
    M = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    rows = {}
    for r in M:
        if r.get("kind") == "round":
            x = rows.setdefault(r["round"], {})
            x["round/max_tokens"] = r.get("max_tokens", 8192)
            for k in ("len_mean", "truncated", "groups_mixed", "groups_all0", "groups_all1", "clipfrac_pos", "clipfrac_neg",
                      "entropy_sample_mean", "t_round", "gen_tokens"):
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool):
                    x[f"round/{k}"] = r[k]
            for k, v in r.get("success", {}).items():
                x[f"round/success/{k}"] = v
            for name in ("ratio_q", "logprob_mismatch_absq"):
                for q in ("q0.001", "q0.99", "q0.999"):
                    if q in r.get(name, {}):
                        x[f"round/{name}/{q}"] = r[name][q]
        elif r.get("kind") == "eval" and r.get("panel") in PANELS:
            x = rows.setdefault(r["round"], {})
            for k in ("pass1", "pass8", "truncated"):
                if isinstance(r.get(k), (int, float)):
                    x[f"eval/{r['panel']}/{k}"] = r[k]
            if r["panel"] == "dev" and jh:
                pp = [p for p in r.get("per_problem", []) if p["id"] in jh]
                if pp:
                    x["eval/probe_J_H/pass1"] = float(np.mean([p["c"] / p["n"] for p in pp]))
                    x["eval/probe_J_H/pass8"] = float(np.mean([grpo.pass_at_k(p["n"], p["c"], 8) for p in pp]))
    for rnd, row in round_rows(d).items():
        x = rows.setdefault(rnd, {})
        for role in ("hard", "aux", "hard_common", "hard_extra"):
            if role in row:
                v = row[role]
                x.update({f"curves/{role}/{k}": v[k] for k in ("all_wrong", "mixed", "all_correct", "success", "mean_tokens")})
                x[f"curves/{role}/capped_frac"] = v["capped"] / v["total"]
    return rows


def push(project, run_id, group, rows, last, tags):
    import wandb
    new = sorted(r for r in rows if r > last)
    if not new:
        return last
    w = wandb.init(project=project, id=run_id, name=run_id, group=group, job_type="derived", tags=tags, resume="allow")
    w.define_metric("round")
    for pre in ("round/*", "curves/*", "eval/*", "rvu/*"):
        w.define_metric(pre, step_metric="round")
    for r in new:
        w.log({"round": r, **rows[r]})
    w.finish()
    return new[-1]


def main(runs, bundle, state_path, project):
    pools = json.load(open(bundle))["pools"]
    jh = {it["id"] for it in pools.get("probe_J_H", [])}
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    started = lambda n: os.path.isfile(p := os.path.join(runs, n, "metrics.jsonl")) and os.path.getsize(p) > 0   # noqa: E731
    names = sorted(n for n in os.listdir(runs) if n.startswith("qis-main-") and started(n))
    att = {}
    for n in names:                                   # attempt id of the CURRENT trajectory (A11: a2 after the restart)
        with open(os.path.join(runs, n, "metrics.jsonl")) as f:
            att[n] = json.loads(f.readline()).get("attempt_id", "a1")
    all_rows = {}
    for n in names:
        all_rows[n] = run_rows(os.path.join(runs, n), jh)
        arm = n.split("-")[2]
        rid = f"{n}-{att[n]}-curves"
        state[rid] = push(project, rid, n, all_rows[n], state.get(rid, -1), ["derived", arm, att[n]])
    for n in names:
        if "-R-" in n and n.replace("-R-", "-U-") in all_rows:
            a, b = round_rows(os.path.join(runs, n)), round_rows(os.path.join(runs, n.replace("-R-", "-U-")))
            rows = {}
            for r in sorted(set(a) & set(b)):
                if "hard_common" in a[r] and "hard_common" in b[r]:
                    ra, rb = a[r]["hard_common"], b[r]["hard_common"]
                    rows[r] = {"rvu/R_success": ra["success"], "rvu/U_success": rb["success"],
                               "rvu/diff_success": ra["success"] - rb["success"],
                               "rvu/R_capped_frac": ra["capped"] / ra["total"], "rvu/U_capped_frac": rb["capped"] / rb["total"],
                               "rvu/R_mixed": ra["mixed"], "rvu/U_mixed": rb["mixed"], "rvu/cap": a[r]["cap"]}
            rid = n.replace("qis-main-R-", "qis-RvU-") + f"-{att[n]}-curves"
            state[rid] = push(project, rid, n.replace("qis-main-R-", "qis-RvU-"), rows, state.get(rid, -1), ["derived", "RvU"])
    json.dump(state, open(state_path, "w"), indent=1)
    print(json.dumps(state, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
