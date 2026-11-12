"""b-train: list segments, events and row multiplicity in each run's metrics files (stdlib only)."""
import json, sys, os, collections, glob, datetime
P = "/data/user_data/vineethb/qis"
def ts(u): return datetime.datetime.utcfromtimestamp(u).strftime("%m-%d %H:%MZ")
files = sorted(glob.glob(f"{P}/runs/qis-main-*/metrics*.jsonl")) + sorted(glob.glob(f"{P}/runs/superseded/*/metrics.jsonl"))
for fn in files:
    print("=====", fn.replace(P, "$P"))
    kinds = collections.Counter(); rounds = collections.Counter(); steps = collections.Counter(); evals = collections.Counter()
    seg = None; attempts = collections.Counter(); bad = 0
    for i, line in enumerate(open(fn)):
        try: r = json.loads(line)
        except Exception: bad += 1; continue
        k = r.get("kind"); kinds[k] += 1; attempts[r.get("attempt_id")] += 1
        if k == "event":
            w = r.get("what")
            if w in ("start", "resume", "preempt_exit", "complete", "nonfinite", "round_held_verifier_unknown", "forced_exit_test", "final_checkpoint_failed"):
                extra = {x: r.get(x) for x in ("job", "restart", "resume_round", "round", "minibatch", "code_sha", "gpu", "opt_step") if x in r}
                print(f"  L{i} {ts(r['utc'])} {r['attempt_id']} event {w} {extra}")
        elif k == "config":
            c = r["config"]; print(f"  L{i} config cap_schedule={c.get('cap_schedule')} max_tokens={c.get('max_tokens')} ckpt_every={c.get('ckpt_every')} eval_rounds={c.get('eval_rounds')} eval_panels={c.get('eval_panels')} seed={c.get('seed')} arm={c.get('arm')} lr={c.get('lr')} retain={c.get('retain_rounds')}")
        elif k == "round": rounds[r["round"]] += 1
        elif k == "step": steps[(r["round"], r["minibatch"])] += 1
        elif k == "eval": evals[(r["panel"], r["round"])] += 1; print(f"  L{i} {ts(r['utc'])} eval {r['panel']} r{r['round']} pass1={r.get('pass1')} cap={r.get('max_tokens')} trunc={r.get('truncated')}")
    print("  kinds", dict(kinds), "attempts", dict(attempts), "bad", bad)
    dup = {k: v for k, v in rounds.items() if v > 1}
    print("  round rows", len(rounds), "min", min(rounds) if rounds else None, "max", max(rounds) if rounds else None, "dup", dup)
    sdup = sorted({k[0] for k, v in steps.items() if v > 1})
    print("  step keys", len(steps), "rounds with dup steps", sdup)
