"""a-data: print field layouts of compute data files (read-only, stdlib)."""
import json, gzip, os, collections
P = "/data/user_data/vineethb/qis"
with open(f"{P}/data/splits-v1/manifest.jsonl") as f:
    r = json.loads(f.readline()); print("splits keys", {k: (str(v)[:60]) for k, v in r.items()})
n = collections.Counter(json.loads(l)["split"] for l in open(f"{P}/data/splits-v1/manifest.jsonl"))
print("splits", dict(n))
b = json.load(open(f"{P}/data/main-v1/bundle_main_v1.json"))
for k, v in b["pools"].items():
    print("bundle", k, len(v), sorted(v[0].keys()))
c = json.load(open(f"{P}/data/calib-panels-k8/bundle.json"))
for k, v in c["pools"].items():
    print("calib", k, len(v), sorted(v[0].keys()))
for run in sorted(os.listdir(f"{P}/runs")):
    if not run.startswith("qis-main"): continue
    kinds = collections.Counter(); ev = []
    for l in open(f"{P}/runs/{run}/metrics.jsonl"):
        x = json.loads(l); kinds[x["kind"]] += 1
        if x["kind"] == "event" and x["what"] in ("start", "resume", "complete"):
            ev.append((x["what"], x.get("job"), x.get("resume_round", x.get("round")), x.get("bundle_sha", "")[:12], x.get("attempt_id"), x.get("code_sha", "")))
        if x["kind"] == "config":
            ev.append(("config", x["bundle_sha"][:12], x["config"].get("bundle"), x["config"].get("seed"), x["config"].get("arm")))
    print(run, dict(kinds)); [print("   ", e) for e in ev]
    with gzip.open(f"{P}/runs/{run}/records/train_r0000.jsonl.gz", "rt") as f:
        r = json.loads(f.readline()); print("   rec keys", sorted(r.keys()))
    print("   records subdirs", [d for d in os.listdir(f"{P}/runs/{run}/records") if not d.endswith(".gz")])
