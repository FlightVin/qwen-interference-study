"""b-train: inspect record formats, non-routine events, diagnostic files and bundle pools (stdlib only)."""
import json, gzip, glob, os, collections, datetime
P = "/data/user_data/vineethb/qis"
def ts(u): return datetime.datetime.fromtimestamp(u, datetime.timezone.utc).strftime("%m-%d %H:%MZ")
for run in ["R-s101", "R-s202", "U-s202", "H-s202"]:
    fn = f"{P}/runs/qis-main-{run}-sym/metrics.jsonl"
    print("==", run)
    for i, l in enumerate(open(fn)):
        r = json.loads(l)
        if r["kind"] == "event" and r["what"] not in ("checkpoint_started",):
            if r["what"] == "checkpoint" and r.get("ok") and r.get("round") not in (16, 60, 64, 80, 124, 128): continue
            print(f"  L{i} {ts(r['utc'])}", {k: v for k, v in r.items() if k not in ("utc", "kind", "run_id", "attempt_id", "argv", "bundle_sha", "torch", "host")})
print("== record formats")
def head(fn):
    with gzip.open(fn, "rt") as f:
        l = f.readline()
    r = json.loads(l); r["ids"] = f"<{len(r['ids'])} ids>"; return r
print(head(f"{P}/runs/qis-main-R-s101-sym/records/train_r0000.jsonl.gz"))
print(head(f"{P}/runs/qis-main-R-s101-sym/records/eval_dev_r0000.jsonl.gz"))
print(head(f"{P}/diag-local/r64/records/r64-none/eval_probe_J_H_r7064.jsonl.gz"))
for lab in ("r0", "r64"):
    for fn in sorted(glob.glob(f"{P}/diag-local/{lab}/diag_*.jsonl")):
        print("==", fn, os.path.getsize(fn))
        for l in open(fn):
            r = json.loads(l)
            ev = r.pop("eval")
            print("  ", r, {p: (round(v["pass1"], 4), v["truncated"]) for p, v in ev.items()})
    print(sorted(os.listdir(f"{P}/diag-local/{lab}/records")))
b = json.load(open(f"{P}/data/main-v1/bundle_main_v1.json"))
print("pools", {k: len(v) for k, v in b["pools"].items()}, "keys", list(b.keys()))
dev = [x["id"] for x in b["pools"]["dev"]]; jh = [x["id"] for x in b["pools"]["probe_J_H"]]
print("J_H == dev[:32]", jh == dev[:32], "J_H subset", set(jh) <= set(dev))
print(sorted(os.listdir(f"{P}/runs/superseded/qis-main-R-s101-sym-a1/records"))[:5], len(os.listdir(f"{P}/runs/superseded/qis-main-R-s101-sym-a1/records")))
print(open(f"{P}/runs/qis-main-H-s202-sym/records_a2orig_r60-64/SHA256SUMS").read())
print(open(f"{P}/runs/qis-main-H-s202-sym/resume_audit_a17b.txt").read()[:1500])
