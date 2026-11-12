"""A26 pass A (a-evals): print one truncated example line of each record kind. Read-only; stdlib only."""
import gzip, json, os, sys
P = "/data/user_data/vineethb/qis"
def short(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 6: out[k] = f"list[{len(v)}]"
        elif isinstance(v, str) and len(v) > 120: out[k] = v[:120] + "..."
        elif isinstance(v, dict): out[k] = short(v)
        else: out[k] = v
    return out
def first(path, n=1, pred=None):
    op = gzip.open if path.endswith(".gz") else open
    k = 0
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if pred and not pred(r): continue
            print("  ", json.dumps(short(r))[:1500]); k += 1
            if k >= n: return
for p in ["test-eval/qis-test-R-s101-sym/records.jsonl.gz", "length-test/R-s202-r128/all/rows.jsonl.gz",
          "length-diag/R-s202-r128/t1/rows.jsonl.gz", "cue-eval/v1/R-s101-cued/records.jsonl.gz"]:
    print("==", p); first(os.path.join(P, p))
print("== extended length row"); first(os.path.join(P, "length-test/R-s202-r128/all/rows.jsonl.gz"), 1, lambda r: r["extended"])
for run in ["qis-main-H-s202-sym", "qis-main-U-s202-sym"]:
    d = os.path.join(P, "runs", run); print("==", d, sorted(os.listdir(d))[:40])
    rd = os.path.join(d, "records"); fs = sorted(os.listdir(rd)); print("   records:", len(fs), fs[:3], fs[-6:])
d = os.path.join(P, "runs", "qis-main-R-s101-sym"); print("==", sorted(os.listdir(d)))
print("== train record"); first(os.path.join(P, "runs/qis-main-R-s101-sym/records/train_r0000.jsonl.gz"))
kinds = {}
with open(os.path.join(P, "runs/qis-main-R-s101-sym/metrics.jsonl")) as f:
    for line in f:
        r = json.loads(line); k = r.get("type") or r.get("event") or r.get("kind") or ",".join(sorted(r)[:4])
        if k not in kinds: kinds[k] = short(r)
for k, v in kinds.items(): print("== metrics kind", k, json.dumps(v)[:1500])
