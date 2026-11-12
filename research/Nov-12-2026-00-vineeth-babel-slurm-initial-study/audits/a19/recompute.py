"""Independent recomputation (A19 §5, CHECKLIST 8.8a): headline numbers from raw attempt records, standard library only,
no project code. pass@1 = mean over problems of correct/attempts; truncated = share of finish == "length";
accuracy on finished = correct / finished attempts. Usage: python3 recompute.py P OUT.json"""
import glob
import gzip
import json
import os
import sys

P, OUT = sys.argv[1], sys.argv[2]
cells = {}
for root in ("test-eval", "test-eval-cgpu", "stress-eval"):
    for d in sorted(glob.glob(os.path.join(P, root, "*"))):
        f = os.path.join(d, "records.jsonl.gz")
        if not os.path.exists(f):
            continue
        by, fin, cor, tr, n = {}, 0, 0, 0, 0
        with gzip.open(f, "rt") as g:
            for line in g:
                r = json.loads(line)
                by.setdefault(r["prompt_id"], []).append(r["reward"] == 1)
                n += 1; cor += r["reward"] == 1
                tr += r["finish"] == "length"; fin += r["finish"] != "length"
        cells[f"{root}/{os.path.basename(d)}"] = {
            "problems": len(by), "attempts": n, "pass1": sum(sum(v) / len(v) for v in by.values()) / len(by),
            "truncated": tr / n, "acc_finished": cor / fin if fin else None,
            "per_problem": {k: sum(v) / len(v) for k, v in by.items()}}
json.dump(cells, open(OUT, "w"))
for k, c in cells.items():
    print(f"{k:45s} problems {c['problems']:4d} attempts {c['attempts']:5d} pass1 {c['pass1']:.4f} "
          f"trunc {c['truncated']:.4f} acc_fin {c['acc_finished']:.4f}")
pairs = [("R-s101", "U-s101"), ("R-s202", "U-s202"), ("R-s101", "H-s101")]
for root, fmt in (("test-eval", "qis-test-{}-sym"), ("test-eval-cgpu", "qis-cgpu-{}"), ("stress-eval", "{}-r128")):
    for a, b in pairs:
        ka, kb = f"{root}/{fmt.format(a)}", f"{root}/{fmt.format(b)}"
        if root == "test-eval-cgpu" and b == "H-s101":
            kb = "test-eval/qis-test-H-s101-sym"
        if ka in cells and kb in cells:
            print(f"{root:15s} {a} − {b}: {cells[ka]['pass1']:.4f} − {cells[kb]['pass1']:.4f} = {cells[ka]['pass1'] - cells[kb]['pass1']:+.4f}")
