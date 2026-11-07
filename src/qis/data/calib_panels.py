"""Calibration control panels (project.md Section 7 "Optimizer calibration").

From the 16-sample baseline screen (stage `calib16`), keep items with 1-5/16 (lower band) and
6-12/16 (higher band) in each calibration split, in the fixed seeded candidate order. Require at
least 16 items per band per split. Items with verifier-unknown attempts are excluded and listed.
The panels are frozen here; evaluation uses independent draws (EVAL_SEED streams), never these
screening samples. Writes the calibration bundle (pools calib_train, calib_val) + summary.
Near-duplicate templates of study items cannot occur: duplicate groups (Jaccard >= 0.7) were
assigned to one split as a whole before any screening (qis.data.splits).
"""
import argparse
import json
import os

BANDS = {"low": (1, 5), "high": (6, 12)}
MIN_PER_BAND = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="calib16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pools, summary = {}, {}
    for sp in ("calib_train", "calib_val"):
        order = [json.loads(line) for line in open(os.path.join(a.cand, f"{sp}.jsonl"))]
        cnt = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(a.screen, sp, f"{a.stage}_counts.jsonl"))}
        panel, info = [], {b: 0 for b in BANDS}
        held = []
        for it in order:
            c = cnt.get(it["id"])
            if c is None:
                continue
            if c["unknown"]:
                held.append(it["id"]); continue
            for b, (lo, hi) in BANDS.items():
                if lo <= c["c"] <= hi:
                    panel.append({**it, "band": b, "baseline_c16": c["c"]}); info[b] += 1
        pools[sp] = panel
        hist = {}
        for c in cnt.values():
            hist[c["c"]] = hist.get(c["c"], 0) + 1
        summary[sp] = {"screened": len(cnt), "panel": len(panel), "per_band": info, "held_unknown": held,
                       "c16_hist": dict(sorted(hist.items())), "ok": all(v >= MIN_PER_BAND for v in info.values())}
    json.dump({"pools": pools, "note": "calibration panels v1 (frozen); bands from calib16 screening"},
              open(os.path.join(a.out, "bundle.json"), "w"))
    json.dump(summary, open(os.path.join(a.out, "panels_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if not all(s["ok"] for s in summary.values()):
        raise SystemExit("calibration panel minimum (16 per band per split) not met")


if __name__ == "__main__":
    main()
