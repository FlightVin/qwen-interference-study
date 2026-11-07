"""Learning curves from per-response training records (read-only; run in a CPU job).

Per run and round, separately for each role (hard_common, hard_extra, aux) and for hard (= hard_common +
hard_extra): groups all-wrong / mixed / all-correct, success, capped/total, mean tokens, and the round's cap
(from metrics `max_tokens`, 8,192 when absent). For each seed with both R and U: per matched round on the shared
hard schedule (role hard_common: identical prompts at identical rounds) the R and U values side by side and their
difference, plus cumulative sums. Every round is kept (early curves are preserved).
Usage: python -m qis.analysis.curves RUNS_DIR OUT.json
"""
import glob
import gzip
import json
import os
import re
import sys

N_RESP = 8


def round_rows(d):
    caps = {}
    mp = os.path.join(d, "metrics.jsonl")
    if os.path.exists(mp):
        for line in open(mp):
            if '"kind": "round"' in line:
                r = json.loads(line)
                caps[r["round"]] = r.get("max_tokens", 8192)
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "records", "train_r*.jsonl.gz"))):
        rnd = int(re.search(r"train_r(\d+)", p).group(1)) + 1          # 1-based round, as in metrics
        try:
            recs = [json.loads(l) for l in gzip.open(p, "rt")]
        except (EOFError, OSError):
            continue
        by_slot = {}
        for r in recs:
            by_slot.setdefault((r["role"], r["slot"]), []).append(r)
        row = {"cap": caps.get(rnd, 8192)}
        for role_key, sel in (("hard_common", lambda x: x == "hard_common"), ("hard_extra", lambda x: x == "hard_extra"),
                              ("aux", lambda x: x == "aux"), ("hard", lambda x: x.startswith("hard"))):
            groups = [g for (role, _), g in by_slot.items() if sel(role)]
            if not groups:
                continue
            c = [sum(x["reward"] == 1 for x in g) for g in groups]
            resp = [x for g in groups for x in g]
            row[role_key] = {"groups": len(groups), "all_wrong": sum(v == 0 for v in c), "mixed": sum(0 < v < N_RESP for v in c),
                             "all_correct": sum(v == N_RESP for v in c), "success": sum(c) / len(resp),
                             "capped": sum(x["finish"] == "length" for x in resp), "total": len(resp),
                             "mean_tokens": sum(x["n_tokens"] for x in resp) / len(resp)}
        rows[rnd] = row
    return rows


def main(runs, out):
    res = {}
    for d in sorted(glob.glob(os.path.join(runs, "qis-main-*"))):
        res[os.path.basename(d)] = round_rows(d)
    paired = {}
    for run in res:
        m = re.match(r"qis-main-R-s(\d+)-(\w+)", run)
        u = run.replace("-R-", "-U-")
        if m and u in res:
            common = sorted(set(res[run]) & set(res[u]))
            per, cum = [], {"R": [0, 0, 0], "U": [0, 0, 0]}
            for rnd in common:
                a, b = res[run][rnd].get("hard_common"), res[u][rnd].get("hard_common")
                if not (a and b):
                    continue
                per.append({"round": rnd, "cap_R": res[run][rnd]["cap"], "cap_U": res[u][rnd]["cap"],
                            "R_success": a["success"], "U_success": b["success"], "diff_success": a["success"] - b["success"],
                            "R_mixed": a["mixed"], "U_mixed": b["mixed"], "R_capped": a["capped"], "U_capped": b["capped"]})
                for k, v in (("R", a), ("U", b)):
                    cum[k][0] += v["success"] * v["total"]; cum[k][1] += v["capped"]; cum[k][2] += v["total"]
            paired[f"s{m.group(1)}-{m.group(2)}"] = {
                "matched_rounds": len(per), "per_round": per,
                "cumulative": {k: {"success": s / t if t else None, "capped": c, "total": t} for k, (s, c, t) in cum.items()}}
    json.dump({"runs": res, "R_vs_U_shared_hard": paired}, open(out, "w"))
    for k, v in paired.items():
        print(k, v["matched_rounds"], json.dumps(v["cumulative"]))
    for run, rows in res.items():
        last = max(rows) if rows else None
        if last:
            print(run, last, {r: {kk: rows[last][r][kk] for kk in ("all_wrong", "mixed", "all_correct")} for r in rows[last] if r != "cap"})


if __name__ == "__main__":
    main(*sys.argv[1:3])
