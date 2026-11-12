#!/usr/bin/env python3
"""A27 / D166: H-s202 r128 sealed test rerun on L40S only, against the original mixed-GPU endpoint.

Reads the merged rerun summary (harvested to slurm-agentic-engineering/results/qis-cgpu-H-s202-k*/summary.json) and the
audited per-problem counts (reports/Nov-12-2026-initial-experiments/tables/v3/perproblem_counts.json). Writes reports/tables/v4/t11_cgpu_h_s202.{json,md}.
Interval: 95% paired problem bootstrap, 10,000 draws, the same resampled problems for both policies.
Usage: python3 cgpu_h_s202.py --root <repo>
"""
import argparse, glob, json, os, random


def boot(a, b, n_att, draws=10000, seed=20261112):
    rng = random.Random(seed)
    m = len(a)
    d = [(a[i] - b[i]) / n_att for i in range(m)]
    est = sum(d) / m
    vals = []
    for _ in range(draws):
        s = 0.0
        for _ in range(m):
            s += d[rng.randrange(m)]
        vals.append(s / m)
    vals.sort()
    return est, vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    cnt = json.load(open(os.path.join(root, "reports/Nov-12-2026-initial-experiments/tables/v3/perproblem_counts.json")))["test"]
    order, n = cnt["order"], cnt["n"]
    cells = {k: v["c"] for k, v in cnt["cells"].items()}
    found = [p for p in glob.glob(os.path.join(root, "slurm-agentic-engineering/results/qis-cgpu-H-s202-k*/summary.json"))]
    assert len(found) == 1, found
    summ = json.load(open(found[0]))
    pp = {r["id"]: r for r in summ["per_problem"]}
    assert set(pp) == set(order) and all(pp[i]["n"] == n for i in order), "panel or n mismatch"
    new = [pp[i]["c"] for i in order]
    gpus = {}
    for g in glob.glob(os.path.join(root, "slurm-agentic-engineering/results/qis-cgpu-H-s202-k*/gpu_name.txt")):
        gpus[g.split("/")[-2]] = open(g).read().strip()
    out = {"source": os.path.relpath(found[0], root), "seed_label": summ.get("seed_label"),
           "ckpt_model_sha256": summ.get("ckpt_model_sha256"), "gpus": gpus,
           "pass1_new": sum(new) / (n * len(order)), "pass1_summary": summ["pass1"],
           "pass1_orig": sum(cells["H-s202"]) / (n * len(order)), "truncated_new": summ.get("truncated"),
           "correct_new": sum(new), "correct_orig": sum(cells["H-s202"])}
    out["new_minus_orig"] = boot(new, cells["H-s202"], n)
    # problems on L40S in the original (6-way interleaved shards k0-k2 = index mod 6 in {0,1,2}) vs RTX A6000 (k3-k5)
    for name, ks in (("orig_L40S_problems", (0, 1, 2)), ("orig_A6000_problems", (3, 4, 5))):
        idx = [i for i in range(len(order)) if i % 6 in ks]
        a = [new[i] for i in idx]; b = [cells["H-s202"][i] for i in idx]
        out[name] = {"problems": len(idx), "new_correct": sum(a), "orig_correct": sum(b),
                     "identical_counts": sum(1 for x, y in zip(a, b) if x == y), "diff": boot(a, b, n)}
    rows = []
    for A, B in (("R-s202", "H-s202"), ("U-s202", "H-s202"), ("cgpu-R-s202", "H-s202"), ("cgpu-U-s202", "H-s202")):
        for hname, h in (("original mixed-GPU H", cells["H-s202"]), ("L40S rerun H", new)):
            e, lo, hi = boot(cells[A], h, n)
            rows.append({"A": A, "B_version": hname, "A_pass1": sum(cells[A]) / (n * len(order)),
                         "B_pass1": sum(h) / (n * len(order)), "diff": e, "lo": lo, "hi": hi})
    out["contrasts"] = rows
    tdir = os.path.join(root, "reports/Nov-12-2026-initial-experiments/tables/v4")
    json.dump(out, open(os.path.join(tdir, "t11_cgpu_h_s202.json"), "w"), indent=1)
    f = lambda x: f"{100 * x:.1f}"
    g = lambda x: (f"{100 * x:+.1f}").replace("-", "−")
    L = ["# t11 — H-s202 r128 sealed test: original (L40S + RTX A6000 shards) vs L40S-only rerun (D166)", "",
         f"128 problems × 32 attempts, cap 12,288, seed label `{out['seed_label']}`. Rerun GPUs: {sorted(set(gpus.values()))}. "
         "Intervals: 95% paired problem bootstrap, 10,000 draws. Source: `" + out["source"] + "`.", "",
         f"Original: {out['correct_orig']:,} / 4,096 correct ({f(out['pass1_orig'])}%). Rerun: {out['correct_new']:,} / 4,096 "
         f"({f(out['pass1_new'])}%). Rerun − original: {g(out['new_minus_orig'][0])} points "
         f"[{g(out['new_minus_orig'][1])}, {g(out['new_minus_orig'][2])}].", "",
         "| Original shard GPU | Problems | Original correct | Rerun correct | Problems with identical counts | Rerun − original (points) | 95% interval |",
         "|---|---|---|---|---|---|---|"]
    for name, lab in (("orig_L40S_problems", "L40S"), ("orig_A6000_problems", "RTX A6000")):
        r = out[name]
        L.append(f"| {lab} | {r['problems']} | {r['orig_correct']:,} | {r['new_correct']:,} | {r['identical_counts']} | "
                 f"{g(r['diff'][0])} | [{g(r['diff'][1])}, {g(r['diff'][2])}] |")
    L += ["", "| Contrast | H version | A pass@1 (%) | H pass@1 (%) | A − H (points) | 95% interval |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['A'].replace('cgpu-', '')}{' (L40S rerun)' if r['A'].startswith('cgpu') else ''} − H-s202 | "
                 f"{r['B_version']} | {f(r['A_pass1'])} | {f(r['B_pass1'])} | {g(r['diff'])} | [{g(r['lo'])}, {g(r['hi'])}] |")
    open(os.path.join(tdir, "t11_cgpu_h_s202.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
