"""A26 x-transfer analysis (stdlib + numpy; own code; laptop-safe: reads only the JSON files below).

  python transfer_analysis.py taxonomy  TAXONOMY_JSON OUT_JSON
  python transfer_analysis.py control   TAXONOMY_JSON CONTROL_EXTRACT_JSON PANEL_JSON OUT_JSON

taxonomy: per run x round x panel class shares; probe gains r0 -> r128 with paired problem bootstrap; gain decomposition by
          failure class; per-problem J_U/J_R changes against the pooled round-0 base (6 runs x 8 = 48 attempts per problem).
control:  pre-registered analysis of research/audit-a26/transfer/PREREG.md §4-§6. Prints markdown tables to stdout.
"""
import json
import sys

import numpy as np

RUNS = ["H-s101", "H-s202", "R-s101", "R-s202", "U-s101", "U-s202"]
ARMS = ["H", "R", "U"]
CLASSES = ["correct", "unknown", "truncated", "no_box", "box_unbalanced", "multi_box_wrong", "single_box_wrong"]
B, SEED = 10_000, 2026


def boot(diff):
    """95% percentile interval of the mean of per-problem differences, problems resampled with replacement."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(tax, run, rnd, pn):
    return tax["cells"][f"qis-main-{run}-sym|{rnd}|{pn}"]


def fmt(x, d=3):
    return "—" if x is None else f"{x:+.{d}f}" if isinstance(x, float) and x < 0 else f"{x:.{d}f}"


def probe_gain(tax, run, pn, key_hi="c", r_lo=0, r_hi=128):
    lo, hi = cell(tax, run, r_lo, pn)["per_problem"], cell(tax, run, r_hi, pn)["per_problem"]
    ids = list(lo)
    return boot([hi[i][key_hi] / hi[i]["n"] - lo[i]["c"] / lo[i]["n"] for i in ids])


def do_taxonomy(tax_path, out_path):
    tax = json.load(open(tax_path))
    out = {"shares": {}, "gains": {}, "decomposition": {}, "per_problem": {}}
    print("## Class shares (attempts; share of all attempts)\n")
    for pn in ("J_U", "J_R", "J_H"):
        print(f"### {pn}\n\n| run | round | n | pass@1 | truncated | no box/unbalanced | multi-box wrong | single-box wrong | lenient rescues (finished any box / truncated any box / last number) | correct ≤ 8,192 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for run in RUNS:
            for rnd in (0, 32, 64, 96, 128):
                c = cell(tax, run, rnd, pn)
                n, k, L = c["attempts"], c["classes"], c["lenient"]
                row = {"n": n, **{x: k[x] / n for x in CLASSES}, "lenient": L, "correct_within_8192": c["correct_within_8192"] / n}
                out["shares"][f"{run}|{rnd}|{pn}"] = row
                print(f"| {run} | {rnd} | {n} | {k['correct']/n:.3f} | {k['truncated']/n:.3f} | {(k['no_box']+k['box_unbalanced'])/n:.3f} | "
                      f"{k['multi_box_wrong']/n:.3f} | {k['single_box_wrong']/n:.3f} | {L['finished_any_box']} / {L['truncated_any_box']} / "
                      f"{L['finished_last_num']} | {c['correct_within_8192']/n:.3f} |")
        print()
    print("## Probe gains, round 0 -> 128 (pass@1 points; paired problem bootstrap 95%)\n")
    print("| run | ΔJ_U | ΔJ_U (r128 correct ≤ 8,192) | ΔJ_R | ΔJ_R (≤ 8,192) | ΔJ_H | ΔJ_H (≤ 8,192) |\n|---|---|---|---|---|---|---|")
    for run in RUNS:
        g = {}
        for pn in ("J_U", "J_R", "J_H"):
            g[pn] = probe_gain(tax, run, pn)
            g[pn + "_8k"] = probe_gain(tax, run, pn, key_hi="c_within_8192")
        out["gains"][run] = g
        print("| " + run + " | " + " | ".join(f"{g[k][0]:+.3f} [{g[k][1]:+.3f}, {g[k][2]:+.3f}]"
                                             for k in ("J_U", "J_U_8k", "J_R", "J_R_8k", "J_H", "J_H_8k")) + " |")
    print("\n## Where the gain comes from (change in share, r128 − r0; negative = fewer failures of that class)\n")
    print("| run | panel | Δ correct | Δ truncated | Δ no box | Δ multi-box wrong | Δ single-box wrong |\n|---|---|---|---|---|---|---|")
    for pn in ("J_U", "J_R", "J_H"):
        for run in RUNS:
            a, b = out["shares"][f"{run}|0|{pn}"], out["shares"][f"{run}|128|{pn}"]
            d = {x: b[x] - a[x] for x in CLASSES}
            d["no_box_all"] = d["no_box"] + d["box_unbalanced"]
            out["decomposition"][f"{run}|{pn}"] = d
            print(f"| {run} | {pn} | {d['correct']:+.3f} | {d['truncated']:+.3f} | {d['no_box_all']:+.3f} | "
                  f"{d['multi_box_wrong']:+.3f} | {d['single_box_wrong']:+.3f} |")
    for pn in ("J_U", "J_R"):
        ids = list(cell(tax, RUNS[0], 0, pn)["per_problem"])
        base = {i: sum(cell(tax, r, 0, pn)["per_problem"][i]["c"] for r in RUNS) / 48 for i in ids}
        end = {r: {i: cell(tax, r, 128, pn)["per_problem"][i]["c"] / 8 for i in ids} for r in RUNS}
        arm = {a: {i: (end[f"{a}-s101"][i] + end[f"{a}-s202"][i]) / 2 - base[i] for i in ids} for a in ARMS}
        corr = {f"{x}-{y}": float(np.corrcoef([arm[x][i] for i in ids], [arm[y][i] for i in ids])[0, 1])
                for x, y in (("H", "R"), ("H", "U"), ("R", "U"))}
        # split-half check: does a seed-101 improvement predict the seed-202 improvement of another arm?
        out["per_problem"][pn] = {"pooled_base": base, "r128": end, "arm_delta": arm, "corr_arm_delta": corr}
        print(f"\n## {pn}: per-problem change, arm mean of r128 c/8 − pooled round-0 base (48 attempts)\n")
        print("Pearson correlation of per-problem arm deltas: " + ", ".join(f"{k} {v:.2f}" for k, v in corr.items()) + "\n")
        print("| problem | base (of 48) | H | R | U |\n|---|---|---|---|---|")
        for i in sorted(ids, key=lambda i: base[i]):
            if base[i] < 0.95:
                print(f"| `{i}` | {round(base[i]*48)} ({base[i]:.2f}) | {arm['H'][i]:+.2f} | {arm['R'][i]:+.2f} | {arm['U'][i]:+.2f} |")
        hi = [i for i in ids if base[i] >= 0.95]
        print(f"\n{len(hi)} of {len(ids)} problems have pooled base ≥ 0.95 (no room); their mean arm deltas: "
              + ", ".join(f"{a} {np.mean([arm[a][i] for i in hi]):+.3f}" for a in ARMS))
    json.dump(out, open(out_path, "w"), indent=1)


def do_control(tax_path, ext_path, panel_path, out_path):
    tax = json.load(open(tax_path))
    ext = json.load(open(ext_path))
    panel = json.load(open(panel_path))["pools"]["control"]
    subj = {x["id"]: x["subject"] for x in panel}
    ids = [x["id"] for x in panel]
    pol = {"base": "base", **{r: f"{r}-r128" for r in RUNS}}
    missing = [p for p in pol.values() if "per_problem" not in ext.get(p, {})]
    if missing:
        print("MISSING policies:", missing)
    out = {"policies": {}, "gains": {}, "contrasts": {}, "decision": {}, "missing": missing}

    def stats(pp, sel):
        n = sum(pp[i]["n"] for i in sel)
        cor = sum(pp[i]["correct"] for i in sel)
        tr = sum(pp[i]["truncated"] for i in sel)
        fin = n - tr
        return {"pass1": float(np.mean([pp[i]["correct"] / pp[i]["n"] for i in sel])), "truncated": tr / n,
                "acc_finished": cor / fin if fin else None, "attempts": n,
                "no_box": sum(pp[i]["no_box"] + pp[i]["box_unbalanced"] for i in sel) / n,
                "multi_box_wrong": sum(pp[i]["multi_box_wrong"] for i in sel) / n,
                "single_box_wrong": sum(pp[i]["single_box_wrong"] for i in sel) / n,
                "unknown": sum(pp[i]["unknown"] for i in sel),
                "pass1_within_8192": float(np.mean([pp[i]["c_within_8192"] / pp[i]["n"] for i in sel]))}
    subjects = sorted(set(subj.values()))
    print("## Control panel per policy (64 problems x 8 attempts, cap 12,288)\n")
    print("| policy | GPU | pass@1 | truncated | acc. among finished | no box | multi-box wrong | single-box wrong | " +
          " | ".join(subjects) + " |\n|---|---|---|---|---|---|---|---|" + "---|" * len(subjects))
    for name, p in pol.items():
        if p in missing:
            continue
        pp = ext[p]["per_problem"]
        st = {"all": stats(pp, ids), **{s: stats(pp, [i for i in ids if subj[i] == s]) for s in subjects}}
        st["gpu"] = ext[p].get("gpu")
        out["policies"][name] = st
        a = st["all"]
        print(f"| {name} | {st['gpu'] or 'see gpu.txt'} | {a['pass1']:.3f} | {a['truncated']:.3f} | {a['acc_finished']:.3f} | {a['no_box']:.3f} | "
              f"{a['multi_box_wrong']:.3f} | {a['single_box_wrong']:.3f} | " + " | ".join(f"{st[s]['pass1']:.3f}" for s in subjects) + " |")
    if "base" in missing:
        json.dump(out, open(out_path, "w"), indent=1)
        return
    bpp = ext["base"]["per_problem"]
    print("\n## Gains vs base (paired problem bootstrap 95%) and comparison with the in-run probe gains\n")
    print("| run | Δ control | Δ control by subject (" + ", ".join(subjects) + ") | ΔJ_U | ΔJ_U ≤ 8,192 | ΔJ_R | ρ = Δcontrol/ΔJ_U |\n|---|---|---|---|---|---|---|")
    for r in RUNS:
        if pol[r] in missing:
            continue
        pp = ext[pol[r]]["per_problem"]
        g = boot([pp[i]["correct"] / pp[i]["n"] - bpp[i]["correct"] / bpp[i]["n"] for i in ids])
        gs = {s: boot([pp[i]["correct"] / 8 - bpp[i]["correct"] / 8 for i in ids if subj[i] == s]) for s in subjects}
        ju, ju8, jr = probe_gain(tax, r, "J_U"), probe_gain(tax, r, "J_U", key_hi="c_within_8192"), probe_gain(tax, r, "J_R")
        rho = g[0] / ju[0] if ju[0] else None
        out["gains"][r] = {"control": g, "control_by_subject": gs, "J_U": ju, "J_U_8k": ju8, "J_R": jr, "rho": rho}
        print(f"| {r} | {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] | " + ", ".join(f"{gs[s][0]:+.3f}" for s in subjects) +
              f" | {ju[0]:+.3f} [{ju[1]:+.3f}, {ju[2]:+.3f}] | {ju8[0]:+.3f} | {jr[0]:+.3f} | {fmt(rho, 2)} |")
    print("\n## Arm contrasts on control, per seed (descriptive)\n\n| seed | H − R | H − U | R − U |\n|---|---|---|---|")
    for s in ("s101", "s202"):
        row = {}
        for x, y in (("H", "R"), ("H", "U"), ("R", "U")):
            px, py = ext.get(pol[f"{x}-{s}"], {}).get("per_problem"), ext.get(pol[f"{y}-{s}"], {}).get("per_problem")
            row[f"{x}-{y}"] = boot([px[i]["correct"] / 8 - py[i]["correct"] / 8 for i in ids]) if px and py else None
        out["contrasts"][s] = row
        print(f"| {s} | " + " | ".join(f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if v else "—" for v in row.values()) + " |")
    print("\n## Pre-registered decision (PREREG §6)\n")
    for a in ARMS:
        gg = [out["gains"].get(f"{a}-{s}") for s in ("s101", "s202")]
        if not all(gg):
            out["decision"][a] = "incomplete"
            continue
        dmean = np.mean([x["control"][0] for x in gg])
        jmean = np.mean([x["J_U"][0] for x in gg])
        rho = dmean / jmean if jmean else None
        general = all(x["control"][0] > 0 for x in gg) and any(x["control"][1] > 0 for x in gg) and rho is not None and rho >= 0.5
        specific = (dmean <= 0.02 or all(x["control"][1] <= 0 for x in gg)) and rho is not None and rho < 0.25
        verdict = "general transfer" if general else "specific transfer" if specific else "mixed"
        # E1 vs E2 split of the seed-mean control gain
        bs = out["policies"]["base"]["all"]
        es = [out["policies"][f"{a}-{s}"]["all"] for s in ("s101", "s202")]
        fmt_drop = np.mean([(bs["truncated"] - e["truncated"]) + (bs["no_box"] - e["no_box"]) +
                            (bs["multi_box_wrong"] - e["multi_box_wrong"]) for e in es])
        single_drop = np.mean([bs["single_box_wrong"] - e["single_box_wrong"] for e in es])
        out["decision"][a] = {"seed_mean_control_gain": float(dmean), "seed_mean_JU_gain": float(jmean), "rho": rho,
                              "verdict": verdict, "format_drop": float(fmt_drop), "single_box_wrong_drop": float(single_drop)}
        print(f"- {a}: seed-mean Δcontrol {dmean:+.3f}, seed-mean ΔJ_U {jmean:+.3f}, ρ {fmt(rho, 2)} → **{verdict}**; "
              f"drop in truncated+no-box+multi-box {fmt_drop:+.3f}, drop in single-box wrong {single_drop:+.3f}")
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "taxonomy":
        do_taxonomy(*sys.argv[2:4])
    elif sys.argv[1] == "control":
        do_control(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
