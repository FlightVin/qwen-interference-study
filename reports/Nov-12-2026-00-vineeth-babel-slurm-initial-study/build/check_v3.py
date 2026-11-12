#!/usr/bin/env python3
"""Value check for the report v3 source tables (A26 section 6). Stdlib only.

Compares every number in reports/tables/v3/t*.json with the corresponding audit pass-B value (the build input) and,
where pass A computed the same quantity, with the pass-A value (pass-a/data/numbers.json, pass-a/data/eval_map.json,
tools/reconcile/compare.json column A). Intervals that the build computed itself (absolute pass@1, per-checkpoint
length gains) have no pass-A/B counterpart; they are recomputed here with an independent stdlib bootstrap
(random.Random, different generator) and must agree within Monte-Carlo tolerance.

Usage: python3 check_v3.py [--root REPO] [--tables DIR]
Prints one line per mismatch and a summary per table. Exit code 1 if any mismatch.
"""
import argparse
import json
import os
import random
import sys

EXACT = 1e-12      # point values: must be identical up to float noise
CI_TOL_A = 0.01    # pass-A intervals used other bootstrap streams: allow 1 point of Monte-Carlo difference
CI_TOL_BOOT = 0.01  # independent stdlib bootstrap vs build interval
BOOT_N = 4000

stats = {}
mism = []


def rec(table, kind, ok, what):
    s = stats.setdefault(table, {})
    k = s.setdefault(kind, [0, 0])
    k[0] += 1
    if not ok:
        k[1] += 1
        mism.append(f"[{table}] {kind}: {what}")


def eq(table, kind, got, ref, what, tol=EXACT):
    if ref is None:
        return
    ok = abs(float(got) - float(ref)) <= tol
    rec(table, kind, ok, f"{what}: table {got} vs {ref} (|diff| {abs(float(got) - float(ref)):.3g}, tol {tol})")


def eq_ci(table, kind, got, ref, what, tol):
    for g, r, side in zip(got, ref, ("lo", "hi")):
        eq(table, kind, g, r, f"{what} {side}", tol)


def boot(xs, seed):
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT_N))
    return [ms[int(0.025 * BOOT_N)], ms[int(0.975 * BOOT_N) - 1]]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--tables")
    a = ap.parse_args()
    T = a.tables or os.path.join(a.root, "reports", "tables", "v3")
    A26 = os.path.join(a.root, "research", "audit-a26")
    L = lambda *p: json.load(open(os.path.join(A26, *p)))
    B = {k: L("pass-b", "data", k + ".json") for k in
         ("eval_test", "eval_stress", "eval_length", "train_curves", "train_diag", "train_rewards")}
    NA = L("pass-a", "data", "numbers.json")
    REC = NA["recomputed"]
    CLAIM = {c["id"]: c for c in NA["claims"]}
    EMAP = {e["evaluation"]: e for e in L("pass-a", "data", "eval_map.json")["evaluations"]}
    _cmp = L("tools", "reconcile", "compare.json")

    class _Pref(dict):
        """compare.json lookup by exact quantity or unique prefix."""
        def __getitem__(self, q):
            hits = [r for r in _cmp if r["quantity"] == q] or [r for r in _cmp if r["quantity"].startswith(q)]
            if len(hits) != 1:
                raise KeyError(f"{q}: {len(hits)} matches")
            return hits[0]

        def __contains__(self, q):
            try:
                self[q]
                return True
            except KeyError:
                return False
    CMP = _Pref()
    tab = lambda n: json.load(open(os.path.join(T, n + ".json")))["rows"]
    cnt = json.load(open(os.path.join(T, "perproblem_counts.json")))

    # ---- t01 endpoints
    t = "t01_endpoints"
    for r in tab(t):
        cell = r["cell"]
        m = B["eval_test"]["cells"][cell]
        eq(t, "B point", r["pass1"], m["pass1"], f"{cell} pass1")
        eq(t, "B point", r["correct"], m["correct"], f"{cell} correct")
        eq(t, "B point", r["truncation_share"], m["truncation_share"], f"{cell} trunc")
        eq(t, "B point", r["pass8_unbiased"], m["pass8_unbiased"], f"{cell} pass8")
        akey = {"H-s202-fb-r116": "test.H-s202-fb"}.get(cell, f"test.{cell}")
        if cell.startswith("cgpu-"):
            eq(t, "A point", r["pass1"], REC[f"cgpu.{cell[5:]}.pass1"], f"{cell} pass1 (A)")
        else:
            eq(t, "A point", r["pass1"], REC[akey + ".pass1"], f"{cell} pass1 (A)")
            eq(t, "A point", r["pass8_unbiased"], REC[akey + ".pass8"], f"{cell} pass8 (A)")
        c = cnt["test"]["cells"][cell]["c"]
        x = [v / 32 for v in c]
        eq(t, "counts", sum(x) / len(x), m["pass1"], f"{cell} per-problem mean")
        eq_ci(t, "indep. bootstrap", r["pass1_ci95"], boot(x, 1), f"{cell} pass1 CI", CI_TOL_BOOT)
        if "gain_over_base" in r:
            arm, s = cell.split("-s")
            g = B["eval_test"]["contrasts"][f"{arm}-base s{s}"]["pass1"]
            eq(t, "B point", r["gain_over_base"], g["diff"], f"{cell} gain")
            eq_ci(t, "B interval", r["gain_over_base_ci95"], g["ci95"], f"{cell} gain CI", EXACT)
            eq(t, "A point", r["gain_over_base"], CMP[f"{cell} - base (point)"]["A"][0], f"{cell} gain (A)")

    # ---- t02 contrasts
    t = "t02_contrasts_test"
    for r in tab(t):
        if r.get("kind") == "two_seed":
            k = r["contrast"]
            eq(t, "B point", r["two_seed_mean"], B["eval_test"]["two_seed_mean"][k], f"{k} mean")
            eq(t, "A point", r["two_seed_mean"], REC[f"test.{k}.mean"]["est"], f"{k} mean (A)")
            continue
        s = r["seed"]
        if r["contrast"].startswith("R-U (L40S"):
            c = B["eval_test"]["contrasts"][f"cgpuR-cgpuU s{s}"]["pass1"]
            ca = CMP[f"cgpu R-U s{s} [95%]"]["A"]
            lab = f"cgpu R-U s{s}"
        else:
            c = B["eval_test"]["contrasts"][f"{r['contrast']} s{s}"]["pass1"]
            ra = REC[f"test.{r['contrast']}.s{s}"]
            ca = [ra["est"], ra["lo"], ra["hi"]]
            lab = f"{r['contrast']} s{s}"
        eq(t, "B point", r["diff"], c["diff"], lab)
        eq_ci(t, "B interval", r["ci95"], c["ci95"], lab, EXACT)
        eq(t, "A point", r["diff"], ca[0], lab + " (A)")
        eq_ci(t, "A interval", r["ci95"], ca[1:], lab + " (A)", CI_TOL_A)
        eq(t, "B point", r["pass1_A"] - r["pass1_B"], r["diff"], lab + " A−B consistency")

    # ---- t03 curves
    t = "t03_curves"
    rows = tab(t)
    by = {(r["run"], r["round"], r["panel"]): r for r in rows}
    for r in rows:
        m = B["train_curves"]["curves"][r["run"]][str(r["round"])][r["panel"]]
        eq(t, "B point", r["pass1"], m["pass1"], f"{r['run']} r{r['round']} {r['panel']}")
        eq(t, "B point", r["cap"], m["cap_canonical_row"], f"{r['run']} r{r['round']} {r['panel']} cap")
        per = 128 if r["run"].startswith("H") else 64
        eq(t, "protocol", r["cum_hard_attempts"], per * r["round"], f"{r['run']} r{r['round']} hard attempts")
        x = [v / m["n_per_problem"][0] for v in m["per_problem_c"]]
        eq_ci(t, "indep. bootstrap", r["pass1_ci95"], boot(x, 2), f"{r['run']} r{r['round']} {r['panel']} CI",
              CI_TOL_BOOT)
    for s in (101, 202):
        for rnd in (0, 32, 64, 96, 128):
            ca = CLAIM[f"3.2.dev.s{s}.r{rnd}"]["recomputed_unrounded"]
            eq(t, "A point", by[(f"R-s{s}", rnd, "dev")]["pass1"], ca[0], f"R-s{s} r{rnd} dev (A)")
            eq(t, "A point", by[(f"U-s{s}", rnd, "dev")]["pass1"], ca[1], f"U-s{s} r{rnd} dev (A)")
        for pan, pn in (("dev", "dev"), ("probe_J_H", "J_H"), ("probe_J_R", "J_R"), ("probe_J_U", "J_U")):
            ref = CMP[f"s{s} {pn}: R-U at rounds 0/32/64/96/128"]["A"]
            for i, rnd in enumerate((0, 32, 64, 96, 128)):
                d = by[(f"R-s{s}", rnd, pan)]["pass1"] - by[(f"U-s{s}", rnd, pan)]["pass1"]
                eq(t, "A derived R−U", d, ref[i], f"s{s} {pn} r{rnd} R−U (A)")
    ref = CMP["H-s202 (a2-r60) dev r64, r96, r128"]["A"]
    for i, rnd in enumerate((64, 96, 128)):
        eq(t, "A point", by[("H-s202", rnd, "dev")]["pass1"], ref[i], f"H-s202 r{rnd} dev (A)")

    # ---- t04 diagnostic
    t = "t04_diag"
    probes = ("probe_J_H", "probe_J_R", "probe_J_U")
    for r in tab(t):
        lab = r["label"]
        ev = B["train_diag"][lab]["evals"][r["branch"]]
        for i, p in enumerate(probes):
            eq(t, "B point", r[p], ev[p]["pass1"], f"{lab} {r['branch']} {p}")
        ad = REC[f"diag.{lab}"]
        if r["type"] == "reference":
            if r["branch"] == "none":
                eq(t, "A point", r["probe_J_H"], ad["none"][0], f"{lab} none J_H (A)")
            else:
                eq(t, "A point", r["probe_J_H"], ad["ref"][0 if r["branch"] == "none_rerun" else 1],
                   f"{lab} {r['branch']} J_H (A)")
            continue
        draw = next(d for d in ad["draws"][r["type"]] if d[0] == r["draw_seed"])
        for i, p in enumerate(probes):
            eq(t, "A point", r[p], draw[1 + i], f"{lab} {r['branch']} {p} (A)")
            eq(t, "B point", r["d_vs_none_" + p],
               B["train_diag"][lab]["branch_types"][p]["per_draw"][r["type"]][r["draw_seed"] - 4101],
               f"{lab} {r['branch']} ΔJ {p}")
        eq(t, "B point", r["grad_norm_preclip"], ev["update_row"]["grad_norm_preclip"], f"{lab} {r['branch']} gnorm")
    for lab in ("r0", "r64"):
        rows = [r for r in tab(t) if r["label"] == lab and r["type"] != "reference"]
        means = [sum(r["d_vs_none_probe_J_H"] for r in rows if r["type"] == ty) / 4
                 for ty in ("hard", "related", "unrelated")]
        ref = CMP[f"{lab}: mean ΔJ_H vs none (hard, related, unrelated)"]["A"]
        for ty, mval, rv in zip(("hard", "related", "unrelated"), means, ref):
            eq(t, "A derived mean", mval, rv, f"{lab} mean ΔJ_H {ty} (A)")
        rr = next(r for r in tab(t) if r["label"] == lab and r["branch"] == "none_rerun")
        eq(t, "A point", rr["d_vs_none_probe_J_H"], CMP[f"{lab}: none_rerun - none (J_H)"]["A"][0],
           f"{lab} rerun − none (A)")
        mo = sorted(r["branch"] for r in rows if r["momentum_only"])
        want = ["related-4101", "related-4102", "unrelated-4103"] if lab == "r64" else []
        rec(t, "momentum-only set", mo == want, f"{lab} momentum-only {mo} vs {want}")

    # ---- t05 length
    t = "t05_length"
    for r in tab(t):
        k = B["eval_length"][r["panel"]][r["run"]]["counts"]
        lab = f"{r['panel']} {r['run']}"
        for f, g in (("pass1_12288", "pass1_12288"), ("pass1_16384", "pass1_16384"), ("capped_at_12288", "capped_at_12288"),
                     ("recovered", "recovered"), ("finished_wrong", "finished_wrong"), ("still_capped", "still_capped")):
            eq(t, "B point", r[f], k[g], f"{lab} {f}")
        ca = CMP[f"{r['panel']} {r['run']}: capped, recovered, finished wrong, still capped"]["A"]
        for f, v in zip(("capped_at_12288", "recovered", "finished_wrong", "still_capped", "pass1_12288",
                         "pass1_16384"), ca):
            eq(t, "A point", r[f], v, f"{lab} {f} (A)")
        pc = cnt["length"][r["panel"]][r["run"]]
        n = r["n"]
        x = [(b - a_) / n for a_, b in zip(pc["c12288"], pc["c16384"])]
        eq(t, "counts", sum(x) / len(x), r["gain"], f"{lab} gain from counts")
        eq_ci(t, "indep. bootstrap", r["gain_ci95"], boot(x, 3), f"{lab} gain CI", CI_TOL_BOOT)

    # ---- t06 length contrasts
    t = "t06_length_contrasts"
    for r in tab(t):
        c = B["eval_length"]["contrasts"][f"{r['panel']} {r['A']} - {r['B']}"]
        lab = f"{r['panel']} {r['A']}−{r['B']}"
        for f, g in (("12288", "at_12288"), ("16384", "at_16384")):
            eq(t, "B point", r["diff_" + f], c[g]["diff"], f"{lab} {f}")
            eq_ci(t, "B interval", r["ci95_" + f], c[g]["ci95"], f"{lab} {f}", EXACT)
        eq(t, "B point", r["change"], c["change"]["diff"], f"{lab} change")
        eq_ci(t, "B interval", r["change_ci95"], c["change"]["ci95"], f"{lab} change", EXACT)
        for f, q in (("12288", "at 12,288"), ("16384", "at 16,384")):
            ca = CMP[f"{r['panel']} {r['A']} - {r['B']} {q} [95%]"]["A"]
            eq(t, "A point", r["diff_" + f], ca[0], f"{lab} {f} (A)")
            eq_ci(t, "A interval", r["ci95_" + f], ca[1:], f"{lab} {f} (A)", CI_TOL_A)
        ca = CMP[f"{r['panel']} {r['A']} - {r['B']} change [95%]"]["A"]
        eq(t, "A point", r["change"], ca[0], f"{lab} change (A)")
        eq_ci(t, "A interval", r["change_ci95"], ca[1:], f"{lab} change (A)", CI_TOL_A)

    # ---- t07 stress
    t = "t07_stress"
    tiers = cnt["stress"]["tier"]
    for r in tab(t):
        m = B["eval_stress"]["cells"][r["cell"]][r["tier"]]
        lab = f"{r['cell']} {r['tier']}"
        eq(t, "B point", r["pass1"], m["pass1"], lab)
        eq(t, "B point", r["correct"], m["correct"], lab + " correct")
        ca = CMP[f"{r['cell']}: pass@1, SuperHard, Harder, trunc, acc finished, solved"]["A"]
        eq(t, "A point", r["pass1"], {"all": ca[0], "superhard": ca[1], "harder": ca[2]}[r["tier"]], lab + " (A)")
        if r["tier"] == "all":
            eq(t, "A point", r["pass1"], EMAP[f"stress/{r['cell']}"]["pass1"], lab + " (A eval_map)")
            eq(t, "A point", r["truncation_share"], ca[3], lab + " trunc (A)")
        else:
            eq(t, "A point", r["correct"], CMP[f"{r['cell']} {r['tier']}: correct, truncated, solved>=1/8, acc finished"]["A"][0],
               lab + " correct (A)")
        c = cnt["stress"]["cells"][r["cell"]]["c"]
        x = [v / 8 for v, ti in zip(c, tiers) if r["tier"] == "all" or ti == r["tier"]]
        eq_ci(t, "indep. bootstrap", r["pass1_ci95"], boot(x, 4), lab + " CI", CI_TOL_BOOT)

    # ---- t08 stress contrasts
    t = "t08_stress_contrasts"
    for r in tab(t):
        c = B["eval_stress"]["contrasts"][f"{r['contrast']} s{r['seed']} {r['tier']}"]["pass1"]
        lab = f"{r['contrast']} s{r['seed']} {r['tier']}"
        eq(t, "B point", r["diff"], c["diff"], lab)
        eq_ci(t, "B interval", r["ci95"], c["ci95"], lab, EXACT)
        eq(t, "B point", r["pass1_A"] - r["pass1_B"], r["diff"], lab + " A−B consistency")
        q = f"{r['contrast']} s{r['seed']} {r['tier']} [95%]"
        if q in CMP:
            ca = CMP[q]["A"]
            eq(t, "A point", r["diff"], ca[0], lab + " (A)")
            eq_ci(t, "A interval", r["ci95"], ca[1:], lab + " (A)", CI_TOL_A)

    # ---- t09 control (new evaluation: no pass-A/B counterpart for the control panel itself)
    t = "t09_control"
    if os.path.exists(os.path.join(T, t + ".json")):
        ctl = L("transfer", "control.json")
        base = ctl["policies"]["base"]["all"]["pass1"]
        for r in tab(t):
            run = r["run"]
            eq(t, "control.json", r["control_endpoint_pass1"], ctl["policies"][run]["all"]["pass1"], f"{run} endpoint")
            eq(t, "control.json", r["control_gain"], ctl["policies"][run]["all"]["pass1"] - base, f"{run} gain = end − base")
            eq_ci(t, "control.json", r["control_gain_ci95"], ctl["gains"][run]["control"][1:], f"{run} gain CI", EXACT)
            cur = B["train_curves"]["curves"][run]
            for key, pan in (("J_U_gain", "probe_J_U"), ("J_R_gain", "probe_J_R")):
                eq(t, "B derived gain", r[key], cur["128"][pan]["pass1"] - cur["0"][pan]["pass1"], f"{run} {key} (B curves)")

    # ---- summary
    print("table | check kind | values checked | mismatches")
    for tname in sorted(stats):
        for kind, (n, bad) in sorted(stats[tname].items()):
            print(f"{tname} | {kind} | {n} | {bad}")
    tot = sum(n for s in stats.values() for n, _ in s.values())
    bad = sum(b for s in stats.values() for _, b in s.values())
    print(f"TOTAL | all | {tot} | {bad}")
    for m in mism:
        print("MISMATCH", m)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
