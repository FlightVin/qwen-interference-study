#!/usr/bin/env python3
"""Report v3 source tables and figures (A26 section 6).

Two steps, both deterministic:

  extract  (compute node only; stdlib)  reads per-attempt records under $P and writes per-problem correct counts
           for the sealed test, the stress panels and the length extension: tables/v3/perproblem_counts.json.
  build    (numpy + matplotlib)         reads the audited inputs (research/audit-a26/pass-b/data, transfer/control.json)
           plus perproblem_counts.json and writes tables/v3/<name>.{json,md} and figures/v3/<name>.png.

Figures read only the table dicts built here, so every plotted value is in a source table.
Inputs are the audited pass-B data files. Pass-A files are used only by check_v3.py.

Usage:
  python build_v3.py extract --out reports/tables/v3/perproblem_counts.json [--P /data/user_data/vineethb/qis]
  python build_v3.py build --root <repo> [--tables DIR] [--figures DIR]
"""
import argparse
import gzip
import json
import os
import sys

P_DEFAULT = "/data/user_data/vineethb/qis"
BOOT_DRAWS = 10000
BOOT_SEED = 20261111  # same generator and seed as audit pass B (pass-b/tools/b-eval/common.py)

ARMS = ("H", "R", "U")
SEEDS = (101, 202)
RUNS = [f"{a}-s{s}" for a in ARMS for s in SEEDS]  # H-s101, H-s202, R-s101, ...
ROUNDS = (0, 32, 64, 96, 128)
PANELS = ("dev", "probe_J_H", "probe_J_R", "probe_J_U")
PANEL_N = {"dev": 53, "probe_J_H": 32, "probe_J_R": 28, "probe_J_U": 32}

TEST_CELLS = ["base", "H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202", "H-s202-fb-r116",
              "cgpu-R-s101", "cgpu-R-s202", "cgpu-U-s101", "cgpu-U-s202"]
STRESS_CELLS = ["base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "H-s202-r128", "R-s202-r128", "U-s202-r128",
                "R-s101-r64", "H-s202-r64"]
LENGTH_POLS = ["H-s101", "R-s101", "U-s101", "R-s202", "U-s202"]


# ----------------------------------------------------------------------------------------------------------------
# extract (compute only)
# ----------------------------------------------------------------------------------------------------------------
def _records(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def _counts(path, order, n, reward_key="reward", finish_key="finish"):
    idx = {p: i for i, p in enumerate(order)}
    c = [0] * len(order)
    t = [0] * len(order)
    seen = set()
    for r in _records(path):
        k = (r["prompt_id"], int(r["sample"]))
        if k in seen:
            raise SystemExit(f"duplicate attempt {k} in {path}")
        seen.add(k)
        i = idx[r["prompt_id"]]
        c[i] += 1 if r[reward_key] == 1 else 0  # unknown (None) counts as incorrect, as in the evaluation
        if finish_key:
            t[i] += 1 if r[finish_key] == "length" else 0
    if len(seen) != len(order) * n:
        raise SystemExit(f"{path}: {len(seen)} attempts, expected {len(order) * n}")
    return c, t


def extract(P, out):
    test_order = [it["id"] for it in json.load(open(P + "/data/main-v1/target_test_sealed.json"))["pools"]["test"]]
    st_items = json.load(open(P + "/data/stress/stress_v1.json"))["pools"]["stress"]
    st_order = [it["id"] for it in st_items]
    te = P + "/test-eval"
    test_paths = {
        "base": te + "/qis-test-base", "H-s101": te + "/qis-test-H-s101-sym", "R-s101": te + "/qis-test-R-s101-sym",
        "U-s101": te + "/qis-test-U-s101-sym", "H-s202": te + "/qis-test-H-s202-sym",
        "R-s202": te + "/qis-test-R-s202-sym", "U-s202": te + "/qis-test-U-s202-sym",
        "H-s202-fb-r116": te + "/qis-test-H-s202-sym-fallback",
        "cgpu-R-s101": P + "/test-eval-cgpu/qis-cgpu-R-s101", "cgpu-R-s202": P + "/test-eval-cgpu/qis-cgpu-R-s202",
        "cgpu-U-s101": P + "/test-eval-cgpu/qis-cgpu-U-s101", "cgpu-U-s202": P + "/test-eval-cgpu/qis-cgpu-U-s202"}
    res = {"generated_by": "reports/build/build_v3.py extract", "P": P,
           "rule": "c = attempts with reward == 1 (verifier unknown counts as incorrect); t = attempts with finish "
                   "'length'. Arrays follow the panel order of the panel file.",
           "test": {"panel_file": "data/main-v1/target_test_sealed.json", "n": 32, "order": test_order, "cells": {}},
           "stress": {"panel_file": "data/stress/stress_v1.json", "n": 8, "order": st_order,
                      "tier": [it["tier"] for it in st_items], "cells": {}},
           "length": {"test": {}, "stress": {}}}
    for name in TEST_CELLS:
        path = test_paths[name] + "/records.jsonl.gz"
        c, t = _counts(path, test_order, 32)
        res["test"]["cells"][name] = {"records": os.path.relpath(path, P), "c": c, "t": t}
    for name in STRESS_CELLS:
        path = f"{P}/stress-eval/{name}/records.jsonl.gz"
        c, t = _counts(path, st_order, 8)
        res["stress"]["cells"][name] = {"records": os.path.relpath(path, P), "c": c, "t": t}
    for pol in LENGTH_POLS:
        for panel, order, n, paths in (
                ("test", test_order, 32, [f"{P}/length-test/{pol}-r128/all/rows.jsonl.gz"]),
                ("stress", st_order, 8, [f"{P}/length-diag/{pol}-r128/t1/rows.jsonl.gz",
                                         f"{P}/length-diag/{pol}-r128/rest/rows.jsonl.gz"])):
            idx = {p: i for i, p in enumerate(order)}
            c1, c2 = [0] * len(order), [0] * len(order)
            seen = set()
            for path in paths:
                for r in _records(path):
                    k = (r["prompt_id"], int(r["sample"]))
                    if k in seen:
                        raise SystemExit(f"duplicate row {k} in {path}")
                    seen.add(k)
                    i = idx[r["prompt_id"]]
                    c1[i] += 1 if r["reward_1"] == 1 else 0
                    c2[i] += 1 if r["reward_2"] == 1 else 0
            if len(seen) != len(order) * n:
                raise SystemExit(f"{pol} {panel}: {len(seen)} rows, expected {len(order) * n}")
            res["length"][panel][pol] = {"rows": [os.path.relpath(p, P) for p in paths], "c12288": c1, "c16384": c2}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=0, sort_keys=False)
    print("wrote", out)


# ----------------------------------------------------------------------------------------------------------------
# build: helpers
# ----------------------------------------------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return json.load(f)


def boot_mean(x, seed=BOOT_SEED):
    """Problem bootstrap of a mean over problems (x = per-problem value or paired per-problem difference).
    Percentile 95% interval, 10,000 draws, numpy default_rng(seed); a fresh generator per quantity."""
    import numpy as np
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(BOOT_DRAWS, len(x)))
    bs = x[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}"


def pts(x, d=1):
    if x is None:
        return "—"
    s = f"{100 * x:+.{d}f}"
    return s.replace("-", "−") if s != "+0.0" and s != "-0.0" else "0.0"


def ci_abs(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def ci_pts(c):
    return f"[{pts(c[0])}, {pts(c[1])}]"


ROUNDING_NOTE = ("Rounding: every displayed number is rounded to one decimal independently from the unrounded value in "
                 "the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the "
                 "displayed scores; the unrounded contrast is the correct one.")


def write_table(tdir, name, title, purpose, sources, rows, md_blocks):
    """rows: list of dicts (unrounded). md_blocks: list of (heading or None, headers, list of row lists, note)."""
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, name + ".json"), "w") as f:
        json.dump({"name": name, "title": title, "purpose": purpose, "sources": sources,
                   "bootstrap": {"draws": BOOT_DRAWS, "rng": f"numpy default_rng({BOOT_SEED})", "unit": "problem",
                                 "interval": "percentile 95%"}, "rows": rows}, f, indent=1)
    lines = [f"# {title}", "", purpose, "", "Sources: " + "; ".join(f"`{s}`" for s in sources) + ".", ""]
    for heading, headers, body, note in md_blocks:
        if heading:
            lines += [f"## {heading}", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for r in body:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")
        if note:
            lines += [note, ""]
    lines += [ROUNDING_NOTE, ""]
    with open(os.path.join(tdir, name + ".md"), "w") as f:
        f.write("\n".join(lines))


def run_label(run):
    return run  # e.g. "R-s101"


# ----------------------------------------------------------------------------------------------------------------
# build: tables
# ----------------------------------------------------------------------------------------------------------------
def t01_endpoints(A, cnt, tdir):
    import numpy as np
    et = A["eval_test"]
    n = cnt["test"]["n"]
    rows = []
    for cell in TEST_CELLS:
        m = et["cells"][cell]
        c = np.array(cnt["test"]["cells"][cell]["c"], float)
        assert int(c.sum()) == int(m["correct"]), cell
        assert abs(float(np.mean(c / n)) - m["pass1"]) < 1e-12, cell
        r = {"cell": cell, "role": ("base" if cell == "base" else "endpoint" if cell in RUNS else "supplementary"),
             "problems": m["problems"], "attempts": m["attempts"], "correct": m["correct"],
             "pass1": m["pass1"], "pass1_ci95": boot_mean(c / n),
             "truncation_share": m["truncation_share"], "pass8_unbiased": m["pass8_unbiased"],
             "coverage_at_32": m["coverage_at_n"]}
        if cell in RUNS:
            arm, s = cell.split("-s")
            g = et["contrasts"][f"{arm}-base s{s}"]["pass1"]
            r["gain_over_base"] = g["diff"]
            r["gain_over_base_ci95"] = g["ci95"]
        rows.append(r)
    names = {"H-s202-fb-r116": "H-s202 r116 fallback", "cgpu-R-s101": "R-s101 L40S rerun",
             "cgpu-R-s202": "R-s202 L40S rerun", "cgpu-U-s101": "U-s101 L40S rerun", "cgpu-U-s202": "U-s202 L40S rerun"}
    body = []
    for r in rows:
        body.append([names.get(r["cell"], r["cell"] + (" r128" if r["cell"] in RUNS else "")),
                     f"{int(r['correct']):,} / {r['attempts']:,}", pct(r["pass1"]), ci_abs(r["pass1_ci95"]),
                     pts(r["gain_over_base"]) if "gain_over_base" in r else "—",
                     ci_pts(r["gain_over_base_ci95"]) if "gain_over_base" in r else "—",
                     pct(r["truncation_share"]), pct(r["pass8_unbiased"])])
    write_table(tdir, "t01_endpoints", "Sealed test: base and round-128 endpoints",
                "Sealed-test pass@1 (128 problems × 32 attempts, cap 12,288) of the base policy and the six round-128 "
                "endpoints, with the round-116 fallback and the four L40S reruns as supplementary rows (not "
                "endpoints). Interval for pass@1: problem bootstrap of the mean of c/32. Gain over base: paired "
                "problem bootstrap (audit pass B).",
                ["research/audit-a26/pass-b/data/eval_test.json", "reports/tables/v3/perproblem_counts.json"],
                rows,
                [(None, ["Policy", "Correct / attempts", "pass@1 (%)", "95% interval", "Gain over base (points)",
                         "95% interval", "Truncated (%)", "pass@8 (%)"], body,
                  "pass@8 = unbiased estimate from 32 attempts per problem (DEFINITIONS §3.C).")])
    return rows


CONTRASTS = [("R", "U"), ("R", "H"), ("U", "H")]


def t02_contrasts(A, tdir):
    et = A["eval_test"]
    rows = []
    for a, b in CONTRASTS:
        for s in SEEDS:
            c = et["contrasts"][f"{a}-{b} s{s}"]
            rows.append({"contrast": f"{a}-{b}", "seed": s, "A": f"{a}-s{s}", "B": f"{b}-s{s}",
                         "pass1_A": c["pass1_A"], "pass1_B": c["pass1_B"], "diff": c["pass1"]["diff"],
                         "ci95": c["pass1"]["ci95"], "share_draws_gt0": c["pass1"]["share_draws_gt0"]})
    for s in SEEDS:
        c = et["contrasts"][f"cgpuR-cgpuU s{s}"]
        rows.append({"contrast": "R-U (L40S reruns)", "seed": s, "A": f"cgpu-R-s{s}", "B": f"cgpu-U-s{s}",
                     "pass1_A": et["cells"][f"cgpu-R-s{s}"]["pass1"], "pass1_B": et["cells"][f"cgpu-U-s{s}"]["pass1"],
                     "diff": c["pass1"]["diff"], "ci95": c["pass1"]["ci95"],
                     "share_draws_gt0": c["pass1"]["share_draws_gt0"]})
    means = [{"contrast": f"{a}-{b}", "two_seed_mean": et["two_seed_mean"][f"{a}-{b}"],
              "two_seed_halfrange": et["two_seed_halfrange"][f"{a}-{b}"]} for a, b in CONTRASTS]
    body = [[r["contrast"].replace("-", " − ", 1), r["seed"], pct(r["pass1_A"]), pct(r["pass1_B"]), pts(r["diff"]),
             ci_pts(r["ci95"])] for r in rows]
    body2 = [[m["contrast"].replace("-", " − "), pts(m["two_seed_mean"]), pts(m["two_seed_halfrange"]).lstrip("+")]
             for m in means]
    write_table(tdir, "t02_contrasts_test", "Sealed test: paired contrasts per training seed",
                "Difference in sealed-test pass@1 between two round-128 endpoints of the same training seed. "
                "Interval: paired problem bootstrap over the 128 problems (same resampled problems for both "
                "policies). It covers problem and attempt sampling only, not training-run variation.",
                ["research/audit-a26/pass-b/data/eval_test.json"], rows + [dict(m, kind="two_seed") for m in means],
                [(None, ["Contrast A − B", "Seed", "A pass@1 (%)", "B pass@1 (%)", "A − B (points)", "95% interval"],
                  body, "The L40S rows re-evaluate the R and U endpoints on one GPU type (reproducibility check)."),
                 ("Two-seed summary (descriptive only)", ["Contrast", "Mean of the two seeds (points)",
                                                          "Half the difference between seeds (points)"], body2,
                  "Two seeds cannot estimate the spread of effects across future runs.")])
    return rows


def t03_curves(A, tdir):
    import numpy as np
    cur = A["train_curves"]["curves"]
    rew = A["train_rewards"]
    rows = []
    for run in RUNS:
        cum = {0: 0}
        for pr in rew[run]["per_round"]:
            cum[pr["round_1based"]] = pr["cum_hard_attempts"]
        for rnd in ROUNDS:
            for pan in PANELS:
                m = cur[run][str(rnd)][pan]
                c = np.array(m["per_problem_c"], float)
                n = m["n_per_problem"][0]
                assert len(m["n_per_problem"]) == 1 and len(c) == PANEL_N[pan]
                assert abs(float(np.mean(c / n)) - m["pass1"]) < 1e-12
                rows.append({"run": run, "round": rnd, "panel": pan, "problems": m["problems"], "n": n,
                             "attempts": m["attempts"], "correct": m["correct"], "pass1": m["pass1"],
                             "pass1_ci95": boot_mean(c / n), "cap": m["cap_canonical_row"],
                             "truncation_share": m["trunc_share"], "cum_hard_attempts": cum[rnd],
                             "trajectory": ("a2-r60" if run == "H-s202" and rnd >= 64 else "a2")})
    by = {(r["run"], r["round"], r["panel"]): r for r in rows}
    blocks = []
    pname = {"dev": "dev (53 problems)", "probe_J_H": "J_H (32 hard dev problems = first 32 of dev)",
             "probe_J_R": "J_R (28 factoring probe problems)", "probe_J_U": "J_U (32 modular-arithmetic problems)"}
    for pan in PANELS:
        body = [[run] + [pct(by[(run, rnd, pan)]["pass1"]) for rnd in ROUNDS] for run in RUNS]
        body.append(["cap (tokens)"] + [f"{by[('H-s101', rnd, pan)]['cap']:,}" for rnd in ROUNDS])
        blocks.append((f"pass@1 (%) on {pname[pan]}, 8 attempts per problem", ["Run"] + [f"r{r}" for r in ROUNDS],
                       body, None))
    body = [[run] + [f"{by[(run, rnd, 'dev')]['cum_hard_attempts']:,}" for rnd in ROUNDS] for run in RUNS]
    blocks.append(("Cumulative hard training attempts after each round", ["Run"] + [f"r{r}" for r in ROUNDS], body,
                   "H trains on 128 hard attempts per round (16 slots × 8), R and U on 64 (8 shared slots × 8)."))
    write_table(tdir, "t03_curves", "In-run evaluations over training rounds",
                "pass@1 of each run at rounds 0, 32, 64, 96 and 128 on dev and the three probes (8 attempts per "
                "problem). Round 0 is the same base policy in every run; its spread is cross-job sampling noise. "
                "The evaluation cap is 8,192 at rounds 0 and 32 and 12,288 from round 64. H-s202 rounds 64–128 are "
                "trajectory a2-r60 (resumed from round 60). Intervals in the JSON: problem bootstrap.",
                ["research/audit-a26/pass-b/data/train_curves.json",
                 "research/audit-a26/pass-b/data/train_rewards.json"], rows, blocks)
    return rows


DIAG_GPU = {("r0", "branch"): "L40", ("r0", "none"): "L40", ("r64", "branch"): "RTX 6000 Ada",
            ("r64", "none"): "RTX 6000 Ada", ("r0", "rerun"): "L40S", ("r64", "rerun"): "L40S"}


def t04_diag(A, tdir):
    d = A["train_diag"]
    rows = []
    probes = ("probe_J_H", "probe_J_R", "probe_J_U")
    for lab in ("r0", "r64"):
        ev = d[lab]["evals"]
        none = {p: ev["none"][p]["pass1"] for p in probes}
        rerun = {p: ev["none_rerun"][p]["pass1"] for p in probes}
        for ref in ("none", "none_rerun", "none_cycled"):
            rows.append({"label": lab, "branch": ref, "type": "reference", "draw_seed": None,
                         "gpu": DIAG_GPU[(lab, "none" if ref == "none" else "rerun")],
                         **{p: ev[ref][p]["pass1"] for p in probes},
                         **{p + "_trunc": ev[ref][p]["trunc_share"] for p in probes},
                         **{"d_vs_none_" + p: ev[ref][p]["pass1"] - none[p] for p in probes}})
        for typ in ("hard", "related", "unrelated"):
            for i, sd in enumerate((4101, 4102, 4103, 4104)):
                key = f"{typ}-{sd}"
                u = ev[key]["update_row"]
                r = {"label": lab, "branch": key, "type": typ, "draw_seed": sd, "gpu": DIAG_GPU[(lab, "branch")],
                     "grad_norm_preclip": u["grad_norm_preclip"], "mixed_groups": u["mixed_groups"],
                     "update_batch_capped": u["capped"], "update_batch_reward": u["batch_reward"],
                     "momentum_only": u["grad_norm_preclip"] == 0.0}
                for p in probes:
                    r[p] = ev[key][p]["pass1"]
                    r[p + "_trunc"] = ev[key][p]["trunc_share"]
                    r["d_vs_none_" + p] = ev[key][p]["pass1"] - none[p]
                    r["d_vs_none_rerun_" + p] = ev[key][p]["pass1"] - rerun[p]
                    assert abs(r["d_vs_none_" + p] - d[lab]["branch_types"][p]["per_draw"][typ][i]) < 1e-12
                rows.append(r)
    blocks = []
    for lab, src in (("r0", "round 0 (base policy, fresh optimizer)"), ("r64", "R-s101 round 64")):
        body = []
        for r in rows:
            if r["label"] != lab:
                continue
            body.append([r["branch"] + (" (momentum-only)" if r.get("momentum_only") else ""), r["gpu"],
                         pct(r["probe_J_H"]), pts(r["d_vs_none_probe_J_H"]),
                         pts(r["d_vs_none_rerun_probe_J_H"]) if r["type"] != "reference" else "—",
                         pct(r["probe_J_R"]), pts(r["d_vs_none_probe_J_R"]),
                         pct(r["probe_J_U"]), pts(r["d_vs_none_probe_J_U"]),
                         "—" if r["type"] == "reference" else f"{r['grad_norm_preclip']:.3f}"])
        blocks.append((f"Source policy: {src}",
                       ["Branch", "GPU", "J_H (%)", "ΔJ_H vs none", "ΔJ_H vs none_rerun", "J_R (%)", "ΔJ_R vs none",
                        "J_U (%)", "ΔJ_U vs none", "Pre-clip grad norm"], body, None))
    write_table(tdir, "t04_diag", "Local one-step diagnostic: individual branch draws",
                "One GRPO step (4 prompts × 8 attempts, one draw seed 4101–4104) of each type (hard, related, "
                "unrelated) from two source policies, then J_H, J_R, J_U at 16 attempts per problem and cap 8,192 "
                "(not the 12,288 cap the r64 source trained with). ΔJ in points against two no-update references: "
                "`none` (same job as the branches) and `none_rerun` (separate job on L40S; `none_cycled` gives "
                "identical values). Momentum-only = pre-clip gradient norm exactly 0 (all four groups non-mixed); "
                "AdamW still stepped with its stored moments. GPUs from DEFINITIONS v1.1 §4 item 1.",
                ["research/audit-a26/pass-b/data/train_diag.json", "research/DEFINITIONS.md v1.1 §4 item 1"], rows,
                blocks)
    return rows


def t05_length(A, cnt, tdir):
    import numpy as np
    el = A["eval_length"]
    rows = []
    for panel in ("test", "stress"):
        n = 32 if panel == "test" else 8
        for pol in LENGTH_POLS:
            k = el[panel][pol]["counts"]
            pc = cnt["length"][panel][pol]
            c1, c2 = np.array(pc["c12288"], float), np.array(pc["c16384"], float)
            assert int(c1.sum()) == k["correct_12288"] and int(c2.sum()) == k["correct_16384"], (panel, pol)
            assert abs(float(np.mean(c1 / n)) - k["pass1_12288"]) < 1e-12
            assert abs(float(np.mean(c2 / n)) - k["pass1_16384"]) < 1e-12
            cap = k["capped_at_12288"]
            assert k["recovered"] + k["finished_wrong"] + k["still_capped"] == cap
            rows.append({"panel": panel, "checkpoint": pol + " r128", "run": pol, "problems": len(c1), "n": n,
                         "attempts": k["attempts"], "pass1_12288": k["pass1_12288"], "pass1_16384": k["pass1_16384"],
                         "gain": k["pass1_16384"] - k["pass1_12288"], "gain_ci95": boot_mean(c2 / n - c1 / n),
                         "capped_at_12288": cap, "recovered": k["recovered"], "finished_wrong": k["finished_wrong"],
                         "still_capped": k["still_capped"], "recovered_share": k["recovered"] / cap,
                         "finished_wrong_share": k["finished_wrong"] / cap, "still_capped_share": k["still_capped"] / cap})
    blocks = []
    for panel, desc in (("test", "Sealed test (128 problems × 32 attempts)"),
                        ("stress", "Stress panels (221 problems × 8 attempts)")):
        body = [[r["checkpoint"], pct(r["pass1_12288"]), pct(r["pass1_16384"]), pts(r["gain"]), ci_pts(r["gain_ci95"]),
                 f"{r['capped_at_12288']:,}", f"{r['recovered']} ({pct(r['recovered_share'])}%)",
                 f"{r['finished_wrong']} ({pct(r['finished_wrong_share'])}%)",
                 f"{r['still_capped']} ({pct(r['still_capped_share'])}%)"] for r in rows if r["panel"] == panel]
        blocks.append((desc, ["Checkpoint", "pass@1 at 12,288 (%)", "pass@1 at 16,384 (%)", "Change (points)",
                              "95% interval", "Capped at 12,288", "Recovered (correct)", "Finished wrong",
                              "Still capped at 16,384"], body, None))
    write_table(tdir, "t05_length", "Length extension 12,288 → 16,384 tokens",
                "Every attempt that hit the 12,288-token cap was continued for up to 4,096 more tokens and regraded; "
                "other attempts keep their score. Change interval: paired problem bootstrap of the per-problem "
                "change. Shares in brackets are of the capped attempts. H-s202 was not extended.",
                ["research/audit-a26/pass-b/data/eval_length.json", "reports/tables/v3/perproblem_counts.json"],
                rows, blocks)
    return rows


def t06_length_contrasts(A, tdir):
    el = A["eval_length"]["contrasts"]
    rows = []
    for panel in ("test", "stress"):
        for a, b in (("R-s101", "U-s101"), ("R-s202", "U-s202"), ("R-s101", "H-s101"), ("U-s101", "H-s101")):
            c = el[f"{panel} {a} - {b}"]
            rows.append({"panel": panel, "A": a, "B": b, "contrast": f"{a[0]}-{b[0]}", "seed": int(a[-3:]),
                         "diff_12288": c["at_12288"]["diff"], "ci95_12288": c["at_12288"]["ci95"],
                         "diff_16384": c["at_16384"]["diff"], "ci95_16384": c["at_16384"]["ci95"],
                         "change": c["change"]["diff"], "change_ci95": c["change"]["ci95"]})
    blocks = []
    for panel, desc in (("test", "Sealed test"), ("stress", "Stress panels")):
        body = [[f"{r['A']} − {r['B']}", pts(r["diff_12288"]), ci_pts(r["ci95_12288"]), pts(r["diff_16384"]),
                 ci_pts(r["ci95_16384"]), pts(r["change"]), ci_pts(r["change_ci95"])]
                for r in rows if r["panel"] == panel]
        blocks.append((desc, ["Contrast", "At 12,288 (points)", "95% interval", "At 16,384 (points)", "95% interval",
                              "Change (points)", "95% interval"], body, None))
    write_table(tdir, "t06_length_contrasts", "Length extension: paired contrasts at both caps",
                "Paired contrasts of pass@1 before (12,288) and after (16,384) the extension, and the change of the "
                "contrast. R − H and U − H exist for seed 101 only (H-s202 was not extended). Interval: paired problem "
                "bootstrap (audit pass B).", ["research/audit-a26/pass-b/data/eval_length.json"], rows, blocks)
    return rows


STRESS_LABEL = {"base": "base", "R-s101-r64": "R-s101 r64", "H-s202-r64": "H-s202 r64 (a2-r60)"}


def stress_label(cell):
    return STRESS_LABEL.get(cell, cell.replace("-r128", " r128"))


def t07_stress(A, cnt, tdir):
    import numpy as np
    es = A["eval_stress"]
    tiers = np.array(cnt["stress"]["tier"])
    rows = []
    for cell in STRESS_CELLS:
        c = np.array(cnt["stress"]["cells"][cell]["c"], float)
        for tier in ("all", "harder", "superhard"):
            m = es["cells"][cell][tier]
            mask = np.ones(len(c), bool) if tier == "all" else tiers == tier
            assert int(c[mask].sum()) == int(m["correct"]), (cell, tier)
            assert abs(float(np.mean(c[mask] / 8)) - m["pass1"]) < 1e-12
            rows.append({"cell": cell, "label": stress_label(cell), "tier": tier, "problems": m["problems"],
                         "attempts": m["attempts"], "correct": m["correct"], "pass1": m["pass1"],
                         "pass1_ci95": boot_mean(c[mask] / 8), "truncation_share": m["truncation_share"],
                         "coverage_at_8": m["coverage_at_n"]})
    by = {(r["cell"], r["tier"]): r for r in rows}
    body = []
    for cell in STRESS_CELLS:
        row = [stress_label(cell)]
        for tier in ("all", "harder", "superhard"):
            r = by[(cell, tier)]
            row += [pct(r["pass1"]), ci_abs(r["pass1_ci95"])]
        row.append(pct(by[(cell, "all")]["truncation_share"]))
        body.append(row)
    write_table(tdir, "t07_stress", "Stress panels: pass@1 per tier",
                "pass@1 on the stress panels (8 attempts per problem, cap 12,288): all 221 problems, Harder (95) and "
                "SuperHard (126). Tiers come from one annotation of solution structure, not from model success. "
                "Interval: problem bootstrap within the tier. The two round-64 cells are the only intermediate "
                "checkpoints evaluated; H-s202 r64 is trajectory a2-r60.",
                ["research/audit-a26/pass-b/data/eval_stress.json", "reports/tables/v3/perproblem_counts.json"], rows,
                [(None, ["Policy", "All (%)", "95% interval", "Harder (%)", "95% interval", "SuperHard (%)",
                         "95% interval", "Truncated, all (%)"], body, None)])
    return rows


def t08_stress_contrasts(A, tdir):
    es = A["eval_stress"]
    rows = []
    for a, b in CONTRASTS:
        for s in SEEDS:
            for tier in ("all", "harder", "superhard"):
                c = es["contrasts"][f"{a}-{b} s{s} {tier}"]
                pa = es["cells"][f"{a}-s{s}-r128"][tier]["pass1"]
                pb = es["cells"][f"{b}-s{s}-r128"][tier]["pass1"]
                rows.append({"contrast": f"{a}-{b}", "seed": s, "tier": tier, "pass1_A": pa, "pass1_B": pb,
                             "diff": c["pass1"]["diff"], "ci95": c["pass1"]["ci95"]})
    body = [[r["contrast"].replace("-", " − "), r["seed"], r["tier"], pct(r["pass1_A"]), pct(r["pass1_B"]),
             pts(r["diff"]), ci_pts(r["ci95"])] for r in rows]
    write_table(tdir, "t08_stress_contrasts", "Stress panels: paired contrasts per seed and tier",
                "Difference in stress pass@1 between round-128 endpoints of the same seed. Interval: paired problem "
                "bootstrap within the tier (audit pass B).", ["research/audit-a26/pass-b/data/eval_stress.json"],
                rows, [(None, ["Contrast A − B", "Seed", "Tier", "A (%)", "B (%)", "A − B (points)", "95% interval"],
                        body, "Under the any-box grading rule (sensitivity check, discrepancies.md §3.2) the seed-202 "
                              "R − U and R − H intervals on all problems change their relation to 0.")])
    return rows


CONTROL_GPU = {"base": "L40S", "H-s101": "L40S", "H-s202": "L40", "R-s101": "L40", "R-s202": "L40",
               "U-s101": "L40S", "U-s202": "L40S"}  # transfer/control.md job table (sacct)


def t09_control(A, tdir):
    ctl = A.get("control")
    if ctl is None:
        return None
    cur = A["train_curves"]["curves"]
    rows = []
    base = ctl["policies"]["base"]["all"]["pass1"]
    for run in RUNS:
        g = ctl["gains"][run]
        ju = g["J_U"]
        jr = g["J_R"]
        # cross-check the probe gains against pass-B in-run curves (r128 - r0 of the same run)
        ju_b = cur[run]["128"]["probe_J_U"]["pass1"] - cur[run]["0"]["probe_J_U"]["pass1"]
        jr_b = cur[run]["128"]["probe_J_R"]["pass1"] - cur[run]["0"]["probe_J_R"]["pass1"]
        assert abs(ju[0] - ju_b) < 1e-12 and abs(jr[0] - jr_b) < 1e-12, run
        rows.append({"run": run, "control_base_pass1": base,
                     "control_endpoint_pass1": ctl["policies"][run]["all"]["pass1"],
                     "control_gain": g["control"][0], "control_gain_ci95": g["control"][1:],
                     "J_U_gain": ju[0], "J_U_gain_ci95": ju[1:], "J_R_gain": jr[0], "J_R_gain_ci95": jr[1:],
                     "J_U_gain_within_8192": g["J_U_8k"][0], "rho_control_over_J_U": g["rho"],
                     "control_gpu": CONTROL_GPU[run]})
    body = [[r["run"], pct(r["control_base_pass1"]), pct(r["control_endpoint_pass1"]), pts(r["control_gain"]),
             ci_pts(r["control_gain_ci95"]), pts(r["J_U_gain"]), ci_pts(r["J_U_gain_ci95"]), pts(r["J_R_gain"]),
             ci_pts(r["J_R_gain_ci95"]), r["control_gpu"]] for r in rows]
    write_table(tdir, "t09_control", "Control panel: gains on subjects no arm practises",
                "Control panel `control_v1`: 64 MATH test problems (16 each from geometry, counting and probability, "
                "precalculus, prealgebra; levels 4–5), none used in any study set; 8 attempts, cap 12,288, base and "
                "the six round-128 endpoints (pre-registered in transfer/PREREG.md). Control gain = endpoint − base. "
                "J_U and J_R gains = in-run pass@1 at round 128 (cap 12,288) − round 0 (cap 8,192) of the same run. "
                "Intervals: paired problem bootstrap, 10,000 draws, numpy default_rng(2026) (transfer agent).",
                ["research/audit-a26/transfer/control.json", "research/audit-a26/transfer/control.md",
                 "research/audit-a26/pass-b/data/train_curves.json (probe gains cross-checked)"], rows,
                [(None, ["Run", "Control base (%)", "Control endpoint (%)", "Control gain (points)", "95% interval",
                         "J_U gain (points)", "95% interval", "J_R gain (points)", "95% interval", "Control GPU"],
                  body, "Base control pass@1 is 50.8%, base J_U and J_R about 81%, so the control panel has more "
                        "room to gain.")])
    return rows


# ----------------------------------------------------------------------------------------------------------------
# build: figures
# ----------------------------------------------------------------------------------------------------------------
COLOR = {"H": "#3F5A87", "R": "#E07B12", "U": "#2BB39A", "base": "#222222"}
MARKER = {101: "o", 202: "s"}
LSTYLE = {101: "-", 202: "--"}
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 12, "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 7,
        "savefig.dpi": 200, "figure.dpi": 100, "svg.hashsalt": "qis", "axes.grid": False,
        "font.family": "DejaVu Sans"})
    return plt


def save(fig, fdir, name):
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, name + ".png"), dpi=200, bbox_inches="tight", facecolor="white",
                metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def arm_seed(name):
    """'R-s101' or 'R-s101-r128' -> ('R', 101)."""
    a, rest = name.split("-s", 1)
    return a, int(rest[:3])


def hgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def dot_ci(ax, y, x, ci, color, marker, ls, filled=True, ms=7):
    ax.plot([ci[0], ci[1]], [y, y], color=color, lw=1.8, ls=ls, solid_capstyle="butt")
    ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc=color if filled else "white", mec=color, mew=1.6,
            ls="none", zorder=3)


def seed_legend_handles(plt):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=MUTED, marker="o", ls="-", label="seed 101"),
            Line2D([], [], color=MUTED, marker="s", ls="--", label="seed 202")]


def arm_legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLOR[a], lw=3, label={"H": "H (hard only)", "R": "R (hard + factoring practice)",
                                                         "U": "U (hard + modular practice)"}[a]) for a in ARMS]


def f01_endpoints(plt, t01, fdir):
    order = ["base", "H-s101", "H-s202", "R-s101", "R-s202", "U-s101", "U-s202"]
    by = {r["cell"]: r for r in t01}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, cell in enumerate(order):
        r = by[cell]
        if cell == "base":
            col, mk, ls = COLOR["base"], "D", "-"
        else:
            a, s = arm_seed(cell)
            col, mk, ls = COLOR[a], MARKER[s], LSTYLE[s]
        dot_ci(ax, i, 100 * r["pass1"], [100 * v for v in r["pass1_ci95"]], col, mk, ls)
    ax.set_yticks(range(len(order)), ["base"] + [c + " r128" for c in order[1:]])
    ax.invert_yaxis()
    ax.set_xlim(45, 90)
    ax.set_xlabel("pass@1 on 128 sealed-test problems, 32 attempts each (%)")
    ax.set_title("Sealed test: base policy and round-128 endpoints", loc="left")
    hgrid(ax)
    save(fig, fdir, "f01_endpoints")


def contrast_rows(ax, rows, xkey, cikey, ytick_labels=True, color=INK, offset=0.0, filled=True):
    labels = []
    y = 0
    ys = []
    for gi, (a, b) in enumerate(CONTRASTS):
        for s in SEEDS:
            r = next(r for r in rows if r["contrast"] == f"{a}-{b}" and r["seed"] == s)
            dot_ci(ax, y + offset, 100 * r[xkey], [100 * v for v in r[cikey]], color, MARKER[s], LSTYLE[s], filled)
            labels.append(f"{a} − {b}, seed {s}")
            ys.append(y)
            y += 1
        y += 0.6
    if ytick_labels:
        ax.set_yticks(ys, labels)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    return ys


def f02_contrasts(plt, t02, fdir):
    rows = [r for r in t02 if r.get("contrast") in ("R-U", "R-H", "U-H")]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    contrast_rows(ax, rows, "diff", "ci95")
    ax.invert_yaxis()
    ax.set_xlabel("Difference in sealed-test pass@1 (percentage points)")
    ax.set_title("Sealed test: paired contrasts within each training seed", loc="left")
    hgrid(ax)
    save(fig, fdir, "f02_contrasts_test")


def _curve_axes(plt, t03, fdir, name, xkey, xlabel, title, panel="dev", ylabel=None):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.7), sharey=True)
    for ax, s in zip(axes, SEEDS):
        for a in ARMS:
            run = f"{a}-s{s}"
            pts_ = sorted([r for r in t03 if r["run"] == run and r["panel"] == panel], key=lambda r: r["round"])
            xs = [r[xkey] / (1000 if xkey == "cum_hard_attempts" else 1) for r in pts_]
            ys = [100 * r["pass1"] for r in pts_]
            ax.plot(xs, ys, color=COLOR[a], ls=LSTYLE[s], lw=2, zorder=2)
            for x, y, r in zip(xs, ys, pts_):
                filled = r["cap"] == 12288
                ax.plot([x], [y], marker=MARKER[s], ms=7, color=COLOR[a], mfc=COLOR[a] if filled else "white",
                        mec=COLOR[a], mew=1.6, ls="none", zorder=3)
        if xkey == "round":
            ax.axvline(32, color=MUTED, lw=1, ls=":", zorder=1)
            ax.text(31, 0.97, "cap\n8,192", transform=ax.get_xaxis_transform(), ha="right", va="top",
                    fontsize=10, color=MUTED)
            ax.text(34, 0.97, "cap 12,288", transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=10, color=MUTED)
            ax.set_xticks(ROUNDS)
            ax.set_xlim(-6, 134)
        else:
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.set_xlim(-0.8, 17)
        ax.set_title(f"Seed {s}", loc="left")
        ax.set_xlabel(xlabel)
        ax.annotate("round 0: same\nbase policy", (0, 41), xytext=(14, -2), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color=MUTED)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel or "pass@1 on 53 dev problems,\n8 attempts each (%)")
    handles = arm_legend_handles() + [
        Line2D([], [], color=MUTED, marker="o", mfc="white", mec=MUTED, ls="none",
               label="open: evaluated at cap 8,192 (rounds 0, 32)"),
        Line2D([], [], color=MUTED, marker="o", ls="none", label="filled: cap 12,288 (rounds 64–128)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, name)


def f03_dev_rounds(plt, t03, fdir):
    _curve_axes(plt, t03, fdir, "f03_dev_rounds", "round", "Training round", "Dev pass@1 over training rounds")


def f04_dev_hard(plt, t03, fdir):
    _curve_axes(plt, t03, fdir, "f04_dev_hard_attempts", "cum_hard_attempts",
                "Hard training attempts (thousands)",
                "Dev pass@1 against cumulative hard training attempts")


def _traj(plt, t03, fdir, name, ypanel, ylabel, title):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.9), sharex=True, sharey=True)
    by = {(r["run"], r["round"], r["panel"]): r for r in t03}
    for ax, s in zip(axes, SEEDS):
        starts = []
        for a in ARMS:
            run = f"{a}-s{s}"
            xs = [100 * by[(run, rnd, "probe_J_H")]["pass1"] for rnd in ROUNDS]
            ys = [100 * by[(run, rnd, ypanel)]["pass1"] for rnd in ROUNDS]
            for i in range(len(ROUNDS) - 1):
                ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[a], lw=1.6, ls=LSTYLE[s],
                                            shrinkA=4, shrinkB=4, mutation_scale=11), zorder=2)
            for i, rnd in enumerate(ROUNDS):
                filled = by[(run, rnd, ypanel)]["cap"] == 12288
                ax.plot([xs[i]], [ys[i]], marker=MARKER[s], ms=6, color=COLOR[a], mfc=COLOR[a] if filled else "white",
                        mec=COLOR[a], mew=1.4, ls="none", zorder=3)
            starts.append((xs[0], ys[0]))
            ax.plot([xs[-1]], [ys[-1]], marker="o", ms=15, mfc="none", mec=INK, mew=1.2, ls="none", zorder=4)
        sx = sum(p[0] for p in starts) / len(starts)
        sy = min(p[1] for p in starts)
        ax.annotate("r0", (sx, sy), xytext=(0, -8), textcoords="offset points", fontsize=10, color=INK,
                    ha="center", va="top")
        ax.margins(0.1)
        ax.set_title(f"Seed {s}", loc="left")
        ax.set_xlabel("J_H: pass@1 on 32 hard dev problems,\n8 attempts each (%)")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    handles = arm_legend_handles() + [
        Line2D([], [], color=MUTED, marker="o", mfc="white", mec=MUTED, ls="none", label="open: cap 8,192 (r0, r32)"),
        Line2D([], [], color=MUTED, marker="o", ls="none", label="filled: cap 12,288 (r64–r128)"),
        Line2D([], [], color=INK, marker="o", ms=13, mfc="none", mew=1.2, ls="none",
               label="ring: round 128 (path end)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, name)


def f05_traj_jr(plt, t03, fdir):
    _traj(plt, t03, fdir, "f05_traj_JH_JR", "probe_J_R",
          "J_R: pass@1 on 28 factoring problems,\n8 attempts each (%)",
          "Training paths in (J_H, J_R): rounds 0 → 32 → 64 → 96 → 128")


def f06_traj_ju(plt, t03, fdir):
    _traj(plt, t03, fdir, "f06_traj_JH_JU", "probe_J_U",
          "J_U: pass@1 on 32 modular-arithmetic\nproblems, 8 attempts each (%)",
          "Training paths in (J_H, J_U): rounds 0 → 32 → 64 → 96 → 128")


def f07_diag(plt, t04, fdir):
    from matplotlib.lines import Line2D
    probes = [("probe_J_H", "ΔJ_H"), ("probe_J_R", "ΔJ_R"), ("probe_J_U", "ΔJ_U")]
    fig, axes = plt.subplots(3, 2, figsize=(6.9, 8.2), sharex=True)
    tcol = {"hard": COLOR["H"], "related": COLOR["R"], "unrelated": COLOR["U"]}
    off = {4101: -0.21, 4102: -0.07, 4103: 0.07, 4104: 0.21}
    for ci, (lab, ctitle) in enumerate((("r0", "Round 0 (base policy)"), ("r64", "R-s101 round 64"))):
        rr = [r for r in t04 if r["label"] == lab]
        rerun = next(r for r in rr if r["branch"] == "none_rerun")
        for ri, (p, plab) in enumerate(probes):
            ax = axes[ri, ci]
            ax.axhline(0, color=INK, lw=1.2, zorder=1)
            ax.axhline(100 * rerun["d_vs_none_" + p], color=MUTED, lw=1.4, ls="--", zorder=1)
            for xi, typ in enumerate(("hard", "related", "unrelated")):
                for r in rr:
                    if r["type"] != typ:
                        continue
                    mo = r["momentum_only"]
                    ax.plot([xi + off[r["draw_seed"]]], [100 * r["d_vs_none_" + p]], marker="X" if mo else "o",
                            ms=8 if mo else 7, color=tcol[typ], mfc="white" if mo else tcol[typ], mec=tcol[typ],
                            mew=1.6, ls="none", zorder=3)
            ax.set_xticks([0, 1, 2], ["hard", "related", "unrelated"])
            ax.set_xlim(-0.5, 2.5)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if ci == 0:
                ax.set_ylabel(f"{plab} vs none\n(points)")
            if ri == 0:
                ax.set_title(ctitle, loc="left")
            if ri == 2:
                ax.set_xlabel("Update type (4 draws each)")
        # same y range within a row
    for ri in range(3):
        lo = min(axes[ri, 0].get_ylim()[0], axes[ri, 1].get_ylim()[0])
        hi = max(axes[ri, 0].get_ylim()[1], axes[ri, 1].get_ylim()[1])
        for ci in range(2):
            axes[ri, ci].set_ylim(lo, hi)
    handles = [Line2D([], [], color=INK, lw=1.2, label="none: no update, same job"),
               Line2D([], [], color=MUTED, lw=1.4, ls="--", label="none_rerun = none_cycled"),
               Line2D([], [], color=MUTED, marker="o", ls="none", label="one update draw"),
               Line2D([], [], color=MUTED, marker="X", mfc="white", ls="none", ms=8,
                      label="momentum-only (zero gradient)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("One optimizer step: change in J against two no-update references", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f07_delta_j")


def _length_dumbbell(ax, rows, ytick=True):
    order = ["H-s101", "R-s101", "U-s101", "R-s202", "U-s202"]
    for i, run in enumerate(order):
        r = next(r for r in rows if r["run"] == run)
        a, s = arm_seed(run)
        x1, x2 = 100 * r["pass1_12288"], 100 * r["pass1_16384"]
        ax.plot([x1, x2], [i, i], color=COLOR[a], lw=2, ls=LSTYLE[s], zorder=2)
        ax.plot([x1], [i], marker=MARKER[s], ms=7, color=COLOR[a], mfc="white", mec=COLOR[a], mew=1.6, ls="none",
                zorder=3)
        ax.plot([x2], [i], marker=MARKER[s], ms=7, color=COLOR[a], ls="none", zorder=3)
    if ytick:
        ax.set_yticks(range(len(order)), [o + " r128" for o in order])
        ax.invert_yaxis()
    hgrid(ax)


def f08_length_pass1(plt, t05, fdir):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), sharey=True)
    for ax, (panel, title, xl) in zip(axes, (("test", "Sealed test", "pass@1, 128 problems × 32 attempts (%)"),
                                             ("stress", "Stress panels", "pass@1, 221 problems × 8 attempts (%)"))):
        _length_dumbbell(ax, [r for r in t05 if r["panel"] == panel], ytick=(panel == "test"))
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
    handles = [Line2D([], [], color=MUTED, marker="o", mfc="white", mec=MUTED, ls="none",
                      label="cap 12,288 (original)"),
               Line2D([], [], color=MUTED, marker="o", ls="none", label="after extending capped attempts to 16,384")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Length extension: pass@1 before and after 4,096 more tokens", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f08_length_pass1")


SEG = [("recovered", "recovered: finished correct", "#2E2E2E"),
       ("finished_wrong", "finished wrong", "#8C8C8C"),
       ("still_capped", "still capped at 16,384", "#D4D4D4")]


def f08b_length_capped(plt, t05, fdir):
    from matplotlib.patches import Patch
    order = ["H-s101", "R-s101", "U-s101", "R-s202", "U-s202"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), sharey=True)
    for ax, (panel, title) in zip(axes, (("test", "Sealed test"), ("stress", "Stress panels"))):
        for i, run in enumerate(order):
            r = next(r for r in t05 if r["panel"] == panel and r["run"] == run)
            left = 0
            for key, _, col in SEG:
                w = 100 * r[key + "_share"]
                ax.barh(i, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.5)
                left += w
            ax.annotate(f"n = {r['capped_at_12288']}", (101, i), xytext=(2, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=10, color=INK, annotation_clip=False)
        if panel == "test":
            ax.set_yticks(range(len(order)), [o + " r128" for o in order])
            ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Share of attempts capped at 12,288 (%)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=col, edgecolor="#888888", label=lab) for _, lab, col in SEG]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle("What happened to capped attempts given 4,096 more tokens", x=0.01, ha="left", y=1.0)
    fig.tight_layout(w_pad=3.0)
    save(fig, fdir, "f08b_length_capped")


def f09_length_contrasts(plt, t06, fdir):
    from matplotlib.lines import Line2D
    order = [("R-s101", "U-s101"), ("R-s202", "U-s202"), ("R-s101", "H-s101"), ("U-s101", "H-s101")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3), sharey=True, sharex=True)
    for ax, (panel, title) in zip(axes, (("test", "Sealed test"), ("stress", "Stress panels"))):
        for i, (a, b) in enumerate(order):
            r = next(r for r in t06 if r["panel"] == panel and r["A"] == a and r["B"] == b)
            s = int(a[-3:])
            dot_ci(ax, i - 0.14, 100 * r["diff_12288"], [100 * v for v in r["ci95_12288"]], INK, MARKER[s],
                   LSTYLE[s], filled=False)
            dot_ci(ax, i + 0.14, 100 * r["diff_16384"], [100 * v for v in r["ci95_16384"]], INK, MARKER[s],
                   LSTYLE[s], filled=True)
        if panel == "test":
            ax.set_yticks(range(len(order)), [f"{a[0]} − {b[0]}, seed {a[-3:]}" for a, b in order])
            ax.invert_yaxis()
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Difference in pass@1 (points)")
        hgrid(ax)
    handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="cap 12,288"),
               Line2D([], [], color=INK, marker="o", ls="none", label="after extension to 16,384")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Length extension: paired contrasts before and after", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f09_length_contrasts")


def f10_stress(plt, t07, fdir):
    order = ["base", "H-s101-r128", "H-s202-r128", "R-s101-r128", "R-s202-r128", "U-s101-r128", "U-s202-r128",
             "R-s101-r64", "H-s202-r64"]
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.2), sharey=True, sharex=True)
    for ax, (tier, title) in zip(axes, (("harder", "Harder (95 problems)"), ("superhard", "SuperHard (126 problems)"))):
        for i, cell in enumerate(order):
            r = next(r for r in t07 if r["cell"] == cell and r["tier"] == tier)
            if cell == "base":
                col, mk, ls, filled = COLOR["base"], "D", "-", True
            else:
                a, s = arm_seed(cell)
                col, mk, ls, filled = COLOR[a], MARKER[s], LSTYLE[s], cell.endswith("r128")
            y = i + (0.5 if cell.endswith("r64") else 0)
            dot_ci(ax, y, 100 * r["pass1"], [100 * v for v in r["pass1_ci95"]], col, mk, ls, filled)
        if tier == "harder":
            ax.set_yticks([i + (0.5 if c.endswith("r64") else 0) for i, c in enumerate(order)],
                          [stress_label(c) for c in order])
            ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xticks([20, 40, 60, 80])
        hgrid(ax)
    fig.supxlabel("pass@1, 8 attempts per problem (%)", fontsize=10, y=0.06)
    handles = [Line2D([], [], color=MUTED, marker="o", ls="none", label="round 128 (endpoint)"),
               Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", label="round 64 (intermediate)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Stress panels: pass@1 per tier", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f10_stress_tiers")


def f11_stress_contrasts(plt, t08, fdir):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.9), sharey=True, sharex=True)
    for k, (ax, (tier, title)) in enumerate(zip(axes, (("all", "All (221)"), ("harder", "Harder (95)"),
                                                     ("superhard", "SuperHard (126)")))):
        contrast_rows(ax, [r for r in t08 if r["tier"] == tier], "diff", "ci95", ytick_labels=(k == 0))
        ax.set_title(title, loc="left")
        hgrid(ax)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("Difference in stress pass@1 (percentage points)")
    fig.suptitle("Stress panels: paired contrasts within each training seed", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f11_stress_contrasts")


def f12_control(plt, t09, fdir):
    order = ["H-s101", "H-s202", "R-s101", "R-s202", "U-s101", "U-s202"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.4), sharey=True, sharex=True)
    for k, (ax, (key, title)) in enumerate(zip(axes, (("control_gain", "Control panel"), ("J_U_gain", "J_U"),
                                                      ("J_R_gain", "J_R")))):
        for i, run in enumerate(order):
            r = next(r for r in t09 if r["run"] == run)
            a, s = arm_seed(run)
            dot_ci(ax, i, 100 * r[key], [100 * v for v in r[key + "_ci95"]], COLOR[a], MARKER[s], LSTYLE[s])
        if k == 0:
            ax.set_yticks(range(len(order)), [o + " r128" for o in order])
        ax.set_xticks([0, 10, 20, 30])
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.set_title(title, loc="left")
        hgrid(ax)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("Gain in pass@1 over the start policy (percentage points)")
    fig.suptitle("Gains on subjects no arm practises, beside the probe gains", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, fdir, "f12_control")


# ----------------------------------------------------------------------------------------------------------------
def build(root, tdir, fdir, counts_path):
    a26 = os.path.join(root, "research", "audit-a26")
    B = os.path.join(a26, "pass-b", "data")
    A = {k: load(os.path.join(B, k + ".json")) for k in
         ("eval_test", "eval_stress", "eval_length", "train_curves", "train_diag", "train_rewards")}
    cpath = os.path.join(a26, "transfer", "control.json")
    A["control"] = load(cpath) if os.path.exists(cpath) and os.path.getsize(cpath) > 0 else None
    cnt = load(counts_path)

    t01 = t01_endpoints(A, cnt, tdir)
    t02 = t02_contrasts(A, tdir)
    t03 = t03_curves(A, tdir)
    t04 = t04_diag(A, tdir)
    t05 = t05_length(A, cnt, tdir)
    t06 = t06_length_contrasts(A, tdir)
    t07 = t07_stress(A, cnt, tdir)
    t08 = t08_stress_contrasts(A, tdir)
    t09 = t09_control(A, tdir)

    plt = setup_mpl()
    f01_endpoints(plt, t01, fdir)
    f02_contrasts(plt, t02, fdir)
    f03_dev_rounds(plt, t03, fdir)
    f04_dev_hard(plt, t03, fdir)
    f05_traj_jr(plt, t03, fdir)
    f06_traj_ju(plt, t03, fdir)
    f07_diag(plt, t04, fdir)
    f08_length_pass1(plt, t05, fdir)
    f08b_length_capped(plt, t05, fdir)
    f09_length_contrasts(plt, t06, fdir)
    f10_stress(plt, t07, fdir)
    f11_stress_contrasts(plt, t08, fdir)
    if t09 is not None:
        f12_control(plt, t09, fdir)
    else:
        print("control.json absent: t09/f12 not built")
    print("tables ->", tdir, "; figures ->", fdir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--P", default=P_DEFAULT)
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    b.add_argument("--tables")
    b.add_argument("--figures")
    b.add_argument("--counts")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.P, args.out)
    else:
        tdir = args.tables or os.path.join(args.root, "reports", "tables", "v3")
        fdir = args.figures or os.path.join(args.root, "reports", "figures", "v3")
        build(args.root, tdir, fdir, args.counts or os.path.join(tdir, "perproblem_counts.json"))


if __name__ == "__main__":
    sys.exit(main())
