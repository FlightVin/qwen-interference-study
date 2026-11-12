"""A26 reconcile: number-by-number comparison of pass A and pass B wherever both computed the same quantity.
Laptop, stdlib only; reads only the two passes' JSON outputs. Usage: python3 compare.py AUDIT_DIR OUT_JSON

Agreement rule: counts and point estimates must be equal within 1e-9 (both passes compute exact rationals);
interval bounds agree when |A - B| <= 0.002 (different bootstrap RNG seeds: Monte-Carlo error)."""
import json
import sys

AD, OUT = sys.argv[1], sys.argv[2]
A = json.load(open(f"{AD}/pass-a/data/numbers.json"))
CL = {c["id"]: c for c in A["claims"]}
RC = A["recomputed"]
SA = json.load(open(f"{AD}/pass-a/data/a-data-schedule_audit.json"))["runs"]
BT = json.load(open(f"{AD}/pass-b/data/eval_test.json"))
BS = json.load(open(f"{AD}/pass-b/data/eval_stress.json"))
BL = json.load(open(f"{AD}/pass-b/data/eval_length.json"))
BC = json.load(open(f"{AD}/pass-b/data/eval_cue.json"))
BA = json.load(open(f"{AD}/pass-b/data/eval_audit.json"))
BG = json.load(open(f"{AD}/pass-b/data/train_groups.json"))
BST = json.load(open(f"{AD}/pass-b/data/train_steps.json"))
BCU = json.load(open(f"{AD}/pass-b/data/train_curves.json"))
BO = json.load(open(f"{AD}/pass-b/data/train_ordering.json"))
BD = json.load(open(f"{AD}/pass-b/data/train_diag.json"))
rows = []


def add(group, q, a, b, kind="exact"):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    if len(a) != len(b):
        ok, maxd = False, None
    else:
        ds = [abs(float(x) - float(y)) for x, y in zip(a, b)]
        maxd = max(ds)
        if kind == "exact":
            ok = maxd <= 1e-9
        else:  # [est, lo, hi]: est exact, bounds within 0.002
            ok = ds[0] <= 1e-9 and max(ds[1:]) <= 0.002
    rows.append({"group": group, "quantity": q, "A": a, "B": b, "kind": kind, "max_abs_diff": maxd, "agree": ok})


def ci(x):
    return [x["diff"], x["ci95"][0], x["ci95"][1]]


# --- sealed test cells ---
cellmapA = {"base": "4.2.base", "H-s101": "4.2.H-s101", "R-s101": "4.2.R-s101", "U-s101": "4.2.U-s101",
            "H-s202": "4.2.H-s202", "R-s202": "4.2.R-s202", "U-s202": "4.2.U-s202"}
for cell, cid in cellmapA.items():
    p1, p8, solved, unsolved, trunc = CL[cid]["recomputed_unrounded"]
    b = BT["cells"][cell]
    add("test cell", f"{cell}: pass@1, pass@8, solved>=1/32, truncated", [p1, p8, solved, trunc],
        [b["pass1"], b["pass8_unbiased"], b["solved_any"], b["truncated"]])
fb = CL["2.fallback"]["recomputed_unrounded"]; b = BT["cells"]["H-s202-fb-r116"]
add("test cell", "H-s202 r116 fallback: pass@1, pass@8, truncation share", fb, [b["pass1"], b["pass8_unbiased"], b["truncation_share"]])
for cell, v in zip(["R-s101", "U-s101", "R-s202", "U-s202"], CL["2.cgpu.cells"]["recomputed_unrounded"]):
    add("test cell", f"cgpu {cell}: pass@1", v, BT["cells"]["cgpu-" + cell]["pass1"])
# --- contrasts ---
for c in ("R-U", "R-H", "U-H"):
    for s in ("s101", "s202"):
        add("test contrast", f"{c} {s} [95%]", CL[f"2.{c}.{s}"]["recomputed_unrounded"], ci(BT["contrasts"][f"{c} {s}"]["pass1"]), "ci")
    add("test contrast", f"{c} two-seed mean (point)", CL[f"2.{c}.mean"]["recomputed_unrounded"][0], BT["two_seed_mean"][c])
for s in ("s101", "s202"):
    add("test contrast", f"cgpu R-U {s} [95%]", CL[f"2.cgpu.R-U.{s}"]["recomputed_unrounded"], ci(BT["contrasts"][f"cgpuR-cgpuU {s}"]["pass1"]), "ci")
for cell, gain in zip(["H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"], [None] * 6):
    s = cell.split("-")[1]
    pa = CL["4.2." + cell]["recomputed_unrounded"][0] - CL["4.2.base"]["recomputed_unrounded"][0]
    add("test contrast", f"{cell} - base (point)", pa, BT["contrasts"][f"{cell[0]}-base {s}"]["pass1"]["diff"])
# --- reruns ---
for r in ("R-s101", "R-s202", "U-s101", "U-s202"):
    ra = CL[f"2.repro.{r}"]["recomputed_unrounded"]; rb = BT["reruns"][r]
    add("cgpu rerun", f"{r}: identical token sequences, problems with equal c", ra, [rb["identical_responses"], rb["problems_identical_count"]])
    if f"2.repro.{r}.delta" in CL:
        add("cgpu rerun", f"{r}: rerun - original [95%]", CL[f"2.repro.{r}.delta"]["recomputed_unrounded"], ci(rb["rerun_minus_orig"]), "ci")
# --- decomposition ---
for s in ("s101", "s202"):
    da = RC[f"test.decomp.{s}"]; db = BT["contrasts"][f"R-U {s}"]["decomposition"]
    add("decomposition", f"R-U {s}: tR, tU, aR, aU, trunc term, acc term",
        [da["tA"], da["tB"], da["aA"], da["aB"], da["trunc_term"]["est"], da["acc_term"]["est"]],
        [db["tA"], db["tB"], db["aA"], db["aB"], db["truncation_term"], db["accuracy_term"]])
# --- coverage s202 ---
cv = CL["4.2.s202reach"]["recomputed_unrounded"]
cb = BT["contrasts"]["R-U s202"].get("coverage_split") or BT["contrasts"]["R-U s202"].get("coverage_vs_reliability")
if cb:
    add("coverage", "R-U s202: both-solved problems, their contribution", [cv[0], cv[1]], [cb["both"]["problems"], cb["both"]["contribution"]])
# --- stress cells and tiers ---
for cell in ("base", "H-s101-r128", "R-s101-r128", "U-s101-r128", "H-s202-r128", "R-s202-r128", "U-s202-r128", "R-s101-r64", "H-s202-r64"):
    a = CL["7.2." + cell]["recomputed_unrounded"]; b = BS["cells"][cell]
    add("stress cell", f"{cell}: pass@1, SuperHard, Harder, trunc, acc finished, solved/221", a,
        [b["all"]["pass1"], b["superhard"]["pass1"], b["harder"]["pass1"], b["all"]["truncation_share"],
         b["all"]["acc_among_finished"], b["all"]["solved_any"]])
    for tier in ("harder", "superhard"):
        ta = CL[f"7.2.tier.{cell}.{tier}"]["recomputed_unrounded"]; tb = b[tier]
        add("stress tier", f"{cell} {tier}: correct, truncated, solved>=1/8, acc finished",
            [ta["correct"], ta["truncated_n"], ta["pass8"], ta["acc_finished"]],
            [tb["correct"], tb["truncated"], tb["coverage_at_n"], tb["acc_among_finished"]])
SPAIR = {"R-s101-r128-U-s101-r128": "R-U s101 all", "R-s202-r128-U-s202-r128": "R-U s202 all",
         "R-s101-r128-H-s101-r128": "R-H s101 all", "U-s101-r128-H-s101-r128": "U-H s101 all",
         "R-s202-r128-H-s202-r128": "R-H s202 all", "U-s202-r128-H-s202-r128": "U-H s202 all",
         "R-s101-r128-R-s101-r64": "R-s101-r128 - R-s101-r64 all"}
for ka, kb in SPAIR.items():
    add("stress contrast", f"{kb} [95%]", CL[f"7.2.c.{ka}"]["recomputed_unrounded"], ci(BS["contrasts"][kb]["pass1"]), "ci")
    db = BS["contrasts"][kb]["decomposition"]
    add("stress contrast", f"{kb}: trunc term, acc term", [CL[f"7.2.c.{ka}.tt"]["recomputed_unrounded"][0], CL[f"7.2.c.{ka}.at"]["recomputed_unrounded"][0]],
        [db["truncation_term"], db["accuracy_term"]])
add("stress contrast", "H-s202 r128 - r64 (point)", -CL["7.2.h202drop"]["recomputed_unrounded"][0], BS["contrasts"]["H-s202-r128 - H-s202-r64 all"]["pass1"]["diff"])
for s in ("s101", "s202"):
    for tier in ("harder", "superhard"):
        add("stress contrast", f"R-U {s} {tier} [95%]", CL[f"Q3.tier.{tier}.{s}"]["recomputed_unrounded"], ci(BS["contrasts"][f"R-U {s} {tier}"]["pass1"]), "ci")
# --- length ---
for panel, sec in (("test", "5.2"), ("stress", "5.3")):
    for ck in ("H-s101", "R-s101", "U-s101", "R-s202", "U-s202"):
        a = CL[f"{sec}.{ck}"]["recomputed_unrounded"]; b = BL[panel][ck]["counts"]
        add("length row", f"{panel} {ck}: capped, recovered, finished wrong, still capped, pass@1 12k, 16k",
            [a["capped_1"], a["recovered"], a["finished_wrong"], a["still_capped"], a["pass1_1"], a["pass1_2"]],
            [b["capped_at_12288"], b["recovered"], b["finished_wrong"], b["still_capped"], b["pass1_12288"], b["pass1_16384"]])
LP = {"test R-s101 - U-s101": "5.test/all/all/R-s101-U-s101", "test R-s202 - U-s202": "5.test/all/all/R-s202-U-s202",
      "test R-s101 - H-s101": "5.test/all/all/R-s101-H-s101", "test U-s101 - H-s101": "5.test/all/all/U-s101-H-s101",
      "stress R-s101 - U-s101": "5.stress/all/all/R-s101-U-s101", "stress R-s202 - U-s202": "5.stress/all/all/R-s202-U-s202",
      "stress R-s101 - H-s101": "5.stress/all/all/R-s101-H-s101", "stress U-s101 - H-s101": "5.stress/all/all/U-s101-H-s101"}
for kb, ka in LP.items():
    for la, lb in (("at 12,288", "at_12288"), ("at 16,384", "at_16384"), ("change", "change")):
        add("length contrast", f"{kb} {la} [95%]", CL[f"{ka}.{la}"]["recomputed_unrounded"], ci(BL["contrasts"][kb][lb]), "ci")
# --- cue ---
ca = CL["8.cells"]["recomputed_unrounded"]  # R-s202 uncued, R-s202 cued, U-s202 uncued, U-s202 cued
add("cue", "s202 cells: R uncued, R cued, U uncued, U cued", ca,
    [BC["cells"][k]["pass1"] for k in ("R-s202-uncued", "R-s202-cued", "U-s202-uncued", "U-s202-cued")])
ga = CL["8.gain"]["recomputed_unrounded"]
gb = [BC["contrasts"]["R-s202 cued - uncued"]["diff"], BC["contrasts"]["U-s202 cued - uncued"]["diff"]]
dd = ["cue gain R - U s101", "cue gain R - U s202"]
gb += [BC["contrasts"][k]["diff"] for k in dd]
add("cue", "cue gain R-s202, U-s202, DiD s101, DiD s202 (" + ", ".join(dd) + ")", ga, gb)
# --- audit ---
aa = CL["7.3.denoms"]["recomputed_unrounded"]
add("audit", "responses, reads, reward-0, capped reward-0", aa[:4], [BA["responses_in_key"], BA["reads_total"], BA["reward0"], BA["reward0_capped"]])
# --- dev / probe ordering (R - U) ---
for s in ("s101", "s202"):
    for pa, pb in (("dev", "dev"), ("J_H", "probe_J_H"), ("J_R", "probe_J_R"), ("J_U", "probe_J_U")):
        oa = RC["ordering"][f"{s}.{pb}"]; ob = BO["ordering"][f"{s}|{pb}"]
        va = [v for _, v in oa["R_minus_U_by_round"]]
        vb = ob["R_minus_U"]
        vb = [vb[str(r)] if isinstance(vb, dict) and str(r) in vb else (vb[r] if isinstance(vb, dict) else None) for r in (0, 32, 64, 96, 128)] \
            if isinstance(vb, dict) else vb
        vb = [x["diff"] if isinstance(x, dict) else x for x in vb]
        add("in-run R-U", f"{s} {pa}: R-U at rounds 0/32/64/96/128", va, vb)
        add("ordering", f"{s} {pa}: earliest round with endpoint sign (A counts r0; B trained rounds only), round from which it holds",
            [oa["earliest_round_with_endpoint_sign"], oa["earliest_round_from_which_sign_holds"]],
            [ob["first_trained_round_same_order_as_128"], ob["earliest_round_from_which_order_stays"]])
h = CL["3.1.h202dev"]["recomputed_unrounded"]
add("dev curve", "H-s202 (a2-r60) dev r64, r96, r128", h,
    [BCU["curves"]["H-s202"][r]["dev"]["correct"] / BCU["curves"]["H-s202"][r]["dev"]["attempts"] for r in ("64", "96", "128")])
# --- groups and momentum-only steps ---
for run in ("H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"):
    ga = SA[f"qis-main-{run}-sym"]
    gb = BG["groups"][run]["totals_all_roles"]
    add("groups", f"{run}: all-wrong, mixed, all-correct", [ga["group_totals"]["all0"], ga["group_totals"]["mixed"], ga["group_totals"]["all1"]],
        [gb.get("all_wrong", gb.get("all0")), gb.get("mixed"), gb.get("all_correct", gb.get("all1"))])
    add("momentum-only", f"{run}: zero-gradient steps, first round", [ga["n_zero_grad_steps"], ga["first_zero_grad_round_1based"]],
        [len(BST[run]["momentum_only_steps"]), BST[run]["first_momentum_only_round_1based"]])
    sa = sorted((r + 1, m) for r, m in ga["zero_grad_steps"])
    sb = sorted((x["round_1based"], x["minibatch"]) for x in BST[run]["momentum_only_steps"])
    add("momentum-only", f"{run}: identical (round, minibatch) set of zero-gradient steps", 1 if sa == sb else 0, 1)
# --- diagnostic ---
for lab in ("r0", "r64"):
    da = RC[f"diag.{lab}"]
    none = da["none"][0]
    bt = BD[lab]["branch_types"]["probe_J_H"]
    vb = [bt["type_means"][t] for t in ("hard", "related", "unrelated")]
    add("diagnostic", f"{lab}: mean ΔJ_H vs none (hard, related, unrelated)", [da["means"][t] - none for t in ("hard", "related", "unrelated")], vb)
    add("diagnostic", f"{lab}: none_rerun - none (J_H)", da["ref"][0] - none, BD[lab]["delta_vs_none"]["none_rerun"]["probe_J_H"]["diff"])
json.dump(rows, open(OUT, "w"), indent=1)
n_ok = sum(r["agree"] for r in rows)
print(f"{len(rows)} comparisons, {n_ok} agree")
for r in rows:
    if not r["agree"]:
        print("DIFF", r["group"], r["quantity"], r["A"], r["B"], r["max_abs_diff"])
