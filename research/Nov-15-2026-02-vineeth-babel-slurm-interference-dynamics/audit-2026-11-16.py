"""Independent audit of report numbers (stdlib only; does not import project code)."""
import json, random, sys
RES = sys.argv[1]
def ev(run):
    E = {}
    for l in open(f"{RES}/{run}/metrics.jsonl"):
        r = json.loads(l)
        if r["kind"] == "eval":
            pp = {}
            for x in r["per_problem"]:
                pp[x["id"]] = x["c"] / x["n"]
            E[(r["panel"], r["round"])] = (r["pass1"], pp, sum(x["n"] for x in r["per_problem"]), len(r["per_problem"]))
    return E
def boot(d, B=2000):
    rnd = random.Random(1); n = len(d)
    bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B) - 1]
def delta(Ea, ka, Eb, kb):           # Ea[ka] - Eb[kb], paired over shared problems
    a, b = Ea[ka][1], Eb[kb][1]; ids = sorted(set(a) & set(b))
    return boot([a[i] - b[i] for i in ids]) + (len(ids),)
rows = []
def check(label, claim, val, tol=0.011):
    ok = abs(val - claim) <= tol
    rows.append((label, claim, round(val, 3), "OK" if ok else "MISMATCH"))
R1, R2 = ev("qis-s02-e1-caesar-s1"), ev("qis-s02-e1-caesar-s2")
J1, J2 = ev("qis-s02-j-caesar-terse-s1"), ev("qis-s02-j-caesar-terse-s2")
JM, RL = ev("qis-s02-j-caesar-mbpp-s1"), ev("qis-s02-e1-caesar-lr1e6-s1")
RM = ev("qis-s02-e1-mbpp-s1")
# F3 / C-1, C-2
check("F3 caesar s12 r32 (s1)", .707, R1[("caesar_s12_sel", 32)][0]); check("F3 caesar s12 r32 (s2)", .64, R2[("caesar_s12_sel", 32)][0], .006)
check("F3 terse delta s1", -.336, delta(R1, ("terse10_k2_sel", 32), R1, ("terse10_k2_sel", 0))[0])
# F6 / F11 D(r16)
for lab, R, J, st, cl in (("F6 D s8 r16", R1, J1, "s8", .18), ("F6 D s12 r16", R1, J1, "s12", .105),
                          ("F11 D s8 r16", R2, J2, "s8", .12), ("F11 D s12 r16", R2, J2, "s12", .15), ("F11 D s8 r32", R2, J2, "s8", .12)):
    rr = 32 if "r32" in lab else 16
    check(lab, cl, delta(R, (f"caesar_{st}_sel", rr), J, (f"caesar_{st}_sel", rr))[0], .006)
# F15 / F17
for st, cl in (("s4", .10), ("s8", .09), ("s12", .12)):
    check(f"F15 D {st} r32", cl, delta(R1, (f"caesar_{st}_sel", 32), JM, (f"caesar_{st}_sel", 32))[0])
check("F15 MBPP D r32", .03, delta(RM, ("mbpp_all_sel", 32), JM, ("mbpp_all_sel", 32))[0])
for st, cl in (("s4", .49), ("s8", .30), ("s12", .30)):
    check(f"F17 half-lr gain {st}", cl, delta(RL, (f"caesar_{st}_sel", 32), RL, (f"caesar_{st}_sel", 0))[0])
# F18
JT, RA = ev("qis-s02-j-add9-terse-s1"), ev("qis-s02-e1-add9-s1")
check("F18 J d6 r32", .52, JT[("add9_d6_sel", 32)][0]); check("F18 J terse r32", .93, JT[("terse10_k2_sel", 32)][0])
check("F18 D d6 r32", -.47, delta(RA, ("add9_d6_sel", 32), JT, ("add9_d6_sel", 32))[0])
# denominators: every sel panel 64 problems x 4
bad = [(run, k) for run, E in (("R1", R1), ("J1", J1), ("JM", JM), ("JT", JT)) for k, v in E.items()
       if k[0].endswith("_sel") and (v[3] != 64 or v[2] != 256)]
for r in rows: print("%-26s claim %+.3f  recomputed %+.3f  %s" % r)
print("rows", len(rows), "mismatches", sum(r[3] != "OK" for r in rows), "| sel panels not 64x4:", bad[:5])
