"""Print the markdown tables of train.md from b-train's JSON outputs (stdlib only). Usage: python3 tables.py DATA_DIR"""
import json
import sys

D = sys.argv[1]
L = lambda n: json.load(open(f"{D}/train_{n}.json"))
RUNS = ["H-s101", "R-s101", "U-s101", "H-s202", "R-s202", "U-s202"]
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:+.4f}"

print("### T1 curves")
c = L("curves")["curves"]
print("| run | round | cap | dev | dev trunc | J_H | J_H trunc | J_R | J_R trunc | J_U | J_U trunc |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    for r in ("0", "32", "64", "96", "128"):
        e = c[run][r]
        cells = [run, r, str(e["dev"]["cap_canonical_row"])]
        for p in ("dev", "probe_J_H", "probe_J_R", "probe_J_U"):
            cells += [f3(e[p]["pass1"]), f3(e[p]["trunc_share"])]
        print("| " + " | ".join(cells) + " |")
x = L("curves")
nc = x["noncanonical_evals"]
print("\n### T1b non-canonical evaluations")
for k, e in nc.items():
    print(k, {p: (f3(e[p]["pass1"]), f3(e[p]["trunc_share"])) for p in ("dev", "probe_J_H", "probe_J_R", "probe_J_U")})
print({p: (f4(v["diff"]), f4(v["lo"]), f4(v["hi"])) for p, v in x["H_s202_r64_orig_minus_r60"].items()})

print("\n### T2 ordering")
o = L("ordering")["ordering"]
print("| seed | panel | R−U r0 | r32 | r64 | r96 | r128 | first trained round with r128 order | stays from | interval at that round |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, v in o.items():
    s, p = k.split("|")
    d = v["R_minus_U"]
    fr = v["first_trained_round_same_order_as_128"]
    iv = d[str(fr)]
    print(f"| {s} | {p.replace('probe_', '')} | " + " | ".join(f4(d[r]["diff"]) for r in ("0", "32", "64", "96", "128")) +
          f" | {fr} | {v['earliest_round_from_which_order_stays']} | {f4(iv['diff'])} [{f4(iv['lo'])}, {f4(iv['hi'])}] |")
print("intervals excluding 0:", [(k, r, f4(d["lo"]), f4(d["hi"])) for k, v in o.items() for r, d in v["R_minus_U"].items() if d["lo"] > 0 or d["hi"] < 0])

print("\n### T3 rewards")
rw = L("rewards")
print("| run | rounds | hard_common success | practice/extra success | hard_common trunc | practice/extra trunc |")
print("|---|---|---|---|---|---|")
for run in RUNS:
    for w, v in rw[run]["windows"].items():
        other = "hard_extra" if run[0] == "H" else "aux"
        print(f"| {run} | {w} | {f3(v['success|hard_common'])} | {f3(v['success|' + other])} | {f3(v['trunc|hard_common'])} | {f3(v['trunc|' + other])} |")
print("equal exposure:", {run: {k: (vv["rounds_1based"][1], f3(vv["success_all_hard_mean"])) for k, vv in rw[run]["equal_hard_exposure"].items()} for run in RUNS})

print("\n### T4 groups and steps")
g = L("groups")
s = L("steps")
print("| run | all-wrong | mixed | all-correct | hard all-wrong / mixed / all-correct | practice or extra all-wrong / mixed / all-correct | momentum-only steps (of 512) | first (round) | by quarter | adv = 0 on all 32 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    t = g["groups"][run]["totals"]
    a = g["groups"][run]["totals_all_roles"]
    other = "hard_extra" if run[0] == "H" else "aux"
    hc = "/".join(str(t.get(f"hard_common|{c}", 0)) for c in ("all_wrong", "mixed", "all_correct"))
    ot = "/".join(str(t.get(f"{other}|{c}", 0)) for c in ("all_wrong", "mixed", "all_correct"))
    st = s[run]
    print(f"| {run} | {a['all_wrong']} | {a['mixed']} | {a['all_correct']} | {hc} | {ot} | {st['zero_grad_observed']} | {st['first_momentum_only_round_1based']} | {st['momentum_only_by_quarter']} | {st['momentum_only_all_adv_zero']} |")
for run in RUNS:
    st = s[run]
    print(run, "vt", st["valid_tokens_match"], "pred", st["zero_grad_predicted"], "obs", st["zero_grad_observed"], "mism", st["pred_obs_mismatch"],
          "clipped", st["grad_clipped_steps"], "med gn", f3(st["grad_norm_nonzero_median"]), "opt", st["opt_step_first_last"],
          "sup", st["superseded_step_rows_valid_tokens_match"], "comp", st["momentum_only_composition"])

print("\n### T5 diag")
dg = L("diag")
for lab in ("r0", "r64"):
    print(f"\n{lab}: identical groups {dg[lab]['identical_eval_groups']}")
    print("| branch | J_H | ΔJ_H | J_R | ΔJ_R | J_U | ΔJ_U | J_H trunc | update: reward, mixed, grad norm, capped |")
    print("|---|---|---|---|---|---|---|---|---|")
    ev, de = dg[lab]["evals"], dg[lab]["delta_vs_none"]
    for k in ["none", "none_rerun", "none_cycled"] + [f"{b}-{sd}" for b in ("hard", "related", "unrelated") for sd in (4101, 4102, 4103, 4104)]:
        cells = [k]
        for p in ("probe_J_H", "probe_J_R", "probe_J_U"):
            cells += [f3(ev[k][p]["pass1"]), "—" if k == "none" else f4(de[k][p]["diff"])]
        u = ev[k]["update_row"]
        cells += [f3(ev[k]["probe_J_H"]["trunc_share"]),
                  "—" if u["batch_reward"] is None else f"{u['batch_reward']:.3f}, {u['mixed_groups']}, {u['grad_norm_preclip']:.3f}, {u['capped']}"]
        print("| " + " | ".join(cells) + " |")
    print("| type | panel | mean ΔJ vs none | sd (4 draws) | mean ΔJ vs none_cycled | range of type means | exact perm p |")
    print("|---|---|---|---|---|---|---|")
    for p, t in dg[lab]["branch_types"].items():
        for bt in ("hard", "related", "unrelated"):
            print(f"| {bt} | {p.replace('probe_', '')} | {f4(t['type_means'][bt])} | {t['type_sd'][bt]:.4f} | {f4(t['type_means_vs_none_cycled'][bt])} | {t['range_of_type_means']:.4f} | {t['perm_p_range']:.3f} |")
        print(f"| none_rerun − none | {p.replace('probe_', '')} | {f4(t['rerun_minus_none'])} | | | | |")
print(dg["compare_inrun"])

print("\n### T6 loss/recovery")
lr = L("loss_recovery")
print("| run | scope | presentations | opportunities (solved, next exists) | losses | rate | recovered | not recovered by 128 | median rounds to recover | expected by sampling: local / pooled |")
print("|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    for sc in ("all_hard", "shared_only"):
        if run[0] != "H" and sc == "shared_only":
            continue
        v = lr[f"{run}|{sc}"]
        print(f"| {run} | {sc} | {v['presentations']} | {v['opportunities']} | {v['loss_episodes']} | {v['loss_rate']:.4f} | {v['recovered']} | {v['not_recovered_by_128']} | {v['median_rounds_to_recover']} | {v['expected_losses_sampling_local']:.1f} / {v['expected_losses_sampling_pooled']:.1f} |")

print("\n### T7 costs")
co = L("costs")
print("| category | GPU-h | CPU-h | jobs | allocation rows | GPU-h by state |")
print("|---|---|---|---|---|---|")
for k, v in co["categories"].items():
    print(f"| {k} | {v['gpu_h']:.2f} | {v['cpu_h']:.2f} | {v['n_jobs']} | {v['alloc_rows']} | " + ", ".join(f"{a} {b:.2f}" for a, b in sorted(v["by_state_gpu_h"].items()) if b > 0.005) + " |")
print(f"| total | {co['total_gpu_h']:.2f} | {co['total_cpu_h']:.2f} | {co['distinct_jobs']} | {co['alloc_rows']} | |")
ov = co["failed_superseded_overlay"]
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ov.items() if not isinstance(v, (list, dict))})
print("| run | all allocations | a1 | jobs without canonical rows | superseded rounds inside canonical jobs (lower bound) | canonical training (upper bound) |")
print("|---|---|---|---|---|---|")
for e, v in co["per_run_summary"].items():
    if v["all_allocations_gpu_h"] > 0:
        print(f"| {e[9:-4]} | {v['all_allocations_gpu_h']:.2f} | {v['a1_gpu_h']:.2f} | {v['jobs_without_canonical_rows_gpu_h']:.2f} | {v['superseded_inside_canonical_jobs_gpu_h_lower_bound']:.2f} | {v['canonical_training_gpu_h_upper_bound']:.2f} |")
