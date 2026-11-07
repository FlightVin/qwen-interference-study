"""Primary-contrast statistics (project.md §9; PROTOCOL §9).

Input: per-run per-problem success rates on one panel, `rates[arm][seed] = {problem_id: c/n}`.
Seeds are paired across arms (arm A seed s vs arm B seed s share hard/pair streams); problems are
paired across all runs. Every run must cover the same problem set, equally weighted.
- `seed_contrasts`: per-seed A − B of the problem-mean pass rate.
- `t_interval`: two-sided paired t interval over seeds (df = n_seeds − 1).
- `crossed_bootstrap`: resample seeds and problems independently with replacement, the same draw
  for both arms (pairing preserved); percentile interval; fixed RNG seed (2027 by protocol).
"""
import numpy as np

# Two-sided 95% Student t critical values t_{0.975, df} (scipy is not in the frozen environment).
T975 = {1: 12.706204736174698, 2: 4.302652729749464, 3: 3.182446305284263, 4: 2.7764451051977934,
        5: 2.570581835636314, 6: 2.4469118511449692, 7: 2.3646242515927844, 8: 2.306004135204166,
        9: 2.2621571628540993, 10: 2.2281388519649385}

BOOT_SEED = 2027
BOOT_DRAWS = 10_000


def matrix(rates, arm, seeds, problems):
    """seeds x problems array of success rates; fails loudly on any missing cell."""
    return np.array([[rates[arm][s][p] for p in problems] for s in seeds], dtype=np.float64)


def common_layout(rates, a, b):
    seeds = sorted(set(rates[a]) & set(rates[b]))
    if seeds != sorted(rates[a]) or seeds != sorted(rates[b]):
        raise ValueError(f"unpaired seeds: {sorted(rates[a])} vs {sorted(rates[b])}")
    problems = sorted(rates[a][seeds[0]])
    for arm in (a, b):
        for s in seeds:
            if sorted(rates[arm][s]) != problems:
                raise ValueError(f"problem set differs for {arm} seed {s}")
    return seeds, problems


def seed_contrasts(rates, a, b):
    seeds, problems = common_layout(rates, a, b)
    d = matrix(rates, a, seeds, problems).mean(1) - matrix(rates, b, seeds, problems).mean(1)
    return dict(zip(seeds, d.tolist()))


def t_interval(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    h = float(T975[n - 1] * se)
    return {"mean": m, "se": se, "lo": m - h, "hi": m + h, "n": n, "df": n - 1}


def crossed_bootstrap(rates, a, b, draws=BOOT_DRAWS, seed=BOOT_SEED, level=0.95):
    seeds, problems = common_layout(rates, a, b)
    D = matrix(rates, a, seeds, problems) - matrix(rates, b, seeds, problems)   # paired cell differences
    rng = np.random.default_rng(seed)
    ns, npb = D.shape
    out = np.empty(draws)
    for i in range(draws):
        si = rng.integers(0, ns, ns)
        pi = rng.integers(0, npb, npb)
        out[i] = D[np.ix_(si, pi)].mean()
    q = (1 - level) / 2
    return {"mean": float(D.mean()), "lo": float(np.quantile(out, q)), "hi": float(np.quantile(out, 1 - q)),
            "draws": draws, "seed": seed, "n_seeds": ns, "n_problems": npb}


def contrast_report(rates, a, b):
    sc = seed_contrasts(rates, a, b)
    return {"contrast": f"{a}-{b}", "per_seed": sc, "t": t_interval(list(sc.values())),
            "bootstrap": crossed_bootstrap(rates, a, b)}


def per_problem_change(rates, a, b):
    """Seed-mean per-problem A − B (for the per-problem change figure)."""
    seeds, problems = common_layout(rates, a, b)
    d = (matrix(rates, a, seeds, problems) - matrix(rates, b, seeds, problems)).mean(0)
    return dict(zip(problems, d.tolist()))
