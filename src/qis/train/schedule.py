"""Per-round prompt schedules (project.md Section 7 "Primary matrix").

Arms: H = 8 common hard + 8 extra hard; R = 8 common hard + 8 related; U = 8 common hard + 8
unrelated; C (calibration) = 16 calibration-train prompts.
Within one seed, R and U share the common-hard stream and the same auxiliary PAIR stream
(pair p -> related[p] in R, unrelated[p] in U). H shares the common-hard stream with R and U.
Streams are shuffled cycles from named seed streams; the schedule is a pure function of
(seed, round), so resume only needs the round number. No prompt repeats within a round.
"""
from qis import seeds


from functools import lru_cache


@lru_cache(maxsize=None)
def _perm(n: int, seed: int, name: str, c: int) -> tuple:
    return tuple(int(i) for i in seeds.rng(seed, f"{name}/cycle{c}").permutation(n))


@lru_cache(maxsize=None)
def _draws(n: int, seed: int, name: str, k: int, rnd: int, excl_name: str = "") -> tuple:
    """Round `rnd` draw of k DISTINCT items from a shuffled-cycle stream over range(n).

    The stream is walked in order; an item already drawn this round (or excluded by the
    `excl_name` stream's draw for the same round) is skipped and will reappear next cycle.
    The stream position before round rnd is reproduced by replaying rounds 0..rnd-1, so the
    draw is a pure function of its arguments."""
    if k > n:
        raise ValueError(f"cannot draw {k} distinct items from {n}")
    pos = 0 if rnd == 0 else _draws(n, seed, name, k, rnd - 1, excl_name)[1]
    excluded = set(_draws(n, seed, excl_name, k, rnd)[0]) if excl_name else set()
    if len(excluded) + k > n:
        raise ValueError("pool too small for exclusion")
    got = []
    while len(got) < k:
        i = _perm(n, seed, name, pos // n)[pos % n]
        pos += 1
        if i in excluded or i in got:
            continue
        got.append(i)
    return tuple(got), pos


def round_plan(arm: str, seed: int, rnd: int, n_hard: int, n_pairs: int = 0, n_calib: int = 0):
    """Returns list of (role, pool, index) for the 16 prompts of round `rnd` (0-based)."""
    if arm == "C":
        return [("calib", "calib_train", i) for i in _draws(n_calib, seed, "sched/calib", 16, rnd)[0]]
    common = _draws(n_hard, seed, "sched/hard_common", 8, rnd)[0]
    plan = [("hard_common", "hard", i) for i in common]
    if arm == "H":
        extra = _draws(n_hard, seed, "sched/hard_extra", 8, rnd, "sched/hard_common")[0]
        return plan + [("hard_extra", "hard", i) for i in extra]
    pairs = _draws(n_pairs, seed, "sched/aux_pairs", 8, rnd)[0]
    pool = {"R": "related", "U": "unrelated"}[arm]
    return plan + [("aux", pool, p) for p in pairs]
