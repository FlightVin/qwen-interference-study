"""Per-round prompt schedules (project.md Section 7 "Primary matrix").

Arms: H = 8 common hard + 8 extra hard; R = 8 common hard + 8 related; U = 8 common hard + 8
unrelated; C (calibration) = 16 calibration-train prompts.
Within one seed, R and U share the common-hard stream and the same auxiliary PAIR stream
(pair p -> related[p] in R, unrelated[p] in U). H shares the common-hard stream with R and U.
Streams are shuffled cycles from named seed streams; the schedule is a pure function of
(seed, round), so resume only needs the round number. No prompt repeats within a round.
"""
from qis import seeds


class Cycle:
    """Infinite stream over range(n): concatenated seeded permutations, one per cycle."""

    def __init__(self, n: int, seed: int, name: str):
        self.n, self.seed, self.name = n, seed, name

    def perm(self, c: int):
        return seeds.rng(self.seed, f"{self.name}/cycle{c}").permutation(self.n)

    def take(self, start: int, k: int) -> list:
        out = []
        for pos in range(start, start + k):
            out.append(int(self.perm(pos // self.n)[pos % self.n]))
        return out


def round_plan(arm: str, seed: int, rnd: int, n_hard: int, n_pairs: int = 0, n_calib: int = 0):
    """Returns list of (role, pool, index) for the 16 prompts of round `rnd` (0-based)."""
    if arm == "C":
        idx = Cycle(n_calib, seed, "sched/calib").take(16 * rnd, 16)
        return [("calib", "calib_train", i) for i in idx]
    common = Cycle(n_hard, seed, "sched/hard_common").take(8 * rnd, 8)
    if len(set(common)) != 8:
        raise ValueError("hard pool smaller than 8 or cycle boundary duplicate")
    if arm == "H":
        # extra stream: walk the extra cycle, skipping items already used this round; the
        # number of positions consumed before round rnd is recomputed deterministically.
        extra_cycle = Cycle(n_hard, seed, "sched/hard_extra")
        pos = 0
        for r in range(rnd + 1):
            used = set(Cycle(n_hard, seed, "sched/hard_common").take(8 * r, 8))
            got = []
            while len(got) < 8:
                i = extra_cycle.take(pos, 1)[0]
                pos += 1
                if i in used or i in got:
                    continue   # skipped item; it reappears in the next cycle
                got.append(i)
        return [("hard_common", "hard", i) for i in common] + [("hard_extra", "hard", i) for i in got]
    pairs = Cycle(n_pairs, seed, "sched/aux_pairs").take(8 * rnd, 8)
    pool = {"R": "related", "U": "unrelated"}[arm]
    return [("hard_common", "hard", i) for i in common] + [("aux", pool, p) for p in pairs]
