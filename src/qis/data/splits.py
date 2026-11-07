"""Near-duplicate grouping and deterministic internal splits (project.md Section 5).

1. Group exact and near-duplicate problems across ALL of MATH (train + test).
2. Drop the train-side members of every group that contains a test problem.
3. Split the remaining official-train groups 60/20/20 (construction / study_dev / calibration),
   stratified by (subject, level) of the group's first member, with the 1701 construction stream.
4. Split calibration groups again into calib_train / calib_val halves.
The official test split is kept whole for final evaluation only.
"""
import re
import zlib

import numpy as np

from qis import seeds

NUM_PERM = 64
BANDS, ROWS = 16, 4
SHINGLE = 5
JACCARD_DUP = 0.7


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-z]+", lambda m: m.group(0), text)
    text = re.sub(r"[^a-z0-9\\]+", " ", text)
    return " ".join(text.split())


def shingles(text: str) -> set:
    toks = normalize(text).split()
    if len(toks) < SHINGLE:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def minhash(sh: set, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ids = np.fromiter((zlib.crc32(s.encode()) for s in sh), dtype=np.uint64, count=len(sh))
    p = np.uint64((1 << 61) - 1)
    h = (np.outer(a, ids) + b[:, None]) % p
    return h.min(axis=1)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.p[max(x, y)] = min(x, y)


def duplicate_groups(problems: list) -> tuple:
    """Return (group index per problem, list of (i, j, jaccard) duplicate pairs)."""
    g = seeds.rng(seeds.CONSTRUCTION_SEED, "minhash")
    a = g.integers(1, (1 << 61) - 1, NUM_PERM, dtype=np.uint64)
    b = g.integers(0, (1 << 61) - 1, NUM_PERM, dtype=np.uint64)
    sh = [shingles(p) for p in problems]
    sig = np.stack([minhash(s, a, b) for s in sh])
    uf, pairs, seen = UnionFind(len(problems)), [], set()
    exact = {}
    for i, p in enumerate(problems):
        k = normalize(p)
        if k in exact:
            pairs.append((exact[k], i, 1.0)); seen.add((exact[k], i)); uf.union(exact[k], i)
        else:
            exact[k] = i
    for band in range(BANDS):
        buckets = {}
        for i in range(len(problems)):
            buckets.setdefault(sig[i, band * ROWS:(band + 1) * ROWS].tobytes(), []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    jac = len(sh[i] & sh[j]) / max(1, len(sh[i] | sh[j]))
                    if jac >= JACCARD_DUP:
                        pairs.append((i, j, round(jac, 3))); uf.union(i, j)
    return [uf.find(i) for i in range(len(problems))], pairs


def assign_splits(items: list) -> dict:
    """items: dicts from math_source.load_math in file order. Returns {id: record}."""
    group, pairs = duplicate_groups([it["problem"] for it in items])
    members = {}
    for k, gid in enumerate(group):
        members.setdefault(gid, []).append(k)
    out = {}
    train_groups = []
    for gid, ks in members.items():
        has_test = any(items[k]["orig_split"] == "test" for k in ks)
        for k in ks:
            it = items[k]
            rec = {"group": items[gid]["id"], "group_size": len(ks)}
            if it["orig_split"] == "test":
                rec["split"] = "test"
            elif has_test:
                rec["split"] = "excluded"; rec["reason"] = "near-duplicate of an official test problem"
            out[it["id"]] = rec
        if not has_test:
            train_groups.append(gid)
    strata = {}
    for gid in sorted(train_groups):
        first = items[gid]
        strata.setdefault((first["subject"], first["level"]), []).append(gid)
    for key in sorted(strata):
        gids = strata[key]
        perm = seeds.rng(seeds.CONSTRUCTION_SEED, f"split/{key[0]}/{key[1]}").permutation(len(gids))
        n = len(gids)
        n_con, n_dev = round(0.6 * n), round(0.2 * n)
        for rank, pi in enumerate(perm):
            gid = gids[pi]
            if rank < n_con:
                s = "construction"
            elif rank < n_con + n_dev:
                s = "study_dev"
            else:
                s = "calib_train" if (rank - n_con - n_dev) % 2 == 0 else "calib_val"
            for k in members[gid]:
                out[items[k]["id"]]["split"] = s
    for it in items:
        if it["gold"] is None and out[it["id"]]["split"] != "test":
            out[it["id"]]["note"] = "no boxed gold"
    dup_pairs = [(items[i]["id"], items[j]["id"], jac) for i, j, jac in pairs]
    return out, dup_pairs
