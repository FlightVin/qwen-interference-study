"""Named, stable random streams derived from a master seed (project.md Section 8).

Never use Python's randomized hash(). Each stream name maps to a SeedSequence spawn key
through SHA-256, so the same (master, name) always gives the same stream.
"""
import hashlib

import numpy as np

CONSTRUCTION_SEED = 1701
BOOTSTRAP_SEED = 2027


def name_key(name: str) -> tuple:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))


def seed_sequence(master: int, name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=master, spawn_key=name_key(name))


def rng(master: int, name: str) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(master, name))


def int_seed(master: int, name: str) -> int:
    """A 31-bit integer seed for libraries that need an int (torch, vLLM requests)."""
    return int(seed_sequence(master, name).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)
