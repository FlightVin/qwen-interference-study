from qis import seeds
from qis.data import splits
from qis.data.math_source import last_boxed


def test_last_boxed_nested():
    assert last_boxed("a \\boxed{1} b \\boxed{\\frac{a}{b}} c") == "\\frac{a}{b}"
    assert last_boxed("\\fbox{3}") == "3"
    assert last_boxed("none") is None
    assert last_boxed("\\boxed{unclosed") is None


def test_seed_streams_stable_and_distinct():
    assert seeds.int_seed(1701, "a") == seeds.int_seed(1701, "a")
    assert seeds.int_seed(1701, "a") != seeds.int_seed(1701, "b")


def _items():
    words = ("apple river seven blue quartz tiger number circle prime square cube even odd "
             "root sum product digit angle line point").split()
    g = seeds.rng(0, "test-words")
    base = [" ".join(words[j] for j in g.integers(0, len(words), 30)) for _ in range(200)]
    items = []
    for i, p in enumerate(base):
        items.append({"id": f"algebra/train/{i:05d}", "subject": "algebra", "orig_split": "train",
                      "level": f"Level {1 + i % 5}", "problem": p, "gold": "1"})
    # a test item duplicating train item 3, and a train duplicate of train item 4
    items.append({"id": "algebra/test/00000", "subject": "algebra", "orig_split": "test",
                  "level": "Level 1", "problem": base[3], "gold": "1"})
    items.append({"id": "algebra/train/00999", "subject": "algebra", "orig_split": "train",
                  "level": "Level 2", "problem": base[4] + " ", "gold": "1"})
    return items


def test_splits_deterministic_and_grouped():
    items = _items()
    r1, p1 = splits.assign_splits(items)
    r2, p2 = splits.assign_splits(items)
    assert r1 == r2 and p1 == p2
    assert r1["algebra/train/00003"]["split"] == "excluded"
    assert r1["algebra/test/00000"]["split"] == "test"
    assert r1["algebra/train/00999"]["split"] == r1["algebra/train/00004"]["split"]
    c = {}
    for v in r1.values():
        c[v["split"]] = c.get(v["split"], 0) + 1
    n = sum(c[s] for s in ("construction", "study_dev", "calib_train", "calib_val"))
    assert abs(c["construction"] / n - 0.6) < 0.05 and abs(c["study_dev"] / n - 0.2) < 0.05
