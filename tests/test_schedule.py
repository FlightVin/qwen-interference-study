from qis.train.schedule import round_plan


def test_rounds_unique_and_deterministic():
    for arm in "HRU":
        for r in range(40):
            p = round_plan(arm, 101, r, n_hard=64, n_pairs=64)
            assert p == round_plan(arm, 101, r, n_hard=64, n_pairs=64)
            assert len(p) == 16
            keys = [(pool, i) for _, pool, i in p]
            assert len(set(keys)) == 16


def test_paired_streams():
    for r in range(20):
        R = round_plan("R", 202, r, 64, 64)
        U = round_plan("U", 202, r, 64, 64)
        H = round_plan("H", 202, r, 64, 64)
        assert [x for x in R if x[0] == "hard_common"] == [x for x in U if x[0] == "hard_common"]
        assert [x for x in R if x[0] == "hard_common"] == [x for x in H if x[0] == "hard_common"]
        assert [i for role, _, i in R if role == "aux"] == [i for role, _, i in U if role == "aux"]
        assert {pool for role, pool, _ in R if role == "aux"} == {"related"}
        assert {pool for role, pool, _ in U if role == "aux"} == {"unrelated"}


def test_cycles_cover_pool():
    seen = []
    for r in range(8):                      # 8 rounds x 8 common = one cycle of 64
        seen += [i for role, _, i in round_plan("R", 7, r, 64, 64) if role == "hard_common"]
    assert sorted(seen) == list(range(64))
    seeds_differ = round_plan("R", 101, 0, 64, 64) != round_plan("R", 202, 0, 64, 64)
    assert seeds_differ


def test_h_extra_balanced_over_time():
    from collections import Counter
    c = Counter()
    for r in range(64):
        c.update(i for _, _, i in round_plan("H", 101, r, 64))
    # 64 rounds x 16 = 1024 slots over 64 items: each item close to 16 exposures
    assert min(c.values()) >= 12 and max(c.values()) <= 20


def test_calibration_arm():
    p = round_plan("C", 11, 3, 0, n_calib=40)
    assert len({i for *_, i in p}) == 16 and {pool for _, pool, _ in p} == {"calib_train"}
