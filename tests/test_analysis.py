import numpy as np
import pytest

from qis.analysis import stats as qs


def toy():
    probs = [f"p{i}" for i in range(5)]
    base = {p: 0.1 * i for i, p in enumerate(probs)}
    rates = {"R": {}, "U": {}}
    for k, s in enumerate((101, 202, 303)):
        rates["U"][s] = dict(base)
        rates["R"][s] = {p: v + 0.05 * (k + 1) for p, v in base.items()}   # R-U = .05,.10,.15
    return rates


def test_seed_contrasts_and_t_interval():
    sc = qs.seed_contrasts(toy(), "R", "U")
    assert sc == pytest.approx({101: 0.05, 202: 0.10, 303: 0.15})
    t = qs.t_interval(list(sc.values()))
    h = 4.302652729749464 * 0.05 / np.sqrt(3)   # t_{0.975,2}
    assert t["mean"] == pytest.approx(0.10) and t["hi"] - t["mean"] == pytest.approx(h)


def test_bootstrap_deterministic_and_paired():
    r = toy()
    b1, b2 = qs.crossed_bootstrap(r, "R", "U", draws=500), qs.crossed_bootstrap(r, "R", "U", draws=500)
    assert b1 == b2 and b1["mean"] == pytest.approx(0.10)
    # problem effects cancel under pairing: interval spans only the seed spread
    assert 0.05 - 1e-12 <= b1["lo"] <= b1["hi"] <= 0.15 + 1e-12


def test_unpaired_inputs_rejected():
    r = toy()
    del r["U"][303]
    with pytest.raises(ValueError):
        qs.seed_contrasts(r, "R", "U")
    r = toy()
    del r["R"][101]["p0"]
    with pytest.raises(ValueError):
        qs.crossed_bootstrap(r, "R", "U", draws=10)
