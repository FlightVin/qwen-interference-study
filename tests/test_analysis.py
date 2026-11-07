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


def _write_run(d, name, b, e, trunc=0.0):
    import json
    with open(d / f"cal8k-{name}.metrics.jsonl", "w") as f:
        for rnd, p in ((0, b), (32, e)):
            f.write(json.dumps({"kind": "eval", "round": rnd, "pass1": p, "truncated": trunc, "unknown": 0}) + "\n")
        f.write(json.dumps({"kind": "step", "loss": 0.1}) + "\n")
        f.write(json.dumps({"kind": "event", "what": "complete"}) + "\n")


def test_calib_select_tie_prefers_1e6_and_rejects_truncation(tmp_path):
    from qis.analysis import calib_select as cs
    gains = {"lr5e-7-sym": .03, "lr5e-7-asym": .0, "lr1e-6-sym": .12, "lr1e-6-asym": .06, "lr2e-6-sym": .125}
    for c, g in gains.items():
        _write_run(tmp_path, f"{c}-s11", .5, .5 + g)
    _write_run(tmp_path, "lr2e-6-asym-s11", .5, .9, trunc=0.07)          # best gain but rejected
    _write_run(tmp_path, "lr2e-6-sym-s22", .5, .62)
    _write_run(tmp_path, "lr1e-6-sym-s22", .5, .62)
    out = cs.select(str(tmp_path))
    assert out["ranking"][:2] == ["lr2e-6-sym", "lr1e-6-sym"] and "lr2e-6-asym" not in out["ranking"]
    assert out["status"] == "selected" and out["tie"] and out["selected"] == "lr1e-6-sym"   # 0.1225 vs 0.12: tie


def test_all_boxed_split_and_nested():
    from qis.analysis.audit_export import all_boxed
    assert all_boxed("so $\\boxed{8}, \\boxed{12}$") == ["8", "12"]
    assert all_boxed("\\boxed{\\frac{1}{2}} and none") == ["\\frac{1}{2}"]
    assert all_boxed("no box") == []
