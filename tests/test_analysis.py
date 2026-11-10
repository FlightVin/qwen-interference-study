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


def test_report_tables_synthetic(tmp_path):
    """A9 table on synthetic summaries: paired differences, pass@8, missing cells (no real test data)."""
    import json
    import numpy as np
    from qis.analysis import report_tables as rt
    rng = np.random.default_rng(0)
    ids = [f"t{i}" for i in range(128)]
    def write(name, shift):
        d = tmp_path / name; d.mkdir()
        pp = [{"id": i, "c": int(min(32, max(0, rng.binomial(32, 0.4) + shift))), "n": 32} for i in ids]
        (d / "summary.json").write_text(json.dumps({"per_problem": pp}))
        return pp
    pps = {(a, s): write(f"qis-test-{a}-s{s}-sym", {"H": 0, "R": 3, "U": 1}[a]) for a in "HRU" for s in (101, 202)
           if (a, s) != ("H", 202)}                                      # H-s202 missing on purpose
    write("qis-test-base", -2)
    rt.main(str(tmp_path), str(tmp_path / "t"))
    res = json.load(open(tmp_path / "t.json"))
    r101 = res["rows"][0]
    exp = np.mean([x["c"] / 32 for x in pps[("R", 101)]]) - np.mean([x["c"] / 32 for x in pps[("U", 101)]])
    assert abs(r101["R-U"]["est"] - exp) < 1e-12 and r101["R-U"]["lo"] <= exp <= r101["R-U"]["hi"]
    assert res["rows"][1]["H"] == "missing" and res["rows"][1]["R-H"] == "missing" and res["mean"]["R-H"] == "missing"
    assert isinstance(res["mean"]["R-U"], dict) and res["agreement"]["R-U"]["same_sign"] in (True, False)
    assert abs(rt.pass8(2, 32) - (1 - (30 * 29 * 28 * 27 * 26 * 25 * 24 * 23) / (32 * 31 * 30 * 29 * 28 * 27 * 26 * 25))) < 1e-12
    assert rt.pass8(30, 32) == 1.0 and rt.pass8(0, 32) == 0.0
    assert "pass@8" in open(tmp_path / "t.md").read()


def _write_gz(path, rows):
    import gzip
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_per_problem_bands_and_pairs(tmp_path):
    import json
    from qis.analysis import per_problem as pp
    ids = [f"t{i}" for i in range(20)]
    def recs(fn):
        return [{"prompt_id": p, "sample": j, "reward": int(fn(i, j))} for i, p in enumerate(ids) for j in range(32)]
    _write_gz(tmp_path / "qis-test-base" / "records.jsonl.gz", recs(lambda i, j: j < i))           # base c = min(i, 32)
    _write_gz(tmp_path / "qis-test-R-s101-sym" / "records.jsonl.gz", recs(lambda i, j: j < i + 4))
    _write_gz(tmp_path / "qis-test-U-s101-sym" / "records.jsonl.gz", recs(lambda i, j: j < i + 1))
    pp.main(str(tmp_path), str(tmp_path / "pp"))
    res = json.load(open(tmp_path / "pp.json"))
    assert res["bands"]["B0"]["size"] == 1 and sum(b["size"] for b in res["bands"].values()) == 20
    assert res["bands"]["B4"]["size"] == 4                                    # i = 16..19 have all of samples 0-15 correct
    assert abs(res["paired"]["R-U/s101"]["pass1"]["est"] - 3 / 32) < 1e-12
    assert res["paired"]["R-H/s101"] == "missing" and res["per_problem"]["t0"]["base"] == "0/32"
    assert res["arms"]["R-s101"]["improved"] == 20 and pp.pass8(0) == 0.0


def test_interference_windows_and_transitions(tmp_path):
    import json
    from qis.analysis import interference as itf
    def rnd_rows(pid, c, capped=0):
        return [{"role": "hard_common", "slot": 0, "prompt_id": pid, "sample": j, "reward": int(j < c),
                 "finish": "length" if j >= 8 - capped else "stop", "n_tokens": 10} for j in range(8)]
    run = tmp_path / "qis-main-R-s101-sym"
    # p1: mixed (r1) -> all-correct (r2) = mastery; p2: mixed (r3) -> all-wrong with a capped answer (r40) = signal loss
    for rnd, rows in ((1, rnd_rows("p1", 3)), (2, rnd_rows("p1", 8)), (3, rnd_rows("p2", 2)), (40, rnd_rows("p2", 0, capped=1))):
        _write_gz(run / "records" / f"train_r{rnd - 1:04d}.jsonl.gz", rows)
    (run / "metrics.jsonl").write_text("")
    itf.main(str(tmp_path), str(tmp_path / "i.json"))
    v = json.load(open(tmp_path / "i.json"))["runs"]["qis-main-R-s101-sym"]
    assert v["windows"]["1-16"]["groups"] == 3 and v["windows"]["33-48"]["all_wrong_capped"] == 1.0
    assert v["from_mixed"]["1-16"]["to_all_correct"] == 1 and v["from_mixed"]["33-48"]["to_all_wrong"] == 1
    assert v["previously_solvable"]["33-48"]["all_wrong_capped"] == 1


def test_cue_gain_and_did(tmp_path, monkeypatch):
    import gzip, json, sys, types
    from qis.analysis import cue
    fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda m: types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "factor " * len(ids))))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    ids = [f"q{i}" for i in range(24)]
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"pools": {"cue": [{"id": i} for i in ids]}}))
    # correct answers per item out of 8: base 2/2, R 2->6 (gain .5), U 2->4 (gain .25); seed 202 missing
    k = {("base", "uncued"): 2, ("base", "cued"): 2, ("R-s101", "uncued"): 2, ("R-s101", "cued"): 6,
         ("U-s101", "uncued"): 2, ("U-s101", "cued"): 4}
    for (pol, form), c in k.items():
        d = tmp_path / "out" / f"{pol}-{form}"; d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
        with gzip.open(d / "records.jsonl.gz", "wt") as f:
            for i in ids:
                for j in range(8):
                    f.write(json.dumps({"prompt_id": i, "sample": j, "n_tokens": 3, "finish": "stop",
                                        "reward": int(j < c), "ids": [1, 2, 3]}) + "\n")
    cue.main(str(tmp_path / "out"), str(panel), "m", str(tmp_path / "cue"))
    r = json.load(open(tmp_path / "cue.json"))
    assert r["cue_gain"]["R-s101"]["est"] == pytest.approx(0.5) and r["cue_gain"]["U-s101"]["est"] == pytest.approx(0.25)
    assert r["did"]["s101"]["est"] == pytest.approx(0.25) and r["did"]["s202"] == "missing"
    assert r["policies"]["base"]["cued"]["mentions_factor"] == 1.0 and len(r["missing"]) == 4
    assert (tmp_path / "cue.txt").read_text().count("=== ") == 6 * 4
