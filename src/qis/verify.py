"""Reward verification (project.md Section 6 "Reward and failures").

Rules:
- Only the LAST \\boxed{...} of the response is the final answer; nothing else is extracted.
- A truncated response (hit the token cap) gets reward 0 whatever its text says.
- Math-Verify runs in a separate worker process. A timeout or crash is UNKNOWN, never wrong:
  the item is retried once in a fresh process with a longer limit; if still unresolved it
  stays "unknown" and the caller must hold the round (never drop or replace the response).
"""
import inspect
import multiprocessing as mp
import re
import time

from qis.data.math_source import last_boxed

TIMEOUT_S = 5.0
RETRY_TIMEOUT_S = 15.0

# status values
CORRECT, WRONG, NO_ANSWER, TRUNCATED, UNKNOWN = "correct", "wrong", "no_answer", "truncated", "unknown"
OVER_LEN = "over_len"


_WRAP = re.compile(r"\\(?:text|mathrm|mathbf)\{(.*)\}")
# trailing base label: _9, _{9}, _{(9)}, _{\text{9}}, _{base 9}, _{\text{base } 9}
_SUB = re.compile(r"(?P<num>.*?)\s*_\{?\s*\(?\s*(?:\\(?:text|mathrm)\{\s*)?(?:base\s*)?(?P<b>\d+)\s*\}?\s*\)?\s*\}?")
# trailing note: (base 9), base 9, \text{ (base 9)}, \text{base } 9
_NOTE = re.compile(r"(?P<num>.*?)\s*(?:\\(?:text|mathrm)\{)?\s*\(?\s*base\s*\}?\s*(?P<b>\d+)\s*\)?\s*\}?")
_SEP = re.compile(r"\\,|\\ |\{,\}|,|~|\s+")


def normalize_digits(s: str, base: int = None):
    """Exact-match normal form for procedural answers (study Nov-15-2026-02).
    Accepts: surrounding $ and one \\text{...} wrapper; a trailing base label (subscript or "(base b)" note), which
    must name `base` when `base` is given; parentheses around the numeral; thousands separators (",", "\\,", "{,}",
    "~", spaces) only in valid 3-digit grouping. Leading zeros are dropped. Returns the ASCII digit string, or None
    if anything else remains (letters, operators, decimals, a wrong base label, misplaced separators)."""
    t = s.strip().strip("$").strip()
    m = _WRAP.fullmatch(t)
    if m:
        t = m.group(1).strip()
    for rx in (_SUB, _NOTE):
        m = rx.fullmatch(t)
        if m:
            if base is not None and int(m.group("b")) != int(base):
                return None
            t = m.group("num").strip()
            break
    m = _WRAP.fullmatch(t)
    if m:
        t = m.group(1).strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    parts = [x for x in _SEP.split(t)]
    if len(parts) > 1:
        if any(not re.fullmatch(r"[0-9]+", x) for x in parts):
            return None
        if not (1 <= len(parts[0]) <= 3 and all(len(x) == 3 for x in parts[1:])):
            return None
        t = "".join(parts)
    if not re.fullmatch(r"[0-9]+", t):
        return None
    return t.lstrip("0") or "0"


def _mv():
    import math_verify
    from math_verify import LatexExtractionConfig, parse, verify
    cfg = [LatexExtractionConfig(boxed_match_priority=0)]
    pkw = {}
    if "parsing_timeout" in inspect.signature(parse).parameters:
        pkw["parsing_timeout"] = None          # outer process limit is authoritative
    vkw = {}
    vsig = inspect.signature(verify).parameters
    if "timeout_seconds" in vsig:
        vkw["timeout_seconds"] = None
    if "raise_on_error" in vsig:
        vkw["raise_on_error"] = True           # an internal error must not look like "wrong"
    return math_verify, parse, verify, cfg, pkw, vkw


def check_pair(gold: str, pred: str) -> dict:
    """Pure check, run inside a worker. Both are boxed CONTENTS (strings)."""
    _, parse, verify, cfg, pkw, vkw = _mv()
    g = parse("\\boxed{" + gold + "}", extraction_config=cfg, **pkw)
    p = parse("\\boxed{" + pred + "}", extraction_config=cfg, **pkw)
    if not g:
        return {"ok": False, "error": "gold_unparsed"}
    if not p:
        return {"ok": True, "correct": False, "pred_parsed": None}
    return {"ok": True, "correct": bool(verify(g, p, **vkw)), "pred_parsed": str(p[0])}


def _worker(conn):
    while True:
        msg = conn.recv()
        if msg is None:
            return
        try:
            if msg[0] == "__qis_test_sleep__":     # test hook: exercises the timeout path
                time.sleep(float(msg[1]))
            conn.send(check_pair(*msg))
        except Exception as e:  # noqa: BLE001 -- reported as unknown, not swallowed
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})


class Verifier:
    """A small pool of killable worker processes (spawn context: no forked CUDA state)."""

    def __init__(self, workers: int = 4):
        self.ctx = mp.get_context("spawn")
        self.n = workers
        self.procs = [self._spawn() for _ in range(workers)]

    def _spawn(self):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker, args=(b,), daemon=True)
        p.start()
        return [p, a]

    def _kill(self, i):
        p, _ = self.procs[i]
        p.kill()
        p.join(5)
        self.procs[i] = self._spawn()

    def close(self):
        for p, c in self.procs:
            try:
                c.send(None)
            except Exception:  # noqa: BLE001
                pass
            p.join(2)
            if p.is_alive():
                p.kill()

    def _run(self, jobs, timeout):
        """jobs: list of (key, gold, pred). Returns {key: result-dict}; timeouts -> error."""
        out, queue, busy = {}, list(jobs), {}
        while queue or busy:
            for i in range(self.n):
                if i not in busy and queue:
                    key, g, p = queue.pop(0)
                    self.procs[i][1].send((g, p))
                    busy[i] = (key, time.monotonic())
            for i, (key, t0) in list(busy.items()):
                conn = self.procs[i][1]
                if conn.poll(0.002):
                    try:
                        out[key] = conn.recv()
                    except EOFError:
                        out[key] = {"ok": False, "error": "worker_crashed"}
                        self._kill(i)
                    del busy[i]
                elif not self.procs[i][0].is_alive():
                    out[key] = {"ok": False, "error": "worker_crashed"}
                    self._kill(i)
                    del busy[i]
                elif time.monotonic() - t0 > timeout:
                    out[key] = {"ok": False, "error": f"timeout_{timeout:g}s"}
                    self._kill(i)
                    del busy[i]
        return out

    def score(self, items):
        """items: list of dicts with gold, response, truncated. Returns list of dicts with
        reward (0/1 or None for unknown), status, extracted, diagnostic.
        Optional item keys (study Nov-15-2026-02): match="digits" grades by exact comparison of normalize_digits()
        in-process (no Math-Verify); len_limit=L with n_tokens gives reward 0 and status "over_len" to a correct
        response longer than L tokens.
        """
        res = [None] * len(items)
        jobs = []
        for k, it in enumerate(items):
            ext = last_boxed(it["response"])
            res[k] = {"extracted": ext}
            if ext is None:
                res[k].update(status=TRUNCATED if it["truncated"] else NO_ANSWER, reward=0)
            elif it.get("match") == "digits":
                ok = normalize_digits(ext, it.get("base")) == normalize_digits(it["gold"])
                if it["truncated"]:
                    res[k].update(status=TRUNCATED, reward=0, would_match=ok)
                elif ok and it.get("len_limit") is not None and it["n_tokens"] > it["len_limit"]:
                    res[k].update(status=OVER_LEN, reward=0)
                else:
                    res[k].update(status=CORRECT if ok else WRONG, reward=1 if ok else 0)
            else:
                jobs.append((k, it["gold"], ext))
        first = self._run(jobs, TIMEOUT_S)
        retry = [(k, g, p) for k, g, p in jobs if not first[k]["ok"]]
        second = self._run(retry, RETRY_TIMEOUT_S) if retry else {}
        for k, g, p in jobs:
            r = second.get(k, first[k])
            diag = [first[k].get("error")] + ([second[k].get("error")] if k in second else [])
            if not r["ok"]:
                res[k].update(status=UNKNOWN, reward=None, diagnostic=[d for d in diag if d])
            elif items[k]["truncated"]:
                # capped response: reward 0 even if the visible fragment matches; log separately
                res[k].update(status=TRUNCATED, reward=0, would_match=r["correct"])
            else:
                res[k].update(status=CORRECT if r["correct"] else WRONG,
                              reward=1 if r["correct"] else 0,
                              retried=k in second, pred_parsed=r.get("pred_parsed"))
        return res


def gold_eligible(verifier: Verifier, gold: str) -> dict:
    """A gold answer is eligible when it parses and verifies against itself."""
    r = verifier.score([{"gold": gold, "response": "\\boxed{" + gold + "}", "truncated": False}])[0]
    return {"eligible": r["status"] == CORRECT, "status": r["status"], "diag": r.get("diagnostic")}
