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


def normalize_string(s: str):
    """Exact-match normal form for string answers: one \text{...} wrapper, quotes and spaces removed, lower case."""
    t = s.strip().strip("$").strip()
    m = _WRAP.fullmatch(t)
    if m:
        t = m.group(1).strip()
    t = t.strip("'\"`").replace(" ", "").lower()
    return t if re.fullmatch(r"[a-z]+", t) else None


def normalize_choice(s: str):
    """Option letter A-D from "B", "(B)", "\text{B}", "B)" or "B. text"; None otherwise."""
    t = s.strip().strip("$").strip()
    m = _WRAP.fullmatch(t)
    if m:
        t = m.group(1).strip()
    m = re.fullmatch(r"\(?\s*([A-Da-d])\s*\)?(?:[.:)]\s*.*)?", t)
    return m.group(1).upper() if m else None


def countdown_ok(expr: str, numbers: list, target: int) -> bool:
    """True when expr uses each of `numbers` exactly once (as a multiset), only + - * / and parentheses, and equals
    `target` exactly (rational arithmetic). A trailing "= target" is allowed. Division by zero is False."""
    import ast
    from fractions import Fraction
    t = expr.strip().strip("$")
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("÷", "/"), ("\\left", ""),
                 ("\\right", ""), ("{", "("), ("}", ")"), ("\\", "")):
        t = t.replace(a, b)
    if "=" in t:                                     # "expr = target" or "target = expr"
        parts = [x.strip() for x in t.split("=")]
        if len(parts) != 2:
            return False
        if re.fullmatch(r"\(?\s*-?[0-9]+\s*\)?", parts[1]) and int(parts[1].strip("() ")) == target:
            t = parts[0]
        elif re.fullmatch(r"\(?\s*-?[0-9]+\s*\)?", parts[0]) and int(parts[0].strip("() ")) == target:
            t = parts[1]
        else:
            return False
    if not re.fullmatch(r"[0-9+\-*/() .]+", t) or len(t) > 200:
        return False
    try:
        tree = ast.parse(t, mode="eval")
    except SyntaxError:
        return False
    used = []

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            used.append(n.value)
            return Fraction(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError
            return a / b
        raise ValueError("disallowed node")
    try:
        val = ev(tree)
    except (ValueError, ZeroDivisionError):
        return False
    return sorted(used) == sorted(numbers) and val == target


_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def last_code_block(text: str):
    """The program of a code answer: every ```python block that defines something (def / class / import / from),
    concatenated in order. Blocks with only usage examples or asserts are skipped (models often end with one).
    Falls back to the last block when no block defines anything; None when there is no block."""
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        return None
    defs = [b for b in blocks if re.search(r"^\s*(def|class|import|from)\s", b, re.M)]
    return "\n\n".join(defs) if defs else blocks[-1]


_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_OPS = re.compile(r"[+\-*/×÷]|\\times|\\div|\\cdot")


def all_boxed(text: str) -> list:
    """Contents of every balanced \\boxed{...} in order."""
    out = []
    for m in _BOXED.finditer(text):
        i, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[i:j - 1].strip())
    return out


_CHOICE_FALLBACK = re.compile(r"(?i:answer)[^A-Za-z0-9]{0,30}(?:(?i:is)[^A-Za-z0-9]{0,10})?\(?([A-D])\)?(?![A-Za-z])")


def choice_fallback(text: str):
    """Option letter stated without a box, e.g. "**Final Answer:** **(D) volcanic**" (last statement wins)."""
    m = list(_CHOICE_FALLBACK.finditer(text[-600:]))
    return m[-1].group(1).upper() if m else None


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def run_code_tests(code: str, tests: list, setup: str = "", timeout: float = 10.0) -> dict:
    """Run `code` then `setup` then every assert of `tests` in a fresh isolated interpreter (python -I, temporary
    working directory, 4 GB address space, 10 s CPU, 1 MB files). Passed only when the process exits 0."""
    import subprocess
    import sys
    import tempfile
    prog = code + "\n\n" + setup + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", prog], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, stdin=subprocess.DEVNULL)
            return {"passed": r.returncode == 0, "rc": r.returncode, "err": r.stderr.decode(errors="replace")[-300:]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "rc": None, "err": "timeout"}


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

    def __init__(self, workers: int = 4, code_workers: int = 8):
        self.code_workers = code_workers
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
        code_jobs = []
        for k, it in enumerate(items):
            if it.get("match") == "code":
                code_jobs.append(k)
                continue
            ext = last_boxed(it["response"])
            res[k] = {"extracted": ext}
            if it.get("match") == "countdown":
                # the expression is the last boxed content that contains an operator: models often box the
                # expression and then box its value ("\\boxed{29}") at the very end
                exprs = [b for b in all_boxed(it["response"]) if _OPS.search(b)]
                if exprs:
                    ext = exprs[-1]
                    res[k] = {"extracted": ext}
            elif it.get("match") == "choice" and ext is None:
                fb = choice_fallback(it["response"])
                if fb is not None:
                    ext = fb
                    res[k] = {"extracted": ext, "fallback": True}
            if ext is None:
                res[k].update(status=TRUNCATED if it["truncated"] else NO_ANSWER, reward=0)
            elif it.get("match") in ("digits", "string", "choice", "countdown"):
                mt = it["match"]
                if mt == "digits":
                    ok = normalize_digits(ext, it.get("base")) == normalize_digits(it["gold"])
                elif mt == "string":
                    ok = normalize_string(ext) is not None and normalize_string(ext) == normalize_string(it["gold"])
                elif mt == "choice":
                    ok = normalize_choice(ext) == it["gold"]
                else:
                    ok = countdown_ok(ext, it["numbers"], it["target"])
                if it["truncated"]:
                    res[k].update(status=TRUNCATED, reward=0, would_match=ok)
                elif ok and it.get("len_limit") is not None and it["n_tokens"] > it["len_limit"]:
                    res[k].update(status=OVER_LEN, reward=0)
                else:
                    res[k].update(status=CORRECT if ok else WRONG, reward=1 if ok else 0)
            else:
                jobs.append((k, it["gold"], ext))
        if code_jobs:
            from concurrent.futures import ThreadPoolExecutor

            def grade(k):
                it = items[k]
                code = last_code_block(it["response"])
                if code is None:
                    return k, {"extracted": None, "status": TRUNCATED if it["truncated"] else NO_ANSWER, "reward": 0}
                r = run_code_tests(code, it["tests"], it.get("test_setup", ""))
                if it["truncated"]:
                    return k, {"extracted": code[-200:], "status": TRUNCATED, "reward": 0, "would_match": r["passed"]}
                return k, {"extracted": code[-200:], "status": CORRECT if r["passed"] else WRONG,
                           "reward": 1 if r["passed"] else 0, "diagnostic": None if r["passed"] else r["err"][-120:]}
            with ThreadPoolExecutor(max(1, self.code_workers)) as ex:
                for k, r in ex.map(grade, code_jobs):
                    res[k] = r
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
