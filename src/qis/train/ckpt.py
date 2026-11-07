"""Full-restart checkpoints: atomic, validated, written in the background (project.md §11; A1).

Layout: <ckpt_dir>/round_NNNN/{model.pt, optim.pt, state.json}; <ckpt_dir>/LATEST.json points
to the newest VALIDATED checkpoint.

Save path (measured on Babel NFS: ~40 MB/s, 20.6 GB per checkpoint, ~8.5 min):
 1. snapshot model + optimizer state to host RAM (training continues afterwards);
 2. a background thread takes the project-wide write lock (one writer at a time across runs),
    writes round_NNNN.tmp while hashing the bytes, fsyncs, re-reads the files to confirm the
    SHA-256, renames to round_NNNN, moves LATEST.json, then deletes older checkpoints.
At most two checkpoints exist per run: the validated newest and one in flight. Per-tensor
checksums (bit-level) are stored and re-checked on load: no BF16 round trip can pass.
"""
import hashlib
import io
import json
import os
import random
import shutil
import threading
import time

import numpy as np
import torch

LOCK_STALE_S = 3600


def tensor_checksums(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        t = v.detach().contiguous()
        bits = t.view(torch.int32) if t.element_size() == 4 else (
            t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int64))
        out[k] = [str(t.dtype), list(t.shape), float(bits.double().sum().item())]
    return out


def _opt_flat(opt_sd: dict) -> dict:
    return {f"{pid}.{n}": v for pid, st in opt_sd["state"].items() for n, v in st.items() if torch.is_tensor(v)}


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s: dict):
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"])


def _to_host(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_host(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_host(v) for v in obj]
    return obj


class _HashWriter(io.RawIOBase):
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def writable(self):
        return True

    def write(self, b):
        self.h.update(b)
        return self.f.write(b)


def _sha_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                return h.hexdigest()
            h.update(b)


class _Lock:
    """Project-wide checkpoint write lock (mkdir is atomic on NFS)."""

    def __init__(self, root):
        self.path = os.path.join(root, ".ckpt-write-lock")

    def __enter__(self):
        while True:
            try:
                os.mkdir(self.path)
                with open(os.path.join(self.path, "owner"), "w") as f:
                    f.write(f"{os.environ.get('SLURM_JOB_ID')} {time.time()}")
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(10)

    def __exit__(self, *a):
        shutil.rmtree(self.path, ignore_errors=True)


class Saver:
    def __init__(self, ckpt_dir: str, lock_root: str, retain=()):
        self.dir, self.lock_root, self.retain = ckpt_dir, lock_root, set(retain)
        self.thread, self.last = None, None
        os.makedirs(ckpt_dir, exist_ok=True)

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def abort(self):
        """On preemption exit: drop the shared write lock if this job holds it (the partial
        .tmp directory is ignored and removed by the next save)."""
        if self.busy():
            lock = os.path.join(self.lock_root, ".ckpt-write-lock")
            try:
                if open(os.path.join(lock, "owner")).read().split()[0] == str(os.environ.get("SLURM_JOB_ID")):
                    shutil.rmtree(lock, ignore_errors=True)
            except (FileNotFoundError, IndexError):
                pass

    def wait(self):
        if self.thread is not None:
            self.thread.join()
        return self.last

    def save(self, rnd: int, model, opt, state: dict, blocking=False):
        self.wait()                                        # never two writes from one run
        t0 = time.time()
        msd, osd = model.state_dict(), opt.state_dict()
        sums = {"model": tensor_checksums(msd), "optim": tensor_checksums(_opt_flat(osd))}
        host = {"model": _to_host(msd), "optim": {"optim": _to_host(osd), "rng": rng_state()}}
        meta = {**state, "round": rnd, "checksums": sums}
        snap_s = time.time() - t0
        self.thread = threading.Thread(target=self._write, args=(rnd, host, meta, snap_s), daemon=True)
        self.thread.start()
        if blocking:
            self.wait()
        return snap_s

    def _write(self, rnd, host, meta, snap_s):
        try:
            t_wait = time.time()
            with _Lock(self.lock_root):
                t0 = time.time()
                final = os.path.join(self.dir, f"round_{rnd:04d}")
                tmp = final + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                hashes = {}
                for name, obj in (("model.pt", host["model"]), ("optim.pt", host["optim"])):
                    with open(os.path.join(tmp, name), "wb") as f:
                        w = _HashWriter(f)
                        torch.save(obj, w)
                        f.flush(); os.fsync(f.fileno())
                    hashes[name] = w.h.hexdigest()
                meta["file_sha256"] = hashes
                meta["saved_utc"] = time.strftime("%FT%TZ", time.gmtime())
                with open(os.path.join(tmp, "state.json"), "w") as f:
                    json.dump(meta, f); f.flush(); os.fsync(f.fileno())
                write_s = time.time() - t0
                for name, hsh in hashes.items():           # bytes on disk == bytes written
                    if _sha_file(os.path.join(tmp, name)) != hsh:
                        raise RuntimeError(f"sha mismatch {tmp}/{name}")
                shutil.rmtree(final, ignore_errors=True)
                os.rename(tmp, final)
                ptr = os.path.join(self.dir, "LATEST.json")
                with open(ptr + ".tmp", "w") as f:
                    json.dump({"round": rnd, "path": final}, f)
                os.replace(ptr + ".tmp", ptr)
                for d in os.listdir(self.dir):
                    if d.startswith("round_") and not d.endswith(".tmp"):
                        r = int(d[6:])
                        if r < rnd and r not in self.retain:
                            shutil.rmtree(os.path.join(self.dir, d), ignore_errors=True)
                self.last = {"round": rnd, "ok": True, "snapshot_s": snap_s, "lock_wait_s": t0 - t_wait,
                             "write_s": write_s, "total_s": time.time() - t_wait,
                             "bytes": sum(os.path.getsize(os.path.join(final, n)) for n in os.listdir(final))}
        except Exception as e:  # noqa: BLE001 -- reported to the training loop, never silent
            self.last = {"round": rnd, "ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


def latest(ckpt_dir: str):
    ptr = os.path.join(ckpt_dir, "LATEST.json")
    return json.load(open(ptr)) if os.path.exists(ptr) else None


def load(path: str, model, opt):
    meta = json.load(open(os.path.join(path, "state.json")))
    for name, hsh in meta.get("file_sha256", {}).items():
        if _sha_file(os.path.join(path, name)) != hsh:
            raise RuntimeError(f"file sha mismatch {path}/{name}")
    msd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    if tensor_checksums(msd) != meta["checksums"]["model"]:
        raise RuntimeError(f"model checksum mismatch in {path}")
    o = torch.load(os.path.join(path, "optim.pt"), map_location="cpu", weights_only=False)
    if tensor_checksums(_opt_flat(o["optim"])) != meta["checksums"]["optim"]:
        raise RuntimeError(f"optimizer checksum mismatch in {path}")
    model.load_state_dict(msd)
    opt.load_state_dict(o["optim"])
    set_rng_state(o["rng"])
    return meta
