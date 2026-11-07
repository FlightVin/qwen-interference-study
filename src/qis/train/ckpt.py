"""Atomic full-restart checkpoints (project.md Section 11; addendum A1).

Layout: <ckpt_dir>/round_NNNN/{model.pt, optim.pt, state.json}; <ckpt_dir>/LATEST.json points
to the newest VALIDATED checkpoint. A checkpoint is written to round_NNNN.tmp, reloaded and
checked against per-tensor checksums computed before the write, then renamed and the pointer
updated. Tensors are saved in their own dtype (FP32 weights and moments): no BF16 round trip.
"""
import json
import os
import random
import shutil
import time

import numpy as np
import torch


def tensor_checksums(sd: dict) -> dict:
    """Cheap exact fingerprint per tensor: dtype, shape, and float64 sum of the raw bits viewed
    as int32/int16 (bit-level, so a BF16 round trip of FP32 data changes it)."""
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
    flat = {}
    for pid, st in opt_sd["state"].items():
        for name, v in st.items():
            if torch.is_tensor(v):
                flat[f"{pid}.{name}"] = v
    return flat


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s: dict):
    random.setstate(s["python"]); np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"])


def save(ckpt_dir: str, rnd: int, model, opt, state: dict, keep: int = 2, retain=()):
    t0 = time.time()
    msd, osd = model.state_dict(), opt.state_dict()
    sums = {"model": tensor_checksums(msd), "optim": tensor_checksums(_opt_flat(osd))}
    final = os.path.join(ckpt_dir, f"round_{rnd:04d}")
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save(msd, os.path.join(tmp, "model.pt"))
    torch.save({"optim": osd, "rng": rng_state()}, os.path.join(tmp, "optim.pt"))
    meta = {**state, "round": rnd, "checksums": sums, "saved_utc": time.strftime("%FT%TZ", time.gmtime())}
    with open(os.path.join(tmp, "state.json"), "w") as f:
        json.dump(meta, f)
    for name in os.listdir(tmp):
        fd = os.open(os.path.join(tmp, name), os.O_RDONLY); os.fsync(fd); os.close(fd)
    write_s = time.time() - t0
    verify(tmp)                                           # raises on mismatch
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    ptr = os.path.join(ckpt_dir, "LATEST.json")
    with open(ptr + ".tmp", "w") as f:
        json.dump({"round": rnd, "path": final}, f)
    os.replace(ptr + ".tmp", ptr)
    # retention: newest `keep` full checkpoints plus explicitly retained rounds
    rounds = sorted(int(d[6:]) for d in os.listdir(ckpt_dir)
                    if d.startswith("round_") and not d.endswith(".tmp"))
    for r in rounds[:-keep]:
        if r not in retain:
            shutil.rmtree(os.path.join(ckpt_dir, f"round_{r:04d}"), ignore_errors=True)
    return {"write_s": write_s, "total_s": time.time() - t0,
            "bytes": sum(os.path.getsize(os.path.join(final, n)) for n in os.listdir(final))}


def verify(path: str) -> dict:
    meta = json.load(open(os.path.join(path, "state.json")))
    msd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    o = torch.load(os.path.join(path, "optim.pt"), map_location="cpu", weights_only=False)
    if tensor_checksums(msd) != meta["checksums"]["model"]:
        raise RuntimeError(f"model checksum mismatch in {path}")
    if tensor_checksums(_opt_flat(o["optim"])) != meta["checksums"]["optim"]:
        raise RuntimeError(f"optimizer checksum mismatch in {path}")
    return meta


def latest(ckpt_dir: str):
    ptr = os.path.join(ckpt_dir, "LATEST.json")
    return json.load(open(ptr)) if os.path.exists(ptr) else None


def load(path: str, model, opt):
    meta = verify(path)
    msd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(msd)                 # copies into the existing FP32 parameters
    o = torch.load(os.path.join(path, "optim.pt"), map_location="cpu", weights_only=False)
    opt.load_state_dict(o["optim"])            # torch casts state to the param dtype (FP32)
    set_rng_state(o["rng"])
    return meta
