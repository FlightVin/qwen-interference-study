"""A26 pass A: inventory and integrity check of every retained checkpoint (read-only; audit code, not production).

Independent of src/qis (no project imports). For every directory under ROOT that holds model.pt or optim.pt:
1. fresh SHA-256 of every file in the directory (streamed), sizes and mtimes;
2. state.json (if any): round, counters, attempt_id, config/bundle sha, file_sha256 → compared with the fresh hashes;
3. model.pt: loaded with torch.load(weights_only=True); tensor names, shapes, dtypes, finiteness, parameter count,
   per-tensor bit checksums (int64 sum of raw bit patterns, the rule documented in src/qis/train/ckpt.py) → compared with
   state.json "checksums.model"; names/shapes/dtypes compared with the base Qwen3-1.7B safetensors; relative L2 distance
   to the base weights;
4. optim.pt: loaded (weights_only=False); param-group hyperparameters, per-parameter step values, finiteness of moments,
   bit checksums → compared with state.json "checksums.optim"; RNG keys present.
Also lists LATEST.json pointers and leftover *.tmp directories.
Usage: python ckpt_audit.py ROOT BASE_DIR OUT_JSONL [--shard K --of N] [--list-only]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

CHUNK = 64 << 20


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bitsum(t):
    import torch
    t = t.detach().contiguous()
    bits = t.view(torch.int32) if t.element_size() == 4 else (
        t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int64))
    return int(bits.to(torch.int64).sum().item())


def discover(root):
    dirs = set()
    for name in ("model.pt", "optim.pt"):
        for p in glob.glob(os.path.join(root, "**", name), recursive=True):
            dirs.add(os.path.dirname(p))
    return sorted(dirs)


def check_model(d, meta, base):
    import torch
    out = {}
    sd = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True)
    out["n_tensors"] = len(sd)
    out["n_params"] = int(sum(v.numel() for v in sd.values()))
    out["dtypes"] = sorted({str(v.dtype) for v in sd.values()})
    nonfinite = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    out["nonfinite_tensors"] = nonfinite[:10]
    out["n_nonfinite_tensors"] = len(nonfinite)
    rec = (meta or {}).get("checksums", {}).get("model")
    if rec is not None:
        mism = [k for k, v in sd.items() if rec.get(k) != [str(v.dtype), list(v.shape), bitsum(v)]]
        out["checksum_mismatch"] = mism[:10]
        out["n_checksum_mismatch"] = len(mism) + len(set(rec) - set(sd))
    else:
        out["checksum_mismatch"] = "no recorded checksums"
    if base is not None:
        bk = set(base)
        mk = set(sd)
        out["names_missing_vs_base"] = sorted(bk - mk)[:10]
        out["names_extra_vs_base"] = sorted(mk - bk)[:10]
        out["shape_mismatch_vs_base"] = [k for k in bk & mk if tuple(base[k].shape) != tuple(sd[k].shape)][:10]
        num = den = 0.0
        for k in bk & mk:
            a, b = sd[k].double(), base[k].double()
            num += float(((a - b) ** 2).sum())
            den += float((b ** 2).sum())
        out["rel_l2_to_base"] = (num / den) ** 0.5 if den else None
    del sd
    return out


def check_optim(d, meta):
    import torch
    out = {}
    o = torch.load(os.path.join(d, "optim.pt"), map_location="cpu", weights_only=False)
    osd = o.get("optim", o)
    out["top_keys"] = sorted(o.keys()) if isinstance(o, dict) else str(type(o))
    out["rng_keys"] = sorted(o.get("rng", {}).keys()) if isinstance(o.get("rng"), dict) else None
    groups = osd.get("param_groups", [])
    out["param_groups"] = [{k: v for k, v in g.items() if k != "params"} | {"n_params": len(g.get("params", []))}
                           for g in groups]
    steps, nonfinite, flat = set(), 0, {}
    for pid, st in osd.get("state", {}).items():
        for n, v in st.items():
            if torch.is_tensor(v):
                flat[f"{pid}.{n}"] = v
                if n == "step":
                    steps.add(float(v.item()))
                elif v.is_floating_point() and not torch.isfinite(v).all():
                    nonfinite += 1
            elif n == "step":
                steps.add(float(v))
    out["n_param_states"] = len(osd.get("state", {}))
    out["step_values"] = sorted(steps)[:5]
    out["n_nonfinite_moment_tensors"] = nonfinite
    rec = (meta or {}).get("checksums", {}).get("optim")
    if rec is not None:
        mism = [k for k, v in flat.items() if rec.get(k) != [str(v.dtype), list(v.shape), bitsum(v)]]
        out["n_checksum_mismatch"] = len(mism) + len(set(rec) - set(flat))
        out["checksum_mismatch"] = mism[:10]
    else:
        out["checksum_mismatch"] = "no recorded checksums"
    del o, osd, flat
    return out


def load_base(base_dir):
    if not base_dir or not os.path.isdir(base_dir):
        return None
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    dirs = discover(a.root)
    extras = {"latest_pointers": {}, "tmp_dirs": sorted(glob.glob(os.path.join(a.root, "**", "*.tmp"), recursive=True))}
    for p in glob.glob(os.path.join(a.root, "**", "LATEST.json"), recursive=True):
        try:
            extras["latest_pointers"][p] = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            extras["latest_pointers"][p] = f"unreadable: {e}"
    if a.list_only:
        rows = []
        for d in dirs:
            rows.append({"dir": d, "files": {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d))
                                             if os.path.isfile(os.path.join(d, f))}})
        json.dump({"dirs": rows, **extras}, open(a.out, "w"), indent=1)
        print(f"{len(dirs)} checkpoint directories")
        return
    mine = [d for i, d in enumerate(dirs) if i % a.of == a.shard]
    base = load_base(a.base)
    with open(a.out, "w") as w:
        if a.shard == 0:
            w.write(json.dumps({"extras": extras, "all_dirs": dirs}) + "\n")
        for d in mine:
            t0 = time.time()
            row = {"dir": d, "files": {}}
            meta = None
            sp = os.path.join(d, "state.json")
            if os.path.exists(sp):
                meta = json.load(open(sp))
                row["state"] = {k: meta.get(k) for k in ("round", "counters", "attempt_id", "config_sha", "bundle_sha",
                                                          "file_sha256", "saved_utc", "wandb_id")}
                row["state_keys"] = sorted(meta.keys())
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    st = os.stat(p)
                    row["files"][f] = {"bytes": st.st_size, "mtime_utc": time.strftime("%FT%TZ", time.gmtime(st.st_mtime)),
                                       "sha256": sha_file(p)}
            rec = (meta or {}).get("file_sha256", {})
            row["hash_vs_state"] = {f: (rec[f] == row["files"].get(f, {}).get("sha256")) if f in row["files"] else "file missing"
                                    for f in rec}
            for name, fn in (("model", lambda: check_model(d, meta, base)), ("optim", lambda: check_optim(d, meta))):
                if os.path.exists(os.path.join(d, f"{name}.pt")):
                    try:
                        row[name] = fn()
                    except Exception as e:  # noqa: BLE001
                        row[name] = {"load_error": f"{type(e).__name__}: {e}"[:400]}
            row["seconds"] = round(time.time() - t0, 1)
            w.write(json.dumps(row) + "\n"); w.flush()
            print(d, row["seconds"], "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
