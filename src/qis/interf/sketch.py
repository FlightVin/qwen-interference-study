"""Count sketch of a full-model vector (study Nov-12-2026-01, understanding interference).

A count sketch S maps a vector a in R^d to R^m: every coordinate i gets a random bucket h(i) in [0, m) and a random
sign s(i); (Sa)[b] = sum over i with h(i) = b of s(i) * a[i]. For two vectors, <Sa, Sb> is an unbiased estimate of
<a, b> with standard deviation about ||a|| * ||b|| / sqrt(m). The hash depends only on (seed, tensor index, chunk
index), so every vector sketched with the same seed and the same parameter order is comparable.

One 40-bit random integer per coordinate gives: bit 0 the sign, bits 1.. the global bucket, bits 24.. the category
bucket (the two bucket fields do not overlap while m_global <= 2**23 and m_cat <= 2**16).
"""
import re

import torch

CHUNK = 1 << 25
TYPES = ("embed", "q", "k", "v", "o", "gate", "up", "down", "norm")


def categories(name: str):
    """(module type, layer label) of a parameter name of Qwen3. Layer label: 'embed', 'final' or the block index."""
    if "embed_tokens" in name or name.startswith("lm_head"):
        return "embed", "embed"
    m = re.search(r"layers\.(\d+)\.", name)
    layer = m.group(1) if m else "final"
    for key, t in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o"),
                   ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        if key in name:
            return t, layer
    return "norm", layer


def layer_labels(names):
    labs = []
    for n in names:
        lab = categories(n)[1]
        if lab not in labs:
            labs.append(lab)
    return labs


def _hash(n, seed, ti, ci, device, mg, mc):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1_000_003 + ti * 4099 + ci)
    r = torch.randint(0, 1 << 40, (n,), generator=gen, device=device, dtype=torch.int64)
    sign = ((r & 1) * 2 - 1).to(torch.float32)
    return sign, (r >> 1) & (mg - 1), (r >> 24) & (mc - 1)


class Sketcher:
    """Sketches a list of tensors (one per parameter, fixed order) into a global sketch and per-category sketches."""

    def __init__(self, names, log2_global=21, log2_cat=15, seed=20261112, device="cuda"):
        assert log2_global <= 23 and log2_cat <= 16
        self.names, self.seed, self.device = list(names), seed, device
        self.mg, self.mc = 1 << log2_global, 1 << log2_cat
        self.types = list(TYPES)
        self.layers = layer_labels(self.names)
        cats = [categories(n) for n in self.names]
        self.t_idx = [self.types.index(t) for t, _ in cats]
        self.l_idx = [self.layers.index(l) for _, l in cats]

    @torch.no_grad()
    def sketch(self, tensors, scale=None, cats=True):
        """tensors: list aligned with names (None = zero). scale: optional list of per-coordinate multipliers; when
        given, a second global sketch of tensor * scale is returned. Returns a dict of float32 tensors on the device:
        'global' [mg], 'type' [n_types, mc], 'layer' [n_layers, mc], 'sq_type', 'sq_layer' (exact squared norms),
        and with scale: 'global_scaled' [mg], 'sq_scaled' (exact squared norm of the scaled vector)."""
        dev = self.device
        out = {"global": torch.zeros(self.mg, device=dev),
               "sq_type": torch.zeros(len(self.types), dtype=torch.float64, device=dev),
               "sq_layer": torch.zeros(len(self.layers), dtype=torch.float64, device=dev)}
        if cats:
            out["type"] = torch.zeros(len(self.types), self.mc, device=dev)
            out["layer"] = torch.zeros(len(self.layers), self.mc, device=dev)
        if scale is not None:
            out["global_scaled"] = torch.zeros(self.mg, device=dev)
            out["sq_scaled"] = torch.zeros((), dtype=torch.float64, device=dev)
        for ti, t in enumerate(tensors):
            if t is None:
                continue
            flat = t.reshape(-1)
            sc = scale[ti].reshape(-1) if scale is not None else None
            for ci, s in enumerate(range(0, flat.numel(), CHUNK)):
                x = flat[s:s + CHUNK].to(dev, torch.float32)
                sq = (x.double() ** 2).sum()
                out["sq_type"][self.t_idx[ti]] += sq
                out["sq_layer"][self.l_idx[ti]] += sq
                sign, hg, hc = _hash(x.numel(), self.seed, ti, ci, dev, self.mg, self.mc)
                v = x * sign
                out["global"].index_add_(0, hg, v)
                if cats:
                    out["type"][self.t_idx[ti]].index_add_(0, hc, v)
                    out["layer"][self.l_idx[ti]].index_add_(0, hc, v)
                if sc is not None:
                    v2 = v * sc[s:s + CHUNK].to(dev, torch.float32)
                    out["global_scaled"].index_add_(0, hg, v2)
                    out["sq_scaled"] += (v2.double() ** 2).sum()
                del x, sign, hg, hc, v
        return out
