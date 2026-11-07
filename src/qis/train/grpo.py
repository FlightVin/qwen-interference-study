"""GRPO mathematics (project.md Section 6 "Algorithm"). Pure functions; FP32 throughout.

- Advantage: A_i = (r_i - mean(r)) / (population_std(r) + 1e-6) per prompt group.
  All-equal groups get exactly zero advantage.
- Objective per token: min(q*A, clip(q, 1-eps_low, 1+eps_high)*A), q = exp(logp - logp_old).
  No dual clip, no log-ratio clamp.
- Reduction: mean over valid response tokens per response, then mean over all responses of the
  minibatch. With micro-batch 1 each response contributes (its token mean) / n_minibatch, so
  accumulation weights every response equally whatever its length.
"""
import torch

ADV_EPS = 1e-6


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: [G, n] float. Returns [G, n] FP32 advantages."""
    r = rewards.to(torch.float32)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    adv = (r - mean) / (std + ADV_EPS)
    equal = (r == r[:, :1]).all(dim=1, keepdim=True)
    return torch.where(equal, torch.zeros_like(adv), adv)


def response_objective(logp: torch.Tensor, logp_old: torch.Tensor, adv: float,
                       eps_low: float, eps_high: float):
    """logp, logp_old: [T] FP32 selected-token log probs over the valid response tokens.
    Returns (mean objective over tokens [scalar, differentiable], stats dict of FP32 tensors)."""
    logp = logp.float()
    ratio = torch.exp(logp - logp_old.float())
    a = torch.as_tensor(adv, dtype=torch.float32, device=logp.device)
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * a
    obj = torch.minimum(unclipped, clipped)
    with torch.no_grad():
        # objective-active clipping: the clipped branch is strictly smaller than the unclipped one
        active = (clipped < unclipped).float()
        stats = {"ratio": ratio.detach(), "clip_active": active,
                 "n_tokens": torch.tensor(float(logp.numel()), device=logp.device)}
    return obj.mean(), stats


def minibatch_plan(roles: list, n_minibatches: int, rng) -> list:
    """Assign complete prompt groups to minibatches with balanced role composition.

    roles: role label per group in round order (e.g. 8 x "hard" + 8 x "aux", or for arm H
    8 x "hard_common" + 8 x "hard_extra"). Each minibatch gets the same number of groups of each
    role. rng: numpy Generator (named stream per round). Returns list of lists of group indices.
    """
    by_role = {}
    for g, r in enumerate(roles):
        by_role.setdefault(r, []).append(g)
    plan = [[] for _ in range(n_minibatches)]
    for r in sorted(by_role):
        gs = by_role[r]
        if len(gs) % n_minibatches:
            raise ValueError(f"role {r}: {len(gs)} groups not divisible by {n_minibatches}")
        perm = rng.permutation(len(gs))
        k = len(gs) // n_minibatches
        for m in range(n_minibatches):
            plan[m] += [gs[i] for i in perm[m * k:(m + 1) * k]]
    order = rng.permutation(n_minibatches)
    return [sorted(plan[m]) for m in order]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n attempts with c successes: 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)
