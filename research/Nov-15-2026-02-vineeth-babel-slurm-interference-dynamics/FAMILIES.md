# Experiment families — study Nov-15-2026-02 (s02)

Purpose: for each experiment family, the question, the competing explanations, the comparison and what is held fixed,
the predicted observations, the smallest useful budget, the expansion / stopping / abandonment criteria, and what the
family cannot establish. Definitions: harness `PROJECT.md` §2. Each entry is written before the family runs and
amended (dated) when evidence changes it.

## E0 — calibration of candidate tasks (base model, `sel` split only)
- **Question.** Which tasks give a target with low but measurable success, and competitors with enough mixed groups to
  produce a learning signal? Which response cap avoids truncation? Does the grader mis-score any answer format?
- **Comparison.** Base pass@1, per-problem success counts, length quantiles, status counts across 18 pools
  (64 problems × 16 samples, cap 4,096, temperature 1).
- **Criteria.** Target strata: base pass@1 from about 3% to about 60%, spanning at least three levels. Competitors:
  pass@1 about 30–70%. Cap: the 99th length percentile below the cap with margin. Grader: no correct answer marked wrong
  in a manual read of 12 wrong examples per pool.
- **Cannot establish.** Anything about learning or interference.

## E1 — canary and pair screen (RQ1 first look; RQ2 dependence on initial success)
- **Question.** Relative to the matched reference R(T), does training C alongside T help, not affect, or harm T's
  acquisition, and is any harm larger or longer-lasting in low-success strata?
- **Arms (seed 1).** R(T) alone; J(T, add10) competing procedure on the same surface; J(T, add7) shared non-decimal
  carry (predicted compatible); J(T, terse10) competing behavioural demand (no working).
- **Held fixed.** T prompts and sampling seeds per round; 8 T groups per round; 8 responses per prompt; 4 AdamW steps
  per round; loss divisor 32; lr; clip; cap; evaluation panels and seeds.
- **Differs.** C's gradients (and with them Adam's second moment, the global clip norm, the minibatch composition).
- **Predictions.** add7: D(t) ≤ 0 (transfer). add10: D(t) > 0 if decimal habits displace the base-9 rule; then the
  wrong-answer pattern shifts toward decimal sums. terse10: D(t) > 0 with falling T response length. RQ2: if feedback
  amplifies harm, the T A0 fraction in J rises above R before D(t) grows, and D(t) is larger in low-p strata.
- **Budget.** Canary 6 rounds. Screen ~40 rounds, eval every 4 rounds (T strata 64 problems × 8 samples; C panel).
- **Expansion.** A pair with D(t) outside the seed-noise band (estimated from a second R seed) → confirmation seed on
  `conf` panels + E2 local effects at the relevant checkpoint + E3 branches.
- **Stopping / abandonment.** If R(T) does not learn (no stratum gains ≥ 10 points in 40 rounds), the target is not
  usable; raise lr or change strata before any interference claim. Abandon a pair whose C does not learn either.
- **Cannot establish.** Mechanism (direct vs optimizer vs feedback); generality beyond these tasks; seed robustness
  (one seed until confirmed).

## E2 — local update effects (RQ1 "actual update"; input to RQ2)
- **Question.** Does one round of C-only updates, from a given model + Adam state, lower T success, and by how much
  relative to a null round (zero gradient: momentum drift only) and a T-only round?
- **Comparison.** Branches from the same state with the same sampled data: weights (T, C) = (0, 1), (1, 0), (0, 0).
  Δ T pass@1 per stratum on paired evaluation seeds.
- **Resolution check.** Two null branches estimate the evaluation noise of Δ (vLLM sampling is not bit-reproducible).
- **Cannot establish.** Long-run consequences; a negative local effect can be repaired by T's own updates.

## E3 — RQ2 discriminating branches (defined after E1/E2)
- Single-stratum confirmation runs (low vs high initial p) for a harmful pair, `conf` panels, new seed.
- Recovery branches from a harmed checkpoint: T-only vs T-only with 4× samples per prompt (more mixed groups at equal
  prompts). Amplification predicts the extra samples help much more after harm than in R at equal p.
- Order: C-first then T vs T-first then C, only if a harmful pair exists.
