# Skill rubric v1 — polynomial factorization (focal skill)

Purpose: fixed labelling rules for every annotated item. Annotators apply this text exactly. The
rubric is frozen with the protocol; changes after the first annotation pass need a dated amendment
and a full relabel of affected items.

## Focal skill
**Polynomial factorization used to solve an equation or derive a requested quantity.** The solution
rewrites a polynomial (or polynomial expression) as a product of lower-degree factors, and that
product form is then used: zero-product rule, cancellation of a common factor, reading roots,
sign analysis of factors, or simplifying a ratio. Recognised forms include: common-factor
extraction, quadratic trinomial factoring, difference of squares/cubes, sum of cubes, grouping,
factor theorem / synthetic division to split off a known root, and factoring in a substitution
variable (e.g. a quadratic in x²).

Not the focal skill by itself: integer prime factorization; the quadratic formula with no
factoring; completing the square; expanding products; Vieta's formulas used without factoring.

## Labels for the reference solution
| label | rule |
|---|---|
| `central` | The reference solution's main line of reasoning contains a factorization step, and the answer is obtained through the factored form. Removing the step would require a different method. |
| `incidental` | A factorization appears but the answer does not depend on it (a side remark, a check, a simplification that a direct computation would replace in one line). |
| `absent` | No polynomial factorization step. |

Record for each item: `label`, `support` (the exact reference-solution step, quoted, ≤200 chars),
`alt_route` (a plausible solution route that avoids factoring, or "none apparent"), `confidence`
(high / medium / low), `direct_practice` (yes if the question is essentially a factoring exercise:
factor an expression, solve a polynomial equation by factoring, simplify a rational expression by
factoring; no if factoring is one step inside a larger derivation or word problem),
`answer_type` (integer / rational / radical / expression / set-or-list / interval / other),
`modular` (number-theory items only: yes / no, per the E_unrelated rule below), and `figure`
(yes if the problem needs an Asymptote figure to be understood).

## Role pools (fixed before screening; success never decides role)
| pool | subjects | rule |
|---|---|---|
| H candidates (H_train, target_dev, target_test) | algebra, intermediate_algebra | `central`, MATH level 4–5 (`direct_practice` may be yes or no) |
| E_related candidates | algebra, intermediate_algebra | `central`, `direct_practice = yes`, MATH level 1–3 |
| E_unrelated candidates | number_theory | `modular = yes` (the task is a direct modular computation: remainder, congruence, last digits, modular inverse, solving a linear congruence) and focal label `absent` |

H_train comes from the construction split, target_dev from study_dev, target_test from the
official test split. Calibration panels come from calib_train / calib_val, from the same three
subjects, any focal label, excluding near-duplicate templates of selected study items.

## Exclusions (any pool)
Gold answer not verifier-eligible; answer requires a figure (Asymptote) to be read; label
`incidental` in the H or E_related pools; `confidence = low` after the second pass; unresolved
disagreement between passes.

## Second pass
A fresh-context agent labels the selected items from this rubric alone, without pass-1 labels,
confidence or any screening or treatment outcome. Disagreements on `label`, `direct_practice` or
`modular` are resolved by a third read that sees both rationales; unresolved items are excluded.
Agent annotation is reported as "agent-annotated and audited", not human-reviewed.

## Amendment v1.1 (2026-11-07, before any screening or model outcome)

**Why.** Pass 1 (1,496 items) showed (a) the v1 role rule gives only 69 E_related candidates in
the construction split, too few to reach 64 after the 12–24/32 band and matching; and (b)
annotators split on edge cases the v1 text left open. The change uses annotation counts only;
no screening or treatment outcome existed. Pass-1 labels are kept unchanged; pass 2 and the
resolution read apply v1.1.

### Role pools (replace the v1 table)
| pool | subjects | rule |
|---|---|---|
| H candidates (H_train, target_dev, target_test) | algebra, intermediate_algebra | `central`, `direct_practice = no`, MATH level 4–5 |
| E_related candidates | algebra, intermediate_algebra | `central`, `direct_practice = yes`, any level |
| E_unrelated candidates | number_theory | `modular = yes`, focal label `absent` |

Roles are disjoint by `direct_practice`. Hard items use factoring as a step inside a larger
derivation; related items practise factoring directly.

### Edge-case clarifications (apply to pass 2 and resolution)
| case | label |
|---|---|
| Polynomial built from known roots, then only expanded or matched to coefficients | `absent` |
| Product form built from roots (factor theorem) and itself evaluated to give the answer | `central` |
| Factor theorem used only as a root/divisibility test, no factorization | `absent` |
| Product already given in the problem; only sign analysis or cancellation | `absent` |
| Identity used in the multiplying direction (e.g. (z−1)(z²+z+1)=z³−1 to get z³) | `absent` |
| Constant pulled out only to complete the square | `incidental` |
| Factoring replaceable by a one-line direct computation (square root, Vieta sum/product) | `incidental` |
| Simon's Favorite Factoring Trick on a polynomial expression | `central` |
| Difference of squares / grouping on plain numbers | `incidental` |
| Word problem whose core is one polynomial equation solved by factoring | `direct_practice = no` |
| Bare task: factor, solve a polynomial equation or inequality, simplify a rational expression | `direct_practice = yes` |

## Amendment v1.2 (2026-11-07): solution-structure families (A4.2; before any stage-2 count was viewed)
Each hard candidate gets one primary `family`, from the reference solution's factorization step:
| family | structure |
|---|---|
| F1 | quadratic trinomial factored, then zero-product / roots |
| F2 | special products: difference of squares, sum/difference of cubes |
| F3 | grouping, including Simon's Favorite Factoring Trick |
| F4 | factor theorem / known root / synthetic division / rational-root search on degree ≥ 3 |
| F5 | substitution: polynomial in an expression (x², 2^x, x+1/x, …), then factored |
| F6 | rational-expression simplification, cancellation, domain or asymptote from factors |
| F7 | polynomial inequality or sign analysis from factors |
| F8 | other (name it) |
Minor wording or number variants of one problem template count as one family member (diversity is
judged by family counts and by near-duplicate groups).

### v1.2 technique definitions with examples (frozen technique list for A5 coverage)
| tag | example (illustrative, not from any study split) |
|---|---|
| F1 quadratic trinomial | x² − 5x + 6 = 0 → (x−2)(x−3) = 0 → x ∈ {2, 3} |
| F2 special products | x⁴ − 16 = (x²−4)(x²+4) = (x−2)(x+2)(x²+4); a³ − 8 = (a−2)(a²+2a+4) |
| F3 grouping / SFFT | xy + 3x + 2y = 10 → (x+2)(y+3) = 16 |
| F4 factor theorem / known root | x³ − 6x² + 11x − 6 has root 1 → (x−1)(x²−5x+6) |
| F5 substitution | x⁴ − 5x² + 4 = 0 with u = x² → (u−1)(u−4) = 0 |
| F6 rational expressions | (x²−9)/(x²+x−6) = (x−3)/(x−2), x ≠ −3; hole at x = −3 |
| F7 sign analysis | x² − x − 6 < 0 → (x−3)(x+2) < 0 → −2 < x < 3 |
| F8 other | name the operation in `family_note` |
Primary = the factoring operation doing the main work; secondary = other factoring operations used.
