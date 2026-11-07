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
