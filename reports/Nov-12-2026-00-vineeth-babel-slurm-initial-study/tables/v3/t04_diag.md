# Local one-step diagnostic: individual branch draws

One GRPO step (4 prompts × 8 attempts, one draw seed 4101–4104) of each type (hard, related, unrelated) from two source policies, then J_H, J_R, J_U at 16 attempts per problem and cap 8,192 (not the 12,288 cap the r64 source trained with). ΔJ in points against two no-update references: `none` (same job as the branches) and `none_rerun` (separate job on L40S; `none_cycled` gives identical values). Momentum-only = pre-clip gradient norm exactly 0 (all four groups non-mixed); AdamW still stepped with its stored moments. GPUs from DEFINITIONS v1.1 §4 item 1.

Sources: `research/audit-a26/pass-b/data/train_diag.json`; `research/DEFINITIONS.md v1.1 §4 item 1`.

## Source policy: round 0 (base policy, fresh optimizer)

| Branch | GPU | J_H (%) | ΔJ_H vs none | ΔJ_H vs none_rerun | J_R (%) | ΔJ_R vs none | J_U (%) | ΔJ_U vs none | Pre-clip grad norm |
|---|---|---|---|---|---|---|---|---|---|
| none | L40 | 40.0 | 0.0 | — | 82.8 | 0.0 | 80.7 | 0.0 | — |
| none_rerun | L40S | 43.2 | +3.1 | — | 80.8 | −2.0 | 79.3 | −1.4 | — |
| none_cycled | L40S | 43.2 | +3.1 | — | 80.8 | −2.0 | 79.3 | −1.4 | — |
| hard-4101 | L40 | 41.0 | +1.0 | −2.1 | 78.8 | −4.0 | 77.5 | −3.1 | 0.563 |
| hard-4102 | L40 | 42.0 | +2.0 | −1.2 | 80.8 | −2.0 | 79.9 | −0.8 | 0.494 |
| hard-4103 | L40 | 40.6 | +0.6 | −2.5 | 80.8 | −2.0 | 82.0 | +1.4 | 0.364 |
| hard-4104 | L40 | 42.2 | +2.1 | −1.0 | 80.6 | −2.2 | 78.1 | −2.5 | 0.679 |
| related-4101 | L40 | 41.6 | +1.6 | −1.6 | 80.4 | −2.5 | 80.1 | −0.6 | 1.150 |
| related-4102 | L40 | 43.9 | +3.9 | +0.8 | 79.7 | −3.1 | 76.4 | −4.3 | 1.041 |
| related-4103 | L40 | 43.4 | +3.3 | +0.2 | 80.8 | −2.0 | 80.1 | −0.6 | 0.679 |
| related-4104 | L40 | 42.0 | +2.0 | −1.2 | 80.6 | −2.2 | 77.3 | −3.3 | 0.824 |
| unrelated-4101 | L40 | 40.6 | +0.6 | −2.5 | 79.9 | −2.9 | 80.9 | +0.2 | 0.925 |
| unrelated-4102 | L40 | 42.6 | +2.5 | −0.6 | 80.6 | −2.2 | 82.2 | +1.6 | 0.827 |
| unrelated-4103 | L40 | 40.4 | +0.4 | −2.7 | 79.7 | −3.1 | 79.5 | −1.2 | 0.614 |
| unrelated-4104 | L40 | 38.5 | −1.6 | −4.7 | 80.4 | −2.5 | 80.3 | −0.4 | 1.083 |

## Source policy: R-s101 round 64

| Branch | GPU | J_H (%) | ΔJ_H vs none | ΔJ_H vs none_rerun | J_R (%) | ΔJ_R vs none | J_U (%) | ΔJ_U vs none | Pre-clip grad norm |
|---|---|---|---|---|---|---|---|---|---|
| none | RTX 6000 Ada | 60.5 | 0.0 | — | 90.2 | 0.0 | 93.9 | 0.0 | — |
| none_rerun | L40S | 63.1 | +2.5 | — | 89.3 | −0.9 | 95.1 | +1.2 | — |
| none_cycled | L40S | 63.1 | +2.5 | — | 89.3 | −0.9 | 95.1 | +1.2 | — |
| hard-4101 | RTX 6000 Ada | 60.0 | −0.6 | −3.1 | 88.6 | −1.6 | 93.4 | −0.6 | 0.272 |
| hard-4102 | RTX 6000 Ada | 60.7 | +0.2 | −2.3 | 88.6 | −1.6 | 93.2 | −0.8 | 0.350 |
| hard-4103 | RTX 6000 Ada | 61.7 | +1.2 | −1.4 | 90.2 | 0.0 | 93.0 | −1.0 | 0.303 |
| hard-4104 | RTX 6000 Ada | 60.2 | −0.4 | −2.9 | 88.6 | −1.6 | 92.8 | −1.2 | 0.242 |
| related-4101 (momentum-only) | RTX 6000 Ada | 61.1 | +0.6 | −2.0 | 89.5 | −0.7 | 93.2 | −0.8 | 0.000 |
| related-4102 (momentum-only) | RTX 6000 Ada | 61.1 | +0.6 | −2.0 | 89.5 | −0.7 | 93.2 | −0.8 | 0.000 |
| related-4103 | RTX 6000 Ada | 62.1 | +1.6 | −1.0 | 90.2 | 0.0 | 93.0 | −1.0 | 0.501 |
| related-4104 | RTX 6000 Ada | 61.9 | +1.4 | −1.2 | 89.3 | −0.9 | 91.6 | −2.3 | 0.198 |
| unrelated-4101 | RTX 6000 Ada | 60.0 | −0.6 | −3.1 | 89.7 | −0.4 | 93.9 | 0.0 | 0.473 |
| unrelated-4102 | RTX 6000 Ada | 63.3 | +2.7 | +0.2 | 89.3 | −0.9 | 93.0 | −1.0 | 0.759 |
| unrelated-4103 (momentum-only) | RTX 6000 Ada | 61.1 | +0.6 | −2.0 | 89.5 | −0.7 | 93.2 | −0.8 | 0.000 |
| unrelated-4104 | RTX 6000 Ada | 59.2 | −1.4 | −3.9 | 88.4 | −1.8 | 93.9 | 0.0 | 0.589 |

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
