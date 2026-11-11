# One-step local update (A17, exploratory)

pre = no-update evaluation of the source checkpoint (original row; rerun row in brackets). post = after ONE GRPO/AdamW step on 4 prompts × 8 answers. Changes are post − pre. J values are pass@1 at 16 samples per problem.

## r0: source base

| No-update row | J_H | J_R | J_U |
|---|---|---|---|
| none_original | 0.4004 | 0.8281 | 0.8066 |
| none_rerun | 0.4316 | 0.8080 | 0.7930 |
| none_cycled | 0.4316 | 0.8080 | 0.7930 |

| Branch | Seed | Mixed groups (of 4) | ΔJ_H = post − pre (original) | ΔJ_H vs rerun | grad norm |
|---|---|---|---|---|---|
| hard | 4101 | 2 | 0.4102 − 0.4004 = +0.0098 | 0.4102 − 0.4316 = -0.0215 | 0.5632 |
| hard | 4102 | 3 | 0.4199 − 0.4004 = +0.0195 | 0.4199 − 0.4316 = -0.0117 | 0.4940 |
| hard | 4103 | 2 | 0.4062 − 0.4004 = +0.0059 | 0.4062 − 0.4316 = -0.0254 | 0.3642 |
| hard | 4104 | 4 | 0.4219 − 0.4004 = +0.0215 | 0.4219 − 0.4316 = -0.0098 | 0.6789 |
| related | 4101 | 3 | 0.4160 − 0.4004 = +0.0156 | 0.4160 − 0.4316 = -0.0156 | 1.1502 |
| related | 4102 | 4 | 0.4395 − 0.4004 = +0.0391 | 0.4395 − 0.4316 = +0.0078 | 1.0407 |
| related | 4103 | 4 | 0.4336 − 0.4004 = +0.0332 | 0.4336 − 0.4316 = +0.0020 | 0.6786 |
| related | 4104 | 3 | 0.4199 − 0.4004 = +0.0195 | 0.4199 − 0.4316 = -0.0117 | 0.8244 |
| unrelated | 4101 | 4 | 0.4062 − 0.4004 = +0.0059 | 0.4062 − 0.4316 = -0.0254 | 0.9250 |
| unrelated | 4102 | 4 | 0.4258 − 0.4004 = +0.0254 | 0.4258 − 0.4316 = -0.0059 | 0.8268 |
| unrelated | 4103 | 3 | 0.4043 − 0.4004 = +0.0039 | 0.4043 − 0.4316 = -0.0273 | 0.6144 |
| unrelated | 4104 | 4 | 0.3848 − 0.4004 = -0.0156 | 0.3848 − 0.4316 = -0.0469 | 1.0831 |

| Comparison (post values) | J_H | J_R | J_U |
|---|---|---|---|
| unrelated − related, seed 4101 | 0.4062 − 0.4160 = -0.0098 | 0.7991 − 0.8036 = -0.0045 | 0.8086 − 0.8008 = +0.0078 |
| unrelated − related, seed 4102 | 0.4258 − 0.4395 = -0.0137 | 0.8058 − 0.7969 = +0.0089 | 0.8223 − 0.7637 = +0.0586 |
| unrelated − related, seed 4103 | 0.4043 − 0.4336 = -0.0293 | 0.7969 − 0.8080 = -0.0112 | 0.7949 − 0.8008 = -0.0059 |
| unrelated − related, seed 4104 | 0.3848 − 0.4199 = -0.0352 | 0.8036 − 0.8058 = -0.0022 | 0.8027 − 0.7734 = +0.0293 |
| mean unrelated − mean related (4 seeds) | 0.4053 − 0.4272 = -0.0220 | 0.8013 − 0.8036 = -0.0022 | 0.8071 − 0.7847 = +0.0225 |
| unrelated − hard, seed 4101 | 0.4062 − 0.4102 = -0.0039 | 0.7991 − 0.7879 = +0.0112 | 0.8086 − 0.7754 = +0.0332 |
| unrelated − hard, seed 4102 | 0.4258 − 0.4199 = +0.0059 | 0.8058 − 0.8080 = -0.0022 | 0.8223 − 0.7988 = +0.0234 |
| unrelated − hard, seed 4103 | 0.4043 − 0.4062 = -0.0020 | 0.7969 − 0.8080 = -0.0112 | 0.7949 − 0.8203 = -0.0254 |
| unrelated − hard, seed 4104 | 0.3848 − 0.4219 = -0.0371 | 0.8036 − 0.8058 = -0.0022 | 0.8027 − 0.7812 = +0.0215 |
| mean unrelated − mean hard (4 seeds) | 0.4053 − 0.4146 = -0.0093 | 0.8013 − 0.8025 = -0.0011 | 0.8071 − 0.7939 = +0.0132 |

| Spread of single rows | J_H min–max (SD) | J_R min–max (SD) | J_U min–max (SD) |
|---|---|---|---|
| hard | 0.4062–0.4219 (0.0075; n=4) | 0.7879–0.8080 (0.0097; n=4) | 0.7754–0.8203 (0.0202; n=4) |
| related | 0.4160–0.4395 (0.0111; n=4) | 0.7969–0.8080 (0.0048; n=4) | 0.7637–0.8008 (0.0190; n=4) |
| unrelated | 0.3848–0.4258 (0.0168; n=4) | 0.7969–0.8058 (0.0041; n=4) | 0.7949–0.8223 (0.0115; n=4) |
| no-update rows | 0.4004–0.4316 (0.0180; n=3) | 0.8080–0.8281 (0.0116; n=3) | 0.7930–0.8066 (0.0079; n=3) |

Draws with zero mixed groups: 0

## r64: source /data/user_data/vineethb/qis/ckpt/qis-main-R-s101-sym/round_0064 (round 64)

| No-update row | J_H | J_R | J_U |
|---|---|---|---|
| none_original | 0.6055 | 0.9018 | 0.9395 |
| none_rerun | 0.6309 | 0.8929 | 0.9512 |
| none_cycled | 0.6309 | 0.8929 | 0.9512 |

| Branch | Seed | Mixed groups (of 4) | ΔJ_H = post − pre (original) | ΔJ_H vs rerun | grad norm |
|---|---|---|---|---|---|
| hard | 4101 | 2 | 0.5996 − 0.6055 = -0.0059 | 0.5996 − 0.6309 = -0.0312 | 0.2719 |
| hard | 4102 | 3 | 0.6074 − 0.6055 = +0.0020 | 0.6074 − 0.6309 = -0.0234 | 0.3497 |
| hard | 4103 | 2 | 0.6172 − 0.6055 = +0.0117 | 0.6172 − 0.6309 = -0.0137 | 0.3035 |
| hard | 4104 | 1 | 0.6016 − 0.6055 = -0.0039 | 0.6016 − 0.6309 = -0.0293 | 0.2417 |
| related | 4101 | 0 | 0.6113 − 0.6055 = +0.0059 | 0.6113 − 0.6309 = -0.0195 | 0.0000 |
| related | 4102 | 0 | 0.6113 − 0.6055 = +0.0059 | 0.6113 − 0.6309 = -0.0195 | 0.0000 |
| related | 4103 | 2 | 0.6211 − 0.6055 = +0.0156 | 0.6211 − 0.6309 = -0.0098 | 0.5011 |
| related | 4104 | 1 | 0.6191 − 0.6055 = +0.0137 | 0.6191 − 0.6309 = -0.0117 | 0.1983 |
| unrelated | 4101 | 1 | 0.5996 − 0.6055 = -0.0059 | 0.5996 − 0.6309 = -0.0312 | 0.4726 |
| unrelated | 4102 | 2 | 0.6328 − 0.6055 = +0.0273 | 0.6328 − 0.6309 = +0.0020 | 0.7593 |
| unrelated | 4103 | 0 | 0.6113 − 0.6055 = +0.0059 | 0.6113 − 0.6309 = -0.0195 | 0.0000 |
| unrelated | 4104 | 2 | 0.5918 − 0.6055 = -0.0137 | 0.5918 − 0.6309 = -0.0391 | 0.5891 |

| Comparison (post values) | J_H | J_R | J_U |
|---|---|---|---|
| unrelated − related, seed 4101 | 0.5996 − 0.6113 = -0.0117 | 0.8973 − 0.8951 = +0.0022 | 0.9395 − 0.9316 = +0.0078 |
| unrelated − related, seed 4102 | 0.6328 − 0.6113 = +0.0215 | 0.8929 − 0.8951 = -0.0022 | 0.9297 − 0.9316 = -0.0020 |
| unrelated − related, seed 4103 | 0.6113 − 0.6211 = -0.0098 | 0.8951 − 0.9018 = -0.0067 | 0.9316 − 0.9297 = +0.0020 |
| unrelated − related, seed 4104 | 0.5918 − 0.6191 = -0.0273 | 0.8839 − 0.8929 = -0.0089 | 0.9395 − 0.9160 = +0.0234 |
| mean unrelated − mean related (4 seeds) | 0.6089 − 0.6157 = -0.0068 | 0.8923 − 0.8962 = -0.0039 | 0.9351 − 0.9272 = +0.0078 |
| unrelated − hard, seed 4101 | 0.5996 − 0.5996 = +0.0000 | 0.8973 − 0.8862 = +0.0112 | 0.9395 − 0.9336 = +0.0059 |
| unrelated − hard, seed 4102 | 0.6328 − 0.6074 = +0.0254 | 0.8929 − 0.8862 = +0.0067 | 0.9297 − 0.9316 = -0.0020 |
| unrelated − hard, seed 4103 | 0.6113 − 0.6172 = -0.0059 | 0.8951 − 0.9018 = -0.0067 | 0.9316 − 0.9297 = +0.0020 |
| unrelated − hard, seed 4104 | 0.5918 − 0.6016 = -0.0098 | 0.8839 − 0.8862 = -0.0022 | 0.9395 − 0.9277 = +0.0117 |
| mean unrelated − mean hard (4 seeds) | 0.6089 − 0.6064 = +0.0024 | 0.8923 − 0.8901 = +0.0022 | 0.9351 − 0.9307 = +0.0044 |

| Spread of single rows | J_H min–max (SD) | J_R min–max (SD) | J_U min–max (SD) |
|---|---|---|---|
| hard | 0.5996–0.6172 (0.0079; n=4) | 0.8862–0.9018 (0.0078; n=4) | 0.9277–0.9336 (0.0025; n=4) |
| related | 0.6113–0.6211 (0.0051; n=4) | 0.8929–0.9018 (0.0039; n=4) | 0.9160–0.9316 (0.0075; n=4) |
| unrelated | 0.5918–0.6328 (0.0179; n=4) | 0.8839–0.8973 (0.0059; n=4) | 0.9297–0.9395 (0.0051; n=4) |
| no-update rows | 0.6055–0.6309 (0.0147; n=3) | 0.8929–0.9018 (0.0052; n=3) | 0.9395–0.9512 (0.0068; n=3) |

Draws with zero mixed groups: 3: related 4101, related 4102, unrelated 4103; identical per-problem counts on all three probes: True

A minibatch whose 4 groups all have equal rewards still takes an AdamW step. Every advantage is 0 (grpo.py:22-23), so every gradient is exactly 0 (recorded grad_norm_preclip 0.0), but diag_local.one_update calls learner.step() unconditionally (diag_local.py:90-93) and torch AdamW (learner.py:50, betas 0.9/0.999, weight_decay 0) moves the weights by lr * m_hat / (sqrt(v_hat) + eps) from its stored moments. From a saved optimizer state (r64) this is a non-zero 'momentum-only' step. From a fresh optimizer (r0, zero moments) the same step would leave the weights unchanged. The training loop does the same (run.py:310-319: learner.step() for every minibatch).

The U-s101 round-64 checkpoint (model and optimizer state) was not retained, so no local-update diagnostic from U-s101 exists (REPORT.md section 3).
