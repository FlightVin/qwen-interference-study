# Data bundles — study s02 (Nov-15-2026-02)

Bundles are built on compute storage by each run's `bundle` phase (`python -m qis.s02.tasks`) and are not
copied here. This file records their identity so results can be traced.

| bundle | path (Babel compute) | SHA-256 | spec | source of the hash |
|---|---|---|---|---|
| s02-v1 | `/data/user_data/vineethb/qis/s02/data/s02-v1.json` | `a761b98149f326e955e4bc461b90674e27ce72ab8a9329970bb1898a4d5b9bf1` | `add9:d4,add9:d6,add9:d8,caesar:s4,caesar:s8,caesar:s12,mbpp:all,arc:all,countdown:n3,countdown:n4,terse10:k2:3000,add10:d14`; n_train 4000, n_eval 64 | `config.json` of qis-s02-j-caesar-terse-s1 (job 11168639) |

Public sources: MBPP = `google-research-datasets/mbpp` (sanitized); ARC = ARC-Challenge. Synthetic families are
generated from seeded streams in `src/qis/s02/tasks.py`.
